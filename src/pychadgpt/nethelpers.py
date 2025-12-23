import contextlib
import json
import time

from logging import Logger
from typing import Any, cast

import requests

from pydantic import BaseModel
from requests import Response, Session

from pychadgpt.constants import DEFAULT_MAX_RETRIES
from pychadgpt.exceptions import (
    ChadGPTConnectionError,
    ChadGPTError,
    ChadGPTHTTPError,
    ChadGPTJSONDecodeError,
    ChadGPTTimeoutError,
    ChadGPTValidationError,
)
from pychadgpt.models import HTTPMethod, TResponse


def should_retry(max_retries: int, exception: Exception, attempt: int) -> bool:
    """
    Определяет, нужно ли повторить запрос при данной ошибке.

    Args:
        exception: Исключение, которое произошло
        attempt: Номер попытки (начиная с 1)

    Returns:
        True, если нужно повторить запрос
    """
    if attempt >= max_retries:
        return False

    # Повторяем при временных ошибках
    if isinstance(exception, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
        return True

    # Повторяем при HTTP ошибках 5xx (серверные ошибки)
    if isinstance(exception, requests.exceptions.HTTPError) and (
        hasattr(exception, "response") and exception.response is not None
    ):
        status_code = exception.response.status_code
        return 500 <= status_code < 600

    return False


def calculate_backoff(attempt: int) -> int:
    """
    Вычисляет время задержки перед повторной попыткой (exponential backoff).

    Args:
        attempt: Номер попытки (начиная с 1)

    Returns:
        Время задержки в секундах (максимум 60 секунд)
    """
    return min(int(2 ** (attempt - 1)), 60)


def validate_http_method(method: str) -> HTTPMethod:
    """
    Валидирует и нормализует HTTP метод.

    Args:
        method: HTTP метод (строка)

    Returns:
        HTTPMethod: Валидированный enum метод

    Raises:
        ChadGPTValidationError: Если метод не поддерживается
    """
    method_normalized = method.lower()
    try:
        return HTTPMethod(method_normalized)
    except ValueError:
        allowed_methods = ", ".join(sorted(m.value for m in HTTPMethod))
        raise ChadGPTValidationError(
            f"Неподдерживаемый HTTP метод '{method}'. Допустимые: {allowed_methods}", "INVALID_HTTP_METHOD"
        ) from None


def prepare_headers(headers: dict[str, Any] | None) -> dict[str, Any]:
    """
    Подготавливает заголовки запроса.

    Args:
        headers: Дополнительные заголовки или None

    Returns:
        dict: Объединенные заголовки с Content-Type по умолчанию
    """
    merged_headers = {"Content-Type": "application/json"}
    if headers:
        merged_headers.update(headers)
    return merged_headers


def prepare_request_kwargs(
    http_method: HTTPMethod,
    url: str,
    headers: dict[str, Any],
    timeout: int,
    payload: dict[str, Any] | None,
    params: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Подготавливает параметры для HTTP запроса.

    Args:
        http_method: HTTP метод
        url: URL запроса
        headers: Заголовки
        timeout: Таймаут в секундах
        payload: Тело запроса
        params: Query параметры

    Returns:
        dict: Параметры для requests.Session.request()
    """
    request_kwargs: dict[str, Any] = {
        "url": url,
        "headers": headers,
        "timeout": timeout,
    }

    if params:
        request_kwargs["params"] = params

    if payload is not None:
        if http_method == HTTPMethod.POST:
            request_kwargs["json"] = payload
        elif http_method == HTTPMethod.GET:
            # Для GET запросов переносим тело в query string
            request_kwargs["params"] = {**params} if params else {}
            request_kwargs["params"].update(payload)

    return request_kwargs


def parse_response(logger: Logger, response: Response, response_type: type[TResponse]) -> TResponse:
    """
    Парсит HTTP ответ в объект указанного типа.

    Args:
        response: HTTP ответ от requests
        response_type: Тип ожидаемого ответа (Pydantic модель)

    Returns:
        Объект response_type с данными ответа
    """
    json_data = response.json()
    logger.debug(f"JSON ответ: {json.dumps(json_data, ensure_ascii=False)[:200]}...")

    if issubclass(response_type, BaseModel):
        return response_type.model_validate(json_data)  # type: ignore

    return cast(TResponse, json_data)


def try_parse_error_response(
    logger: Logger, response: Response | None, response_type: type[TResponse]
) -> TResponse | None:
    """
    Пытается распарсить ответ с ошибкой как валидный JSON.

    Некоторые API возвращают JSON даже при HTTP ошибках.

    Args:
        response: HTTP ответ или None
        response_type: Тип ожидаемого ответа

    Returns:
        Распарсенный ответ или None, если не удалось
    """
    if response is None:
        return None

    with contextlib.suppress(json.JSONDecodeError):
        return parse_response(logger=logger, response=response, response_type=response_type)

    return None


def handle_request_exception(
    exception: Exception, response: Response | None, response_type: type[TResponse], request_timeout: int
) -> ChadGPTError:
    """
    Преобразует исключения requests в кастомные исключения ChadGPT.

    Args:
        exception: Исключение от requests
        response: HTTP ответ (если есть)
        response_type: Тип ожидаемого ответа
        request_timeout: Таймаут запроса

    Returns:
        Кастомное исключение ChadGPT
    """
    if isinstance(exception, requests.exceptions.HTTPError):
        return ChadGPTHTTPError(
            f"HTTP ошибка: {exception}. Ответ: {response.text if response else 'N/A'}",
            status_code=response.status_code if response else 0,
            error_code="HTTP_ERROR",
        )

    if isinstance(exception, requests.exceptions.ConnectionError):
        return ChadGPTConnectionError(f"Не удалось подключиться к API: {exception}", "CONNECTION_ERROR")

    if isinstance(exception, requests.exceptions.Timeout):
        return ChadGPTTimeoutError(f"Превышено время ожидания ({request_timeout}s): {exception}", "TIMEOUT_ERROR")

    if isinstance(exception, json.JSONDecodeError):
        return ChadGPTJSONDecodeError(
            f"Не удалось декодировать JSON: {exception}. Ответ: {response.text if response else 'N/A'}",
            "JSON_DECODE_ERROR",
        )

    return ChadGPTError(f"Непредвиденная ошибка при запросе: {exception}", "REQUEST_ERROR")


def execute_single_request(
    logger: Logger,
    session: Session,
    http_method: HTTPMethod,
    request_kwargs: dict[str, Any],
    response_type: type[TResponse],
) -> tuple[Response, TResponse]:
    """
    Выполняет один HTTP запрос без retry логики.

    Args:
        http_method: HTTP метод
        request_kwargs: Параметры запроса
        response_type: Тип ожидаемого ответа

    Returns:
        Кортеж (Response, распарсенный ответ)

    Raises:
        requests.exceptions.HTTPError: При HTTP ошибке
        requests.exceptions.ConnectionError: При ошибке соединения
        requests.exceptions.Timeout: При превышении таймаута
        json.JSONDecodeError: При ошибке парсинга JSON
    """
    response = session.request(method=http_method.value, **request_kwargs)
    logger.debug(f"Получен ответ: status={response.status_code}, size={len(response.content)} bytes")
    response.raise_for_status()  # Вызовет HTTPError для статусов 4xx/5xx
    parsed_response = parse_response(logger=logger, response=response, response_type=response_type)
    return response, parsed_response


def send_request_with_retry(
    logger: Logger,
    session: Session,
    http_method: HTTPMethod,
    request_kwargs: dict[str, Any],
    response_type: type[TResponse],
    request_timeout: int,
    masked_api_key: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> TResponse:
    """
    Выполняет HTTP запрос с retry логикой.

    Args:
        http_method: HTTP метод
        request_kwargs: Параметры запроса
        response_type: Тип ожидаемого ответа
        request_timeout: Таймаут запроса
        masked_api_key: Маскированный API ключ для логирования

    Returns:
        Объект response_type с данными ответа

    Raises:
        ChadGPTHTTPError: При HTTP ошибке после всех попыток
        ChadGPTConnectionError: При ошибке соединения после всех попыток
        ChadGPTTimeoutError: При превышении таймаута после всех попыток
        ChadGPTJSONDecodeError: При ошибке парсинга JSON
    """
    response: Response | None = None
    last_exception: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            response, parsed_response = execute_single_request(
                logger=logger,
                session=session,
                http_method=http_method,
                request_kwargs=request_kwargs,
                response_type=response_type,
            )
            return parsed_response

        except requests.exceptions.HTTPError as http_err:
            last_exception = http_err
            logger.error(f"HTTP ошибка: {http_err} (API key: {masked_api_key})")

            # Пытаемся распарсить ответ как валидный JSON (некоторые API возвращают JSON при ошибках)
            parsed_error_response = try_parse_error_response(logger=logger, response=response, response_type=response_type)
            if parsed_error_response is not None:
                # Type narrowing: после проверки is not None mypy знает, что это TResponse
                return parsed_error_response

            # Если не нужно повторять (4xx ошибка), сразу выбрасываем исключение
            if not should_retry(max_retries=max_retries, exception=http_err, attempt=attempt):
                raise handle_request_exception(http_err, response, response_type, request_timeout) from None

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as network_err:
            last_exception = network_err
            error_type = (
                "Ошибка соединения" if isinstance(network_err, requests.exceptions.ConnectionError) else "Таймаут"
            )
            logger.error(f"{error_type}: {network_err} (API key: {masked_api_key})")

            if not should_retry(max_retries=max_retries, exception=network_err, attempt=attempt):
                raise handle_request_exception(network_err, response, response_type, request_timeout) from None

        except json.JSONDecodeError as json_err:
            # JSON ошибки не повторяем
            logger.error(f"Ошибка парсинга JSON: {json_err}")
            raise handle_request_exception(json_err, response, response_type, request_timeout) from None

        except requests.exceptions.RequestException as req_err:
            last_exception = req_err
            logger.error(f"Ошибка запроса: {req_err}")
            if not should_retry(max_retries=max_retries, exception=req_err, attempt=attempt):
                raise handle_request_exception(req_err, response, response_type, request_timeout) from None

        # Если нужно повторить, ждем перед следующей попыткой
        if attempt < max_retries and should_retry(max_retries=max_retries, exception=last_exception, attempt=attempt):
            backoff = calculate_backoff(attempt)
            logger.warning(f"Повторная попытка {attempt + 1}/{max_retries} через {backoff}s...")
            time.sleep(backoff)

    # Все попытки исчерпаны - выбрасываем последнее исключение
    if last_exception:
        raise handle_request_exception(last_exception, response, response_type, request_timeout) from last_exception

    raise ChadGPTError(f"Ошибка после {max_retries} попыток", "REQUEST_ERROR")
