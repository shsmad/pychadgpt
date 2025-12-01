"""
ChadGPT API Client - улучшенная версия с автогенерацией методов,
кастомными исключениями, логированием и валидацией через Pydantic.
"""


import contextlib
import json
import logging
import warnings

from collections.abc import Callable
from enum import Enum
from typing import Any, TypeVar, cast

import requests

from pydantic import BaseModel
from requests import Response, Session

from pychadgpt.exceptions import (
    ChadGPTAPIError,
    ChadGPTConnectionError,
    ChadGPTError,
    ChadGPTHTTPError,
    ChadGPTJSONDecodeError,
    ChadGPTTimeoutError,
    ChadGPTValidationError,
)
from pychadgpt.image_models_config import MODEL_VALIDATORS
from pychadgpt.models import (
    AskParameters,
    ChatHistoryMessage,
    ChatResponse,
    CheckResponse,
    ImagineParameters,
    ImagineResponse,
    WordsResponse,
)

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO) -> None:
    """
    Настраивает логирование для библиотеки.

    Args:
        level: Уровень логирования (по умолчанию INFO)

    Example:
        >>> from pychadgpt import setup_logging
        >>> setup_logging(logging.DEBUG)
    """
    logging.basicConfig(level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


# Определяем TypeVar для универсального типа возвращаемого значения.
# `bound=TypedDict` гарантирует, что T будет TypedDict.
TResponse = TypeVar("TResponse", bound=BaseModel)


class HTTPMethod(str, Enum):
    """Поддерживаемые HTTP методы для запросов к API."""

    GET = "get"
    POST = "post"


# Константы для валидации и обработки
MAX_PROMPT_LENGTH = 1000
DEFAULT_TIMEOUT_SECONDS = 30

# ============================================================================
# МЕТАКЛАСС ДЛЯ АВТОГЕНЕРАЦИИ МЕТОДОВ
# ============================================================================


class ModelMethodsMeta(type):
    """
    Метакласс для автоматической генерации методов ask_* для каждой модели.

    Создает методы вида ask_gpt5(), ask_claude_4_5_sonnet() и т.д.
    на основе списка моделей из атрибута _MODELS класса.
    """

    def __new__(mcs, name: str, bases: tuple[type, ...], namespace: dict[str, Any]) -> type:
        cls = super().__new__(mcs, name, bases, namespace)

        # Генерируем методы только для ChadGPTClient
        if name == "ChadGPTClient" and "_MODELS" in namespace:
            models = namespace["_MODELS"]
            deprecated_models = namespace.get("_DEPRECATED_MODELS", set())

            for model_name in models:
                method_name = f"ask_{model_name.replace('-', '_').replace('.', '_')}"

                # Создаем метод динамически
                def create_method(model: str, is_deprecated: bool, m_name: str) -> Callable[..., ChatResponse]:
                    def method(
                        self: "ChadGPTClient",
                        message: str,
                        history: list[ChatHistoryMessage] | None = None,
                        temperature: float | None = None,
                        max_tokens: int | None = None,
                        images: list[str] | None = None,
                    ) -> ChatResponse:
                        return self.ask(model, message, history, temperature, max_tokens, images)

                    docstring = f"""
                    Отправляет запрос модели {model}.
                    {" (УСТАРЕВШАЯ - рекомендуется использовать более новую модель)" if is_deprecated else ""}

                    Args:
                        message: Запрос к нейросети
                        history: История предыдущих сообщений
                        temperature: Температура генерации (0-2)
                        max_tokens: Максимальное количество токенов
                        images: Список изображений (URL или base64)

                    Returns:
                        ChatResponse: Ответ от модели

                    Raises:
                        ChadGPTValidationError: При неверных параметрах
                        ChadGPTConnectionError: При ошибке соединения
                        ChadGPTTimeoutError: При превышении таймаута

                    Example:
                        >>> client = ChadGPTClient("your-api-key")
                        >>> response = client.{m_name}("Привет!")
                        >>> if response.is_success:
                        ...     print(response.response)
                    """

                    method.__doc__ = docstring
                    method.__name__ = m_name
                    return method

                is_deprecated = model_name in deprecated_models
                setattr(cls, method_name, create_method(model_name, is_deprecated, method_name))

        return cls

class ChadGPTBaseClient:
    """
    Базовый клиент для взаимодействия с ChadGPT API.

    Предоставляет общие методы для HTTP-запросов и управления сессией.

    Example:
        >>> with ChadGPTBaseClient("your-api-key") as client:
        ...     stat = client.get_stat_info()
        ...     print(f"Осталось слов: {stat.remaining_words}")
    """

    BASE_URL = "https://ask.chadgpt.ru/api/public/"

    def __init__(self, api_key: str):
        """
        Инициализирует клиент ChadGPT.

        Args:
            api_key (str): Ваш персональный API-ключ.

        Raises:
            ChadGPTValidationError: Если API ключ пустой
        """

        if not api_key or not api_key.strip():
            raise ChadGPTValidationError("API key не может быть пустым.", "INVALID_API_KEY")

        self.api_key = api_key
        self._session = Session()
        logger.info("ChadGPT клиент инициализирован")

    def close(self) -> None:
        """Закрывает HTTP-сессию и освобождает соединения."""
        self._session.close()
        logger.info("ChadGPT клиент закрыт")

    def __enter__(self) -> "ChadGPTBaseClient":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def _send_request(
        self,
        response_type: type[TResponse],
        url: str,
        method: str = "post",
        headers: dict[str, Any] | None = None,
        timeout: int | None = None,
        payload: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> TResponse:
        """
        Отправляет HTTP-запрос к API.

        Args:
            response_type: Тип ожидаемого ответа (Pydantic модель)
            url: URL для запроса
            method: HTTP метод (get/post)
            headers: Дополнительные заголовки
            timeout: Таймаут запроса в секундах
            payload: Тело запроса (для POST)
            params: Query параметры (для GET)

        Returns:
            Объект response_type с данными ответа

        Raises:
            ChadGPTHTTPError: При HTTP ошибке
            ChadGPTConnectionError: При ошибке соединения
            ChadGPTTimeoutError: При превышении таймаута
            ChadGPTJSONDecodeError: При ошибке парсинга JSON
        """

        method_normalized = method.lower()
        try:
            http_method = HTTPMethod(method_normalized)
        except ValueError:
            allowed_methods = ", ".join(sorted(m.value for m in HTTPMethod))
            raise ChadGPTValidationError(
                f"Неподдерживаемый HTTP метод '{method}'. Допустимые: {allowed_methods}", "INVALID_HTTP_METHOD"
            ) from None

        merged_headers = {"Content-Type": "application/json"}
        if headers:
            merged_headers.update(headers)

        request_timeout = timeout if timeout is not None else DEFAULT_TIMEOUT_SECONDS
        response: Response | None = None

        logger.debug(
            f"Отправка {http_method.value.upper()} запроса к {url} "
            f"(timeout={request_timeout}s, payload_size={len(str(payload)) if payload else 0})"
        )

        try:
            request_kwargs: dict[str, Any] = {
                "url": url,
                "headers": merged_headers,
                "timeout": request_timeout,
            }
            if params:
                request_kwargs["params"] = params
            if payload is not None and http_method == HTTPMethod.POST:
                request_kwargs["json"] = payload
            elif payload is not None and http_method == HTTPMethod.GET:
                # Для GET запросов переносим тело в query string, чтобы не нарушать HTTP-стандарты.
                request_kwargs["params"] = {**params} if params else {}
                request_kwargs["params"].update(payload)

            response = self._session.request(method=http_method.value, **request_kwargs)
            logger.debug(f"Получен ответ: status={response.status_code}, size={len(response.content)} bytes")
            response.raise_for_status()  # Вызовет HTTPError для статусов 4xx/5xx
            json_data = response.json()
            logger.debug(f"JSON ответ: {json.dumps(json_data, ensure_ascii=False)[:200]}...")

            # Валидируем через Pydantic если это BaseModel
            if issubclass(response_type, BaseModel):
                return response_type.model_validate(json_data)  # type: ignore

            return cast(TResponse, json_data)
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP ошибка: {http_err}")
            # Попытка парсинга ошибки из ответа API, даже если это HTTP-ошибка
            with contextlib.suppress(json.JSONDecodeError):
                if response is not None:
                    json_data = response.json()
                    if issubclass(response_type, BaseModel):
                        return response_type.model_validate(json_data)  # type: ignore
                    return cast(TResponse, response.json())
                raise

            raise ChadGPTHTTPError(
                f"HTTP ошибка: {http_err}. Ответ: {response.text if response else 'N/A'}",
                status_code=response.status_code if response else 0,
                error_code="HTTP_ERROR",
            ) from None

        except requests.exceptions.ConnectionError as conn_err:
            logger.error(f"Ошибка соединения: {conn_err}")
            raise ChadGPTConnectionError(f"Не удалось подключиться к API: {conn_err}", "CONNECTION_ERROR") from None

        except requests.exceptions.Timeout as timeout_err:
            logger.error(f"Таймаут: {timeout_err}")
            raise ChadGPTTimeoutError(
                f"Превышено время ожидания ({request_timeout}s): {timeout_err}", "TIMEOUT_ERROR"
            ) from None

        except requests.exceptions.RequestException as req_err:
            logger.error(f"Ошибка запроса: {req_err}")
            raise ChadGPTError(f"Непредвиденная ошибка при запросе: {req_err}", "REQUEST_ERROR") from None

        except json.JSONDecodeError as json_err:
            logger.error(f"Ошибка парсинга JSON: {json_err}")
            raise ChadGPTJSONDecodeError(
                f"Не удалось декодировать JSON: {json_err}. Ответ: {response.text if response else 'N/A'}",
                "JSON_DECODE_ERROR",
            ) from None

    def get_stat_info(self, timeout: int | None = None) -> WordsResponse:
        """
        Позволяет определить общее количество слов, использованных,
        зарезервированных (в текущих генерациях контента) и оставшихся.

        Args:
            timeout: Таймаут запроса в секундах

        Returns:
            WordsResponse: Статистика по словам

        Example:
            >>> client = ChadGPTClient("your-api-key")
            >>> stat = client.get_stat_info()
            >>> print(f"Использовано: {stat.used_words}/{stat.total_words}")
            >>> print(f"Осталось: {stat.remaining_words}")
        """

        url = f"{self.BASE_URL}words"

        payload: dict[str, Any] = {
            "api_key": self.api_key,
        }
        response = self._send_request(
            response_type=WordsResponse, url=url, method="post", timeout=timeout, payload=payload
        )
        if not response.is_success and response.error_code:
            raise ChadGPTAPIError(response.error_message or "Неизвестная ошибка API", response.error_code)

        return response


# ============================================================================
# КЛИЕНТ ДЛЯ CHAT API
# ============================================================================


class ChadGPTClient(ChadGPTBaseClient, metaclass=ModelMethodsMeta):
    """
    Клиент для взаимодействия с ChadGPT API.

    Позволяет отправлять запросы различным моделям GPT и Claude,
    передавать историю сообщений, изображения и управлять параметрами запроса.

    Example:
        >>> # Базовое использование
        >>> client = ChadGPTClient("your-api-key")
        >>> response = client.ask_gpt5("Расскажи про Python")
        >>> if response.is_success:
        ...     print(response.response)

        >>> # С историей и параметрами
        >>> history = [
        ...     ChatHistoryMessage(role="user", content="Привет!"),
        ...     ChatHistoryMessage(role="assistant", content="Здравствуйте!")
        ... ]
        >>> response = client.ask_claude_4_5_sonnet(
        ...     message="Продолжи разговор",
        ...     history=history,
        ...     temperature=0.7,
        ...     max_tokens=1000
        ... )

        >>> # Context manager
        >>> with ChadGPTClient("your-api-key") as client:
        ...     response = client.ask_gemini_2_5_pro("Hello!")
    """

    # Словарь сопоставления названий моделей и их API-путей
    _MODELS = {
        "gpt-5",
        "gpt-5-mini",
        "gpt-5-nano",
        "gpt-4o-mini",
        "gpt-4o",
        "claude-3-haiku",
        "claude-3-opus",
        "claude-4.1-opus",
        "claude-4.5-sonnet",
        "claude-3.7-sonnet-thinking",
        "gemini-2.0-flash",
        "gemini-2.5-pro",
        "deepseek-v3.1",
    }

    # Модели, которые помечены как устаревшие в описании API
    _DEPRECATED_MODELS = {
        "gpt-4o-mini",
        "gpt-4o",
        "claude-3-haiku",
        "claude-3-opus",
        "claude-3.7-sonnet-thinking",
    }

    def ask(
        self,
        model_name: str,
        message: str,
        history: list[ChatHistoryMessage] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        images: list[str] | None = None,
        timeout: int | None = None,
    ) -> ChatResponse:
        """
        Отправляет POST-запрос к API ChadGPT.

        Args:
            model_name (str): Название модели (например, "gpt-5", "claude-4-1-opus").
            message (str): Запрос к нейросети.
            history (list, optional): История предыдущих сообщений. Defaults to None.
                Формат: [{"role": "user", "content": "Текст"}, {"role": "assistant", "content": "Текст"}].
            temperature (float, optional): Настройка температуры GPT (от 0 до 2). Defaults to None.
            max_tokens (int, optional): Максимальная длина ответа в токенах. Defaults to None.
            images (list, optional): Список URL или base64 Data URL изображений. Defaults to None.
            timeout: Таймаут запроса в секундах

        Returns:
            ChatResponse: Ответ от модели

        Raises:
            ChadGPTValidationError: При неверных параметрах
            ChadGPTConnectionError: При ошибке соединения
            ChadGPTTimeoutError: При превышении таймаута
            ChadGPTAPIError: При ошибке от API

        Example:
            >>> client = ChadGPTClient("your-api-key")
            >>> response = client.ask(
            ...     model_name="gpt-5",
            ...     message="Что такое Python?",
            ...     temperature=0.7,
            ...     max_tokens=500
            ... )
        """

        if model_name not in self._MODELS:
            available = ", ".join(sorted(self._MODELS))
            raise ChadGPTValidationError(
                f"Неподдерживаемая модель: '{model_name}'. Доступные: {available}", "INVALID_MODEL"
            )

        if model_name in self._DEPRECATED_MODELS:
            warnings.warn(
                f"Модель '{model_name}' устарела и может быть удалена в будущем. "
                "Рекомендуется использовать более новые модели.",
                DeprecationWarning,
                stacklevel=2,
            )
            logger.warning(f"Использована устаревшая модель: {model_name}")

        # Валидация параметров через Pydantic
        try:
            params = AskParameters(
                message=message, history=history, temperature=temperature, max_tokens=max_tokens, images=images
            )
        except Exception as e:
            raise ChadGPTValidationError(f"Ошибка валидации параметров: {e}", "VALIDATION_ERROR") from e

        url = f"{self.BASE_URL}{model_name}"

        payload: dict[str, Any] = {
            "message": params.message,
            "api_key": self.api_key,
        }
        if params.history is not None:
            payload["history"] = [msg.model_dump(exclude_none=True) for msg in params.history]
        if params.temperature is not None:
            payload["temperature"] = params.temperature
        if params.max_tokens is not None:
            payload["max_tokens"] = params.max_tokens
        if params.images is not None:
            payload["images"] = params.images

        response = self._send_request(
            response_type=ChatResponse, url=url, method="post", timeout=timeout, payload=payload
        )

        if not response.is_success and response.error_code:
            logger.error(f"API вернул ошибку: {response.error_code} - {response.error_message}")
            raise ChadGPTAPIError(response.error_message or "Неизвестная ошибка API", response.error_code)

        if response.deprecated:
            logger.warning(f"Модель устарела: {response.deprecated}")

        return response


# ============================================================================
# КЛИЕНТ ДЛЯ IMAGE API
# ============================================================================


class ChadGPTImageClient(ChadGPTBaseClient):
    """
    Клиент для работы с генерацией изображений через ChadGPT API.

    Поддерживает различные модели генерации изображений и позволяет
    отслеживать статус генерации.

    Example:
        >>> client = ChadGPTImageClient("your-api-key")
        >>>
        >>> # Запускаем генерацию
        >>> response = client.imagine(
        ...     model_name="imagen-4",
        ...     prompt="Beautiful sunset over mountains"
        ... )
        >>>
        >>> if response.status == "starting":
        ...     content_id = response.content_id
        ...
        ...     # Проверяем статус
        ...     import time
        ...     while True:
        ...         status = client.check_status(content_id)
        ...         if status.status == "completed":
        ...             print(f"Готово! Изображения: {status.output}")
        ...             break
        ...         elif status.status == "failed":
        ...             print(f"Ошибка: {status.error_message}")
        ...             break
        ...         time.sleep(2)
    """

    @property
    def check_url(self) -> str:
        """URL для проверки статуса генерации изображений."""
        return f"{self.BASE_URL}check"

    def _validate_imagine_params(self, model_name: str, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """
        Валидирует параметры для генерации изображения.

        Args:
            model_name: Имя модели
            prompt: Текстовое описание
            kwargs: Дополнительные параметры модели

        Returns:
            dict: Валидированный payload

        Raises:
            ChadGPTValidationError: При неверных параметрах
        """

        if model_name not in MODEL_VALIDATORS:
            available = ", ".join(sorted(MODEL_VALIDATORS.keys()))
            raise ChadGPTValidationError(
                f"Неподдерживаемая модель: '{model_name}'. Доступные: {available}", "INVALID_MODEL"
            )

        # Базовая валидация через Pydantic
        try:
            base_params = ImagineParameters(prompt=prompt)
        except Exception as e:
            raise ChadGPTValidationError(f"Ошибка валидации промпта: {e}", "VALIDATION_ERROR") from e

        payload = {"api_key": self.api_key, "prompt": base_params.prompt}

        # Валидация специфичных для модели параметров
        validator = MODEL_VALIDATORS[model_name]
        try:
            params = validator(**kwargs)
            params_dict = params.model_dump(exclude_none=True)
            payload.update(params_dict)
        except Exception as e:
            raise ChadGPTValidationError(
                f"Ошибка валидации параметров модели: {e}", "MODEL_PARAMS_VALIDATION_ERROR"
            ) from e

        return payload

    def imagine(self, model_name: str, prompt: str, timeout: int | None = None, **kwargs: Any) -> ImagineResponse:
        """
        Запускает генерацию изображения.

        Args:
            model_name: Имя модели (например, "imagen-4", "mj-6")
            prompt: Текстовое описание изображения (1-1000 символов)
            timeout: Таймаут запроса в секундах
            **kwargs: Параметры, специфичные для модели

        Returns:
            ImagineResponse: Ответ с content_id и статусом

        Raises:
            ChadGPTValidationError: При неверных параметрах
            ChadGPTAPIError: При ошибке от API

        Example:
            >>> client = ChadGPTImageClient("your-api-key")
            >>> response = client.imagine(
            ...     model_name="imagen-4",
            ...     prompt="A cat wearing a hat",
            ...     aspect_ratio="1:1",
            ...     number_of_images=2
            ... )
            >>> print(response.content_id)
        """

        payload = self._validate_imagine_params(model_name, prompt, **kwargs)
        url = f"{self.BASE_URL}{model_name}/imagine"

        response = self._send_request(
            response_type=ImagineResponse, url=url, method="post", payload=payload, timeout=timeout
        )

        if response.status == "failed" and response.error_code:
            logger.error(f"Генерация завершилась с ошибкой: {response.error_code} - {response.error_message}")
            raise ChadGPTAPIError(response.error_message or "Ошибка генерации изображения", response.error_code)

        logger.info(f"Генерация запущена: content_id={response.content_id}")
        return response

    def check_status(self, content_id: str, timeout: int | None = None) -> CheckResponse:
        """
        Проверяет статус генерации изображения.

        Args:
            content_id: ID контента из ответа imagine()
            timeout: Таймаут запроса в секундах

        Returns:
            CheckResponse: Статус генерации и ссылки на изображения

        Raises:
            ChadGPTValidationError: Если content_id пустой
            ChadGPTAPIError: При ошибке от API

        Example:
            >>> status = client.check_status("abc123")
            >>> if status.status == "completed":
            ...     for url in status.output:
            ...         print(f"Изображение: {url}")
            >>> elif status.status == "pending":
            ...     print("Генерация в процессе...")
        """

        if not isinstance(content_id, str) or not content_id.strip():
            raise ChadGPTValidationError("Content ID должен быть непустой строкой", "INVALID_CONTENT_ID")

        payload = {"api_key": self.api_key, "content_id": content_id}
        response = self._send_request(
            response_type=CheckResponse, url=self.check_url, method="get", params=payload, timeout=timeout
        )

        if response.status == "failed" and response.error_code:
            logger.error(f"Генерация завершилась с ошибкой: {response.error_code} - {response.error_message}")
            raise ChadGPTAPIError(response.error_message or "Ошибка генерации изображения", response.error_code)

        logger.debug(f"Статус генерации {content_id}: {response.status}")
        return response
