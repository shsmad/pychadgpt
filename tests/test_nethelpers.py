"""Тесты для вспомогательных функций из nethelpers.py."""

import json

from unittest.mock import Mock

import pytest
import requests

from pytest_mock import MockerFixture

from pychadgpt.exceptions import (
    ChadGPTConnectionError,
    ChadGPTError,
    ChadGPTHTTPError,
    ChadGPTJSONDecodeError,
    ChadGPTTimeoutError,
    ChadGPTValidationError,
)
from pychadgpt.models import ChatResponse, HTTPMethod
from pychadgpt.nethelpers import (
    calculate_backoff,
    execute_single_request,
    handle_request_exception,
    parse_response,
    prepare_headers,
    prepare_request_kwargs,
    send_request_with_retry,
    should_retry,
    try_parse_error_response,
    validate_http_method,
)


class TestShouldRetry:
    """Тесты функции should_retry."""

    def test_should_retry_connection_error(self) -> None:
        """Проверка, что ConnectionError должен повторяться."""
        exception = requests.exceptions.ConnectionError("Connection refused")
        assert should_retry(max_retries=3, exception=exception, attempt=1) is True
        assert should_retry(max_retries=3, exception=exception, attempt=2) is True

    def test_should_retry_timeout(self) -> None:
        """Проверка, что Timeout должен повторяться."""
        exception = requests.exceptions.Timeout("Request timeout")
        assert should_retry(max_retries=3, exception=exception, attempt=1) is True
        assert should_retry(max_retries=3, exception=exception, attempt=2) is True

    def test_should_retry_http_5xx(self) -> None:
        """Проверка, что HTTP 5xx ошибки должны повторяться."""
        mock_response = Mock()
        mock_response.status_code = 500
        exception = requests.exceptions.HTTPError("Internal Server Error")
        exception.response = mock_response

        assert should_retry(max_retries=3, exception=exception, attempt=1) is True
        assert should_retry(max_retries=3, exception=exception, attempt=2) is True

    def test_should_not_retry_http_4xx(self) -> None:
        """Проверка, что HTTP 4xx ошибки не должны повторяться."""
        mock_response = Mock()
        mock_response.status_code = 400
        exception = requests.exceptions.HTTPError("Bad Request")
        exception.response = mock_response

        assert should_retry(max_retries=3, exception=exception, attempt=1) is False

    def test_should_not_retry_after_max_attempts(self) -> None:
        """Проверка, что не повторяем после максимального количества попыток."""
        exception = requests.exceptions.ConnectionError("Connection refused")
        assert should_retry(max_retries=3, exception=exception, attempt=3) is False
        assert should_retry(max_retries=3, exception=exception, attempt=4) is False

    def test_should_not_retry_http_error_without_response(self) -> None:
        """Проверка, что HTTPError без response не повторяем."""
        exception = requests.exceptions.HTTPError("HTTP Error")
        exception.response = None
        assert should_retry(max_retries=3, exception=exception, attempt=1) is False

    def test_should_not_retry_other_exceptions(self) -> None:
        """Проверка, что другие исключения не повторяем."""
        exception = ValueError("Some error")
        assert should_retry(max_retries=3, exception=exception, attempt=1) is False


class TestCalculateBackoff:
    """Тесты функции calculate_backoff."""

    def test_backoff_exponential(self) -> None:
        """Проверка экспоненциального роста backoff."""
        assert calculate_backoff(1) == 1  # 2^0 = 1
        assert calculate_backoff(2) == 2  # 2^1 = 2
        assert calculate_backoff(3) == 4  # 2^2 = 4
        assert calculate_backoff(4) == 8  # 2^3 = 8
        assert calculate_backoff(5) == 16  # 2^4 = 16

    def test_backoff_max_limit(self) -> None:
        """Проверка, что backoff не превышает 60 секунд."""
        # Для attempt=7: 2^6 = 64, но должно быть 60
        assert calculate_backoff(7) == 60
        assert calculate_backoff(10) == 60
        assert calculate_backoff(100) == 60


class TestValidateHTTPMethod:
    """Тесты функции validate_http_method."""

    def test_valid_methods(self) -> None:
        """Проверка валидных HTTP методов."""
        assert validate_http_method("get") == HTTPMethod.GET
        assert validate_http_method("GET") == HTTPMethod.GET
        assert validate_http_method("Get") == HTTPMethod.GET
        assert validate_http_method("post") == HTTPMethod.POST
        assert validate_http_method("POST") == HTTPMethod.POST
        assert validate_http_method("Post") == HTTPMethod.POST

    def test_invalid_method_raises_error(self) -> None:
        """Проверка, что невалидный метод вызывает ошибку."""
        with pytest.raises(ChadGPTValidationError) as exc_info:
            validate_http_method("put")

        assert exc_info.value.error_code == "INVALID_HTTP_METHOD"

        with pytest.raises(ChadGPTValidationError) as exc_info:
            validate_http_method("DELETE")

        assert exc_info.value.error_code == "INVALID_HTTP_METHOD"


class TestPrepareHeaders:
    """Тесты функции prepare_headers."""

    def test_prepare_headers_with_default(self) -> None:
        """Проверка подготовки заголовков с дефолтным Content-Type."""
        headers = prepare_headers(None)
        assert headers == {"Content-Type": "application/json"}

    def test_prepare_headers_with_custom(self) -> None:
        """Проверка подготовки заголовков с кастомными заголовками."""
        custom_headers = {"Authorization": "Bearer token"}
        headers = prepare_headers(custom_headers)
        assert headers["Content-Type"] == "application/json"
        assert headers["Authorization"] == "Bearer token"

    def test_prepare_headers_merges(self) -> None:
        """Проверка, что кастомные заголовки объединяются с дефолтными."""
        custom_headers = {"X-Custom-Header": "value"}
        headers = prepare_headers(custom_headers)
        assert len(headers) == 2
        assert headers["Content-Type"] == "application/json"
        assert headers["X-Custom-Header"] == "value"


class TestPrepareRequestKwargs:
    """Тесты функции prepare_request_kwargs."""

    def test_prepare_post_request(self) -> None:
        """Проверка подготовки POST запроса."""
        kwargs = prepare_request_kwargs(
            http_method=HTTPMethod.POST,
            url="https://example.com/api",
            headers={"Content-Type": "application/json"},
            timeout=30,
            payload={"message": "Hello"},
            params=None,
        )

        assert kwargs["url"] == "https://example.com/api"
        assert kwargs["headers"]["Content-Type"] == "application/json"
        assert kwargs["timeout"] == 30
        assert kwargs["json"] == {"message": "Hello"}
        assert "params" not in kwargs

    def test_prepare_get_request(self) -> None:
        """Проверка подготовки GET запроса."""
        kwargs = prepare_request_kwargs(
            http_method=HTTPMethod.GET,
            url="https://example.com/api",
            headers={"Content-Type": "application/json"},
            timeout=30,
            payload={"api_key": "test-key"},
            params={"param1": "value1"},
        )

        assert kwargs["url"] == "https://example.com/api"
        assert kwargs["timeout"] == 30
        assert "json" not in kwargs
        assert "params" in kwargs
        assert kwargs["params"]["api_key"] == "test-key"
        assert kwargs["params"]["param1"] == "value1"

    def test_prepare_request_without_payload(self) -> None:
        """Проверка подготовки запроса без payload."""
        kwargs = prepare_request_kwargs(
            http_method=HTTPMethod.GET,
            url="https://example.com/api",
            headers={},
            timeout=30,
            payload=None,
            params={"param1": "value1"},
        )

        assert "json" not in kwargs
        assert kwargs["params"] == {"param1": "value1"}


class TestParseResponse:
    """Тесты функции parse_response."""

    def test_parse_pydantic_response(self, mocker: MockerFixture, requests_mock) -> None:
        """Проверка парсинга Pydantic модели."""
        logger = mocker.Mock()
        requests_mock.get("https://example.com", json={"is_success": True, "response": "test"})

        response = requests.get("https://example.com")
        result = parse_response(logger, response, ChatResponse)

        assert isinstance(result, ChatResponse)
        assert result.is_success is True
        assert result.response == "test"

    def test_parse_dict_response(self, mocker: MockerFixture, requests_mock) -> None:
        """Проверка парсинга обычного словаря."""
        logger = mocker.Mock()
        requests_mock.get("https://example.com", json={"test": "data"})

        response = requests.get("https://example.com")
        result = parse_response(logger, response, dict)

        assert result == {"test": "data"}


class TestTryParseErrorResponse:
    """Тесты функции try_parse_error_response."""

    def test_parse_successful_error_response(self, mocker: MockerFixture, requests_mock) -> None:
        """Проверка успешного парсинга ответа с ошибкой."""
        logger = mocker.Mock()
        requests_mock.get("https://example.com", json={"is_success": False, "error_code": "API-001"})

        response = requests.get("https://example.com")
        result = try_parse_error_response(logger, response, ChatResponse)

        assert result is not None
        assert isinstance(result, ChatResponse)
        assert result.is_success is False
        assert result.error_code == "API-001"

    def test_parse_failed_error_response(self, mocker: MockerFixture, requests_mock) -> None:
        """Проверка неудачного парсинга (не JSON)."""
        logger = mocker.Mock()
        requests_mock.get("https://example.com", text="Not JSON", headers={"Content-Type": "application/json"})

        response = requests.get("https://example.com")
        result = try_parse_error_response(logger, response, ChatResponse)

        assert result is None

    def test_parse_none_response(self, mocker: MockerFixture) -> None:
        """Проверка парсинга None ответа."""
        logger = mocker.Mock()
        result = try_parse_error_response(logger, None, ChatResponse)

        assert result is None


class TestHandleRequestException:
    """Тесты функции handle_request_exception."""

    def test_handle_http_error(self, requests_mock) -> None:
        """Проверка обработки HTTP ошибки."""
        requests_mock.get("https://example.com", status_code=500, text="Internal Server Error")

        try:
            response = requests.get("https://example.com")
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            result = handle_request_exception(e, e.response, ChatResponse, 30)

            assert isinstance(result, ChadGPTHTTPError)
            assert result.error_code == "HTTP_ERROR"

    def test_handle_connection_error(self) -> None:
        """Проверка обработки ConnectionError."""
        exception = requests.exceptions.ConnectionError("Connection refused")

        result = handle_request_exception(exception, None, ChatResponse, 30)

        assert isinstance(result, ChadGPTConnectionError)
        assert result.error_code == "CONNECTION_ERROR"

    def test_handle_timeout_error(self) -> None:
        """Проверка обработки Timeout."""
        exception = requests.exceptions.Timeout("Request timeout")

        result = handle_request_exception(exception, None, ChatResponse, 30)

        assert isinstance(result, ChadGPTTimeoutError)
        assert result.error_code == "TIMEOUT_ERROR"

    def test_handle_json_decode_error(self, requests_mock) -> None:
        """Проверка обработки JSONDecodeError."""
        requests_mock.get("https://example.com", text="Not JSON", headers={"Content-Type": "application/json"})

        response = requests.get("https://example.com")
        try:
            response.json()
        except json.JSONDecodeError as e:
            result = handle_request_exception(e, response, ChatResponse, 30)

            assert isinstance(result, ChadGPTJSONDecodeError)
            assert result.error_code == "JSON_DECODE_ERROR"

    def test_handle_generic_error(self) -> None:
        """Проверка обработки общего исключения."""
        exception = ValueError("Some error")

        result = handle_request_exception(exception, None, ChatResponse, 30)

        assert isinstance(result, ChadGPTError)
        assert result.error_code == "REQUEST_ERROR"


class TestExecuteSingleRequest:
    """Тесты функции execute_single_request."""

    def test_execute_successful_request(self, mocker: MockerFixture, requests_mock) -> None:
        """Проверка успешного выполнения запроса."""
        logger = mocker.Mock()
        session = requests.Session()

        requests_mock.post("https://example.com", json={"is_success": True, "response": "test"})

        response, parsed = execute_single_request(
            logger=logger,
            session=session,
            http_method=HTTPMethod.POST,
            request_kwargs={"url": "https://example.com"},
            response_type=ChatResponse,
        )

        assert response.status_code == 200
        assert isinstance(parsed, ChatResponse)
        assert parsed.is_success is True

    def test_execute_request_raises_http_error(self, mocker: MockerFixture, requests_mock) -> None:
        """Проверка, что HTTP ошибка пробрасывается."""
        logger = mocker.Mock()
        session = requests.Session()

        requests_mock.post("https://example.com", status_code=500, text="Internal Server Error")

        with pytest.raises(requests.exceptions.HTTPError):
            execute_single_request(
                logger=logger,
                session=session,
                http_method=HTTPMethod.POST,
                request_kwargs={"url": "https://example.com"},
                response_type=ChatResponse,
            )

    def test_execute_request_sets_response_on_http_error(self, mocker: MockerFixture, requests_mock) -> None:
        """Проверка, что response устанавливается в HTTPError, если его там нет."""
        logger = mocker.Mock()
        session = requests.Session()

        # Создаем response с ошибкой
        requests_mock.post("https://example.com", status_code=500, text="Internal Server Error")
        response = requests.post("https://example.com")

        # Мокаем raise_for_status, чтобы он выбрасывал HTTPError без response
        def mock_raise_for_status():
            # Создаем HTTPError без response
            http_err = requests.exceptions.HTTPError("HTTP Error")
            # Не устанавливаем response, чтобы покрыть строку 261
            if hasattr(http_err, "response"):
                delattr(http_err, "response")
            raise http_err

        # Мокаем response.raise_for_status
        mocker.patch.object(response, "raise_for_status", side_effect=mock_raise_for_status)

        # Мокаем session.request, чтобы вернуть наш response
        mocker.patch.object(session, "request", return_value=response)

        with pytest.raises(requests.exceptions.HTTPError) as exc_info:
            execute_single_request(
                logger=logger,
                session=session,
                http_method=HTTPMethod.POST,
                request_kwargs={"url": "https://example.com"},
                response_type=ChatResponse,
            )

        # Проверяем, что response был установлен в исключении (строка 261)
        assert hasattr(exc_info.value, "response")
        assert exc_info.value.response is not None
        assert exc_info.value.response.status_code == 500

class TestSendRequestWithRetry:
    """Тесты функции send_request_with_retry."""

    def test_successful_request_no_retry(self, mocker: MockerFixture, requests_mock) -> None:
        """Проверка успешного запроса без retry."""
        logger = mocker.Mock()
        session = requests.Session()

        requests_mock.post("https://example.com", json={"is_success": True, "response": "test"})

        result = send_request_with_retry(
            logger=logger,
            session=session,
            http_method=HTTPMethod.POST,
            request_kwargs={"url": "https://example.com"},
            response_type=ChatResponse,
            request_timeout=30,
            masked_api_key="test***key",
            max_retries=3,
        )

        assert isinstance(result, ChatResponse)
        assert result.is_success is True
        assert requests_mock.call_count == 1

    def test_retry_on_connection_error(self, mocker: MockerFixture, requests_mock) -> None:
        """Проверка retry при ConnectionError."""
        logger = mocker.Mock()
        session = requests.Session()

        # Первые две попытки - ошибка, третья - успех
        requests_mock.post(
            "https://example.com",
            [
                {"exc": requests.exceptions.ConnectionError("Connection refused")},
                {"exc": requests.exceptions.ConnectionError("Connection refused")},
                {"json": {"is_success": True, "response": "test"}},
            ],
        )

        mocker.patch("time.sleep")  # Мокаем sleep для скорости

        result = send_request_with_retry(
            logger=logger,
            session=session,
            http_method=HTTPMethod.POST,
            request_kwargs={"url": "https://example.com"},
            response_type=ChatResponse,
            request_timeout=30,
            masked_api_key="test***key",
            max_retries=3,
        )

        assert isinstance(result, ChatResponse)
        assert result.is_success is True
        assert requests_mock.call_count == 3

    def test_retry_on_http_5xx(self, mocker: MockerFixture, requests_mock) -> None:
        """Проверка retry при HTTP 5xx ошибке."""
        logger = mocker.Mock()
        session = requests.Session()

        # Первая попытка - 500, вторая - успех
        requests_mock.post(
            "https://example.com",
            [
                {"status_code": 500, "text": "Internal Server Error"},
                {"json": {"is_success": True, "response": "test"}},
            ],
        )

        mocker.patch("time.sleep")

        result = send_request_with_retry(
            logger=logger,
            session=session,
            http_method=HTTPMethod.POST,
            request_kwargs={"url": "https://example.com"},
            response_type=ChatResponse,
            request_timeout=30,
            masked_api_key="test***key",
            max_retries=3,
        )

        assert isinstance(result, ChatResponse)
        assert result.is_success is True
        assert requests_mock.call_count == 2

    def test_no_retry_on_http_4xx(self, mocker: MockerFixture, requests_mock) -> None:
        """Проверка, что HTTP 4xx ошибки не повторяются."""
        logger = mocker.Mock()
        session = requests.Session()

        requests_mock.post("https://example.com", status_code=400, text="Bad Request")

        with pytest.raises(ChadGPTHTTPError):
            send_request_with_retry(
                logger=logger,
                session=session,
                http_method=HTTPMethod.POST,
                request_kwargs={"url": "https://example.com"},
                response_type=ChatResponse,
                request_timeout=30,
                masked_api_key="test***key",
                max_retries=3,
            )

        # Должна быть только одна попытка
        assert requests_mock.call_count == 1

    def test_max_retries_exceeded(self, mocker: MockerFixture, requests_mock) -> None:
        """Проверка, что после исчерпания попыток выбрасывается исключение."""
        logger = mocker.Mock()
        session = requests.Session()

        # Все попытки заканчиваются ошибкой
        requests_mock.post(
            "https://example.com",
            exc=requests.exceptions.ConnectionError("Connection refused"),
        )

        mocker.patch("time.sleep")

        with pytest.raises(ChadGPTConnectionError) as exc_info:
            send_request_with_retry(
                logger=logger,
                session=session,
                http_method=HTTPMethod.POST,
                request_kwargs={"url": "https://example.com"},
                response_type=ChatResponse,
                request_timeout=30,
                masked_api_key="test***key",
                max_retries=3,
            )

        assert exc_info.value.error_code == "CONNECTION_ERROR"
        assert requests_mock.call_count == 3

    def test_parse_error_response_success(self, mocker: MockerFixture, requests_mock) -> None:
        """Проверка, что успешно парсится ответ с ошибкой (JSON при HTTP ошибке)."""
        logger = mocker.Mock()
        session = requests.Session()

        # HTTP ошибка, но ответ содержит валидный JSON
        # requests_mock автоматически устанавливает response в HTTPError
        requests_mock.post(
            "https://example.com",
            status_code=500,
            json={"is_success": False, "error_code": "API-001"},
        )

        result = send_request_with_retry(
            logger=logger,
            session=session,
            http_method=HTTPMethod.POST,
            request_kwargs={"url": "https://example.com"},
            response_type=ChatResponse,
            request_timeout=30,
            masked_api_key="test***key",
            max_retries=3,
        )

        # Должен вернуть распарсенный ответ, несмотря на HTTP ошибку
        assert isinstance(result, ChatResponse)
        assert result.is_success is False
        assert result.error_code == "API-001"
        assert requests_mock.call_count == 1

    def test_no_retry_on_json_decode_error(self, mocker: MockerFixture, requests_mock) -> None:
        """Проверка, что JSON decode ошибки не повторяются."""
        logger = mocker.Mock()
        session = requests.Session()

        # Ответ с невалидным JSON
        requests_mock.post("https://example.com", text="Not JSON", headers={"Content-Type": "application/json"})

        with pytest.raises(ChadGPTJSONDecodeError):
            send_request_with_retry(
                logger=logger,
                session=session,
                http_method=HTTPMethod.POST,
                request_kwargs={"url": "https://example.com"},
                response_type=ChatResponse,
                request_timeout=30,
                masked_api_key="test***key",
                max_retries=3,
            )

        # Должна быть только одна попытка
        assert requests_mock.call_count == 1

    def test_all_retries_exhausted_without_last_exception(self, mocker: MockerFixture) -> None:
        """Проверка обработки случая, когда все попытки исчерпаны без установленного last_exception"""
        logger = mocker.Mock()
        session = mocker.Mock()

        def mock_execute(*args, **kwargs):
            # Выбрасываем KeyboardInterrupt, который не перехватывается блоками except Exception
            raise KeyboardInterrupt("Interrupted")

        mocker.patch("pychadgpt.nethelpers.execute_single_request", side_effect=mock_execute)
        mocker.patch("time.sleep")

        with pytest.raises(ChadGPTError) as exc_info:
            send_request_with_retry(
                logger=logger,
                session=session,
                http_method=HTTPMethod.POST,
                request_kwargs={"url": "https://example.com"},
                response_type=ChatResponse,
                request_timeout=30,
                masked_api_key="test***key",
                max_retries=0,  # Цикл не выполнится, last_exception останется None
            )

        assert exc_info.value.error_code == "REQUEST_ERROR"
        assert "после 0 попыток" in str(exc_info.value)
