"""Тесты для edge cases и дополнительных сценариев."""

import json

import pytest
import requests

from pytest_mock import MockerFixture

from pychadgpt.client import ChadGPTBaseClient, ChadGPTClient, ChadGPTImageClient, ChatHistoryMessage
from pychadgpt.exceptions import ChadGPTAPIError, ChadGPTValidationError
from pychadgpt.models import ChatResponse


class TestEdgeCases:
    """Тесты для edge cases и граничных случаев."""

    def test_ask_with_invalid_pydantic_params(self, mocker: MockerFixture) -> None:
        """Проверка обработки ошибок валидации Pydantic параметров."""
        client = ChadGPTClient("test-api-key")

        # Невалидная temperature (больше 2)
        with pytest.raises(ChadGPTValidationError) as exc_info:
            client.ask("gpt-5", "Hello", temperature=3.0)

        assert exc_info.value.error_code == "VALIDATION_ERROR"

        # Невалидный max_tokens (отрицательный)
        with pytest.raises(ChadGPTValidationError) as exc_info:
            client.ask("gpt-5", "Hello", max_tokens=-1)

        assert exc_info.value.error_code == "VALIDATION_ERROR"

    def test_ask_with_empty_message_after_strip(self, mocker: MockerFixture) -> None:
        """Проверка, что сообщение из пробелов вызывает ошибку."""
        client = ChadGPTClient("test-api-key")

        with pytest.raises(ChadGPTValidationError) as exc_info:
            client.ask("gpt-5", "   ")

        assert exc_info.value.error_code == "VALIDATION_ERROR"

    def test_ask_with_api_error_response(self, mocker: MockerFixture) -> None:
        """Проверка обработки ответа с ошибкой от API."""
        client = ChadGPTClient("test-api-key")
        mock_response = mocker.Mock()
        mock_response.json.return_value = {
            "is_success": False,
            "error_code": "API-001",
            "error_message": "Invalid API key",
        }
        mock_response.content = json.dumps(mock_response.json.return_value)
        mock_response.raise_for_status = mocker.Mock()
        mocker.patch.object(client._session, "request", return_value=mock_response)

        with pytest.raises(ChadGPTAPIError) as exc_info:
            client.ask("gpt-5", "Hello")

        assert exc_info.value.error_code == "API-001"
        assert "Invalid API key" in str(exc_info.value)

    def test_ask_with_deprecated_response(self, mocker: MockerFixture) -> None:
        """Проверка обработки ответа с предупреждением об устаревшей модели."""
        client = ChadGPTClient("test-api-key")
        mock_response = mocker.Mock()
        mock_response.json.return_value = {
            "is_success": True,
            "response": "test",
            "deprecated": "This model will be removed soon",
        }
        mock_response.content = json.dumps(mock_response.json.return_value)
        mock_response.raise_for_status = mocker.Mock()
        mocker.patch.object(client._session, "request", return_value=mock_response)

        result = client.ask("gpt-5", "Hello")

        assert result.is_success is True
        assert result.deprecated == "This model will be removed soon"

    def test_imagine_with_failed_status(self, mocker: MockerFixture) -> None:
        """Проверка обработки failed статуса в imagine."""
        client = ChadGPTImageClient("test-api-key")
        mock_response = mocker.Mock()
        mock_response.json.return_value = {
            "status": "failed",
            "error_code": "GEN-001",
            "error_message": "Generation failed",
        }
        mock_response.content = json.dumps(mock_response.json.return_value)
        mock_response.raise_for_status = mocker.Mock()
        mocker.patch.object(client._session, "request", return_value=mock_response)

        with pytest.raises(ChadGPTAPIError) as exc_info:
            client.imagine("imagen-4", "A landscape")

        assert exc_info.value.error_code == "GEN-001"

    def test_check_status_with_failed_status(self, mocker: MockerFixture) -> None:
        """Проверка обработки failed статуса в check_status."""
        client = ChadGPTImageClient("test-api-key")
        mock_response = mocker.Mock()
        mock_response.json.return_value = {
            "status": "failed",
            "error_code": "CHECK-001",
            "error_message": "Check failed",
        }
        mock_response.content = json.dumps(mock_response.json.return_value)
        mock_response.raise_for_status = mocker.Mock()
        mocker.patch.object(client._session, "request", return_value=mock_response)

        with pytest.raises(ChadGPTAPIError) as exc_info:
            client.check_status("test-content-id")

        assert exc_info.value.error_code == "CHECK-001"

    def test_get_stat_info_with_error(self, mocker: MockerFixture) -> None:
        """Проверка обработки ошибки в get_stat_info."""
        client = ChadGPTClient("test-api-key")
        mock_response = mocker.Mock()
        mock_response.json.return_value = {
            "is_success": False,
            "error_code": "STATS-001",
            "error_message": "Stats error",
        }
        mock_response.content = json.dumps(mock_response.json.return_value)
        mock_response.raise_for_status = mocker.Mock()
        mocker.patch.object(client._session, "request", return_value=mock_response)

        with pytest.raises(ChadGPTAPIError) as exc_info:
            client.get_stat_info()

        assert exc_info.value.error_code == "STATS-001"

    def test_history_validation(self, mocker: MockerFixture) -> None:
        """Проверка валидации истории сообщений."""
        client = ChadGPTClient("test-api-key")

        # Невалидная роль - Pydantic валидирует при создании объекта
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ChatHistoryMessage(role="invalid_role", content="test")  # type: ignore[arg-type]

        # Пустое содержимое - Pydantic валидирует при создании объекта
        with pytest.raises(ValidationError):
            ChatHistoryMessage(role="user", content="")  # Пустое содержимое

    def test_all_autogenerated_methods_exist(self) -> None:
        """Проверка, что все автогенерируемые методы существуют."""
        client = ChadGPTClient("test-api-key")

        # Проверяем наличие методов для всех моделей
        models = [
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-nano",
            "claude-4.5-sonnet",
            "claude-4.1-opus",
            "gemini-2.5-pro",
        ]

        for model in models:
            method_name = f"ask_{model.replace('-', '_').replace('.', '_')}"
            assert hasattr(client, method_name), f"Method {method_name} should exist for model {model}"

    def test_autogenerated_method_with_timeout(self, mocker: MockerFixture) -> None:
        """Проверка, что автогенерируемые методы поддерживают timeout."""
        client = ChadGPTClient("test-api-key")
        mock_response = mocker.Mock()
        mock_response.json.return_value = {"is_success": True, "response": "test"}
        mock_response.content = json.dumps(mock_response.json.return_value)
        mock_response.raise_for_status = mocker.Mock()
        mock_request = mocker.patch.object(client._session, "request", return_value=mock_response)

        # Используем автогенерируемый метод
        client.ask_gpt_5("Hello", timeout=60)  # type: ignore

        call_args = mock_request.call_args
        assert call_args is not None
        assert call_args.kwargs["timeout"] == 60

    def test_autogenerated_method_with_all_params(self, mocker: MockerFixture) -> None:
        """Проверка автогенерируемого метода со всеми параметрами."""
        client = ChadGPTClient("test-api-key")
        mock_response = mocker.Mock()
        mock_response.json.return_value = {"is_success": True, "response": "test"}
        mock_response.content = json.dumps(mock_response.json.return_value)
        mock_response.raise_for_status = mocker.Mock()
        mock_request = mocker.patch.object(client._session, "request", return_value=mock_response)

        history = [ChatHistoryMessage(role="user", content="Previous")]
        images = ["https://example.com/image.jpg"]

        client.ask_gpt_5(  # type: ignore
            "Hello",
            history=history,
            temperature=0.7,
            max_tokens=100,
            images=images,
            timeout=45,
        )

        call_args = mock_request.call_args
        assert call_args is not None
        payload = call_args.kwargs["json"]
        assert payload["temperature"] == 0.7
        assert payload["max_tokens"] == 100
        assert payload["images"] == images
        assert call_args.kwargs["timeout"] == 45


class TestRetryMechanism:
    """Тесты для retry механизма в реальных сценариях."""

    def test_retry_on_connection_error_success(self, mocker: MockerFixture, requests_mock) -> None:
        """Проверка успешного retry после ConnectionError."""
        client = ChadGPTClient("test-api-key")
        client.max_retries = 3

        # Первые две попытки - ошибка, третья - успех
        requests_mock.post(
            "https://ask.chadgpt.ru/api/public/gpt-5",
            [
                {"exc": requests.exceptions.ConnectionError("Connection refused")},
                {"exc": requests.exceptions.ConnectionError("Connection refused")},
                {"json": {"is_success": True, "response": "test"}},
            ],
        )

        mocker.patch("time.sleep")  # Мокаем sleep для скорости

        result = client.ask("gpt-5", "Hello")

        assert result.is_success is True
        assert requests_mock.call_count == 3

    def test_retry_on_http_5xx_success(self, mocker: MockerFixture) -> None:
        """Проверка успешного retry после HTTP 5xx."""
        client = ChadGPTClient("test-api-key")
        client.max_retries = 3

        # Первая попытка - 500, вторая - успех
        mock_error_response = mocker.Mock()
        mock_error_response.status_code = 500
        mock_error_response.text = "Internal Server Error"
        mock_error_response.content = b"Error"
        http_error = requests.exceptions.HTTPError("Server Error")
        http_error.response = mock_error_response
        mock_error_response.raise_for_status.side_effect = http_error

        mock_success_response = mocker.Mock()
        mock_success_response.json.return_value = {"is_success": True, "response": "test"}
        mock_success_response.content = json.dumps(mock_success_response.json.return_value).encode()
        mock_success_response.raise_for_status = mocker.Mock()

        mock_request = mocker.patch.object(
            client._session,
            "request",
            side_effect=[mock_error_response, mock_success_response],
        )

        mocker.patch("time.sleep")

        result = client.ask("gpt-5", "Hello")

        assert result.is_success is True
        assert mock_request.call_count == 2

    def test_no_retry_on_http_4xx(self, mocker: MockerFixture, requests_mock) -> None:
        """Проверка, что HTTP 4xx ошибки не повторяются."""
        client = ChadGPTClient("test-api-key")
        client.max_retries = 3

        requests_mock.post(
            "https://ask.chadgpt.ru/api/public/gpt-5",
            status_code=400,
            text="Bad Request",
        )

        from pychadgpt.exceptions import ChadGPTHTTPError

        with pytest.raises(ChadGPTHTTPError):
            client.ask("gpt-5", "Hello")

        # Должна быть только одна попытка
        assert requests_mock.call_count == 1

    def test_retry_exhausted_raises_error(self, mocker: MockerFixture, requests_mock) -> None:
        """Проверка, что после исчерпания попыток выбрасывается исключение."""
        client = ChadGPTClient("test-api-key")
        client.max_retries = 2

        # Все попытки заканчиваются ошибкой
        requests_mock.post(
            "https://ask.chadgpt.ru/api/public/gpt-5",
            exc=requests.exceptions.ConnectionError("Connection refused"),
        )

        mocker.patch("time.sleep")

        from pychadgpt.exceptions import ChadGPTConnectionError

        with pytest.raises(ChadGPTConnectionError) as exc_info:
            client.ask("gpt-5", "Hello")

        assert exc_info.value.error_code == "CONNECTION_ERROR"
        # max_retries=2 означает максимум 3 попытки (1 начальная + 2 retry)
        assert requests_mock.call_count == 3


class TestResponseParsing:
    """Тесты для парсинга различных типов ответов."""

    def test_parse_dict_response(self, mocker: MockerFixture) -> None:
        """Проверка парсинга обычного словаря (не Pydantic модели)."""
        client = ChadGPTBaseClient("test-api-key")
        mock_response = mocker.Mock()
        mock_response.json.return_value = {"test": "data", "number": 42}
        mock_response.content = json.dumps(mock_response.json.return_value)
        mock_response.raise_for_status = mocker.Mock()
        mocker.patch.object(client._session, "request", return_value=mock_response)

        result = client._send_request(
            response_type=dict,
            url="https://example.com/api",
            method="get",
        )

        assert isinstance(result, dict)
        assert result["test"] == "data"
        assert result["number"] == 42

    def test_parse_pydantic_response(self, mocker: MockerFixture) -> None:
        """Проверка парсинга Pydantic модели."""
        client = ChadGPTClient("test-api-key")
        mock_response = mocker.Mock()
        mock_response.json.return_value = {
            "is_success": True,
            "response": "Hello!",
            "used_words_count": 10,
        }
        mock_response.content = json.dumps(mock_response.json.return_value)
        mock_response.raise_for_status = mocker.Mock()
        mocker.patch.object(client._session, "request", return_value=mock_response)

        result = client.ask("gpt-5", "Hello")

        assert isinstance(result, ChatResponse)
        assert result.is_success is True
        assert result.response == "Hello!"
        assert result.used_words_count == 10
