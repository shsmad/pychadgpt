"""Тесты для валидации параметров клиента ChadGPT."""

import json

import pytest

from pytest_mock import MockerFixture

from pychadgpt.client import ChadGPTBaseClient, ChadGPTClient, ChadGPTImageClient, ChatHistoryMessage
from pychadgpt.exceptions import (
    ChadGPTValidationError,
)


class TestChadGPTBaseClientValidation:
    """Тесты валидации для базового клиента."""

    def test_empty_api_key_raises_error(self) -> None:
        """Проверка, что пустой API ключ вызывает ошибку."""
        with pytest.raises(ChadGPTValidationError) as exc_info:
            ChadGPTBaseClient("")

        assert exc_info.value.error_code == "INVALID_API_KEY"

        with pytest.raises(ChadGPTValidationError) as exc_info:
            ChadGPTBaseClient("   ")

        assert exc_info.value.error_code == "INVALID_API_KEY"

    def test_valid_api_key_initializes(self) -> None:
        """Проверка, что валидный API ключ инициализирует клиент."""
        client = ChadGPTBaseClient("test-api-key")
        assert client.api_key == "test-api-key"
        assert client._session is not None

    def test_unsupported_http_method_raises_error(self, mocker: MockerFixture) -> None:
        """Проверка, что неподдерживаемый HTTP метод вызывает ошибку."""
        client = ChadGPTBaseClient("test-api-key")
        mocker.patch.object(client._session, "request")

        with pytest.raises(ChadGPTValidationError) as exc_info:
            client._send_request(
                response_type=dict,
                url="https://example.com",
                method="put",
            )

        assert exc_info.value.error_code == "INVALID_HTTP_METHOD"

        with pytest.raises(ChadGPTValidationError) as exc_info:
            client._send_request(
                response_type=dict,
                url="https://example.com",
                method="DELETE",
            )

        assert exc_info.value.error_code == "INVALID_HTTP_METHOD"

    def test_case_insensitive_http_method(self, mocker: MockerFixture) -> None:
        """Проверка, что HTTP методы нечувствительны к регистру."""
        client = ChadGPTBaseClient("test-api-key")
        mock_response = mocker.Mock()
        mock_response.json.return_value = {"test": "data"}
        mock_response.content = json.dumps(mock_response.json.return_value)
        mock_response.raise_for_status = mocker.Mock()
        mocker.patch.object(client._session, "request", return_value=mock_response)

        # Проверяем разные варианты регистра
        result1 = client._send_request(
            response_type=dict,
            url="https://example.com",
            method="GET",
        )
        assert result1 == {"test": "data"}

        result2 = client._send_request(
            response_type=dict,
            url="https://example.com",
            method="Post",
        )
        assert result2 == {"test": "data"}


class TestChadGPTClientValidation:
    """Тесты валидации для чат-клиента."""

    def test_unsupported_model_raises_error(self) -> None:
        """Проверка, что неподдерживаемая модель вызывает ошибку."""
        client = ChadGPTClient("test-api-key")

        with pytest.raises(ChadGPTValidationError) as exc_info:
            client.ask("invalid-model", "Hello")

        assert exc_info.value.error_code == "INVALID_MODEL"

    def test_deprecated_model_shows_warning(self, mocker: MockerFixture) -> None:
        """Проверка, что использование устаревшей модели показывает предупреждение."""
        client = ChadGPTClient("test-api-key")
        mock_response = mocker.Mock()
        mock_response.json.return_value = {"is_success": True, "response": "test"}
        mock_response.content = json.dumps(mock_response.json.return_value)
        mock_response.raise_for_status = mocker.Mock()
        mocker.patch.object(client._session, "request", return_value=mock_response)

        with pytest.warns(DeprecationWarning, match="устарела"):
            client.ask("gpt-4o-mini", "Hello")

    def test_valid_model_no_warning(self, mocker: MockerFixture) -> None:
        """Проверка, что валидная модель не показывает предупреждение."""
        import warnings

        client = ChadGPTClient("test-api-key")
        mock_response = mocker.Mock()
        mock_response.json.return_value = {"is_success": True, "response": "test"}
        mock_response.content = json.dumps(mock_response.json.return_value)
        mock_response.raise_for_status = mocker.Mock()
        mocker.patch.object(client._session, "request", return_value=mock_response)

        # Проверяем, что нет предупреждений при использовании валидной модели
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = client.ask("gpt-5", "Hello")
            assert result.is_success is True
            # Проверяем, что предупреждения не были вызваны
            assert len(w) == 0, f"Unexpected warnings: {[str(warning.message) for warning in w]}"

    def test_ask_with_history(self, mocker: MockerFixture) -> None:
        """Проверка передачи истории сообщений."""
        client = ChadGPTClient("test-api-key")
        mock_response = mocker.Mock()
        mock_response.json.return_value = {"is_success": True, "response": "test"}
        mock_response.content = json.dumps(mock_response.json.return_value)
        mock_response.raise_for_status = mocker.Mock()
        mock_request = mocker.patch.object(client._session, "request", return_value=mock_response)

        history: list[ChatHistoryMessage] = [
            ChatHistoryMessage(role="user", content="Previous message"),
            ChatHistoryMessage(role="assistant", content="Previous response"),
        ]
        client.ask("gpt-5", "Hello", history=history)

        # Проверяем, что история передана в payload
        call_args = mock_request.call_args
        assert call_args is not None
        assert "json" in call_args.kwargs
        assert call_args.kwargs["json"]["history"] == [x.model_dump(exclude_none=True) for x in history]

    def test_ask_with_optional_params(self, mocker: MockerFixture) -> None:
        """Проверка передачи опциональных параметров."""
        client = ChadGPTClient("test-api-key")
        mock_response = mocker.Mock()
        mock_response.json.return_value = {"is_success": True, "response": "test"}
        mock_response.content = json.dumps(mock_response.json.return_value)
        mock_response.raise_for_status = mocker.Mock()
        mock_request = mocker.patch.object(client._session, "request", return_value=mock_response)

        client.ask(
            "gpt-5",
            "Hello",
            temperature=0.7,
            max_tokens=100,
            images=["https://example.com/image.jpg"],
        )

        call_args = mock_request.call_args
        assert call_args is not None
        payload = call_args.kwargs["json"]
        assert payload["temperature"] == 0.7
        assert payload["max_tokens"] == 100
        assert payload["images"] == ["https://example.com/image.jpg"]


class TestChadGPTImageClientValidation:
    """Тесты валидации для клиента генерации изображений."""

    def test_unsupported_image_model_raises_error(self) -> None:
        """Проверка, что неподдерживаемая модель изображений вызывает ошибку."""
        client = ChadGPTImageClient("test-api-key")

        with pytest.raises(ChadGPTValidationError) as exc_info:
            client.imagine("invalid-model", "A beautiful landscape")

        assert exc_info.value.error_code == "INVALID_MODEL"

    def test_empty_prompt_raises_error(self) -> None:
        """Проверка, что пустой промпт вызывает ошибку."""
        client = ChadGPTImageClient("test-api-key")

        with pytest.raises(ChadGPTValidationError) as exc_info:
            client.imagine("imagen-4", "")

        assert exc_info.value.error_code == "VALIDATION_ERROR"

        with pytest.raises(ChadGPTValidationError) as exc_info:
            client.imagine("imagen-4", "   ")

        assert exc_info.value.error_code == "VALIDATION_ERROR"

    def test_prompt_too_long_raises_error(self) -> None:
        """Проверка, что слишком длинный промпт вызывает ошибку."""
        client = ChadGPTImageClient("test-api-key")
        long_prompt = "a" * 1001  # Превышает лимит в 1000 символов

        with pytest.raises(ChadGPTValidationError) as exc_info:
            client.imagine("imagen-4", long_prompt)

        assert exc_info.value.error_code == "VALIDATION_ERROR"

    def test_prompt_stripped(self, mocker: MockerFixture) -> None:
        """Проверка, что пробелы в начале и конце промпта удаляются."""
        client = ChadGPTImageClient("test-api-key")
        mock_response = mocker.Mock()
        mock_response.json.return_value = {"status": "starting", "content_id": "test-id"}
        mock_response.content = json.dumps(mock_response.json.return_value)
        mock_response.raise_for_status = mocker.Mock()
        mock_request = mocker.patch.object(client._session, "request", return_value=mock_response)

        client.imagine("imagen-4", "  A beautiful landscape  ")

        call_args = mock_request.call_args
        assert call_args is not None
        payload = call_args.kwargs["json"]
        assert payload["prompt"] == "A beautiful landscape"

    def test_empty_content_id_raises_error(self) -> None:
        """Проверка, что пустой content_id вызывает ошибку."""
        client = ChadGPTImageClient("test-api-key")

        with pytest.raises(ChadGPTValidationError) as exc_info:
            client.check_status("")

        assert exc_info.value.error_code == "INVALID_CONTENT_ID"

        with pytest.raises(ChadGPTValidationError) as exc_info:
            client.check_status("   ")

        assert exc_info.value.error_code == "INVALID_CONTENT_ID"

    def test_invalid_content_id_type_raises_error(self) -> None:
        """Проверка, что неверный тип content_id вызывает ошибку."""
        client = ChadGPTImageClient("test-api-key")

        with pytest.raises(ChadGPTValidationError) as exc_info:
            client.check_status(123)  # type: ignore[arg-type]

        assert exc_info.value.error_code == "INVALID_CONTENT_ID"
