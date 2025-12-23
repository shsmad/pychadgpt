"""Тесты для валидации моделей Pydantic."""

import pytest

from pydantic import ValidationError

from pychadgpt.models import (
    AskParameters,
    ChatHistoryMessage,
    ChatResponse,
    ImagineParameters,
    validate_image_format,
)


class TestChatHistoryMessage:
    """Тесты для модели ChatHistoryMessage."""

    def test_valid_message(self) -> None:
        """Проверка создания валидного сообщения."""
        msg = ChatHistoryMessage(role="user", content="Hello!")
        assert msg.role == "user"
        assert msg.content == "Hello!"

    def test_valid_roles(self) -> None:
        """Проверка всех валидных ролей."""
        for role in ["user", "assistant", "system"]:
            msg = ChatHistoryMessage(role=role, content="Test")  # type: ignore
            assert msg.role == role

    def test_invalid_role(self) -> None:
        """Проверка невалидной роли."""
        with pytest.raises(ValidationError):
            ChatHistoryMessage(role="invalid", content="Test")  # type: ignore

    def test_empty_content(self) -> None:
        """Проверка, что пустое содержимое вызывает ошибку."""
        with pytest.raises(ValidationError):
            ChatHistoryMessage(role="user", content="")

    def test_whitespace_content_stripped(self) -> None:
        """Проверка, что пробелы в начале и конце удаляются."""
        msg = ChatHistoryMessage(role="user", content="  Hello  ")
        assert msg.content == "Hello"

    def test_valid_images_url(self) -> None:
        """Проверка валидных URL изображений."""
        msg = ChatHistoryMessage(
            role="user",
            content="Test",
            images=["https://example.com/image.jpg", "http://example.com/image.png"],
        )
        assert len(msg.images) == 2

    def test_valid_images_base64(self) -> None:
        """Проверка валидных base64 изображений."""
        msg = ChatHistoryMessage(
            role="user",
            content="Test",
            images=[
                "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
            ],
        )
        assert len(msg.images) == 1

    def test_invalid_image_format(self) -> None:
        """Проверка невалидного формата изображения."""
        with pytest.raises(ValidationError):
            ChatHistoryMessage(role="user", content="Test", images=["invalid-format"])

    def test_images_not_list(self) -> None:
        """Проверка, что images должен быть списком."""
        with pytest.raises(ValidationError):
            ChatHistoryMessage(role="user", content="Test", images="not-a-list")  # type: ignore

    def test_images_element_not_string(self) -> None:
        """Проверка, что элементы images должны быть строками."""
        with pytest.raises(ValidationError):
            ChatHistoryMessage(role="user", content="Test", images=[123])  # type: ignore

    def test_images_none(self) -> None:
        """Проверка, что images может быть None."""
        msg = ChatHistoryMessage(role="user", content="Test", images=None)
        assert msg.images is None


class TestAskParameters:
    """Тесты для модели AskParameters."""

    def test_valid_parameters(self) -> None:
        """Проверка создания валидных параметров."""
        params = AskParameters(message="Hello!")
        assert params.message == "Hello!"

    def test_empty_message(self) -> None:
        """Проверка, что пустое сообщение вызывает ошибку."""
        with pytest.raises(ValidationError):
            AskParameters(message="")

    def test_message_too_long(self) -> None:
        """Проверка, что слишком длинное сообщение вызывает ошибку."""
        long_message = "a" * 50001  # Превышает лимит в 50000
        with pytest.raises(ValidationError):
            AskParameters(message=long_message)

    def test_valid_temperature(self) -> None:
        """Проверка валидной температуры."""
        params = AskParameters(message="Hello", temperature=0.7)
        assert params.temperature == 0.7

    def test_temperature_bounds(self) -> None:
        """Проверка границ температуры."""
        # Минимум
        params = AskParameters(message="Hello", temperature=0.0)
        assert params.temperature == 0.0

        # Максимум
        params = AskParameters(message="Hello", temperature=2.0)
        assert params.temperature == 2.0

    def test_invalid_temperature(self) -> None:
        """Проверка невалидной температуры."""
        with pytest.raises(ValidationError):
            AskParameters(message="Hello", temperature=-0.1)

        with pytest.raises(ValidationError):
            AskParameters(message="Hello", temperature=2.1)

    def test_valid_max_tokens(self) -> None:
        """Проверка валидного max_tokens."""
        params = AskParameters(message="Hello", max_tokens=100)
        assert params.max_tokens == 100

    def test_invalid_max_tokens(self) -> None:
        """Проверка невалидного max_tokens."""
        with pytest.raises(ValidationError):
            AskParameters(message="Hello", max_tokens=0)

        with pytest.raises(ValidationError):
            AskParameters(message="Hello", max_tokens=-1)

    def test_valid_images(self) -> None:
        """Проверка валидных изображений."""
        params = AskParameters(
            message="Hello",
            images=["https://example.com/image.jpg", "data:image/png;base64,iVBORw0KGgo="],
        )
        assert len(params.images) == 2

    def test_invalid_images(self) -> None:
        """Проверка невалидных изображений."""
        with pytest.raises(ValidationError):
            AskParameters(message="Hello", images=["invalid-format"])

    def test_images_not_list(self) -> None:
        """Проверка, что images должен быть списком."""
        with pytest.raises(ValidationError):
            AskParameters(message="Hello", images="not-a-list")  # type: ignore

    def test_images_element_not_string(self) -> None:
        """Проверка, что элементы images должны быть строками."""
        with pytest.raises(ValidationError):
            AskParameters(message="Hello", images=[123])  # type: ignore


class TestImagineParameters:
    """Тесты для модели ImagineParameters."""

    def test_valid_prompt(self) -> None:
        """Проверка валидного промпта."""
        params = ImagineParameters(prompt="A beautiful landscape")
        assert params.prompt == "A beautiful landscape"

    def test_empty_prompt(self) -> None:
        """Проверка, что пустой промпт вызывает ошибку."""
        with pytest.raises(ValidationError):
            ImagineParameters(prompt="")

    def test_prompt_too_long(self) -> None:
        """Проверка, что слишком длинный промпт вызывает ошибку."""
        long_prompt = "a" * 1001  # Превышает лимит в 1000
        with pytest.raises(ValidationError):
            ImagineParameters(prompt=long_prompt)

    def test_prompt_stripped(self) -> None:
        """Проверка, что пробелы удаляются."""
        params = ImagineParameters(prompt="  A landscape  ")
        assert params.prompt == "A landscape"


class TestValidateImageFormat:
    """Тесты для функции validate_image_format."""

    def test_valid_http_url(self) -> None:
        """Проверка валидного HTTP URL."""
        validate_image_format("http://example.com/image.jpg")

    def test_valid_https_url(self) -> None:
        """Проверка валидного HTTPS URL."""
        validate_image_format("https://example.com/image.png")

    def test_valid_base64_data_url(self) -> None:
        """Проверка валидного base64 data URL."""
        validate_image_format("data:image/png;base64,iVBORw0KGgo=")

    def test_invalid_format(self) -> None:
        """Проверка невалидного формата."""
        with pytest.raises(ValueError, match="Неверный формат изображения"):
            validate_image_format("not-a-valid-url")

    def test_invalid_base64_format(self) -> None:
        """Проверка невалидного base64 формата."""
        with pytest.raises(ValueError):  # noqa: PT011
            validate_image_format("data:image/png;base64,invalid-base64!")

    def test_missing_protocol(self) -> None:
        """Проверка URL без протокола."""
        with pytest.raises(ValueError):  # noqa: PT011
            validate_image_format("example.com/image.jpg")


class TestChatResponse:
    """Тесты для модели ChatResponse."""

    def test_valid_success_response(self) -> None:
        """Проверка валидного успешного ответа."""
        response = ChatResponse(is_success=True, response="Hello!")
        assert response.is_success is True
        assert response.response == "Hello!"

    def test_valid_error_response(self) -> None:
        """Проверка валидного ответа с ошибкой."""
        response = ChatResponse(is_success=False, error_code="API-001", error_message="Error")
        assert response.is_success is False
        assert response.error_code == "API-001"

    def test_success_without_response(self) -> None:
        """Проверка, что при is_success=True должно быть поле response."""
        with pytest.raises(ValueError, match="При is_success=True должно быть заполнено поле response"):
            ChatResponse(is_success=True, response=None)

    def test_error_without_error_code(self) -> None:
        """Проверка, что при is_success=False должно быть поле error_code."""
        with pytest.raises(ValueError, match="При is_success=False должно быть заполнено поле error_code"):
            ChatResponse(is_success=False, error_code=None)
