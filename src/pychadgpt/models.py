from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ============================================================================
# PYDANTIC МОДЕЛИ ДЛЯ ВАЛИДАЦИИ
# ============================================================================


class ChatHistoryMessage(BaseModel):
    """
    Модель для сообщения в истории чата.

    Example:
        >>> msg = ChatHistoryMessage(role="user", content="Hello!")
        >>> msg = ChatHistoryMessage(
        ...     role="assistant",
        ...     content="Hi there!",
        ...     images=["https://example.com/image.jpg"]
        ... )
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
    )

    # Кто отправил сообщение. Допустимые значения: "user", "assistant", "system".
    # Значение "system" в GPT моделях даёт приоритет этому промту,
    # а в Claude не отличается от "user".
    role: Literal["user", "assistant", "system"] = Field(..., description="Роль отправителя сообщения")
    content: str = Field(..., min_length=1, description="Содержимое сообщения")
    images: list[str] | None = Field(default=None, description="Список URL или base64 Data URL изображений")


class AskParameters(BaseModel):
    """
    Параметры для метода ask().

    Example:
        >>> params = AskParameters(
        ...     message="Привет!",
        ...     temperature=0.7,
        ...     max_tokens=1000
        ... )
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
    )

    message: str = Field(..., min_length=1, max_length=50000, description="Запрос к нейросети")
    history: list[ChatHistoryMessage] | None = Field(default=None, description="История предыдущих сообщений")
    temperature: float | None = Field(default=None, ge=0.0, le=2.0, description="Температура генерации (0-2)")
    max_tokens: int | None = Field(default=None, gt=0, description="Максимальное количество токенов в ответе")
    images: list[str] | None = Field(default=None, description="Список изображений (URL или base64)")



class ImagineParameters(BaseModel):
    """
    Параметры для метода imagine().

    Example:
        >>> params = ImagineParameters(
        ...     prompt="A beautiful sunset over mountains",
        ...     aspect_ratio="16:9"
        ... )
    """

    model_config = ConfigDict(
        str_strip_whitespace=True,
    )

    prompt: str = Field(..., min_length=1, max_length=1000, description="Описание изображения")


# ============================================================================
# ТИПИЗИРОВАННЫЕ ОТВЕТЫ
# ============================================================================

class ChatResponse(BaseModel):
    """
    Ответ от chat API.

    Example:
        >>> response = client.ask_gpt5("Hello!")
        >>> if response.is_success:
        ...     print(response.response)
    """

    # Успешен ли ответ
    is_success: bool
    # Текст ответа от нейросети, в случае если is_success = True.
    response: str | None = None
    # Количество израсходованных слов на персональном тарифе.
    # Если используется корпоративный аккаунт, всегда возвращает 0.
    # Отправляется только если is_success = True.
    used_words_count: int = 0
    # Количество израсходованных токенов на корпоративном тарифе.
    # Если используется персональный тариф, всегда возвращает 0.
    # Отправляется только если is_success = True.
    used_tokens_count: int = 0
    # Код ошибки, если is_success = False.
    error_code: str | None = None
    # Название ошибки, если is_success = False
    error_message: str | None = None
    # Если присутствует в ответе, значит, модель устарела и скоро может быть удалена.
    # Содержит текст сообщения с более подробной информацией.
    # Если в ответах на ваши запросы стало появляться такое поле, значит, во
    # избежание проблем в будущем стоит указывать более новую модель.
    deprecated: str | None = None

    @model_validator(mode="after")
    def check_success_fields(self) -> "ChatResponse":
        if self.is_success and not self.response:
            raise ValueError("При is_success=True должно быть заполнено поле response")
        if not self.is_success and not self.error_code:
            raise ValueError("При is_success=False должно быть заполнено поле error_code")
        return self


class WordsResponse(BaseModel):
    """Ответ от /words API."""

    is_success: bool
    used_words: int = 0
    total_words: int = 0
    remaining_words: int = 0
    reserved_words: int = 0
    error_code: str | None = None
    error_message: str | None = None


class ImagineResponse(BaseModel):
    """Ответ от /imagine API."""

    status: str
    content_id: str | None = None
    created_at: str | None = None
    input: Any = None
    model: str | None = None
    error_code: str | None = None
    error_message: str | None = None


class CheckResponse(BaseModel):
    """Ответ от /check API."""

    # Возможные статусы:
    # - pending: Генерация в процессе
    # - completed: Генерация завершена успешно
    # - failed: Генерация завершилась с ошибкой
    # - cancelled: Генерация отменена

    status: Literal["pending", "completed", "failed", "cancelled"]
    # Временные метки представлены в формате ISO 8601 (UTC)
    created_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    output: list[str] = Field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None


# ============================================================================
# КОНСТАНТЫ И ENUMS
# ============================================================================


class HTTPMethod(str, Enum):
    """Поддерживаемые HTTP методы для запросов к API."""

    GET = "get"
    POST = "post"
