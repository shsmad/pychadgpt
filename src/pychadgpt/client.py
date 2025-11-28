import json
import warnings

from collections.abc import Mapping
from enum import Enum
from typing import Any, Required, TypedDict, TypeVar, cast

import requests

from requests import Response, Session

from pychadgpt.image_models_config import MODEL_VALIDATORS

# Определяем TypeVar для универсального типа возвращаемого значения.
# `bound=TypedDict` гарантирует, что T будет TypedDict.
TResponse = TypeVar("TResponse", bound=Mapping[str, Any])


class ChatHistoryRow(TypedDict, total=False):
    # Кто отправил сообщение. Допустимые значения: "user", "assistant", "system".
    # Значение "system" в GPT моделях даёт приоритет этому промту, а в Claude не отличается от "user".
    role: Required[str]
    # Содержимое сообщения
    content: Required[str]
    # Картинки для передачи в нейросеть: строки, которые могут быть в одном из следующих форматов:
    # - URL на картинку (https)
    # - base64 dataUrl (документация)
    images: list[str]


class ChatResponse(TypedDict, total=False):
    # Успешен ли ответ
    is_success: Required[bool]
    # Текст ответа от нейросети, в случае если is_success = True.
    response: str
    # Количество израсходованных слов на персональном тарифе.
    # Если используется корпоративный аккаунт, всегда возвращает 0.
    # Отправляется только если is_success = True.
    used_words_count: int
    # Количество израсходованных токенов на корпоративном тарифе.
    # Если используется персональный тариф, всегда возвращает 0.
    # Отправляется только если is_success = True.
    used_tokens_count: int
    # Код ошибки, если is_success = False.
    error_code: str
    # Название ошибки, если is_success = False
    error_message: str
    # Если присутствует в ответе, значит, модель устарела и скоро может быть удалена.
    # Содержит текст сообщения с более подробной информацией.
    # Если в ответах на ваши запросы стало появляться такое поле, значит, во
    # избежание проблем в будущем стоит указывать более новую модель.
    deprecated: str


class WordsResponse(TypedDict, total=False):
    # Успешен ли ответ
    is_success: Required[bool]
    used_words: int
    total_words: int
    remaining_words: int
    reserved_words: int
    error_code: str
    error_message: str


class ImagineResponse(TypedDict, total=False):
    status: Required[str]  # starting / failed
    content_id: str
    created_at: str
    input: Any
    model: str
    error_code: str
    error_message: str


class CheckResponse(TypedDict, total=False):
    # Возможные статусы:
    # - pending: Генерация в процессе
    # - completed: Генерация завершена успешно
    # - failed: Генерация завершилась с ошибкой
    # - cancelled: Генерация отменена
    status: Required[str]
    # Временные метки представлены в формате ISO 8601 (UTC)
    created_at: str
    started_at: str
    completed_at: str
    output: list[str]
    error_code: str
    error_message: str


class HTTPMethod(str, Enum):
    """Поддерживаемые HTTP методы для запросов к API."""

    GET = "get"
    POST = "post"


# Константы для валидации и обработки
MAX_PROMPT_LENGTH = 1000
DEFAULT_TIMEOUT_SECONDS = 30

# Коды ошибок клиента
class ErrorCodes:
    """Коды ошибок клиента."""

    HTTP_ERROR_NO_JSON = "CLI-001"
    CONNECTION_ERROR = "CLI-002"
    TIMEOUT_ERROR = "CLI-003"
    REQUEST_ERROR = "CLI-004"
    JSON_DECODE_ERROR = "CLI-005"


class ChadGPTBaseClient:
    BASE_URL = "https://ask.chadgpt.ru/api/public/"

    def __init__(self, api_key: str):
        """
        Инициализирует клиент ChadGPT.

        Args:
            api_key (str): Ваш персональный API-ключ.
        """
        if not api_key or not api_key.strip():
            raise ValueError("API key не может быть пустым.")
        self.api_key = api_key
        self._session = Session()

    def close(self) -> None:
        """Закрывает HTTP-сессию и освобождает соединения."""
        self._session.close()

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
        method_normalized = method.lower()
        try:
            http_method = HTTPMethod(method_normalized)
        except ValueError:
            allowed_methods = ", ".join(sorted(m.value for m in HTTPMethod))
            raise ValueError(f"Unsupported HTTP method '{method}'. Allowed: {allowed_methods}") from None

        merged_headers = {"Content-Type": "application/json"}
        if headers:
            merged_headers.update(headers)

        request_timeout = timeout if timeout is not None else DEFAULT_TIMEOUT_SECONDS
        response: Response | None = None

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
            response.raise_for_status()  # Вызовет HTTPError для статусов 4xx/5xx
            return cast(TResponse, response.json())
        except requests.exceptions.HTTPError as http_err:
            # Попытка парсинга ошибки из ответа API, даже если это HTTP-ошибка
            try:
                if response is not None:
                    return cast(TResponse, response.json())
                raise
            except json.JSONDecodeError:
                return {
                    "is_success": False,
                    "error_code": ErrorCodes.HTTP_ERROR_NO_JSON,
                    "error_message": f"HTTP-ошибка, но не удалось разобрать JSON: {http_err}. "
                    f"Текст ответа: {response.text if response is not None else 'N/A'}",
                }  # type: ignore
        except requests.exceptions.ConnectionError as conn_err:
            return {"is_success": False, "error_code": ErrorCodes.CONNECTION_ERROR, "error_message": f"Ошибка соединения: {conn_err}"}  # type: ignore
        except requests.exceptions.Timeout as timeout_err:
            return {
                "is_success": False,
                "error_code": ErrorCodes.TIMEOUT_ERROR,
                "error_message": f"Превышено время ожидания запроса: {timeout_err}",
            }  # type: ignore
        except requests.exceptions.RequestException as req_err:
            return {
                "is_success": False,
                "error_code": ErrorCodes.REQUEST_ERROR,
                "error_message": f"Непредвиденная ошибка запроса: {req_err}",
            }  # type: ignore
        except json.JSONDecodeError as json_err:
            return {
                "is_success": False,
                "error_code": ErrorCodes.JSON_DECODE_ERROR,
                "error_message": f"Не удалось декодировать JSON ответ: {json_err}. "
                f"Текст ответа: {response.text if response is not None else 'N/A'}",
            }  # type: ignore

    def get_stat_info(self, timeout: int | None = None) -> WordsResponse:
        """
        Позволяет определить общее количество слов, использованных,
        зарезервированных (в текущих генерациях контента) и оставшихся.
        """
        url = f"{self.BASE_URL}words"

        payload: dict[str, Any] = {
            "api_key": self.api_key,
        }
        return self._send_request(response_type=WordsResponse, url=url, method="post", timeout=timeout, payload=payload)


class ChadGPTClient(ChadGPTBaseClient):
    """
    Клиент для взаимодействия с ChadGPT API.

    Позволяет отправлять запросы различным моделям GPT и Claude,
    передавать историю сообщений, изображения и управлять параметрами запроса.
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

    def _validate_history(self, history: list[ChatHistoryRow] | None) -> None:
        """
        Валидирует формат истории сообщений.

        Args:
            history: История сообщений для валидации.

        Raises:
            ValueError: Если формат истории неверный.
        """
        if history is None:
            return

        valid_roles = {"user", "assistant", "system"}
        for i, message in enumerate(history):
            if not isinstance(message, dict):
                raise ValueError(f"История должна содержать словари. Элемент {i} имеет тип {type(message).__name__}")

            role = message.get("role")
            if not role or role not in valid_roles:
                raise ValueError(
                    f"Неверная роль '{role}' в сообщении {i}. Допустимые роли: {', '.join(sorted(valid_roles))}"
                )

            content = message.get("content")
            if not content or not isinstance(content, str):
                raise ValueError(f"Сообщение {i} должно содержать непустое строковое поле 'content'")

    def ask(
        self,
        model_name: str,
        message: str,
        history: list[ChatHistoryRow] | None = None,
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

        Returns:
            dict: JSON-ответ от API.

        Raises:
            ValueError: Если указана неподдерживаемая модель или неверный формат истории.
        """
        if model_name not in self._MODELS:
            raise ValueError(f"Неподдерживаемая модель: '{model_name}'. Доступные модели: {', '.join(self._MODELS)}")

        if model_name in self._DEPRECATED_MODELS:
            warnings.warn(
                f"Модель '{model_name}' устарела и может быть удалена в будущем. "
                "Рассмотрите возможность использования более новых моделей.",
                DeprecationWarning,
                stacklevel=1,
            )

        self._validate_history(history)

        url = f"{self.BASE_URL}{model_name}"

        payload: dict[str, Any] = {
            "message": message,
            "api_key": self.api_key,
        }
        if history is not None:
            payload["history"] = history
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if images is not None:
            payload["images"] = images

        return self._send_request(response_type=ChatResponse, url=url, method="post", timeout=timeout, payload=payload)

    # --- Методы для каждой поддерживаемой модели ---

    def ask_gpt5(
        self,
        message: str,
        history: list[ChatHistoryRow] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        images: list[str] | None = None,
    ) -> ChatResponse:
        """Отправляет запрос модели GPT-5."""
        return self.ask("gpt-5", message, history, temperature, max_tokens, images)

    def ask_gpt5_mini(
        self,
        message: str,
        history: list[ChatHistoryRow] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        images: list[str] | None = None,
    ) -> ChatResponse:
        """Отправляет запрос модели GPT-5 Mini."""
        return self.ask("gpt-5-mini", message, history, temperature, max_tokens, images)

    def ask_gpt5_nano(
        self,
        message: str,
        history: list[ChatHistoryRow] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        images: list[str] | None = None,
    ) -> ChatResponse:
        """Отправляет запрос модели GPT-5 Nano."""
        return self.ask("gpt-5-nano", message, history, temperature, max_tokens, images)

    def ask_gpt4o_mini(
        self,
        message: str,
        history: list[ChatHistoryRow] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        images: list[str] | None = None,
    ) -> ChatResponse:
        """Отправляет запрос модели GPT-4o Mini (устаревшая)."""
        return self.ask("gpt-4o-mini", message, history, temperature, max_tokens, images)

    def ask_gpt4o(
        self,
        message: str,
        history: list[ChatHistoryRow] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        images: list[str] | None = None,
    ) -> ChatResponse:
        """Отправляет запрос модели GPT-4o (устаревшая)."""
        return self.ask("gpt-4o", message, history, temperature, max_tokens, images)

    def ask_claude_3_haiku(
        self,
        message: str,
        history: list[ChatHistoryRow] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        images: list[str] | None = None,
    ) -> ChatResponse:
        """Отправляет запрос модели Claude 3 Haiku (устаревшая)."""
        return self.ask("claude-3-haiku", message, history, temperature, max_tokens, images)

    def ask_claude_3_opus(
        self,
        message: str,
        history: list[ChatHistoryRow] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        images: list[str] | None = None,
    ) -> ChatResponse:
        """Отправляет запрос модели Claude 3 Opus (устаревшая)."""
        return self.ask("claude-3-opus", message, history, temperature, max_tokens, images)

    def ask_claude_4_1_opus(
        self,
        message: str,
        history: list[ChatHistoryRow] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        images: list[str] | None = None,
    ) -> ChatResponse:
        """Отправляет запрос модели Claude 4.1 Opus."""
        return self.ask("claude-4.1-opus", message, history, temperature, max_tokens, images)

    def ask_claude_4_5_sonnet(
        self,
        message: str,
        history: list[ChatHistoryRow] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        images: list[str] | None = None,
    ) -> ChatResponse:
        """Отправляет запрос модели Claude 4.5 Sonnet."""
        return self.ask("claude-4.5-sonnet", message, history, temperature, max_tokens, images)

    def ask_claude_3_7_sonnet_thinking(
        self,
        message: str,
        history: list[ChatHistoryRow] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        images: list[str] | None = None,
    ) -> ChatResponse:
        """Отправляет запрос модели Claude 3.7 Sonnet Thinking (устаревшая)."""
        return self.ask("claude-3.7-sonnet-thinking", message, history, temperature, max_tokens, images)

    def ask_gemini_2_0_flash(
        self,
        message: str,
        history: list[ChatHistoryRow] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        images: list[str] | None = None,
    ) -> ChatResponse:
        """Отправляет запрос модели Gemini 2.0 Flash."""
        return self.ask("gemini-2.0-flash", message, history, temperature, max_tokens, images)

    def ask_gemini_2_5_pro(
        self,
        message: str,
        history: list[ChatHistoryRow] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        images: list[str] | None = None,
    ) -> ChatResponse:
        """Отправляет запрос модели Gemini 2.5 Pro."""
        return self.ask("gemini-2.5-pro", message, history, temperature, max_tokens, images)

    def ask_deepseek_v3_1(
        self,
        message: str,
        history: list[ChatHistoryRow] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        images: list[str] | None = None,
    ) -> ChatResponse:
        """Отправляет запрос модели Deepseek v3.1."""
        return self.ask("deepseek-v3.1", message, history, temperature, max_tokens, images)


class ChadGPTImageClient(ChadGPTBaseClient):
    @property
    def check_url(self) -> str:
        """URL для проверки статуса генерации изображений."""
        return f"{self.BASE_URL}check"

    def _validate_imagine_params(self, model_name: str, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """
        Приватный метод для валидации параметров запроса генерации изображения.

        Args:
            model_name (str): Имя модели.
            prompt (str): Промпт.
            kwargs (dict): Дополнительные параметры.

        Returns:
            dict: Валидированный payload для отправки.

        Raises:
            ValueError: Если параметры не соответствуют требованиям.
        """

        if model_name not in MODEL_VALIDATORS:
            raise ValueError(
                f"Unsupported model: '{model_name}'. Available models: {', '.join(MODEL_VALIDATORS.keys())}"
            )

        prompt_stripped = prompt.strip()
        if not prompt_stripped or len(prompt_stripped) > MAX_PROMPT_LENGTH:
            raise ValueError(
                f"Prompt must be a non-empty string with max {MAX_PROMPT_LENGTH} characters. Current length: {len(prompt_stripped)}"
            )

        payload = {"api_key": self.api_key, "prompt": prompt_stripped}
        validator = MODEL_VALIDATORS[model_name]
        params = validator(**kwargs)
        # Конвертируем Pydantic модель в словарь, исключая None значения
        params_dict = params.model_dump(exclude_none=True)
        payload.update(params_dict)
        return payload

    def imagine(self, model_name: str, prompt: str, **kwargs: Any) -> ImagineResponse:
        """
        Запускает генерацию изображения с использованием указанной модели.

        Args:
            model_name (str): Имя модели для генерации (например, "imagen-4", "mj-6").
            prompt (str): Текстовое описание изображения (1-1000 символов).
            **kwargs: Дополнительные параметры, специфичные для модели.

        Returns:
            dict: JSON-ответ от API, содержащий статус и content_id.

        Raises:
            ValueError: Если API ключ, промпт или параметры недействительны, или при ошибке API.
        """
        payload = self._validate_imagine_params(model_name, prompt, **kwargs)
        url = f"{self.BASE_URL}{model_name}/imagine"
        return self._send_request(response_type=ImagineResponse, url=url, method="post", payload=payload)

    def check_status(self, content_id: str) -> CheckResponse:
        """
        Проверяет статус генерации изображения по его content_id.

        Args:
            content_id (str): ID контента, полученный из ответа imagine.

        Returns:
            CheckResponse: JSON-ответ от API, содержащий статус генерации и, при завершении, ссылки на изображения.

        Raises:
            ValueError: Если API ключ или content_id недействительны, или при ошибке API.
        """
        if not isinstance(content_id, str) or not content_id.strip():
            raise ValueError("Content ID must be a non-empty string.")

        payload = {"api_key": self.api_key, "content_id": content_id}
        return self._send_request(response_type=CheckResponse, url=self.check_url, method="get", params=payload)
