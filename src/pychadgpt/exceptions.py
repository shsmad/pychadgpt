# ============================================================================
# КАСТОМНЫЕ ИСКЛЮЧЕНИЯ
# ============================================================================

class ChadGPTError(Exception):
    """Базовое исключение для всех ошибок ChadGPT API."""

    def __init__(self, message: str, error_code: str | None = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

class ChadGPTConnectionError(ChadGPTError):
    """Ошибка соединения с API."""
    pass

class ChadGPTTimeoutError(ChadGPTError):
    """Превышено время ожидания запроса."""
    pass

class ChadGPTHTTPError(ChadGPTError):
    """HTTP ошибка при запросе к API."""

    def __init__(self, message: str, status_code: int, error_code: str | None = None):
        self.status_code = status_code
        super().__init__(message, error_code)

class ChadGPTValidationError(ChadGPTError):
    """Ошибка валидации параметров запроса."""
    pass

class ChadGPTJSONDecodeError(ChadGPTError):
    """Ошибка декодирования JSON ответа."""
    pass

class ChadGPTAPIError(ChadGPTError):
    """Ошибка, возвращенная API (is_success=False)."""
    pass
