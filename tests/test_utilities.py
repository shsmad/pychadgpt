"""Тесты для утилитарных функций."""

import logging

from pychadgpt.client import _mask_api_key, setup_logging


class TestMaskAPIKey:
    """Тесты функции _mask_api_key."""

    def test_mask_short_key(self) -> None:
        """Проверка маскировки короткого ключа."""
        assert _mask_api_key("") == "***"
        assert _mask_api_key("short") == "***"
        assert _mask_api_key("1234567") == "***"  # 7 символов

    def test_mask_long_key(self) -> None:
        """Проверка маскировки длинного ключа."""
        key = "abcdefghijklmnopqrstuvwxyz"
        masked = _mask_api_key(key)
        assert masked == "abcd...wxyz"
        assert len(masked) < len(key)

    def test_mask_medium_key(self) -> None:
        """Проверка маскировки ключа средней длины."""
        key = "12345678"  # 8 символов
        masked = _mask_api_key(key)
        assert masked == "1234...5678"

    def test_mask_whitespace_key(self) -> None:
        """Проверка маскировки ключа из пробелов."""
        # Ключ из пробелов считается валидным, если длина >= 8
        assert _mask_api_key("        ") == "    ...    "  # 8 пробелов
        assert _mask_api_key("   ") == "***"  # 3 пробела - короткий


class TestSetupLogging:
    """Тесты функции setup_logging."""

    def test_setup_logging_default_level(self) -> None:
        """Проверка настройки логирования с уровнем по умолчанию."""
        # Сбрасываем настройки логирования
        logging.root.handlers = []
        logging.root.setLevel(logging.WARNING)

        setup_logging()

        assert logging.root.level == logging.INFO

    def test_setup_logging_custom_level(self) -> None:
        """Проверка настройки логирования с кастомным уровнем."""
        logging.root.handlers = []
        logging.root.setLevel(logging.WARNING)

        setup_logging(logging.DEBUG)

        assert logging.root.level == logging.DEBUG

    def test_setup_logging_warning_level(self) -> None:
        """Проверка настройки логирования с уровнем WARNING."""
        logging.root.handlers = []
        logging.root.setLevel(logging.INFO)

        setup_logging(logging.WARNING)

        assert logging.root.level == logging.WARNING
