"""Тесты для обработки сетевых ошибок клиента ChadGPT."""

import pytest
import requests

from pychadgpt.client import (
    ChadGPTClient,
    ChadGPTImageClient,
)
from pychadgpt.exceptions import (
    ChadGPTAPIError,
    ChadGPTConnectionError,
    ChadGPTError,
    ChadGPTHTTPError,
    ChadGPTJSONDecodeError,
    ChadGPTTimeoutError,
)


@pytest.fixture
def mock_client() -> ChadGPTClient:
    return ChadGPTClient("test-api-key", base_url="mock://example.com/")


@pytest.fixture
def mock_image_client() -> ChadGPTImageClient:
    return ChadGPTImageClient("test-api-key", base_url="mock://example.com/")


class TestNetworkErrors:
    """Тесты обработки различных сетевых ошибок."""

    def test_connection_error(self, mock_client, requests_mock) -> None:
        """Проверка обработки ошибки соединения."""

        requests_mock.post("mock://example.com/gpt-5", exc=requests.exceptions.ConnectionError("Connection refused"))

        with pytest.raises(ChadGPTConnectionError) as exc_info:
            mock_client.ask("gpt-5", "Hello")

        assert exc_info.value.error_code == "CONNECTION_ERROR"

    def test_timeout_error(self, mock_client, requests_mock) -> None:
        """Проверка обработки ошибки таймаута."""

        requests_mock.post(
            "mock://example.com/gpt-5",
            exc=requests.exceptions.Timeout("Request timeout"),
        )

        with pytest.raises(ChadGPTTimeoutError) as exc_info:
            mock_client.ask("gpt-5", "Hello")

        assert exc_info.value.error_code == "TIMEOUT_ERROR"

    def test_http_error_with_json_response(self, mock_client, requests_mock) -> None:
        """Проверка обработки HTTP ошибки с JSON ответом."""

        requests_mock.post(
            "mock://example.com/gpt-5",
            json={"is_success": False, "error_code": "API-001", "error_message": "Invalid API key"},
            headers={"Content-type": "application/json"},
        )

        with pytest.raises(ChadGPTAPIError) as exc_info:
            mock_client.ask("gpt-5", "Hello")

        assert exc_info.value.error_code == "API-001"

    def test_http_error_without_json_response(self, mock_client, requests_mock) -> None:
        """Проверка обработки HTTP ошибки без JSON ответа."""

        requests_mock.post(
            "mock://example.com/gpt-5",
            status_code=500,
            text="Internal Server Error",
        )

        with pytest.raises(ChadGPTHTTPError) as exc_info:
            mock_client.ask("gpt-5", "Hello")

        assert exc_info.value.error_code == "HTTP_ERROR"

    def test_http_error_no_response_object(self, mock_client, requests_mock) -> None:
        """Проверка обработки HTTP ошибки без объекта ответа."""

        requests_mock.post(
            "mock://example.com/gpt-5",
            exc=requests.exceptions.HTTPError("500 Internal Server Error"),
        )

        with pytest.raises(ChadGPTHTTPError) as exc_info:
            mock_client.ask("gpt-5", "Hello")

        assert exc_info.value.error_code == "HTTP_ERROR"

    def test_json_decode_error(self, mock_client, requests_mock) -> None:
        """Проверка обработки ошибки декодирования JSON."""

        requests_mock.post(
            "mock://example.com/gpt-5",
            text="Not a JSON response",
            headers={"Content-type": "application/json"},
        )

        with pytest.raises(ChadGPTJSONDecodeError) as exc_info:
            mock_client.ask("gpt-5", "Hello")

        assert exc_info.value.error_code == "JSON_DECODE_ERROR"

    def test_generic_request_exception(self, mock_client, requests_mock) -> None:
        """Проверка обработки общей ошибки запроса."""

        requests_mock.post(
            "mock://example.com/gpt-5",
            exc=requests.exceptions.RequestException("Generic request error"),
        )

        with pytest.raises(ChadGPTError) as exc_info:
            mock_client.ask("gpt-5", "Hello")

        assert exc_info.value.error_code == "REQUEST_ERROR"


class TestBaseClientNetworkErrors:
    """Тесты сетевых ошибок для базового клиента."""

    def test_get_stat_info_connection_error(self, mock_client, requests_mock) -> None:
        """Проверка обработки ошибки соединения в get_stat_info."""

        requests_mock.post("mock://example.com/words", exc=requests.exceptions.ConnectionError("Connection refused"))

        with pytest.raises(ChadGPTConnectionError) as exc_info:
            mock_client.get_stat_info()

        assert exc_info.value.error_code == "CONNECTION_ERROR"


class TestImageClientNetworkErrors:
    """Тесты сетевых ошибок для клиента изображений."""

    def test_imagine_connection_error(self, mock_image_client, requests_mock) -> None:
        """Проверка обработки ошибки соединения в imagine."""

        requests_mock.post(
            "mock://example.com/imagen-4/imagine",
            exc=requests.exceptions.ConnectionError("Connection refused"),
        )

        with pytest.raises(ChadGPTConnectionError) as exc_info:
            mock_image_client.imagine("imagen-4", "A beautiful landscape")

        assert exc_info.value.error_code == "CONNECTION_ERROR"

    def test_check_status_timeout(self, mock_image_client, requests_mock) -> None:
        """Проверка обработки таймаута в check_status."""

        requests_mock.post(
            "mock://example.com/check",
            exc=requests.exceptions.Timeout("Request timeout"),
        )

        with pytest.raises(ChadGPTTimeoutError) as exc_info:
            mock_image_client.check_status("test-content-id")

        assert exc_info.value.error_code == "TIMEOUT_ERROR"
