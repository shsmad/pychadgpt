"""Тесты для обработки сетевых ошибок клиента ChadGPT."""

import json

import pytest
import requests

from pytest_mock import MockerFixture

from pychadgpt.client import (
    ChadGPTBaseClient,
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


class TestNetworkErrors:
    """Тесты обработки различных сетевых ошибок."""

    def test_connection_error(self, mocker: MockerFixture) -> None:
        """Проверка обработки ошибки соединения."""
        client = ChadGPTClient("test-api-key")
        mocker.patch.object(
            client._session,
            "request",
            side_effect=requests.exceptions.ConnectionError("Connection refused"),
        )

        with pytest.raises(ChadGPTConnectionError) as exc_info:
            client.ask("gpt-5", "Hello")

        assert exc_info.value.error_code == "CONNECTION_ERROR"

    def test_timeout_error(self, mocker: MockerFixture) -> None:
        """Проверка обработки ошибки таймаута."""
        client = ChadGPTClient("test-api-key")
        mocker.patch.object(
            client._session,
            "request",
            side_effect=requests.exceptions.Timeout("Request timeout"),
        )

        with pytest.raises(ChadGPTTimeoutError) as exc_info:
            client.ask("gpt-5", "Hello")

        assert exc_info.value.error_code == "TIMEOUT_ERROR"

    def test_http_error_with_json_response(self, mocker: MockerFixture) -> None:
        """Проверка обработки HTTP ошибки с JSON ответом."""
        client = ChadGPTClient("test-api-key")
        mock_response = mocker.Mock()
        mock_response.json.return_value = {
            "is_success": False,
            "error_code": "API-001",
            "error_message": "Invalid API key",
        }
        mock_response.content = json.dumps(mock_response.json.return_value)
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "400 Bad Request", response=mock_response
        )
        mocker.patch.object(client._session, "request", return_value=mock_response)

        with pytest.raises(ChadGPTAPIError) as exc_info:
            client.ask("gpt-5", "Hello")

        assert exc_info.value.error_code == "API-001"

    def test_http_error_without_json_response(self, mocker: MockerFixture) -> None:
        """Проверка обработки HTTP ошибки без JSON ответа."""
        client = ChadGPTClient("test-api-key")
        mock_response = mocker.Mock()
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_response.text = "Internal Server Error"
        mock_response.content = b"Internal Server Error"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "500 Internal Server Error", response=mock_response
        )
        mocker.patch.object(client._session, "request", return_value=mock_response)

        with pytest.raises(ChadGPTHTTPError) as exc_info:
            client.ask("gpt-5", "Hello")

        assert exc_info.value.error_code == "HTTP_ERROR"

    def test_http_error_no_response_object(self, mocker: MockerFixture) -> None:
        """Проверка обработки HTTP ошибки без объекта ответа."""
        client = ChadGPTClient("test-api-key")
        http_error = requests.exceptions.HTTPError("500 Internal Server Error")
        http_error.response = None  # type: ignore[assignment]
        mocker.patch.object(client._session, "request", side_effect=http_error)

        with pytest.raises(requests.exceptions.HTTPError):
            client.ask("gpt-5", "Hello")

    def test_json_decode_error(self, mocker: MockerFixture) -> None:
        """Проверка обработки ошибки декодирования JSON."""
        client = ChadGPTClient("test-api-key")
        mock_response = mocker.Mock()
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_response.text = "Not a JSON response"
        mock_response.content = b"Not a JSON response"
        mock_response.raise_for_status = mocker.Mock()
        mocker.patch.object(client._session, "request", return_value=mock_response)

        with pytest.raises(ChadGPTJSONDecodeError) as exc_info:
            client.ask("gpt-5", "Hello")

        assert exc_info.value.error_code == "JSON_DECODE_ERROR"

    def test_generic_request_exception(self, mocker: MockerFixture) -> None:
        """Проверка обработки общей ошибки запроса."""
        client = ChadGPTClient("test-api-key")
        mocker.patch.object(
            client._session,
            "request",
            side_effect=requests.exceptions.RequestException("Generic request error"),
        )

        with pytest.raises(ChadGPTError) as exc_info:
            client.ask("gpt-5", "Hello")

        assert exc_info.value.error_code == "REQUEST_ERROR"


class TestBaseClientNetworkErrors:
    """Тесты сетевых ошибок для базового клиента."""

    def test_get_stat_info_connection_error(self, mocker: MockerFixture) -> None:
        """Проверка обработки ошибки соединения в get_stat_info."""
        client = ChadGPTBaseClient("test-api-key")
        mocker.patch.object(
            client._session,
            "request",
            side_effect=requests.exceptions.ConnectionError("Connection refused"),
        )

        with pytest.raises(ChadGPTConnectionError) as exc_info:
            client.get_stat_info()

        assert exc_info.value.error_code == "CONNECTION_ERROR"


class TestImageClientNetworkErrors:
    """Тесты сетевых ошибок для клиента изображений."""

    def test_imagine_connection_error(self, mocker: MockerFixture) -> None:
        """Проверка обработки ошибки соединения в imagine."""
        client = ChadGPTImageClient("test-api-key")
        mocker.patch.object(
            client._session,
            "request",
            side_effect=requests.exceptions.ConnectionError("Connection refused"),
        )
        with pytest.raises(ChadGPTConnectionError) as exc_info:
            client.imagine("imagen-4", "A beautiful landscape")

        assert exc_info.value.error_code == "CONNECTION_ERROR"

    def test_check_status_timeout(self, mocker: MockerFixture) -> None:
        """Проверка обработки таймаута в check_status."""
        client = ChadGPTImageClient("test-api-key")
        mocker.patch.object(
            client._session,
            "request",
            side_effect=requests.exceptions.Timeout("Request timeout"),
        )
        with pytest.raises(ChadGPTTimeoutError) as exc_info:
            client.check_status("test-content-id")

        assert exc_info.value.error_code == "TIMEOUT_ERROR"
