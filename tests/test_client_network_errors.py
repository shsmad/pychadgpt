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

        result = client.ask("gpt-5", "Hello")

        assert result["is_success"] is False
        assert result["error_code"] == "CLI-002"
        assert "Ошибка соединения" in result["error_message"]

    def test_timeout_error(self, mocker: MockerFixture) -> None:
        """Проверка обработки ошибки таймаута."""
        client = ChadGPTClient("test-api-key")
        mocker.patch.object(
            client._session,
            "request",
            side_effect=requests.exceptions.Timeout("Request timeout"),
        )

        result = client.ask("gpt-5", "Hello")

        assert result["is_success"] is False
        assert result["error_code"] == "CLI-003"
        assert "Превышено время ожидания" in result["error_message"]

    def test_http_error_with_json_response(self, mocker: MockerFixture) -> None:
        """Проверка обработки HTTP ошибки с JSON ответом."""
        client = ChadGPTClient("test-api-key")
        mock_response = mocker.Mock()
        mock_response.json.return_value = {
            "is_success": False,
            "error_code": "API-001",
            "error_message": "Invalid API key",
        }
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "400 Bad Request", response=mock_response
        )
        mocker.patch.object(client._session, "request", return_value=mock_response)

        result = client.ask("gpt-5", "Hello")

        # Должен вернуться JSON ответ с ошибкой от API
        assert result["is_success"] is False
        assert result["error_code"] == "API-001"
        assert "Invalid API key" in result["error_message"]

    def test_http_error_without_json_response(self, mocker: MockerFixture) -> None:
        """Проверка обработки HTTP ошибки без JSON ответа."""
        client = ChadGPTClient("test-api-key")
        mock_response = mocker.Mock()
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_response.text = "Internal Server Error"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "500 Internal Server Error", response=mock_response
        )
        mocker.patch.object(client._session, "request", return_value=mock_response)

        result = client.ask("gpt-5", "Hello")

        assert result["is_success"] is False
        assert result["error_code"] == "CLI-001"
        assert "HTTP-ошибка" in result["error_message"]
        assert "Internal Server Error" in result["error_message"]

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
        mock_response.raise_for_status = mocker.Mock()
        mocker.patch.object(client._session, "request", return_value=mock_response)

        result = client.ask("gpt-5", "Hello")

        assert result["is_success"] is False
        assert result["error_code"] == "CLI-005"
        assert "Не удалось декодировать JSON" in result["error_message"]
        assert "Not a JSON response" in result["error_message"]

    def test_generic_request_exception(self, mocker: MockerFixture) -> None:
        """Проверка обработки общей ошибки запроса."""
        client = ChadGPTClient("test-api-key")
        mocker.patch.object(
            client._session,
            "request",
            side_effect=requests.exceptions.RequestException("Generic request error"),
        )

        result = client.ask("gpt-5", "Hello")

        assert result["is_success"] is False
        assert result["error_code"] == "CLI-004"
        assert "Непредвиденная ошибка запроса" in result["error_message"]


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

        result = client.get_stat_info()

        assert result["is_success"] is False
        assert result["error_code"] == "CLI-002"


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

        result = client.imagine("imagen-4", "A beautiful landscape")

        assert result.get("status") == "failed" or result.get("error_code") == "CLI-002"

    def test_check_status_timeout(self, mocker: MockerFixture) -> None:
        """Проверка обработки таймаута в check_status."""
        client = ChadGPTImageClient("test-api-key")
        mocker.patch.object(
            client._session,
            "request",
            side_effect=requests.exceptions.Timeout("Request timeout"),
        )

        result = client.check_status("test-content-id")

        # Результат зависит от реализации, но должен быть в формате CheckResponse
        assert "error_code" in result or "status" in result
