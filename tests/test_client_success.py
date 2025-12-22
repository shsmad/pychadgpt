"""Тесты для успешных сценариев работы клиента ChadGPT."""

import json

from pytest_mock import MockerFixture

from pychadgpt.client import (
    DEFAULT_TIMEOUT_SECONDS,
    ChadGPTBaseClient,
    ChadGPTClient,
    ChadGPTImageClient,
    ChatHistoryMessage,
)


class TestSuccessfulScenarios:
    """Тесты успешных сценариев работы клиента."""

    def test_successful_ask_request(self, mocker: MockerFixture) -> None:
        """Проверка успешного запроса к модели."""
        client = ChadGPTClient("test-api-key")
        mock_response = mocker.Mock()
        mock_response.json.return_value = {
            "is_success": True,
            "response": "Hello! How can I help you?",
            "used_words_count": 10,
        }
        mock_response.content = (
            b'{"is_success": true, "response": "Hello! How can I help you?", "used_words_count": 10}'
        )
        mock_response.raise_for_status = mocker.Mock()
        mock_request = mocker.patch.object(client._session, "request", return_value=mock_response)

        result = client.ask("gpt-5", "Hello")

        assert result.is_success is True
        assert result.response == "Hello! How can I help you?"
        assert result.used_words_count == 10

        # Проверяем, что запрос был сделан с правильными параметрами
        call_args = mock_request.call_args
        assert call_args is not None
        assert call_args.kwargs["json"]["message"] == "Hello"
        assert call_args.kwargs["json"]["api_key"] == "test-api-key"

    def test_successful_get_stat_info(self, mocker: MockerFixture) -> None:
        """Проверка успешного получения статистики."""
        client = ChadGPTBaseClient("test-api-key")
        mock_response = mocker.Mock()
        mock_response.json.return_value = {
            "is_success": True,
            "used_words": 100,
            "total_words": 1000,
            "remaining_words": 900,
            "reserved_words": 0,
        }
        mock_response.content = (
            b'{"is_success": true, "used_words": 100, "total_words": 1000, "remaining_words": 900, "reserved_words": 0}'
        )
        mock_response.raise_for_status = mocker.Mock()
        mock_request = mocker.patch.object(client._session, "request", return_value=mock_response)

        result = client.get_stat_info()

        assert result.is_success is True
        assert result.used_words == 100
        assert result.total_words == 1000
        assert result.remaining_words == 900

        # Проверяем, что запрос был сделан на правильный URL
        call_args = mock_request.call_args
        assert call_args is not None
        assert "words" in call_args.kwargs["url"]

    def test_successful_imagine_request(self, mocker: MockerFixture) -> None:
        """Проверка успешного запроса генерации изображения."""
        client = ChadGPTImageClient("test-api-key")
        mock_response = mocker.Mock()
        mock_response.json.return_value = {
            "status": "starting",
            "content_id": "test-content-id-123",
            "created_at": "2024-01-01T00:00:00Z",
        }
        mock_response.content = (
            b'{"status": "starting", "content_id": "test-content-id-123", "created_at": "2024-01-01T00:00:00Z"}'
        )
        mock_response.raise_for_status = mocker.Mock()
        mock_request = mocker.patch.object(client._session, "request", return_value=mock_response)

        result = client.imagine("imagen-4", "A beautiful sunset", aspect_ratio="16:9")

        assert result.status == "starting"
        assert result.content_id == "test-content-id-123"

        # Проверяем, что параметры переданы корректно
        call_args = mock_request.call_args
        assert call_args is not None
        assert call_args.kwargs["json"]["prompt"] == "A beautiful sunset"
        assert call_args.kwargs["json"]["aspect_ratio"] == "16:9"
        assert call_args.kwargs["json"]["api_key"] == "test-api-key"

    def test_successful_check_status(self, mocker: MockerFixture) -> None:
        """Проверка успешной проверки статуса генерации."""
        client = ChadGPTImageClient("test-api-key")
        mock_response = mocker.Mock()
        mock_response.json.return_value = {
            "status": "completed",
            "created_at": "2024-01-01T00:00:00Z",
            "started_at": "2024-01-01T00:01:00Z",
            "completed_at": "2024-01-01T00:02:00Z",
            "output": ["https://example.com/image1.jpg", "https://example.com/image2.jpg"],
        }
        mock_response.content = json.dumps(mock_response.json.return_value)
        mock_response.raise_for_status = mocker.Mock()
        mock_request = mocker.patch.object(client._session, "request", return_value=mock_response)

        result = client.check_status("test-content-id-123")

        assert result.status == "completed"
        assert len(result.output) == 2
        assert "image1.jpg" in result.output[0]

        # Проверяем, что GET запрос сделан с правильными параметрами
        call_args = mock_request.call_args
        assert call_args is not None
        assert call_args.kwargs["method"] == "post"
        assert "api_key" in call_args.kwargs["json"]
        assert call_args.kwargs["json"]["content_id"] == "test-content-id-123"

    def test_custom_timeout(self, mocker: MockerFixture) -> None:
        """Проверка использования пользовательского таймаута."""
        client = ChadGPTClient("test-api-key")
        mock_response = mocker.Mock()
        mock_response.json.return_value = {"is_success": True, "response": "test"}
        mock_response.raise_for_status = mocker.Mock()
        mock_response.content = json.dumps(mock_response.json.return_value)
        mock_request = mocker.patch.object(client._session, "request", return_value=mock_response)

        client.ask("gpt-5", "Hello", timeout=60)

        call_args = mock_request.call_args
        assert call_args is not None
        assert call_args.kwargs["timeout"] == 60

    def test_default_timeout(self, mocker: MockerFixture) -> None:
        """Проверка использования таймаута по умолчанию."""
        client = ChadGPTClient("test-api-key")
        mock_response = mocker.Mock()
        mock_response.json.return_value = {"is_success": True, "response": "test"}
        mock_response.raise_for_status = mocker.Mock()
        mock_response.content = json.dumps(mock_response.json.return_value)
        mock_request = mocker.patch.object(client._session, "request", return_value=mock_response)

        client.ask("gpt-5", "Hello")

        call_args = mock_request.call_args
        assert call_args is not None
        assert call_args.kwargs["timeout"] == DEFAULT_TIMEOUT_SECONDS

    def test_get_request_with_params(self, mocker: MockerFixture) -> None:
        """Проверка GET запроса с параметрами в query string."""
        client = ChadGPTBaseClient("test-api-key")
        mock_response = mocker.Mock()
        mock_response.json.return_value = {"test": "data"}
        mock_response.raise_for_status = mocker.Mock()
        mock_response.content = json.dumps(mock_response.json.return_value)
        mock_request = mocker.patch.object(client._session, "request", return_value=mock_response)

        client._send_request(
            response_type=dict,
            url="https://example.com/api",
            method="get",
            params={"param1": "value1"},
            payload={"api_key": "test-key"},
        )

        call_args = mock_request.call_args
        assert call_args is not None
        assert call_args.kwargs["method"] == "get"
        # Проверяем, что payload перенесен в params для GET
        assert "params" in call_args.kwargs
        assert call_args.kwargs["params"]["api_key"] == "test-key"
        assert call_args.kwargs["params"]["param1"] == "value1"
        # Для GET не должно быть json в теле запроса
        assert "json" not in call_args.kwargs

    def test_post_request_with_payload(self, mocker: MockerFixture) -> None:
        """Проверка POST запроса с payload в теле."""
        client = ChadGPTBaseClient("test-api-key")
        mock_response = mocker.Mock()
        mock_response.json.return_value = {"test": "data"}
        mock_response.raise_for_status = mocker.Mock()
        mock_response.content = json.dumps(mock_response.json.return_value)
        mock_request = mocker.patch.object(client._session, "request", return_value=mock_response)

        payload = {"message": "Hello", "api_key": "test-key"}
        client._send_request(
            response_type=dict,
            url="https://example.com/api",
            method="post",
            payload=payload,
        )

        call_args = mock_request.call_args
        assert call_args is not None
        assert call_args.kwargs["method"] == "post"
        # Проверяем, что payload в json для POST
        assert "json" in call_args.kwargs
        assert call_args.kwargs["json"] == payload


class TestContextManager:
    """Тесты использования клиента как контекстного менеджера."""

    def test_context_manager_closes_session(self, mocker: MockerFixture) -> None:
        """Проверка, что контекстный менеджер закрывает сессию."""
        client = ChadGPTBaseClient("test-api-key")
        session_close = mocker.spy(client._session, "close")

        with client:
            assert client._session is not None

        # После выхода из контекста сессия должна быть закрыта
        session_close.assert_called_once()

    def test_context_manager_returns_self(self) -> None:
        """Проверка, что контекстный менеджер возвращает сам объект."""
        client = ChadGPTBaseClient("test-api-key")

        with client as ctx:
            assert ctx is client

    def test_manual_close(self, mocker: MockerFixture) -> None:
        """Проверка ручного закрытия сессии."""
        client = ChadGPTBaseClient("test-api-key")
        session_close = mocker.spy(client._session, "close")

        client.close()

        session_close.assert_called_once()


class TestModelSpecificMethods:
    """Тесты методов для конкретных моделей."""

    def test_ask_gpt_5(self, mocker: MockerFixture) -> None:
        """Проверка метода ask_gpt5."""
        client = ChadGPTClient("test-api-key")
        mock_response = mocker.Mock()
        mock_response.json.return_value = {"is_success": True, "response": "test"}
        mock_response.raise_for_status = mocker.Mock()
        mock_response.content = json.dumps(mock_response.json.return_value)
        mock_request = mocker.patch.object(client._session, "request", return_value=mock_response)

        client.ask_gpt_5("Hello") # type: ignore

        call_args = mock_request.call_args
        assert call_args is not None
        assert "gpt-5" in call_args.kwargs["url"]

    def test_ask_claude_4_5_sonnet(self, mocker: MockerFixture) -> None:
        """Проверка метода ask_claude_4_5_sonnet."""
        client = ChadGPTClient("test-api-key")
        mock_response = mocker.Mock()
        mock_response.json.return_value = {"is_success": True, "response": "test"}
        mock_response.raise_for_status = mocker.Mock()
        mock_response.content = json.dumps(mock_response.json.return_value)
        mock_request = mocker.patch.object(client._session, "request", return_value=mock_response)

        client.ask_claude_4_5_sonnet("Hello") # type: ignore

        call_args = mock_request.call_args
        assert call_args is not None
        assert "claude-4.5-sonnet" in call_args.kwargs["url"]

    def test_ask_with_all_params(self, mocker: MockerFixture) -> None:
        """Проверка запроса со всеми опциональными параметрами."""
        client = ChadGPTClient("test-api-key")
        mock_response = mocker.Mock()
        mock_response.json.return_value = {"is_success": True, "response": "test"}
        mock_response.raise_for_status = mocker.Mock()
        mock_response.content = json.dumps(mock_response.json.return_value)
        mock_request = mocker.patch.object(client._session, "request", return_value=mock_response)

        history: list[ChatHistoryMessage] = [
            ChatHistoryMessage(role="user", content="Previous"),
            ChatHistoryMessage(role="assistant", content="Response"),
        ]
        images = ["https://example.com/img1.jpg", "https://example.com/img2.jpg"]

        client.ask(
            "gpt-5",
            "Hello",
            history=history,
            temperature=0.8,
            max_tokens=200,
            images=images,
            timeout=45,
        )

        call_args = mock_request.call_args
        assert call_args is not None
        payload = call_args.kwargs["json"]
        assert payload["history"] == [x.model_dump(exclude_none=True) for x in history]
        assert payload["temperature"] == 0.8
        assert payload["max_tokens"] == 200
        assert payload["images"] == images
        assert call_args.kwargs["timeout"] == 45
