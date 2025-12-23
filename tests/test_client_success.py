"""Тесты для успешных сценариев работы клиента ChadGPT."""


from pytest_mock import MockerFixture

from pychadgpt.client import (
    ChadGPTBaseClient,
    ChadGPTClient,
    ChadGPTImageClient,
    ChatHistoryMessage,
)
from pychadgpt.constants import DEFAULT_TIMEOUT_SECONDS


class TestSuccessfulScenarios:
    """Тесты успешных сценариев работы клиента."""

    def test_successful_ask_request(self, requests_mock) -> None:
        """Проверка успешного запроса к модели."""
        client = ChadGPTClient("test-api-key")
        requests_mock.post(
            "https://ask.chadgpt.ru/api/public/gpt-5",
            json={
                "is_success": True,
                "response": "Hello! How can I help you?",
                "used_words_count": 10,
            },
        )

        result = client.ask("gpt-5", "Hello")

        assert result.is_success is True
        assert result.response == "Hello! How can I help you?"
        assert result.used_words_count == 10

        # Проверяем, что запрос был сделан с правильными параметрами
        assert requests_mock.call_count == 1
        request = requests_mock.request_history[0]
        assert request.json()["message"] == "Hello"
        assert request.json()["api_key"] == "test-api-key"

    def test_successful_get_stat_info(self, requests_mock) -> None:
        """Проверка успешного получения статистики."""
        client = ChadGPTBaseClient("test-api-key")
        requests_mock.post(
            "https://ask.chadgpt.ru/api/public/words",
            json={
                "is_success": True,
                "used_words": 100,
                "total_words": 1000,
                "remaining_words": 900,
                "reserved_words": 0,
            },
        )

        result = client.get_stat_info()

        assert result.is_success is True
        assert result.used_words == 100
        assert result.total_words == 1000
        assert result.remaining_words == 900

        # Проверяем, что запрос был сделан на правильный URL
        assert requests_mock.call_count == 1
        assert "words" in requests_mock.request_history[0].url

    def test_successful_imagine_request(self, requests_mock) -> None:
        """Проверка успешного запроса генерации изображения."""
        client = ChadGPTImageClient("test-api-key")
        requests_mock.post(
            "https://ask.chadgpt.ru/api/public/imagen-4/imagine",
            json={
                "status": "starting",
                "content_id": "test-content-id-123",
                "created_at": "2024-01-01T00:00:00Z",
            },
        )

        result = client.imagine("imagen-4", "A beautiful sunset", aspect_ratio="16:9")

        assert result.status == "starting"
        assert result.content_id == "test-content-id-123"

        # Проверяем, что параметры переданы корректно
        assert requests_mock.call_count == 1
        request = requests_mock.request_history[0]
        assert request.json()["prompt"] == "A beautiful sunset"
        assert request.json()["aspect_ratio"] == "16:9"
        assert request.json()["api_key"] == "test-api-key"

    def test_successful_check_status(self, requests_mock) -> None:
        """Проверка успешной проверки статуса генерации."""
        client = ChadGPTImageClient("test-api-key")
        requests_mock.post(
            "https://ask.chadgpt.ru/api/public/check",
            json={
                "status": "completed",
                "created_at": "2024-01-01T00:00:00Z",
                "started_at": "2024-01-01T00:01:00Z",
                "completed_at": "2024-01-01T00:02:00Z",
                "output": ["https://example.com/image1.jpg", "https://example.com/image2.jpg"],
            },
        )

        result = client.check_status("test-content-id-123")

        assert result.status == "completed"
        assert len(result.output) == 2
        assert "image1.jpg" in result.output[0]

        # Проверяем, что POST запрос сделан с правильными параметрами
        assert requests_mock.call_count == 1
        request = requests_mock.request_history[0]
        assert request.method == "POST"
        assert "api_key" in request.json()
        assert request.json()["content_id"] == "test-content-id-123"

    def test_custom_timeout(self, mocker: MockerFixture, requests_mock) -> None:
        """Проверка использования пользовательского таймаута."""
        client = ChadGPTClient("test-api-key")
        requests_mock.post(
            "https://ask.chadgpt.ru/api/public/gpt-5",
            json={"is_success": True, "response": "test"},
        )

        # Мокаем time.sleep для проверки таймаута
        mocker.patch("time.sleep")

        client.ask("gpt-5", "Hello", timeout=60)

        # Проверяем, что запрос был сделан
        assert requests_mock.call_count == 1
        # Таймаут передается в session.request, но requests_mock его не перехватывает
        # Проверяем через spy на session.request
        spy = mocker.spy(client._session, "request")
        client.ask("gpt-5", "Hello", timeout=60)
        assert spy.call_args.kwargs["timeout"] == 60

    def test_default_timeout(self, mocker: MockerFixture, requests_mock) -> None:
        """Проверка использования таймаута по умолчанию."""
        client = ChadGPTClient("test-api-key")
        requests_mock.post(
            "https://ask.chadgpt.ru/api/public/gpt-5",
            json={"is_success": True, "response": "test"},
        )

        # Мокаем time.sleep для проверки таймаута
        mocker.patch("time.sleep")

        spy = mocker.spy(client._session, "request")
        client.ask("gpt-5", "Hello")

        assert spy.call_args.kwargs["timeout"] == DEFAULT_TIMEOUT_SECONDS

    def test_get_request_with_params(self, requests_mock) -> None:
        """Проверка GET запроса с параметрами в query string."""
        client = ChadGPTBaseClient("test-api-key")
        requests_mock.get(
            "https://example.com/api",
            json={"test": "data"},
        )

        client._send_request(
            response_type=dict,
            url="https://example.com/api",
            method="get",
            params={"param1": "value1"},
            payload={"api_key": "test-key"},
        )

        # Проверяем, что запрос был сделан
        assert requests_mock.call_count == 1
        request = requests_mock.request_history[0]
        assert request.method == "GET"
        # Проверяем, что payload перенесен в params для GET
        assert "api_key" in request.qs
        assert "param1" in request.qs
        assert request.qs["api_key"] == ["test-key"]
        assert request.qs["param1"] == ["value1"]
        # Для GET не должно быть json в теле запроса
        assert request.body is None or request.body == b""

    def test_post_request_with_payload(self, requests_mock) -> None:
        """Проверка POST запроса с payload в теле."""
        client = ChadGPTBaseClient("test-api-key")
        requests_mock.post(
            "https://example.com/api",
            json={"test": "data"},
        )

        payload = {"message": "Hello", "api_key": "test-key"}
        client._send_request(
            response_type=dict,
            url="https://example.com/api",
            method="post",
            payload=payload,
        )

        # Проверяем, что запрос был сделан
        assert requests_mock.call_count == 1
        request = requests_mock.request_history[0]
        assert request.method == "POST"
        # Проверяем, что payload в json для POST
        assert request.json() == payload


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

    def test_ask_gpt_5(self, requests_mock) -> None:
        """Проверка метода ask_gpt5."""
        client = ChadGPTClient("test-api-key")
        requests_mock.post(
            "https://ask.chadgpt.ru/api/public/gpt-5",
            json={"is_success": True, "response": "test"},
        )

        client.ask_gpt_5("Hello")  # type: ignore

        assert requests_mock.call_count == 1
        assert "gpt-5" in requests_mock.request_history[0].url

    def test_ask_claude_4_5_sonnet(self, requests_mock) -> None:
        """Проверка метода ask_claude_4_5_sonnet."""
        client = ChadGPTClient("test-api-key")
        requests_mock.post(
            "https://ask.chadgpt.ru/api/public/claude-4.5-sonnet",
            json={"is_success": True, "response": "test"},
        )

        client.ask_claude_4_5_sonnet("Hello")  # type: ignore

        assert requests_mock.call_count == 1
        assert "claude-4.5-sonnet" in requests_mock.request_history[0].url

    def test_ask_with_all_params(self, requests_mock) -> None:
        """Проверка запроса со всеми опциональными параметрами."""
        client = ChadGPTClient("test-api-key")
        requests_mock.post(
            "https://ask.chadgpt.ru/api/public/gpt-5",
            json={"is_success": True, "response": "test"},
        )

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

        assert requests_mock.call_count == 1
        request = requests_mock.request_history[0]
        payload = request.json()
        assert payload["history"] == [x.model_dump(exclude_none=True) for x in history]
        assert payload["temperature"] == 0.8
        assert payload["max_tokens"] == 200
        assert payload["images"] == images
