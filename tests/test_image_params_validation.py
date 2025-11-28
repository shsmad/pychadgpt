"""Тесты для валидации параметров генерации изображений."""

import pytest
from pydantic import ValidationError
from pytest_mock import MockerFixture

from pychadgpt.client import ChadGPTImageClient


class TestImageParamsValidation:
    """Тесты валидации параметров для различных моделей генерации изображений."""

    def test_imagen_params_validation(self, mocker: MockerFixture) -> None:
        """Проверка валидации параметров для Imagen."""
        client = ChadGPTImageClient("test-api-key")
        mock_response = mocker.Mock()
        mock_response.json.return_value = {"status": "starting", "content_id": "test-id"}
        mock_response.raise_for_status = mocker.Mock()
        mock_request = mocker.patch.object(client._session, "request", return_value=mock_response)

        # Валидный aspect_ratio
        client.imagine("imagen-4", "A landscape", aspect_ratio="16:9")
        call_args = mock_request.call_args
        assert call_args is not None
        assert call_args.kwargs["json"]["aspect_ratio"] == "16:9"

        # Проверка невалидного aspect_ratio
        with pytest.raises((ValidationError, ValueError)):
            client.imagine("imagen-4", "A landscape", aspect_ratio="invalid-ratio")

    def test_midjourney_params_validation(self, mocker: MockerFixture) -> None:
        """Проверка валидации параметров для Midjourney."""
        client = ChadGPTImageClient("test-api-key")
        mock_response = mocker.Mock()
        mock_response.json.return_value = {"status": "starting", "content_id": "test-id"}
        mock_response.raise_for_status = mocker.Mock()
        mocker.patch.object(client._session, "request", return_value=mock_response)

        # Валидные параметры (aspect_ratio обязателен)
        client.imagine(
            "mj-6",
            "A beautiful scene",
            aspect_ratio="16:9",
            chaos=50,
            seed=12345,
            stop=50,
        )

        # Проверка невалидного chaos (должен быть 0-100)
        with pytest.raises((ValidationError, ValueError)):
            client.imagine("mj-6", "A scene", aspect_ratio="16:9", chaos=150)

        # Проверка невалидного seed (должен быть 0-4000000)
        with pytest.raises((ValidationError, ValueError)):
            client.imagine("mj-6", "A scene", aspect_ratio="16:9", seed=5000000)

        # Проверка невалидного stop (должен быть 10-100)
        with pytest.raises((ValidationError, ValueError)):
            client.imagine("mj-6", "A scene", aspect_ratio="16:9", stop=5)

    def test_gemini_params_validation(self, mocker: MockerFixture) -> None:
        """Проверка валидации параметров для Gemini."""
        client = ChadGPTImageClient("test-api-key")
        mock_response = mocker.Mock()
        mock_response.json.return_value = {"status": "starting", "content_id": "test-id"}
        mock_response.raise_for_status = mocker.Mock()
        mocker.patch.object(client._session, "request", return_value=mock_response)

        # Валидные параметры (максимум 5 изображений)
        client.imagine(
            "gemini-2.5-flash-image",
            "A scene",
            image_urls=["url1", "url2", "url3"],
            image_base64s=["base64_1", "base64_2"],
        )

        # Проверка превышения лимита image_urls (максимум 5)
        with pytest.raises((ValidationError, ValueError)):
            client.imagine(
                "gemini-2.5-flash-image",
                "A scene",
                image_urls=["url1", "url2", "url3", "url4", "url5", "url6"],
            )

        # Проверка превышения лимита image_base64s (максимум 5)
        with pytest.raises((ValidationError, ValueError)):
            client.imagine(
                "gemini-2.5-flash-image",
                "A scene",
                image_base64s=["b1", "b2", "b3", "b4", "b5", "b6"],
            )

    def test_flux_params_validation(self, mocker: MockerFixture) -> None:
        """Проверка валидации параметров для Flux."""
        client = ChadGPTImageClient("test-api-key")
        mock_response = mocker.Mock()
        mock_response.json.return_value = {"status": "starting", "content_id": "test-id"}
        mock_response.raise_for_status = mocker.Mock()
        mocker.patch.object(client._session, "request", return_value=mock_response)

        # FluxSimple - валидные параметры (все поля обязательны)
        client.imagine("flux-1-schnell", "A scene", aspect_ratio="16:9", images=3)

        # Проверка невалидного images (должен быть 1-5)
        with pytest.raises((ValidationError, ValueError)):
            client.imagine("flux-1-schnell", "A scene", aspect_ratio="16:9", images=0)

        with pytest.raises((ValidationError, ValueError)):
            client.imagine("flux-1-schnell", "A scene", aspect_ratio="16:9", images=6)

        # FluxPro - валидные параметры (все поля обязательны)
        client.imagine("flux-1.1-pro", "A scene", aspect_ratio="16:9", seed=1000, is_raw=True)

        # Проверка невалидного seed для FluxPro
        with pytest.raises((ValidationError, ValueError)):
            client.imagine("flux-1.1-pro", "A scene", aspect_ratio="16:9", seed=5000000, is_raw=True)

    def test_flux_kontext_params_validation(self, mocker: MockerFixture) -> None:
        """Проверка валидации параметров для Flux Kontext."""
        client = ChadGPTImageClient("test-api-key")
        mock_response = mocker.Mock()
        mock_response.json.return_value = {"status": "starting", "content_id": "test-id"}
        mock_response.raise_for_status = mocker.Mock()
        mocker.patch.object(client._session, "request", return_value=mock_response)

        # Валидные параметры
        client.imagine(
            "flux-kontext-pro",
            "A scene",
            aspect_ratio="16:9",
            seed=2000,
            image_url="https://example.com/image.jpg",
        )

    def test_dalle_params_validation(self, mocker: MockerFixture) -> None:
        """Проверка валидации параметров для DALL-E."""
        client = ChadGPTImageClient("test-api-key")
        mock_response = mocker.Mock()
        mock_response.json.return_value = {"status": "starting", "content_id": "test-id"}
        mock_response.raise_for_status = mocker.Mock()
        mocker.patch.object(client._session, "request", return_value=mock_response)

        # Валидные параметры
        client.imagine("gpt-img-high", "A scene", aspect_ratio="16:9")

        # Проверка невалидного aspect_ratio
        with pytest.raises((ValidationError, ValueError)):
            client.imagine("gpt-img-high", "A scene", aspect_ratio="4:3")

    def test_seedream_params_validation(self, mocker: MockerFixture) -> None:
        """Проверка валидации параметров для Seedream."""
        client = ChadGPTImageClient("test-api-key")
        mock_response = mocker.Mock()
        mock_response.json.return_value = {"status": "starting", "content_id": "test-id"}
        mock_response.raise_for_status = mocker.Mock()
        mocker.patch.object(client._session, "request", return_value=mock_response)

        # Валидные параметры
        client.imagine(
            "seedream-4",
            "A scene",
            aspect_ratio="21:9",
            size_preset="4K",
            image_urls=["url1", "url2"],
        )

        # Проверка невалидного aspect_ratio
        with pytest.raises((ValidationError, ValueError)):
            client.imagine("seedream-4", "A scene", aspect_ratio="invalid", size_preset="4K", image_urls=[])

        # Проверка невалидного size_preset
        with pytest.raises((ValidationError, ValueError)):
            client.imagine("seedream-4", "A scene", aspect_ratio="16:9", size_preset="8K", image_urls=[])

    def test_seededit_params_validation(self, mocker: MockerFixture) -> None:
        """Проверка валидации параметров для Seededit."""
        client = ChadGPTImageClient("test-api-key")
        mock_response = mocker.Mock()
        mock_response.json.return_value = {"status": "starting", "content_id": "test-id"}
        mock_response.raise_for_status = mocker.Mock()
        mocker.patch.object(client._session, "request", return_value=mock_response)

        # Валидные параметры с image_url
        client.imagine(
            "seededit-3",
            "Edit this image",
            seed=3000,
            guidance_scale=5.0,
            image_url="https://example.com/image.jpg",
        )

        # Валидные параметры с image_base64
        client.imagine(
            "seededit-3",
            "Edit this image",
            seed=3000,
            guidance_scale=7.5,
            image_base64="base64string",
        )

        # Проверка невалидного guidance_scale (должен быть 1.0-10.0)
        with pytest.raises((ValidationError, ValueError)):
            client.imagine(
                "seededit-3",
                "Edit this image",
                seed=3000,
                guidance_scale=0.5,
                image_url="https://example.com/image.jpg",
            )

        with pytest.raises((ValidationError, ValueError)):
            client.imagine(
                "seededit-3",
                "Edit this image",
                seed=3000,
                guidance_scale=11.0,
                image_url="https://example.com/image.jpg",
            )

        # Проверка невалидного seed
        with pytest.raises((ValidationError, ValueError)):
            client.imagine(
                "seededit-3",
                "Edit this image",
                seed=-1,
                guidance_scale=5.0,
                image_url="https://example.com/image.jpg",
            )

        # Проверка, что нельзя передать оба поля одновременно
        with pytest.raises((ValidationError, ValueError)):
            client.imagine(
                "seededit-3",
                "Edit this image",
                seed=3000,
                guidance_scale=5.0,
                image_url="https://example.com/image.jpg",
                image_base64="base64string",
            )

        # Проверка, что нельзя не передать ни одно из полей
        with pytest.raises((ValidationError, ValueError)):
            client.imagine(
                "seededit-3",
                "Edit this image",
                seed=3000,
                guidance_scale=5.0,
            )

    def test_generic_params_validation(self, mocker: MockerFixture) -> None:
        """Проверка валидации для моделей с GenericParams."""
        client = ChadGPTImageClient("test-api-key")
        mock_response = mocker.Mock()
        mock_response.json.return_value = {"status": "starting", "content_id": "test-id"}
        mock_response.raise_for_status = mocker.Mock()
        mocker.patch.object(client._session, "request", return_value=mock_response)

        # GenericParams не имеет обязательных полей, должен работать без дополнительных параметров
        client.imagine("recraft-v3-svg", "A simple SVG icon")
