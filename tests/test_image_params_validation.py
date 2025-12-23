"""Тесты для валидации параметров генерации изображений."""

import pytest

from pychadgpt.client import ChadGPTImageClient
from pychadgpt.exceptions import (
    ChadGPTValidationError,
)


class TestImageParamsValidation:
    """Тесты валидации параметров для различных моделей генерации изображений."""

    def test_imagen_params_validation(self, requests_mock) -> None:
        """Проверка валидации параметров для Imagen."""
        client = ChadGPTImageClient("test-api-key")
        requests_mock.post(
            "https://ask.chadgpt.ru/api/public/imagen-4/imagine",
            json={"status": "starting", "content_id": "test-id"},
        )

        # Валидный aspect_ratio
        client.imagine("imagen-4", "A landscape", aspect_ratio="16:9")
        assert requests_mock.call_count == 1
        request = requests_mock.request_history[0]
        assert request.json()["aspect_ratio"] == "16:9"

        # Проверка невалидного aspect_ratio
        with pytest.raises(ChadGPTValidationError) as exc_info:
            client.imagine("imagen-4", "A landscape", aspect_ratio="invalid-ratio")

        assert exc_info.value.error_code == "MODEL_PARAMS_VALIDATION_ERROR"

    def test_midjourney_params_validation(self, requests_mock) -> None:
        """Проверка валидации параметров для Midjourney."""
        client = ChadGPTImageClient("test-api-key")
        requests_mock.post(
            "https://ask.chadgpt.ru/api/public/mj-6/imagine",
            json={"status": "starting", "content_id": "test-id"},
        )

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
        with pytest.raises(ChadGPTValidationError) as exc_info:
            client.imagine("mj-6", "A scene", aspect_ratio="16:9", chaos=150)

        assert exc_info.value.error_code == "MODEL_PARAMS_VALIDATION_ERROR"

        # Проверка невалидного seed (должен быть 0-4000000)
        with pytest.raises(ChadGPTValidationError) as exc_info:
            client.imagine("mj-6", "A scene", aspect_ratio="16:9", seed=5000000)

        assert exc_info.value.error_code == "MODEL_PARAMS_VALIDATION_ERROR"

        # Проверка невалидного stop (должен быть 10-100)
        with pytest.raises(ChadGPTValidationError) as exc_info:
            client.imagine("mj-6", "A scene", aspect_ratio="16:9", stop=5)

        assert exc_info.value.error_code == "MODEL_PARAMS_VALIDATION_ERROR"

    def test_gemini_params_validation(self, requests_mock) -> None:
        """Проверка валидации параметров для Gemini."""
        client = ChadGPTImageClient("test-api-key")
        requests_mock.post(
            "https://ask.chadgpt.ru/api/public/gemini-2.5-flash-image/imagine",
            json={"status": "starting", "content_id": "test-id"},
        )

        # Валидные параметры (максимум 5 изображений)
        client.imagine(
            "gemini-2.5-flash-image",
            "A scene",
            image_urls=["url1", "url2", "url3"],
            image_base64s=["base64_1", "base64_2"],
        )

        # Проверка превышения лимита image_urls (максимум 5)
        with pytest.raises(ChadGPTValidationError) as exc_info:
            client.imagine(
                "gemini-2.5-flash-image",
                "A scene",
                image_urls=["url1", "url2", "url3", "url4", "url5", "url6"],
            )

        assert exc_info.value.error_code == "MODEL_PARAMS_VALIDATION_ERROR"

        # Проверка превышения лимита image_base64s (максимум 5)
        with pytest.raises(ChadGPTValidationError) as exc_info:
            client.imagine(
                "gemini-2.5-flash-image",
                "A scene",
                image_base64s=["b1", "b2", "b3", "b4", "b5", "b6"],
            )

        assert exc_info.value.error_code == "MODEL_PARAMS_VALIDATION_ERROR"

    def test_flux_params_validation(self, requests_mock) -> None:
        """Проверка валидации параметров для Flux."""
        client = ChadGPTImageClient("test-api-key")
        requests_mock.post(
            "https://ask.chadgpt.ru/api/public/flux-1-schnell/imagine",
            json={"status": "starting", "content_id": "test-id"},
        )
        requests_mock.post(
            "https://ask.chadgpt.ru/api/public/flux-1.1-pro/imagine",
            json={"status": "starting", "content_id": "test-id"},
        )

        # FluxSimple - валидные параметры (все поля обязательны)
        client.imagine("flux-1-schnell", "A scene", aspect_ratio="16:9", images=3)

        # Проверка невалидного images (должен быть 1-5)
        with pytest.raises(ChadGPTValidationError) as exc_info:
            client.imagine("flux-1-schnell", "A scene", aspect_ratio="16:9", images=0)

        assert exc_info.value.error_code == "MODEL_PARAMS_VALIDATION_ERROR"

        with pytest.raises(ChadGPTValidationError) as exc_info:
            client.imagine("flux-1-schnell", "A scene", aspect_ratio="16:9", images=6)

        assert exc_info.value.error_code == "MODEL_PARAMS_VALIDATION_ERROR"

        # FluxPro - валидные параметры (все поля обязательны)
        client.imagine("flux-1.1-pro", "A scene", aspect_ratio="16:9", seed=1000, is_raw=True)

        # Проверка невалидного seed для FluxPro
        with pytest.raises(ChadGPTValidationError) as exc_info:
            client.imagine("flux-1.1-pro", "A scene", aspect_ratio="16:9", seed=5000000, is_raw=True)

        assert exc_info.value.error_code == "MODEL_PARAMS_VALIDATION_ERROR"

    def test_flux_kontext_params_validation(self, requests_mock) -> None:
        """Проверка валидации параметров для Flux Kontext."""
        client = ChadGPTImageClient("test-api-key")
        requests_mock.post(
            "https://ask.chadgpt.ru/api/public/flux-kontext-pro/imagine",
            json={"status": "starting", "content_id": "test-id"},
        )

        # Валидные параметры
        client.imagine(
            "flux-kontext-pro",
            "A scene",
            aspect_ratio="16:9",
            seed=2000,
            image_url="https://example.com/image.jpg",
        )

    def test_dalle_params_validation(self, requests_mock) -> None:
        """Проверка валидации параметров для DALL-E."""
        client = ChadGPTImageClient("test-api-key")
        requests_mock.post(
            "https://ask.chadgpt.ru/api/public/gpt-img-high/imagine",
            json={"status": "starting", "content_id": "test-id"},
        )

        # Валидные параметры
        client.imagine("gpt-img-high", "A scene", aspect_ratio="16:9")

        # Проверка невалидного aspect_ratio
        with pytest.raises(ChadGPTValidationError) as exc_info:
            client.imagine("gpt-img-high", "A scene", aspect_ratio="4:3")

        assert exc_info.value.error_code == "MODEL_PARAMS_VALIDATION_ERROR"

    def test_seedream_params_validation(self, requests_mock) -> None:
        """Проверка валидации параметров для Seedream."""
        client = ChadGPTImageClient("test-api-key")
        requests_mock.post(
            "https://ask.chadgpt.ru/api/public/seedream-4/imagine",
            json={"status": "starting", "content_id": "test-id"},
        )

        # Валидные параметры
        client.imagine(
            "seedream-4",
            "A scene",
            aspect_ratio="21:9",
            size_preset="4K",
            image_urls=["url1", "url2"],
        )

        # Проверка невалидного aspect_ratio
        with pytest.raises(ChadGPTValidationError) as exc_info:
            client.imagine("seedream-4", "A scene", aspect_ratio="invalid", size_preset="4K", image_urls=[])

        assert exc_info.value.error_code == "MODEL_PARAMS_VALIDATION_ERROR"

        # Проверка невалидного size_preset
        with pytest.raises(ChadGPTValidationError) as exc_info:
            client.imagine("seedream-4", "A scene", aspect_ratio="16:9", size_preset="8K", image_urls=[])

        assert exc_info.value.error_code == "MODEL_PARAMS_VALIDATION_ERROR"

    def test_seededit_params_validation(self, requests_mock) -> None:
        """Проверка валидации параметров для Seededit."""
        client = ChadGPTImageClient("test-api-key")
        requests_mock.post(
            "https://ask.chadgpt.ru/api/public/seededit-3/imagine",
            json={"status": "starting", "content_id": "test-id"},
        )

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
        with pytest.raises(ChadGPTValidationError) as exc_info:
            client.imagine(
                "seededit-3",
                "Edit this image",
                seed=3000,
                guidance_scale=0.5,
                image_url="https://example.com/image.jpg",
            )

        assert exc_info.value.error_code == "MODEL_PARAMS_VALIDATION_ERROR"

        with pytest.raises(ChadGPTValidationError) as exc_info:
            client.imagine(
                "seededit-3",
                "Edit this image",
                seed=3000,
                guidance_scale=11.0,
                image_url="https://example.com/image.jpg",
            )

        assert exc_info.value.error_code == "MODEL_PARAMS_VALIDATION_ERROR"

        # Проверка невалидного seed
        with pytest.raises(ChadGPTValidationError) as exc_info:
            client.imagine(
                "seededit-3",
                "Edit this image",
                seed=-1,
                guidance_scale=5.0,
                image_url="https://example.com/image.jpg",
            )

        assert exc_info.value.error_code == "MODEL_PARAMS_VALIDATION_ERROR"

        # Проверка, что нельзя передать оба поля одновременно
        with pytest.raises(ChadGPTValidationError) as exc_info:
            client.imagine(
                "seededit-3",
                "Edit this image",
                seed=3000,
                guidance_scale=5.0,
                image_url="https://example.com/image.jpg",
                image_base64="base64string",
            )

        assert exc_info.value.error_code == "MODEL_PARAMS_VALIDATION_ERROR"

        # Проверка, что нельзя не передать ни одно из полей
        with pytest.raises(ChadGPTValidationError) as exc_info:
            client.imagine(
                "seededit-3",
                "Edit this image",
                seed=3000,
                guidance_scale=5.0,
            )

        assert exc_info.value.error_code == "MODEL_PARAMS_VALIDATION_ERROR"

    def test_generic_params_validation(self, requests_mock) -> None:
        """Проверка валидации для моделей с GenericParams."""
        client = ChadGPTImageClient("test-api-key")
        requests_mock.post(
            "https://ask.chadgpt.ru/api/public/recraft-v3-svg/imagine",
            json={"status": "starting", "content_id": "test-id"},
        )

        # GenericParams не имеет обязательных полей, должен работать без дополнительных параметров
        client.imagine("recraft-v3-svg", "A simple SVG icon")
