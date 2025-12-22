from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class GenericParams(BaseModel): ...


class ImagenParams(GenericParams):
    aspect_ratio: Literal["1:1", "16:9", "9:16", "4:3", "3:4"] = "1:1"


class MidjourneyParams(GenericParams):
    aspect_ratio: Literal["1:1", "3:2", "2:3", "16:9", "9:16"] = "1:1"
    chaos: Annotated[int, Field(ge=0, le=100)] = 50
    no: str = ""
    quality: str = ""
    seed: Annotated[int, Field(ge=0, le=4000000)] = 0
    stop: Annotated[int, Field(ge=10, le=100)] = 100
    style: str = ""
    stylize: int = 0
    tile: bool = False
    weird: int = 0


class GeminiParams(GenericParams):
    image_urls: Annotated[list[str], Field(max_length=5)] = []
    image_base64s: Annotated[list[str], Field(max_length=5)] = []


class FluxSimpleParams(GenericParams):
    aspect_ratio: str = "1:1"
    images: Annotated[int, Field(ge=1, le=5)] = 1


class FluxProParams(GenericParams):
    aspect_ratio: str = "1:1"
    seed: Annotated[int, Field(ge=0, le=4000000)] = 0
    # is_raw: bool = False


class FluxKontextParams(GenericParams):
    aspect_ratio: str = "1:1"
    seed: Annotated[int, Field(ge=0, le=4000000)] = 0
    image_url: str = ""


class DallEParams(GenericParams):
    aspect_ratio: Literal["1:1", "16:9", "9:16"] = "1:1"


class SeedreamParams(GenericParams):
    aspect_ratio: Literal["1:1", "4:3", "3:4", "16:9", "9:16", "3:2", "2:3", "21:9"] = "1:1"
    size_preset: Literal["1K", "2K", "4K"] = "1K"
    image_urls: Annotated[list[str], Field(max_length=5)] = []


class SeededitParams(GenericParams):
    seed: Annotated[int, Field(ge=0, le=4000000)] = 0
    guidance_scale: Annotated[float, Field(ge=1.0, le=10.0)] = 1.0
    image_url: str | None = None
    image_base64: str | None = None
    # "For 'seededit-3', you must provide either 'image_url' or 'image_base64', but not both."

    @model_validator(mode="after")
    def validate_image_source(self) -> "SeededitParams":
        """Валидация: должно быть предоставлено ровно одно из полей image_url или image_base64."""
        has_url = bool(self.image_url and self.image_url.strip())
        has_base64 = bool(self.image_base64 and self.image_base64.strip())

        if not has_url and not has_base64:
            raise ValueError("For 'seededit-3', you must provide either 'image_url' or 'image_base64'")
        if has_url and has_base64:
            raise ValueError("For 'seededit-3', you must provide either 'image_url' or 'image_base64', but not both.")
        return self


MODEL_VALIDATORS: dict[str, type] = {
    "imagen-4": ImagenParams,
    "imagen-4-fast": ImagenParams,
    "imagen-4-ultra": ImagenParams,
    "mj-5.2": MidjourneyParams,
    "mj-6": MidjourneyParams,
    "mj-6.1": MidjourneyParams,
    "mj-7": MidjourneyParams,
    "gemini-2.5-flash-image": GeminiParams,
    "flux-1-schnell": FluxSimpleParams,
    "flux-1.1-pro": FluxProParams,
    "flux-1.1-pro-ultra": FluxProParams,
    "flux-kontext-pro": FluxKontextParams,
    "flux-kontext-max": FluxKontextParams,
    "flux-kontext-multi": FluxKontextParams,
    "recraft-v3-svg": GenericParams,
    "gpt-img-low": DallEParams,
    "gpt-img-medium": DallEParams,
    "gpt-img-high": DallEParams,
    "seedream-4": SeedreamParams,
    "seededit-3": SeededitParams,
}
