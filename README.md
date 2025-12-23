# pychadgpt

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Python библиотека для работы с ChadGPT API - унифицированный интерфейс для GPT, Claude, Gemini и других AI моделей.

## Особенности

- 🚀 **Автогенерация методов** - методы `ask_*` создаются автоматически для каждой модели
- ✅ **Валидация через Pydantic** - все параметры валидируются автоматически
- 🛡️ **Кастомные исключения** - понятная обработка ошибок
- 📝 **Полная типизация** - поддержка type hints и mypy
- 🖼️ **Генерация изображений** - поддержка Imagen, Midjourney, Flux, DALL-E и других
- 📊 **Статистика использования** - отслеживание слов и токенов
- 🔄 **Контекстный менеджер** - автоматическое управление сессией

## Установка

```bash
pip install pychadgpt
```

Или через PDM:

```bash
pdm add pychadgpt
```

## Быстрый старт

```python
from pychadgpt import ChadGPTClient

# Инициализация клиента
client = ChadGPTClient("your-api-key")

# Простой запрос
response = client.ask_gpt5("Привет! Расскажи про Python")
if response.is_success:
    print(response.response)
```

## Основное использование

### Работа с различными моделями

Библиотека автоматически генерирует методы для каждой модели:

```python
# GPT модели
response = client.ask_gpt5("Привет!")
response = client.ask_gpt5_mini("Привет!")

# Claude модели
response = client.ask_claude_4_5_sonnet("Привет!")
response = client.ask_claude_4_1_opus("Привет!")

# Gemini модели
response = client.ask_gemini_2_5_pro("Привет!")
```

### Универсальный метод

```python
response = client.ask(
    model_name="gpt-5",
    message="Привет!",
    temperature=0.7,
    max_tokens=1000,
    timeout=60
)
```

### Работа с историей сообщений

```python
from pychadgpt.models import ChatHistoryMessage

history = [
    ChatHistoryMessage(role="user", content="Привет!"),
    ChatHistoryMessage(role="assistant", content="Здравствуйте!"),
]

response = client.ask_gpt5(
    message="Продолжи разговор",
    history=history,
    temperature=0.8
)
```

### Контекстный менеджер

```python
with ChadGPTClient("your-api-key") as client:
    response = client.ask_gpt5("Hello!")
    # Сессия автоматически закроется
```

### Параметры запроса

```python
response = client.ask_gpt5(
    message="Расскажи про Python",
    temperature=0.7,      # Температура генерации (0-2)
    max_tokens=1000,      # Максимальное количество токенов
    timeout=60,           # Таймаут в секундах
    images=["https://example.com/image.jpg"]  # Изображения
)
```

## Генерация изображений

```python
from pychadgpt import ChadGPTImageClient
import time

image_client = ChadGPTImageClient("your-api-key")

# Запуск генерации
result = image_client.imagine(
    model_name="imagen-4",
    prompt="A beautiful sunset over mountains",
    aspect_ratio="16:9"
)

if result.status == "starting":
    content_id = result.content_id

    # Проверка статуса
    status = image_client.check_status(content_id)
    while status.status == "pending":
        time.sleep(5)
        status = image_client.check_status(content_id)

    if status.status == "completed":
        for image_url in status.output:
            print(f"Изображение: {image_url}")
```

## Получение статистики

```python
from pychadgpt import ChadGPTBaseClient

client = ChadGPTBaseClient("your-api-key")
stat = client.get_stat_info()

print(f"Использовано: {stat.used_words}/{stat.total_words}")
print(f"Осталось: {stat.remaining_words}")
```

## Обработка ошибок

```python
from pychadgpt import (
    ChadGPTClient,
    ChadGPTValidationError,
    ChadGPTConnectionError,
    ChadGPTTimeoutError,
    ChadGPTAPIError,
)

client = ChadGPTClient("your-api-key")

try:
    response = client.ask_gpt5("Привет!")
    if not response.is_success:
        raise ChadGPTAPIError(response.error_message, response.error_code)
except ChadGPTValidationError as e:
    print(f"Ошибка валидации: {e.message}")
except ChadGPTConnectionError as e:
    print(f"Ошибка соединения: {e.message}")
except ChadGPTTimeoutError as e:
    print(f"Таймаут: {e.message}")
```

## Логирование

```python
from pychadgpt.client import setup_logging
import logging

setup_logging(logging.DEBUG)
client = ChadGPTClient("your-api-key")
response = client.ask_gpt5("Привет!")
```

## Продвинутое использование

### Кастомный таймаут

```python
# Глобальный таймаут по умолчанию - 30 секунд
# Можно переопределить для конкретного запроса
response = client.ask_gpt5("Привет!", timeout=120)
```

### Валидация параметров

Библиотека автоматически валидирует все параметры через Pydantic:

```python
from pychadgpt.models import AskParameters

# Валидация перед отправкой
params = AskParameters(
    message="Привет!",
    temperature=0.7,
    max_tokens=1000
)
```

### Работа с устаревшими моделями

Библиотека предупреждает об использовании устаревших моделей:

```python
import warnings

# Предупреждение будет показано
response = client.ask_gpt4o("Привет!")

# Или отключить предупреждения
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    response = client.ask_gpt4o("Привет!")
```

## Поддерживаемые модели

### Chat модели

- **GPT**: gpt-5, gpt-5-mini, gpt-5-nano, gpt-4o, gpt-4o-mini
- **Claude**: claude-4.5-sonnet, claude-4.1-opus, claude-3.7-sonnet-thinking, claude-3-opus, claude-3-haiku
- **Gemini**: gemini-2.5-pro, gemini-2.0-flash
- **Deepseek**: deepseek-v3.1

### Image модели

- **Imagen**: imagen-4, imagen-4-fast, imagen-4-ultra
- **Midjourney**: mj-7, mj-6.1, mj-6, mj-5.2
- **Flux**: flux-1.1-pro-ultra, flux-1.1-pro, flux-1-schnell, flux-kontext-pro, flux-kontext-max, flux-kontext-multi
- **DALL-E**: gpt-img-high, gpt-img-medium, gpt-img-low
- **Seedream**: seedream-4
- **Seededit**: seededit-3
- **Recraft**: recraft-v3-svg
- **Gemini**: gemini-2.5-flash-image

## Разработка

### Установка зависимостей

```bash
pdm install --group dev
```

### Запуск тестов

```bash
pdm run pytest
pdm run pytest -v
pdm run pytest --cov=pychadgpt
```

### Линтинг

```bash
pdm run ruff check .
pdm run mypy src/
```

### Документация сервиса chadgpt

- [Chad API](https://chadgpt.ru/api-docs)
- [Chad Image API](https://chadgpt.ru/image-api-docs)

## Лицензия

MIT License - см. [LICENSE](LICENSE)

## Автор

shsmad (<shsmad@gmail.com>)

## Ссылки

- [Документация](docs/README.md)
- [Changelog](CHANGELOG.md)
- [Issues](https://github.com/shsmad/pychadgpt/issues)
