# pychadgpt

Python library to use chadgpt API.

## Установка

```bash
pdm install
```

## Использование

```python
from pychadgpt import ChadGPTClient

client = ChadGPTClient("your-api-key")
response = client.ask("gpt-5", "Hello, world!")
print(response["response"])
```

## Тестирование

Проект содержит unit-тесты, покрывающие:
- Валидацию параметров (API ключ, модели, HTTP методы, промпты)
- Обработку сетевых ошибок (ConnectionError, Timeout, HTTPError, JSONDecodeError)
- Успешные сценарии работы всех клиентов
- Валидацию параметров генерации изображений через Pydantic

Для запуска тестов:

```bash
# Установка зависимостей для разработки
pdm install --group dev

# Запуск всех тестов
pdm run pytest

# Запуск с подробным выводом
pdm run pytest -v

# Запуск конкретного файла тестов
pdm run pytest tests/test_client_validation.py
```

Тесты используют моки для избежания реальных сетевых вызовов.
