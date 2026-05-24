# Setup

Инструкция по развёртыванию окружения проекта.

## Требования

- **OS:** macOS или Linux
- **Python:** 3.11
- **Package manager:** [`uv`](https://docs.astral.sh/uv/)
- **Git** (любая современная версия)
- Опционально для тренировки: GPU 16+ ГБ (CUDA 12.x)

## Установка системных инструментов

Один раз на машине:

```bash
brew install python@3.11 uv
```

Альтернативные способы установки `uv`:

```bash
# Универсальный скрипт
curl -LsSf https://astral.sh/uv/install.sh | sh

# Через pipx
pipx install uv
```

Проверка:

```bash
python3.11 --version && uv --version && git --version
```

## Клонирование и установка зависимостей

```bash
git clone https://github.com/<user>/disaster-tweets-classifier.git
cd disaster-tweets-classifier

uv venv --python 3.11
uv sync --all-groups
```

`uv sync` создаёт `.venv/` и устанавливает все зависимости строго по `uv.lock`.

Проверка корректности установки:

```bash
uv run python -c "import torch, pytorch_lightning, transformers, hydra, mlflow, dvc; print('OK')"
```

## Pre-commit хуки

```bash
uv run pre-commit install
uv run pre-commit run -a
```

Первый запуск долгий (скачивает хуки) и может переформатировать файлы — это нормально. Запусти `uv run pre-commit run -a` второй раз, пока не станет зелёным.

## MLflow сервер для логирования

Для локального трекинга экспериментов:

```bash
uv run mlflow server --host 127.0.0.1 --port 8080
```

UI откроется на <http://127.0.0.1:8080>. Адрес уже прописан в `configs/logging/mlflow.yaml`.

## DVC remote

По умолчанию remote — локальная папка `../dvc-storage` (см. `.dvc/config`). Для смены remote на Google Drive / S3 / Я.Диск см. [официальную документацию DVC](https://dvc.org/doc/user-guide/data-management/remote-storage).

## Что дальше

Команды запуска тренировки и инференса описаны в [README.md](README.md) в разделе **Train**.
