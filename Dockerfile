# syntax=docker/dockerfile:1
# BuildKit 缓存加速重复构建：export DOCKER_BUILDKIT=1（Docker 24+ 默认已开启）
FROM python:3.13-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

WORKDIR /app

# 系统依赖：apt 使用 BuildKit 缓存，重复构建时 deb 包可命中缓存
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 重依赖单独一层：仅当 requirements-base.txt 变更时才重装 PyTorch 等
COPY requirements-base.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r requirements-base.txt

# 应用依赖：日常改版本只重建这一层
COPY requirements-app.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-app.txt

# 再拷贝项目代码（改代码不触发 pip）
COPY app ./app
COPY evals ./evals

EXPOSE 8090

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8090"]
