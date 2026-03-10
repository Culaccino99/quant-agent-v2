FROM python:3.13-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

WORKDIR /app

# 系统依赖（如果后续装不上某些包，可在这里追加）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 先拷贝依赖文件，利用 Docker 缓存
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 再拷贝项目代码
COPY app ./app
COPY evals ./evals

# 默认使用环境变量注入敏感配置（DEEPSEEK_API_KEY / FEISHU_* / LANGCHAIN_API_KEY 等）

EXPOSE 8090

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8090"]

