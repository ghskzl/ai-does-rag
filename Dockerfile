FROM python:3.12-slim

WORKDIR /app

# 使用阿里云镜像加速 apt
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources && \
    apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv 并配置国内镜像
ENV UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
ENV UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple/

COPY pyproject.toml uv.lock ./

# 使用 BuildKit 缓存加速重复构建
RUN --mount=type=cache,target=/root/.cache/uv \
    pip install --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ uv && \
    uv sync --frozen

COPY src/ src/

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
