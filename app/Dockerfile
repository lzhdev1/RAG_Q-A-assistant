# 使用 Python 3.11 官方镜像作为基础
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装系统级依赖（PDF解析需要的库）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖文件，利用 Docker 缓存层加速后续构建
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制应用代码
COPY . .

# 创建文档和数据库目录
RUN mkdir -p /app/docs /app/chroma_data

# 暴露 FastAPI 端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]