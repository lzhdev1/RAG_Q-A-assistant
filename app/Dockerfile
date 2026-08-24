# ==========================================
# 阶段 1: 构建依赖 (Builder)
# ==========================================
FROM python:3.10-slim AS builder

WORKDIR /app

# 优化点1: 设置环境变量，防止生成 .pyc 文件，减少体积
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 优化点2: 先只复制依赖文件，利用 Docker 缓存
# 只要 requirements.txt 不变，这一层就不会重新构建
COPY requirements.txt .

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ==========================================
# 阶段 2: 运行环境 (Runner)
# ==========================================
FROM python:3.10-slim AS runner

WORKDIR /app

# 优化点3: 从 builder 阶段复制纯净的 Python 包
COPY --from=builder /install /usr/local

# 优化点4: 创建非 root 用户，提升安全性
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# 复制项目代码 (放在最后，因为代码变动最频繁)
COPY --chown=appuser:appuser . .

# 优化点5: 声明数据卷，保护 ChromaDB 数据
VOLUME ["/app/chroma_data"]

# 暴露端口 (根据你的 app.py 或 main.py 修改)
EXPOSE 8000

# 启动命令
CMD ["python", "main.py"]