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

# 优化点2.1: 配置 pip 使用阿里云镜像 + 超时重试
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ \
    && pip config set global.trusted-host mirrors.aliyun.com \
    && pip config set global.timeout 300 \
    && pip config set install.retries 10

# 优化点2.2: 预装 CPU-only PyTorch (ragas 依赖，~200MB 而非 ~526MB)
RUN pip install --no-cache-dir --prefix=/install \
    torch --extra-index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ==========================================
# 阶段 2: 运行环境 (Runner)
# ==========================================
FROM python:3.10-slim AS runner

WORKDIR /app

# 优化点3: 从 builder 阶段复制纯净的 Python 包
COPY --from=builder /install /usr/local

# 优化点4: 安装 gosu 用于 entrypoint 中切换用户，创建非 root 用户
RUN apt-get update && apt-get install -y --no-install-recommends gosu \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 appuser

# 复制 entrypoint 脚本
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# 复制项目代码 (放在最后，因为代码变动最频繁)
COPY --chown=appuser:appuser . .

# 优化点5: 声明数据卷，保护 ChromaDB 数据
VOLUME ["/app/chroma_data"]

# 暴露端口
EXPOSE 8000

# 使用 entrypoint 脚本启动（先修复权限，再切换用户）
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "main.py"]