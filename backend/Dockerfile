# ==========================================
# 阶段 1: 构建依赖 (Builder)
# ==========================================
FROM python:3.10-slim AS builder

WORKDIR /app

# 优化点1: 防止生成 .pyc 文件，减少体积
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 优化点2: 先只复制依赖文件，利用 Docker 缓存
COPY requirements.txt .

# 优化点3: pip 使用清华镜像 + 超时重试
# （原阿里云源 mirrors.aliyun.com 于 2026-11 起对部分 wheel 返回 403，导致构建卡死）
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/ \
    && pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn \
    && pip config set global.timeout 300 \
    && pip config set install.retries 10

# 优化点4: 直接安装依赖
# 本项目 embedding / rerank / LLM 全部走 DashScope 云端 API，代码中无本地模型推理，
# 因此不安装 torch / transformers / sentence-transformers（早期遗留依赖，已移除）。
# 由此也不再需要 CPU 版 torch 预下载、constraints 固定版本和 torch 行过滤。
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ==========================================
# 阶段 2: 运行环境 (Runner)
# ==========================================
FROM python:3.10-slim AS runner

WORKDIR /app

# 优化点5: 从 builder 阶段复制纯净的 Python 包
COPY --from=builder /install /usr/local

# 优化点6: 安装 gosu 用于 entrypoint 中切换用户，创建非 root 用户
# 服务器直连 deb.debian.org 极慢/卡死，apt 源切换为清华镜像
RUN sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || true; \
    apt-get update && apt-get install -y --no-install-recommends gosu \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 appuser

# 复制 entrypoint 脚本
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# 复制项目代码 (放在最后，因为代码变动最频繁)
COPY --chown=appuser:appuser . .

# 优化点7: 声明数据卷，保护 ChromaDB 数据
VOLUME ["/app/chroma_data"]

# 暴露端口
EXPOSE 8000

# 使用 entrypoint 脚本启动（先修复权限，再切换用户）
ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "main.py"]
