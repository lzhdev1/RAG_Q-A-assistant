# ==========================================
# 阶段 1: 构建依赖 (Builder)
# ==========================================
FROM python:3.10-slim AS builder

WORKDIR /app

# 优化点1: 设置环境变量，防止生成 .pyc 文件，减少体积
# PYTHONPATH 指向 --prefix=/install 的 site-packages，让后续 pip 安装时能识别已装包，
# 避免同一依赖被重复解析/下载（否则 sentence-transformers 会重新拉 GPU 版 torch）
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/install/lib/python3.10/site-packages

# 优化点2: 先只复制依赖文件，利用 Docker 缓存
# 只要 requirements.txt 不变，这一层就不会重新构建
COPY requirements.txt .

# 优化点2.1: 配置 pip 使用清华镜像 + 超时重试
# （原阿里云源 mirrors.aliyun.com 于 2026-11 起对 wheel 下载返回 403 Forbidden，导致 torch 大包构建卡死，改为清华源）
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/ \
    && pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn \
    && pip config set global.timeout 300 \
    && pip config set install.retries 10

# 优化点2.2: PyTorch 使用 CPU 轮子（服务器无 GPU）
# 为什么不装 requirements 里的 torch==2.14.1：GPU 版会连带下载 nvidia-cudnn-cu13(553MB)、
# cusparselt(170MB)、cublas 等 2GB+ CUDA 依赖，服务器实测仅 ~350KB/s，构建会长时间卡死；
# CPU 版 torch-2.6.0+cpu 仅 178MB，无 nvidia 依赖，服务器实测 1.8MB/s。
# 下载用 python urllib（slim 镜像无 curl），文件按 PEP 427 规范名保存，避免 pip 拒绝非规范 wheel 文件名。
RUN python -c "import urllib.request; urllib.request.urlretrieve('https://download.pytorch.org/whl/cpu/torch-2.6.0%2Bcpu-cp310-cp310-linux_x86_64.whl', '/tmp/torch-2.6.0+cpu-cp310-cp310-linux_x86_64.whl')" \
    && pip install --no-cache-dir --prefix=/install '/tmp/torch-2.6.0+cpu-cp310-cp310-linux_x86_64.whl' \
    && rm -f '/tmp/torch-2.6.0+cpu-cp310-cp310-linux_x86_64.whl'

# 优化点2.3: SQLAlchemy 2.0.55 曾因阿里云镜像 wheel 与官方 hash 不一致需单独用清华源安装；
# 现全局源已切清华，该约束安装时已满足，保留原步骤避免重复下载
RUN pip install --no-cache-dir --prefix=/install -i https://pypi.tuna.tsinghua.edu.cn/simple "SQLAlchemy==2.0.55"

# 优化点2.4: 过滤 requirements.txt 中的 torch 行后安装其余依赖
# （torch 已在上一步装好 CPU 版，保留 torch==2.14.1 会让 pip 重新下载 GPU 版并拖入 CUDA 依赖）
# constraints.txt 固定 transformers 版本，避免 pip 在 transformers 上无限回溯（实测下载 180+ 次 10MB 轮子）
COPY constraints.txt /tmp/constraints.txt
RUN grep -v -E '^torch' requirements.txt > /tmp/requirements-no-torch.txt \
    && pip install --no-cache-dir --prefix=/install -c /tmp/constraints.txt -r /tmp/requirements-no-torch.txt

# ==========================================
# 阶段 2: 运行环境 (Runner)
# ==========================================
FROM python:3.10-slim AS runner

WORKDIR /app

# 优化点3: 从 builder 阶段复制纯净的 Python 包
COPY --from=builder /install /usr/local

# 优化点4: 安装 gosu 用于 entrypoint 中切换用户，创建非 root 用户
# 服务器直连 deb.debian.org 极慢/卡死，apt 源切换为清华镜像（deb822 sources 格式，替换域名即可）
RUN sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list.d/debian.sources 2>/dev/null || true; \
    apt-get update && apt-get install -y --no-install-recommends gosu \
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