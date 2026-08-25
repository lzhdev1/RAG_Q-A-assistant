#!/bin/bash
set -e

# 修复挂载目录的权限，确保 appuser (uid 1000) 可以写入
chown -R appuser:appuser /app/chroma_data 2>/dev/null || true

# 以 appuser 身份执行后续命令
exec gosu appuser "$@"
