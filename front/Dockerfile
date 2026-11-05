# ==========================================
# 前端构建（Next.js standalone）
# ==========================================
FROM node:20-alpine AS builder

WORKDIR /app

# 依赖层（package.json 不变时复用缓存）
COPY package.json package-lock.json* ./
# 生产环境用国内镜像加速（服务器访问 npm 官方源慢）
RUN npm config set registry https://registry.npmmirror.com \
    && npm install --no-audit --no-fund

# 源码层
COPY . .

# 构建 standalone 产物（注入后端代理目标：容器内用服务名）
ENV NEXT_TELEMETRY_DISABLED=1
ENV API_PROXY_TARGET=http://rag-app:8000
RUN npm run build

# ==========================================
# 运行环境（仅复制 standalone 产物）
# ==========================================
FROM node:20-alpine AS runner

WORKDIR /app
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public

EXPOSE 3000

ENV PORT=3000
ENV HOSTNAME=0.0.0.0

CMD ["node", "server.js"]
