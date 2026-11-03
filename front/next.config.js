/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: 'standalone',

  // 服务端代理：浏览器只访问前端同源 /api/*，Next.js 转发到后端容器
  // 容器内：API_PROXY_TARGET=http://rag-app:8000（compose 构建期注入）
  // 本地开发：API_PROXY_TARGET=http://localhost:8000
  async rewrites() {
    const target = process.env.API_PROXY_TARGET || 'http://rag-app:8000'
    return [
      {
        source: '/api/:path*',
        destination: `${target}/:path*`,
      },
    ]
  },
}

module.exports = nextConfig
