import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'RAG Q&A Assistant',
  description: '企业知识库问答助手',
}

// hydration 前同步主题（读 localStorage，无记录默认暗色），避免闪烁
const themeScript = `(function(){try{var t=localStorage.getItem('rag-theme');if(!t)t='dark';if(t==='dark')document.documentElement.classList.add('dark')}catch(e){document.documentElement.classList.add('dark')}})()`

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="zh-CN" suppressHydrationWarning>
      <body>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
        {children}
      </body>
    </html>
  )
}
