'use client'

export interface Source {
  content: string
  metadata?: Record<string, unknown>
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: Source[]
  streaming?: boolean
}

export interface DocInfo {
  name: string
  size: number
}

// 所有请求走前端同源 /api/*，由 Next.js rewrites 服务端代理到后端容器
// 生产同域名路径分流：带 /rag-assistant 前缀（与 basePath 一致）
const API_BASE = '/rag-assistant/api'

/** SSE 流式问答：回调返回 sources、每个 token、完成、错误 */
export async function askStream(
  question: string,
  callbacks: {
    onSources?: (sources: Source[]) => void
    onToken?: (text: string) => void
    onDone?: () => void
    onError?: (detail: string) => void
  },
): Promise<void> {
  try {
    const res = await fetch(`${API_BASE}/ask/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    })

    if (!res.ok || !res.body) {
      const errText = await res.text().catch(() => '')
      callbacks.onError?.(errText || `HTTP ${res.status}`)
      return
    }

    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })

      // SSE 帧以空行分隔
      let sepIdx: number
      while ((sepIdx = buffer.indexOf('\n\n')) >= 0) {
        const frame = buffer.slice(0, sepIdx)
        buffer = buffer.slice(sepIdx + 2)
        const eventLine = frame.split('\n').find((l) => l.startsWith('event:'))
        const dataLine = frame.split('\n').find((l) => l.startsWith('data:'))
        if (!eventLine || !dataLine) continue

        const event = eventLine.slice(6).trim()
        const dataStr = dataLine.slice(5).trim()
        let data: unknown
        try {
          data = JSON.parse(dataStr)
        } catch {
          continue
        }

        if (event === 'sources') {
          callbacks.onSources?.(data as Source[])
        } else if (event === 'token') {
          const text = (data as { text: string }).text
          callbacks.onToken?.(text)
        } else if (event === 'done') {
          callbacks.onDone?.()
        } else if (event === 'error') {
          callbacks.onError?.((data as { detail: string }).detail)
        }
      }
    }
  } catch (e) {
    callbacks.onError?.(e instanceof Error ? e.message : String(e))
  }
}

/** 上传文档 */
export async function uploadDocument(file: File): Promise<{ filename: string; status: string }> {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${API_BASE}/upload`, { method: 'POST', body: form })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error((err as { detail?: string }).detail || `上传失败 HTTP ${res.status}`)
  }
  return res.json()
}

/** 列出文档 */
export async function listDocuments(): Promise<DocInfo[]> {
  const res = await fetch(`${API_BASE}/documents`)
  if (!res.ok) return []
  const data = (await res.json()) as { documents: DocInfo[] }
  return data.documents || []
}

/** 删除文档 */
export async function deleteDocument(name: string): Promise<void> {
  const res = await fetch(`${API_BASE}/documents/${encodeURIComponent(name)}`, {
    method: 'DELETE',
  })
  if (!res.ok) throw new Error(`删除失败 HTTP ${res.status}`)
}
