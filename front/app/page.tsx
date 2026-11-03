'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import {
  askStream,
  deleteDocument,
  listDocuments,
  uploadDocument,
  type ChatMessage,
  type DocInfo,
  type Source,
} from '../lib/api'

interface Session {
  id: string
  title: string
  messages: ChatMessage[]
}

let sessionSeq = 0
function newSessionId(): string {
  sessionSeq += 1
  return `s_${Date.now()}_${sessionSeq}`
}

function newMsgId(): string {
  return `m_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`
}

// RAGAS 评测（最新 v6 · HyDE 配置，口径：30 题评估集，来源 docs/ERROR_ANALYSIS.md 附录 C，2026-11 存档）
const RAGAS_METRICS = [
  { key: '忠实性 F', value: '0.9542' },
  { key: '答案相关 AR', value: '0.8490' },
  { key: '上下文精度 CP', value: '0.9472' },
] as const

export default function Home() {
  const [sessions, setSessions] = useState<Session[]>([])
  const [activeId, setActiveId] = useState<string | null>(null)
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const [docs, setDocs] = useState<DocInfo[]>([])
  const [uploadStatus, setUploadStatus] = useState('')

  // 主题：初始固定（与 SSR 一致，避免 hydration 不匹配），
  // mount 后从 html.dark（由 layout 内联脚本提前设置）同步真实主题
  const [dark, setDark] = useState<boolean>(false)

  // 挂载后同步：读 html class（themeScript 已在 hydration 前设置）
  useEffect(() => {
    setDark(document.documentElement.classList.contains('dark'))
  }, [])

  const chatAreaRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  // ── 打字机节奏队列 ─────────────────────────────
  // 后端逐 token 到达（可能非常快），前端缓冲后按固定速率逐字打出，
  // 形成打字机观感；done 后等待队列清空再结束 streaming。
  const typeStateRef = useRef<{ sid: string; msgId: string; pending: string; done: boolean } | null>(null)
  const typeTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const TICK_MS = 30
  const CHARS_PER_TICK = 3

  const flushTyping = useCallback(() => {
    const st = typeStateRef.current
    if (!st) return
    if (st.pending.length > 0) {
      const chunk = st.pending.slice(0, CHARS_PER_TICK)
      st.pending = st.pending.slice(CHARS_PER_TICK)
      setSessions((prev) =>
        prev.map((s) =>
          s.id === st.sid
            ? {
                ...s,
                messages: s.messages.map((m) =>
                  m.id === st.msgId ? { ...m, content: m.content + chunk } : m,
                ),
              }
            : s,
        ),
      )
      requestAnimationFrame(() => {
        const el = chatAreaRef.current
        if (el) el.scrollTop = el.scrollHeight
      })
    }
    if (st.pending.length === 0 && st.done) {
      if (typeTimerRef.current) clearInterval(typeTimerRef.current)
      typeTimerRef.current = null
      setSessions((prev) =>
        prev.map((s) =>
          s.id === st.sid
            ? {
                ...s,
                messages: s.messages.map((m) =>
                  m.id === st.msgId ? { ...m, streaming: false } : m,
                ),
              }
            : s,
        ),
      )
      setSending(false)
      typeStateRef.current = null
    }
  }, [])

  const ensureTyping = useCallback(() => {
    if (!typeTimerRef.current && typeStateRef.current) {
      typeTimerRef.current = setInterval(flushTyping, TICK_MS)
    }
  }, [flushTyping])

  // 卸载时清理打字定时器
  useEffect(() => {
    return () => {
      if (typeTimerRef.current) clearInterval(typeTimerRef.current)
    }
  }, [])

  const activeSession = sessions.find((s) => s.id === activeId) || null
  const isEmpty = !activeSession || activeSession.messages.length === 0

  const toggleTheme = useCallback(() => {
    setDark((v) => {
      const next = !v
      document.documentElement.classList.toggle('dark', next)
      try {
        localStorage.setItem('rag-theme', next ? 'dark' : 'light')
      } catch {
        /* 忽略存储不可用 */
      }
      return next
    })
  }, [])

  const scrollToBottom = useCallback(() => {
    requestAnimationFrame(() => {
      const el = chatAreaRef.current
      if (el) el.scrollTop = el.scrollHeight
    })
  }, [])

  const refreshDocs = useCallback(async () => {
    try {
      setDocs(await listDocuments())
    } catch {
      setDocs([])
    }
  }, [])

  useEffect(() => {
    refreshDocs()
  }, [refreshDocs])

  const patchSession = useCallback((sessionId: string, updater: (s: Session) => Session) => {
    setSessions((prev) => prev.map((s) => (s.id === sessionId ? updater(s) : s)))
  }, [])

  const startNewChat = useCallback(() => {
    const id = newSessionId()
    setSessions((prev) => [...prev, { id, title: '新对话', messages: [] }])
    setActiveId(id)
    setInput('')
  }, [])

  const switchSession = useCallback((id: string) => {
    setActiveId(id)
    setInput('')
  }, [])

  const handleSend = useCallback(
    async (text?: string) => {
      const question = (text ?? input).trim()
      if (!question || sending) return

      // 无会话则新建
      let sid: string = activeId ?? ''
      if (!sid) {
        sid = newSessionId()
        setSessions((prev) => [...prev, { id: sid, title: question.slice(0, 20), messages: [] }])
        setActiveId(sid)
      } else {
        patchSession(sid, (s) => ({ ...s, title: s.title === '新对话' ? question.slice(0, 20) : s.title }))
      }

      setInput('')
      setSending(true)

      const userMsg: ChatMessage = { id: newMsgId(), role: 'user', content: question }
      const assistantMsg: ChatMessage = { id: newMsgId(), role: 'assistant', content: '', streaming: true }

      patchSession(sid, (s) => ({ ...s, messages: [...s.messages, userMsg, assistantMsg] }))

      await askStream(
        question,
        {
          onSources: (sources: Source[]) => {
            patchSession(sid, (s) => ({
              ...s,
              messages: s.messages.map((m) => (m.id === assistantMsg.id ? { ...m, sources } : m)),
            }))
          },
          onToken: (tok: string) => {
            // 缓冲进打字队列（不直接渲染，由 flushTyping 逐字打出）
            if (!typeStateRef.current) {
              typeStateRef.current = { sid, msgId: assistantMsg.id, pending: '', done: false }
            }
            typeStateRef.current.pending += tok
            ensureTyping()
          },
          onDone: () => {
            if (typeStateRef.current) {
              typeStateRef.current.done = true
              if (typeStateRef.current.pending.length === 0) flushTyping()
            }
          },
          onError: (detail: string) => {
            patchSession(sid, (s) => ({
              ...s,
              messages: s.messages.map((m) =>
                m.id === assistantMsg.id
                  ? { ...m, content: `⚠️ 请求失败：${detail}`, streaming: false }
                  : m,
              ),
            }))
            setSending(false)
          },
        },
      )
    },
    [activeId, input, sending, patchSession, scrollToBottom],
  )

  const handleUpload = useCallback(
    async (file: File) => {
      setUploadStatus(`正在上传 ${file.name}...`)
      try {
        const r = await uploadDocument(file)
        setUploadStatus(`✓ ${r.filename} 已索引`)
        await refreshDocs()
      } catch (e) {
        setUploadStatus(`✗ ${e instanceof Error ? e.message : String(e)}`)
      }
    },
    [refreshDocs],
  )

  const handleDeleteDoc = useCallback(
    async (name: string) => {
      if (!window.confirm(`删除文档「${name}」？将同时清除其向量。`)) return
      try {
        await deleteDocument(name)
        await refreshDocs()
        setUploadStatus(`✓ 已删除 ${name}`)
      } catch (e) {
        setUploadStatus(`✗ 删除失败：${e instanceof Error ? e.message : String(e)}`)
      }
    },
    [refreshDocs],
  )

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault()
        handleSend()
      }
    },
    [handleSend],
  )

  return (
    <div className="flex h-screen overflow-hidden bg-surface text-content transition-colors">
      {/* ── 侧边栏 ── */}
      <aside className="flex w-[260px] min-w-[260px] flex-col border-r border-border-soft bg-surface-2 backdrop-blur-xl">
        <div className="flex items-center gap-2 border-b border-border-soft px-4 py-4 text-[15px] font-semibold tracking-wide text-content">
          <span className="flex h-6 w-6 items-center justify-center rounded-md bg-gradient-to-br from-blue-400 to-blue-600 text-white shadow-lg shadow-blue-500/20">
            <LogoIcon size={14} />
          </span>
          <span className="flex-1">RAG Q&amp;A</span>
          <button
            className="flex h-8 w-8 items-center justify-center rounded-lg border border-border-strong text-content-2 transition hover:bg-surface-2 hover:text-content"
            onClick={toggleTheme}
            title={dark ? '切换到亮色' : '切换到暗色'}
          >
            {dark ? (
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                <circle cx="12" cy="12" r="4" />
                <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41" />
              </svg>
            ) : (
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
              </svg>
            )}
          </button>
        </div>

        <button
          className="mx-3 mb-1 mt-3 flex items-center gap-1.5 rounded-lg border border-border-strong px-3.5 py-2.5 text-left text-[13px] text-content transition hover:border-border-strong hover:bg-content-2/10"
          onClick={startNewChat}
        >
          <PlusIcon size={13} />
          新建对话
        </button>

        <div className="flex-1 overflow-y-auto px-3 py-1">
          {sessions.length === 0 && (
            <div className="px-3 py-2 text-xs text-content-muted">暂无历史会话</div>
          )}
          {sessions.map((s) => (
            <div
              key={s.id}
              className={`mb-0.5 flex cursor-pointer items-center gap-1.5 rounded-lg px-3 py-2 text-[13px] transition ${
                s.id === activeId
                  ? 'bg-content-2/15 text-content shadow-sm'
                  : 'text-content-2 hover:bg-content-2/10 hover:text-content'
              }`}
              onClick={() => switchSession(s.id)}
            >
              <ChatIcon size={13} />
              <span className="truncate">{s.title}</span>
            </div>
          ))}
        </div>

        {/* ── RAGAS 评测 ── */}
        <div className="mx-3 mb-1 mt-3 rounded-lg border border-border-soft bg-surface-2 p-3">
          <div className="mb-1.5 flex items-center justify-between">
            <span className="text-[11px] font-semibold uppercase tracking-[0.08em] text-content-3">
              RAGAS 评测
            </span>
            <span className="rounded bg-content-2/10 px-1.5 py-0.5 text-[10px] font-medium text-content-2">
              v6 · HyDE
            </span>
          </div>
          <div className="flex gap-2">
            {RAGAS_METRICS.map((m) => (
              <div key={m.key} className="flex-1 rounded-md bg-surface px-1 py-1.5 text-center">
                <div className="text-[13px] font-semibold tabular-nums leading-tight text-content">
                  {m.value}
                </div>
                <div className="mt-0.5 text-[10px] leading-tight text-content-muted">{m.key}</div>
              </div>
            ))}
          </div>
        </div>

        {/* ── 文档管理 ── */}
        <div className="flex max-h-[260px] flex-col gap-2 overflow-y-auto border-t border-border-soft px-3 py-3">
          <div className="text-[11px] font-semibold uppercase tracking-[0.08em] text-content-3">
            知识库文档
          </div>

          <div className="flex items-center gap-2">
            <input
              type="file"
              ref={fileInputRef}
              className="hidden"
              accept=".pdf,.docx,.doc,.txt,.md"
              onChange={(e) => {
                const f = e.target.files?.[0]
                if (f) handleUpload(f)
                e.target.value = ''
              }}
            />
            <button
              className="flex flex-shrink-0 items-center gap-1.5 rounded-md bg-blue-500/90 px-3.5 py-1.5 text-xs font-semibold text-white shadow-md shadow-blue-500/20 transition hover:bg-blue-500"
              onClick={() => fileInputRef.current?.click()}
            >
              <UploadIcon size={13} />
              上传文档
            </button>
          </div>

          {uploadStatus && (
            <div className="truncate text-[11px] text-content-2">{uploadStatus}</div>
          )}

          {docs.length === 0 && <div className="text-[11px] text-content-muted">知识库暂无文档</div>}
          {docs.map((d) => (
            <div key={d.name} className="flex items-center justify-between gap-2 text-xs text-content-2">
              <span className="flex min-w-0 flex-1 items-center gap-1.5 truncate" title={`${(d.size / 1024).toFixed(1)} KB`}>
                <DocIcon size={13} />
                <span className="truncate">{d.name}</span>
              </span>
              <button
                className="flex flex-shrink-0 items-center justify-center rounded px-1.5 py-0.5 text-content-muted transition hover:bg-content-2/10 hover:text-red-400"
                title="删除"
                onClick={() => handleDeleteDoc(d.name)}
              >
                <CloseIcon size={13} />
              </button>
            </div>
          ))}
        </div>
      </aside>
      <main className={`relative flex flex-1 flex-col overflow-hidden ${isEmpty ? 'justify-center' : ''}`}>
        {/* 背景光晕 */}
        <div
          aria-hidden
          className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_60%_50%_at_50%_-10%,rgba(59,130,246,0.08),transparent)]"
        />

        <div className={`chat-area relative flex-1 overflow-y-auto scroll-smooth ${isEmpty ? 'flex-none overflow-visible' : ''}`} ref={chatAreaRef}>
          <div className={`mx-auto ${isEmpty ? 'flex max-w-none flex-col items-center p-0' : 'max-w-[768px] px-5 pb-32 pt-6'}`}>
            {isEmpty ? (
              <div className="flex h-full w-full flex-col items-center justify-center gap-2.5 text-center">
                <h2 className="m-0 bg-gradient-to-r from-content via-content-2 to-content-muted bg-clip-text text-[22px] font-semibold tracking-[0.01em] text-transparent">
                  企业知识库问答助手
                </h2>
              </div>
            ) : (
              activeSession.messages.map((m) => (
                <div
                  key={m.id}
                  className={`msg-in mb-6 flex gap-3 ${m.role === 'user' ? 'flex-row-reverse' : ''}`}
                >
                  <div
                    className={`flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-lg text-xs font-bold tracking-wide ${
                      m.role === 'assistant'
                        ? 'bg-gradient-to-br from-blue-400 to-blue-600 text-white shadow-lg shadow-blue-500/25'
                        : 'border border-border-strong bg-surface-2 text-content-2'
                    }`}
                  >
                    {m.role === 'assistant' ? 'AI' : '我'}
                  </div>
                  <div className="flex max-w-[85%] flex-col gap-2">
                    {m.role === 'assistant' ? (
                      <>
                        <div className="rounded-2xl rounded-tl-md border border-border-soft bg-bubble-ai px-3.5 py-2.5 text-sm leading-[1.7] text-content">
                          {m.content ? (
                            <div className="markdown">
                              <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>
                            </div>
                          ) : m.streaming ? (
                            <span className="thinking text-content-muted">思考中…</span>
                          ) : null}
                          {m.streaming && m.content && <span className="cursor" />}
                        </div>

                        {m.sources && m.sources.length > 0 && (
                          <div className="rounded-xl border border-border-soft bg-surface-2 px-3.5 py-2.5">
                            <div className="mb-1.5 text-xs font-semibold tracking-wide text-content-2">
                              📎 引用来源（Rerank Top-{m.sources.length}）
                            </div>
                            {m.sources.map((src, i) => (
                              <div
                                key={i}
                                className="line-clamp-3 border-t border-border-soft py-1 text-xs leading-[1.6] text-content-2 first:border-t-0"
                              >
                                <span className="mr-1 font-bold tabular-nums text-content">{i + 1}</span>
                                {src.content}
                              </div>
                            ))}
                          </div>
                        )}
                      </>
                    ) : (
                      <div className="rounded-2xl rounded-tr-md bg-gradient-to-br from-blue-500 to-blue-600 px-3.5 py-2.5 text-sm leading-[1.7] text-white shadow-lg shadow-blue-500/20">
                        {m.content}
                      </div>
                    )}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* ── 输入区 ── */}
        <div
          className={`${isEmpty ? 'relative z-10 mx-auto w-full max-w-[768px] px-5 pt-6' : 'absolute inset-x-0 bottom-0 bg-gradient-to-t from-surface via-surface/85 to-transparent px-5 pb-5 pt-8'}`}
        >
          <div className="mx-auto flex max-w-[768px] items-end gap-2.5 rounded-3xl border border-border-strong bg-input-bg px-5 py-3 shadow-[0_8px_30px_rgba(0,0,0,0.12)] backdrop-blur-xl transition focus-within:border-blue-400/60 focus-within:shadow-[0_0_0_4px_rgba(59,130,246,0.12),0_8px_30px_rgba(0,0,0,0.12)]">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="输入你的问题，Enter 发送，Shift+Enter 换行"
              rows={1}
              disabled={sending}
              className="max-h-[200px] flex-1 resize-none bg-transparent px-1 py-1 text-[15px] leading-[1.6] text-content outline-none placeholder:text-content-muted"
            />
            <button
              className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-blue-400 to-blue-600 text-white shadow-lg shadow-blue-500/30 transition hover:scale-105 hover:shadow-blue-500/40 active:scale-95 disabled:cursor-not-allowed disabled:bg-border-strong disabled:text-content-muted disabled:shadow-none"
              disabled={sending || !input.trim()}
              onClick={() => handleSend()}
              title="发送"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M22 2L11 13" />
                <path d="M22 2L15 22L11 13L2 9L22 2Z" />
              </svg>
            </button>
          </div>
        </div>
      </main>
    </div>
  )
}

/* ── 极简线性图标集（1.8px 描边，currentColor 跟随主题） ── */
const iconProps = {
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.8,
  strokeLinecap: 'round' as const,
  strokeLinejoin: 'round' as const,
}

/** 打开的书（知识库 Logo） */
function LogoIcon({ size = 14 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" {...iconProps}>
      <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z" />
      <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z" />
    </svg>
  )
}

/** 聊天气泡（会话条目） */
function ChatIcon({ size = 13 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" {...iconProps}>
      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
    </svg>
  )
}

/** 文档（知识库条目） */
function DocIcon({ size = 13 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" {...iconProps}>
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <path d="M14 2v6h6" />
      <path d="M16 13H8" />
      <path d="M16 17H8" />
    </svg>
  )
}

/** 关闭（删除） */
function CloseIcon({ size = 13 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" {...iconProps}>
      <path d="M18 6L6 18" />
      <path d="M6 6l12 12" />
    </svg>
  )
}

/** 加号（新建对话） */
function PlusIcon({ size = 13 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" {...iconProps}>
      <path d="M12 5v14" />
      <path d="M5 12h14" />
    </svg>
  )
}

/** 上传（上传文档） */
function UploadIcon({ size = 13 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" {...iconProps}>
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
      <path d="M17 8l-5-5-5 5" />
      <path d="M12 3v12" />
    </svg>
  )
}
