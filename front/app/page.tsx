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

const WELCOME_SUGGESTIONS = [
  '在评估行人重识别系统时，两个广泛使用的度量标准是什么？',
  '视频行人重识别系统由哪两个主要组件构成？',
  '时间特征会带来哪两个副作用？',
]

export default function Home() {
  const [sessions, setSessions] = useState<Session[]>([])
  const [activeId, setActiveId] = useState<string | null>(null)
  const [input, setInput] = useState('')
  const [sending, setSending] = useState(false)
  const [docs, setDocs] = useState<DocInfo[]>([])
  const [uploadStatus, setUploadStatus] = useState('')

  const chatAreaRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const activeSession = sessions.find((s) => s.id === activeId) || null
  const isEmpty = !activeSession || activeSession.messages.length === 0

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
            patchSession(sid, (s) => ({
              ...s,
              messages: s.messages.map((m) =>
                m.id === assistantMsg.id ? { ...m, content: m.content + tok } : m,
              ),
            }))
            scrollToBottom()
          },
          onDone: () => {
            patchSession(sid, (s) => ({
              ...s,
              messages: s.messages.map((m) =>
                m.id === assistantMsg.id ? { ...m, streaming: false } : m,
              ),
            }))
            setSending(false)
            scrollToBottom()
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
    <div className="flex h-screen overflow-hidden bg-zinc-950 text-zinc-200">
      {/* ── 侧边栏 ── */}
      <aside className="flex w-[260px] min-w-[260px] flex-col border-r border-white/10 bg-white/[0.03] backdrop-blur-xl">
        <div className="flex items-center gap-2 border-b border-white/10 px-4 py-4 text-[15px] font-semibold tracking-wide text-zinc-100">
          <span className="flex h-6 w-6 items-center justify-center rounded-md bg-gradient-to-br from-blue-400 to-blue-600 text-[13px] shadow-lg shadow-blue-500/20">
            📚
          </span>
          RAG Q&amp;A
        </div>

        <button
          className="mx-3 mb-1 mt-3 rounded-lg border border-white/15 px-3.5 py-2.5 text-left text-[13px] text-zinc-200 transition hover:border-white/25 hover:bg-white/5"
          onClick={startNewChat}
        >
          ＋ 新建对话
        </button>

        <div className="flex-1 overflow-y-auto px-3 py-1">
          {sessions.length === 0 && (
            <div className="px-3 py-2 text-xs text-zinc-500">暂无历史会话</div>
          )}
          {sessions.map((s) => (
            <div
              key={s.id}
              className={`mb-0.5 cursor-pointer truncate rounded-lg px-3 py-2 text-[13px] transition ${
                s.id === activeId
                  ? 'bg-white/10 text-white shadow-sm'
                  : 'text-zinc-400 hover:bg-white/5 hover:text-zinc-100'
              }`}
              onClick={() => switchSession(s.id)}
            >
              💬 {s.title}
            </div>
          ))}
        </div>

        {/* ── 文档管理 ── */}
        <div className="flex max-h-[260px] flex-col gap-2 overflow-y-auto border-t border-white/10 px-3 py-3">
          <div className="text-[11px] font-semibold uppercase tracking-[0.08em] text-zinc-500">
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
              className="flex-shrink-0 rounded-md bg-blue-500/90 px-3.5 py-1.5 text-xs font-semibold text-white shadow-md shadow-blue-500/20 transition hover:bg-blue-500"
              onClick={() => fileInputRef.current?.click()}
            >
              ＋ 上传文档
            </button>
          </div>

          {uploadStatus && (
            <div className="truncate text-[11px] text-zinc-400">{uploadStatus}</div>
          )}

          {docs.length === 0 && <div className="text-[11px] text-zinc-500">知识库暂无文档</div>}
          {docs.map((d) => (
            <div key={d.name} className="flex items-center justify-between gap-2 text-xs text-zinc-400">
              <span className="flex-1 truncate" title={`${(d.size / 1024).toFixed(1)} KB`}>
                📄 {d.name}
              </span>
              <button
                className="rounded px-1.5 py-0.5 text-[13px] text-zinc-500 transition hover:bg-white/5 hover:text-red-400"
                title="删除"
                onClick={() => handleDeleteDoc(d.name)}
              >
                ✕
              </button>
            </div>
          ))}
        </div>
      </aside>

      {/* ── 主聊天区 ── */}
      <main className={`relative flex flex-1 flex-col overflow-hidden ${isEmpty ? 'justify-center' : ''}`}>
        {/* 背景光晕 */}
        <div
          aria-hidden
          className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_60%_50%_at_50%_-10%,rgba(59,130,246,0.08),transparent)]"
        />

        <div className={`chat-area relative flex-1 overflow-y-auto ${isEmpty ? 'flex-none overflow-visible' : ''}`} ref={chatAreaRef}>
          <div className={`mx-auto ${isEmpty ? 'flex max-w-none flex-col items-center p-0' : 'max-w-[768px] px-5 pb-32 pt-6'}`}>
            {isEmpty ? (
              <div className="flex h-full w-full flex-col items-center justify-center gap-2.5 text-center text-zinc-500">
                <h2 className="m-0 bg-gradient-to-r from-white via-zinc-200 to-zinc-400 bg-clip-text text-[22px] font-semibold tracking-[0.01em] text-transparent">
                  企业知识库问答助手
                </h2>
                <p className="m-0 text-[13px] text-zinc-500">基于混合检索 + Rerank + HyDE 的论文知识问答</p>
                <div className="mt-5 flex w-full max-w-[420px] flex-col gap-2">
                  {WELCOME_SUGGESTIONS.map((q) => (
                    <button
                      key={q}
                      className="m-0 rounded-lg border border-white/15 px-3.5 py-2.5 text-left text-[13px] text-zinc-300 transition hover:border-white/25 hover:bg-white/5"
                      onClick={() => handleSend(q)}
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              activeSession.messages.map((m) => (
                <div
                  key={m.id}
                  className={`mb-6 flex gap-3 ${m.role === 'user' ? 'justify-end' : ''}`}
                >
                  <div
                    className={`flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-lg text-xs font-bold tracking-wide ${
                      m.role === 'assistant'
                        ? 'bg-gradient-to-br from-blue-400 to-blue-600 text-white shadow-lg shadow-blue-500/25'
                        : 'border border-white/10 bg-white/10 text-zinc-300'
                    }`}
                  >
                    {m.role === 'assistant' ? 'AI' : '我'}
                  </div>
                  <div className="flex max-w-[85%] flex-col gap-2">
                    {m.role === 'assistant' ? (
                      <>
                        <div className="rounded-2xl rounded-tl-md border border-white/10 bg-white/[0.06] px-3.5 py-2.5 text-sm leading-[1.7] text-zinc-100">
                          {m.content ? (
                            <div className="markdown">
                              <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>
                            </div>
                          ) : m.streaming ? (
                            <span className="text-zinc-500">思考中…</span>
                          ) : null}
                          {m.streaming && m.content && <span className="cursor">▍</span>}
                        </div>

                        {m.sources && m.sources.length > 0 && (
                          <div className="rounded-xl border border-white/10 bg-white/[0.03] px-3.5 py-2.5">
                            <div className="mb-1.5 text-xs font-semibold tracking-wide text-zinc-400">
                              📎 引用来源（Rerank Top-{m.sources.length}）
                            </div>
                            {m.sources.map((src, i) => (
                              <div
                                key={i}
                                className="line-clamp-3 border-t border-white/10 py-1 text-xs leading-[1.6] text-zinc-400 first:border-t-0"
                              >
                                <span className="mr-1 font-bold tabular-nums text-zinc-200">[{i + 1}]</span>
                                {src.content}
                              </div>
                            ))}
                          </div>
                        )}
                      </>
                    ) : (
                      <div className="rounded-2xl rounded-tr-md bg-blue-600 px-3.5 py-2.5 text-sm leading-[1.7] text-white shadow-lg shadow-blue-500/20">
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
          className={`${isEmpty ? 'relative z-10 mx-auto w-full max-w-[768px] px-5 pt-6' : 'absolute inset-x-0 bottom-0 bg-gradient-to-t from-zinc-950 via-zinc-950/85 to-transparent px-5 pb-5 pt-8'}`}
        >
          <div className="mx-auto flex max-w-[768px] items-end gap-2.5 rounded-3xl border border-white/15 bg-zinc-900/80 px-5 py-3 shadow-[0_8px_30px_rgba(0,0,0,0.35)] backdrop-blur-xl transition focus-within:border-blue-400/50 focus-within:shadow-[0_0_0_4px_rgba(59,130,246,0.12),0_8px_30px_rgba(0,0,0,0.35)]">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="输入你的问题，Enter 发送，Shift+Enter 换行"
              rows={1}
              disabled={sending}
              className="max-h-[200px] flex-1 resize-none bg-transparent px-1 py-1 text-[15px] leading-[1.6] text-zinc-100 outline-none placeholder:text-zinc-500"
            />
            <button
              className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-blue-400 to-blue-600 text-white shadow-lg shadow-blue-500/30 transition hover:scale-105 hover:shadow-blue-500/40 active:scale-95 disabled:cursor-not-allowed disabled:bg-zinc-700 disabled:text-zinc-500 disabled:shadow-none"
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
