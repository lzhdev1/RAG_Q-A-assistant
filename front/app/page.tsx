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
    <div className="layout">
      {/* ── 侧边栏 ── */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <span style={{ fontSize: 18 }}>📚</span> RAG Q&amp;A
        </div>

        <button className="new-chat-btn" onClick={startNewChat}>
          ＋ 新建对话
        </button>

        <div className="session-list">
          {sessions.length === 0 && (
            <div style={{ fontSize: 12, color: '#555', padding: '8px 12px' }}>暂无历史会话</div>
          )}
          {sessions.map((s) => (
            <div
              key={s.id}
              className={`session-item ${s.id === activeId ? 'active' : ''}`}
              onClick={() => switchSession(s.id)}
            >
              💬 {s.title}
            </div>
          ))}
        </div>

        {/* ── 文档管理 ── */}
        <div className="doc-panel">
          <div className="doc-panel-title">知识库文档</div>

          <div className="upload-row">
            <input
              type="file"
              ref={fileInputRef}
              className="upload-input"
              accept=".pdf,.docx,.doc,.txt,.md"
              onChange={(e) => {
                const f = e.target.files?.[0]
                if (f) handleUpload(f)
                e.target.value = ''
              }}
            />
            <button className="upload-btn" onClick={() => fileInputRef.current?.click()}>
              ＋ 上传文档
            </button>
          </div>

          {uploadStatus && <div className="upload-status">{uploadStatus}</div>}

          {docs.length === 0 && (
            <div style={{ fontSize: 11, color: '#555' }}>知识库暂无文档</div>
          )}
          {docs.map((d) => (
            <div key={d.name} className="doc-item">
              <span className="doc-item-name" title={`${(d.size / 1024).toFixed(1)} KB`}>
                📄 {d.name}
              </span>
              <button
                className="doc-delete"
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
      <main className={`main ${isEmpty ? 'main-centered' : ''}`}>
        <div className="chat-area" ref={chatAreaRef}>
          <div className="chat-inner">
            {isEmpty ? (
              <div className="empty-state">
                <h2>企业知识库问答助手</h2>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginTop: 20, width: '100%', maxWidth: 420 }}>
                  {WELCOME_SUGGESTIONS.map((q) => (
                    <button
                      key={q}
                      className="new-chat-btn"
                      style={{ margin: 0 }}
                      onClick={() => handleSend(q)}
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              activeSession.messages.map((m) => (
                <div key={m.id} className={`msg-row ${m.role}`}>
                  <div className={`msg-avatar ${m.role}`}>{m.role === 'assistant' ? 'AI' : '我'}</div>
                  <div className="msg-body">
                    {m.role === 'assistant' ? (
                      <>
                        <div className="msg-bubble">
                          {m.content ? (
                            <div className="markdown">
                              <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>
                            </div>
                          ) : m.streaming ? (
                            <span style={{ color: '#777' }}>思考中…</span>
                          ) : null}
                          {m.streaming && m.content && <span className="cursor">▍</span>}
                        </div>

                        {m.sources && m.sources.length > 0 && (
                          <div className="sources-box">
                            <div className="sources-title">📎 引用来源（Rerank Top-{m.sources.length}）</div>
                            {m.sources.map((src, i) => (
                              <div key={i} className="source-item">
                                <span className="source-num">[{i + 1}]</span>
                                {src.content}
                              </div>
                            ))}
                          </div>
                        )}
                      </>
                    ) : (
                      <div className="msg-bubble">{m.content}</div>
                    )}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* ── 输入区 ── */}
        <div className={`input-bar-wrap ${isEmpty ? 'input-bar-centered' : ''}`}>
          <div className="input-bar">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="输入你的问题，Enter 发送，Shift+Enter 换行"
              rows={1}
              disabled={sending}
            />
            <button
              className="send-btn"
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
