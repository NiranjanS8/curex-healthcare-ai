import { useEffect, useRef, useState } from 'react'
import {
  AlertCircle,
  ArrowRight,
  BookOpenCheck,
  Database,
  Menu,
  ShieldCheck,
  Stethoscope,
  X,
} from 'lucide-react'
import { ChatInput } from './components/ChatInput'
import { ChatMessage } from './components/ChatMessage'
import { CitationCard } from './components/CitationCard'
import { Sidebar } from './components/Sidebar'
import './App.css'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const DEFAULT_METRICS = [
  { name: 'Faithfulness', value: null, level: 'unknown' },
  { name: 'Answer Relevancy', value: null, level: 'unknown' },
  { name: 'Context Precision', value: null, level: 'unknown' },
  { name: 'Context Recall', value: null, level: 'unknown' },
]

function nowLabel() {
  return new Date().toLocaleTimeString('en-US', {
    hour: 'numeric',
    minute: '2-digit',
  })
}

function metricLevel(value) {
  if (value >= 0.8) return 'high'
  if (value >= 0.6) return 'medium'
  return 'low'
}

function sessionTitleFromMessage(message) {
  const cleaned = message.trim().replace(/\s+/g, ' ')
  if (!cleaned) return 'Healthcare Conversation'
  return cleaned.length > 42 ? `${cleaned.slice(0, 42)}...` : cleaned
}

function LandingPage({ metrics, onStart }) {
  return (
    <main className="landing-page">
      <nav className="landing-nav" aria-label="Primary">
        <div className="brand-mark">
          <span>CX</span>
          <strong>CureX</strong>
        </div>
        <button type="button" className="nav-action" onClick={onStart}>
          Open assistant
          <ArrowRight size={16} />
        </button>
      </nav>

      <section className="landing-hero">
        <div className="hero-copy">
          <p className="eyebrow">Healthcare RAG Assistant</p>
          <h1>Evidence-grounded answers for clinical research questions.</h1>
          <p className="hero-lede">
            CureX combines retrieval, citations, safety checks, and faithfulness scoring so medical
            information stays traceable to the source material.
          </p>
          <div className="hero-actions">
            <button type="button" className="primary-landing-action" onClick={onStart}>
              Start a session
              <ArrowRight size={18} />
            </button>
            <a href="#how-it-works" className="secondary-landing-action">
              View workflow
            </a>
          </div>
        </div>

        <div className="hero-console" aria-label="CureX workflow preview">
          <div className="console-header">
            <span />
            <span />
            <span />
          </div>
          <div className="console-body">
            <div className="pipeline-row">
              <Database size={18} />
              <span>Retrieve medical context from pgvector</span>
            </div>
            <div className="pipeline-row">
              <BookOpenCheck size={18} />
              <span>Rank evidence and attach citations</span>
            </div>
            <div className="pipeline-row">
              <ShieldCheck size={18} />
              <span>Apply safety and faithfulness checks</span>
            </div>
            <div className="answer-preview">
              <small>Assistant response contract</small>
              <p>Answer only when retrieved evidence supports the claim.</p>
              <span>[Source: title, chunk id]</span>
            </div>
          </div>
        </div>
      </section>

      <section className="landing-metrics" aria-label="Evaluation metrics">
        {metrics.map((metric) => (
          <div key={metric.name} className="landing-metric">
            <span>{metric.name}</span>
            <strong>{metric.value === null ? '--' : `${(metric.value * 100).toFixed(0)}%`}</strong>
          </div>
        ))}
      </section>

      <section id="how-it-works" className="landing-bands">
        <article>
          <Stethoscope size={22} />
          <h2>Built for healthcare questions</h2>
          <p>Routes intent, recalls session context, and keeps responses informational.</p>
        </article>
        <article>
          <Database size={22} />
          <h2>Local vector retrieval</h2>
          <p>Uses pgvector-backed retrieval with reranking and cited source chunks.</p>
        </article>
        <article>
          <ShieldCheck size={22} />
          <h2>Safety-first generation</h2>
          <p>Flags unsafe requests and scores whether each answer is grounded.</p>
        </article>
      </section>

      <footer className="landing-footer">
        <AlertCircle size={16} />
        <span>For informational purposes only. Not a substitute for professional medical advice.</span>
      </footer>
    </main>
  )
}

function App() {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [activeSessionId, setActiveSessionId] = useState(null)
  const [selectedCitationId, setSelectedCitationId] = useState(null)
  const [messages, setMessages] = useState([])
  const [sessions, setSessions] = useState([])
  const [metrics, setMetrics] = useState(DEFAULT_METRICS)
  const [citationDetails, setCitationDetails] = useState({})
  const chatEndRef = useRef(null)

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  useEffect(() => {
    let cancelled = false
    fetch(`${API_BASE_URL}/eval/metrics`)
      .then((response) => (response.ok ? response.json() : null))
      .then((payload) => {
        if (cancelled || !payload?.metrics) return
        const nextMetrics = DEFAULT_METRICS.map((metric) => {
          const key = metric.name.toLowerCase().replaceAll(' ', '_')
          const value = payload.metrics[key]
          return value === undefined
            ? metric
            : { ...metric, value, level: metricLevel(value) }
        })
        setMetrics(nextMetrics)
      })
      .catch(() => {})
    return () => {
      cancelled = true
    }
  }, [])

  const createSession = async () => {
    let sessionId = `session-${Date.now()}`
    try {
      const response = await fetch(`${API_BASE_URL}/session/new`, { method: 'POST' })
      if (response.ok) {
        const payload = await response.json()
        sessionId = payload.session_id || sessionId
      }
    } catch {
      // A local session lets the interface remain usable while the API starts.
    }
    return sessionId
  }

  const handleNewSession = async () => {
    const sessionId = await createSession()
    setMessages([])
    setActiveSessionId(sessionId)
    setSelectedCitationId(null)
    setCitationDetails({})
    setSessions((current) => [
      { id: sessionId, title: 'New Healthcare Conversation', timestamp: 'Just now' },
      ...current,
    ])
    setIsMobileMenuOpen(false)
  }

  const handleSelectSession = async (id) => {
    setActiveSessionId(id)
    setSelectedCitationId(null)
    setIsMobileMenuOpen(false)
    try {
      const response = await fetch(`${API_BASE_URL}/session/${id}/history`)
      if (!response.ok) return
      const payload = await response.json()
      setMessages(
        (payload.messages || []).map((message, index) => ({
          id: `${id}-${index}`,
          role: message.type === 'human' ? 'user' : 'assistant',
          content: message.content,
          timestamp: '',
        })),
      )
    } catch {
      setMessages([])
    }
  }

  const handleCitationClick = (citationId) => {
    setSelectedCitationId(citationId)
  }

  const applyDonePayload = (assistantId, payload) => {
    if (!payload) return
    const citations = (payload.citations || []).map((citation, index) => {
      const id = citation.chunk_id || `citation-${assistantId}-${index}`
      setCitationDetails((current) => ({
        ...current,
        [id]: {
          id,
          title: citation.title || 'Retrieved medical source',
          docType: citation.doc_type || 'Medical Source',
          date: citation.date || 'Unknown date',
          specialty: citation.specialty || 'General Medicine',
          excerpt: citation.excerpt || '',
          source: citation.source_url || citation.metadata?.source || 'Retrieved context',
          metadata: citation.metadata || {},
        },
      }))
      return {
        id,
        source: citation.title || 'Medical Source',
        docType: citation.doc_type || 'Medical Source',
        chunkId: citation.chunk_id || id,
      }
    })

    setMessages((current) =>
      current.map((message) =>
        message.id === assistantId
          ? {
              ...message,
              citations,
              faithfulnessScore: payload.faithfulness_score ?? message.faithfulnessScore,
            }
          : message,
      ),
    )
  }

  const sendViaApi = async (sessionId, content, assistantId) => {
    const response = await fetch(`${API_BASE_URL}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, message: content }),
    })
    if (!response.ok || !response.body) throw new Error('Unable to stream response')

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const events = buffer.split('\n\n')
      buffer = events.pop() || ''

      for (const event of events) {
        const dataLine = event
          .split('\n')
          .find((line) => line.startsWith('data: '))
          ?.slice(6)
        if (!dataLine || dataLine === '[DONE]') continue
        const payload = JSON.parse(dataLine)
        if (event.startsWith('event: done')) {
          applyDonePayload(assistantId, payload)
          continue
        }
        if (payload.token) {
          setMessages((current) =>
            current.map((message) =>
              message.id === assistantId
                ? { ...message, content: `${message.content}${payload.token}` }
                : message,
            ),
          )
        }
      }
    }
  }

  const handleSendMessage = async (content) => {
    const sessionId = activeSessionId || (await createSession())
    if (!activeSessionId) {
      setActiveSessionId(sessionId)
      setSessions((current) => [
        { id: sessionId, title: sessionTitleFromMessage(content), timestamp: 'Just now' },
        ...current,
      ])
    } else {
      setSessions((current) =>
        current.map((session) =>
          session.id === sessionId && session.title === 'New Healthcare Conversation'
            ? { ...session, title: sessionTitleFromMessage(content) }
            : session,
        ),
      )
    }

    const userMessage = {
      id: `msg-${Date.now()}`,
      role: 'user',
      content,
      timestamp: nowLabel(),
    }
    const assistantId = `msg-${Date.now()}-assistant`
    const assistantMessage = {
      id: assistantId,
      role: 'assistant',
      content: '',
      citations: [],
      timestamp: nowLabel(),
    }

    setMessages((current) => [...current, userMessage, assistantMessage])
    setIsLoading(true)
    setSelectedCitationId(null)

    try {
      await sendViaApi(sessionId, content, assistantId)
    } catch {
      setMessages((current) =>
        current.map((message) =>
          message.id === assistantId
            ? {
                ...message,
                content:
                  'Unable to connect to the healthcare assistant API. Start the backend service and try again.',
              }
            : message,
        ),
      )
    } finally {
      setIsLoading(false)
    }
  }

  const selectedCitation = selectedCitationId ? citationDetails[selectedCitationId] : null

  if (!activeSessionId) {
    return <LandingPage metrics={metrics} onStart={handleNewSession} />
  }

  return (
    <div className="app-shell">
      <button
        type="button"
        aria-label="Toggle navigation"
        onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
        className="mobile-menu-button"
      >
        {isMobileMenuOpen ? <X size={20} /> : <Menu size={20} />}
      </button>

      <div className={`sidebar-layer ${isMobileMenuOpen ? 'sidebar-layer-open' : ''}`}>
        <Sidebar
          sessions={sessions}
          activeSessionId={activeSessionId}
          metrics={metrics}
          onNewSession={handleNewSession}
          onSelectSession={handleSelectSession}
        />
      </div>

      {isMobileMenuOpen && (
        <button
          type="button"
          aria-label="Close navigation overlay"
          className="mobile-overlay"
          onClick={() => setIsMobileMenuOpen(false)}
        />
      )}

      <main className="main-panel">
        <div className="disclaimer-banner">
          <AlertCircle size={16} className="disclaimer-icon" />
          <p>
            <strong>Disclaimer:</strong> For informational purposes only. Not a substitute for
            professional medical advice.
          </p>
        </div>

        <section className="chat-scroll">
          <div className="chat-content">
            {messages.length === 0 ? (
              <div className="empty-state">
                <div className="empty-state-icon">
                  <AlertCircle size={32} />
                </div>
                <h2>Start a Healthcare Conversation</h2>
                <p>
                  Ask questions about clinical guidelines, treatment protocols, or medical research.
                  Responses are grounded in evidence-based sources.
                </p>
              </div>
            ) : (
              <>
                {messages.map((message) => (
                  <ChatMessage
                    key={message.id}
                    message={message}
                    onCitationClick={handleCitationClick}
                  />
                ))}
                <div ref={chatEndRef} />
              </>
            )}

            {selectedCitation && (
              <div className="citation-details">
                <h3>
                  <span />
                  Citation Details
                </h3>
                <CitationCard citation={selectedCitation} />
              </div>
            )}
          </div>
        </section>

        <ChatInput onSend={handleSendMessage} isLoading={isLoading} />
      </main>
    </div>
  )
}

export default App
