import { useEffect, useRef, useState } from 'react'
import {
  AlertCircle,
  ArrowRight,
  BookOpenCheck,
  BrainCircuit,
  ClipboardCheck,
  Database,
  Eye,
  EyeOff,
  FileText,
  LockKeyhole,
  LogOut,
  Menu,
  ShieldCheck,
  Stethoscope,
  UploadCloud,
  UserPlus,
  X,
} from 'lucide-react'
import { ChatInput } from './components/ChatInput'
import { ChatMessage } from './components/ChatMessage'
import { CitationCard } from './components/CitationCard'
import { Sidebar } from './components/Sidebar'
import './App.css'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
const TOKEN_STORAGE_KEY = 'curex_auth'

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

function createAgentTrace({ status = 'pending', citations = [], faithfulnessScore = null, error = null } = {}) {
  const hasCitations = citations.length > 0
  return [
    {
      label: 'Query router',
      detail: status === 'pending' ? 'Classifying healthcare intent' : 'Intent routed',
      status: error ? 'complete' : status === 'pending' ? 'active' : 'complete',
    },
    {
      label: hasCitations ? 'Retriever' : 'Retriever / tools',
      detail:
        status === 'pending'
          ? 'Waiting for evidence retrieval'
          : hasCitations
            ? `${citations.length} cited source${citations.length === 1 ? '' : 's'} returned`
            : 'No cited context returned',
      status: status === 'pending' ? 'pending' : hasCitations ? 'complete' : error ? 'blocked' : 'pending',
    },
    {
      label: 'Safety check',
      detail: error ? 'Response blocked by connection state' : 'Medical safety constraints applied',
      status: status === 'pending' ? 'pending' : error ? 'blocked' : 'complete',
    },
    {
      label: 'Faithfulness check',
      detail:
        faithfulnessScore === null
          ? 'Score unavailable'
          : `Grounding score ${(faithfulnessScore * 100).toFixed(0)}%`,
      status: status === 'pending' ? 'pending' : faithfulnessScore === null ? 'pending' : 'complete',
    },
    {
      label: 'Response',
      detail: error || (status === 'pending' ? 'Streaming answer' : 'Answer delivered'),
      status: error ? 'blocked' : status === 'pending' ? 'active' : 'complete',
    },
  ]
}

function readStoredAuth() {
  try {
    return JSON.parse(localStorage.getItem(TOKEN_STORAGE_KEY) || 'null')
  } catch {
    return null
  }
}

function storeAuth(auth) {
  localStorage.setItem(TOKEN_STORAGE_KEY, JSON.stringify(auth))
}

function clearStoredAuth() {
  localStorage.removeItem(TOKEN_STORAGE_KEY)
}

function LandingPage({ metrics, onStart, onSignIn, isAuthenticated }) {
  return (
    <main className="landing-page">
      <nav className="landing-nav" aria-label="Primary">
        <div className="brand-mark">
          <span>CX</span>
          <strong>CureX</strong>
        </div>
        <div className="landing-nav-actions">
          {!isAuthenticated && (
            <button type="button" className="nav-action" onClick={onSignIn}>
              Sign in
            </button>
          )}
          <button type="button" className="nav-action" onClick={onStart}>
            Open assistant
            <ArrowRight size={16} />
          </button>
        </div>
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
          <p>
            Ask medication, condition, guideline, and clinical research questions in a focused
            assistant experience designed for healthcare information workflows.
          </p>
        </article>
        <article>
          <Database size={22} />
          <h2>Local vector retrieval</h2>
          <p>
            Retrieves relevant passages from a local pgvector knowledge base, reranks evidence,
            and keeps source chunks available beside every answer.
          </p>
        </article>
        <article>
          <ShieldCheck size={22} />
          <h2>Safety-first generation</h2>
          <p>
            Applies medical safety guardrails, blocks unsafe clinical instructions, and checks
            whether responses stay grounded in retrieved evidence.
          </p>
        </article>
      </section>

      <section className="landing-detail">
        <div>
          <p className="eyebrow">What CureX does</p>
          <h2>Turns healthcare documents into cited, auditable assistant responses.</h2>
        </div>
        <div className="detail-grid">
          <article>
            <BookOpenCheck size={20} />
            <h3>Evidence-backed answers</h3>
            <p>
              CureX answers from retrieved medical context instead of unsupported recall, then
              shows citations so reviewers can inspect the source material quickly.
            </p>
          </article>
          <article>
            <BrainCircuit size={20} />
            <h3>Agentic workflow</h3>
            <p>
              The assistant can expose its reasoning workflow on demand, showing routing,
              retrieval, safety review, grounding checks, and response generation.
            </p>
          </article>
          <article>
            <ClipboardCheck size={20} />
            <h3>Evaluation built in</h3>
            <p>
              Faithfulness, relevancy, context precision, and recall metrics help track whether
              the RAG pipeline is producing dependable healthcare responses.
            </p>
          </article>
        </div>
      </section>

      <section className="landing-workflow" aria-label="How CureX works">
        <div className="workflow-copy">
          <p className="eyebrow">How it works</p>
          <h2>From question to answer, every step is grounded.</h2>
          <p>
            CureX receives a healthcare question, retrieves relevant medical passages, validates
            the response against safety and grounding rules, then streams a cited answer back to
            the user.
          </p>
        </div>
        <ol className="workflow-steps">
          <li>
            <span>01</span>
            <strong>Route the request</strong>
            <p>The system classifies the healthcare intent and prepares the right retrieval path.</p>
          </li>
          <li>
            <span>02</span>
            <strong>Retrieve and rank evidence</strong>
            <p>Relevant chunks are pulled from pgvector and ordered for answer quality.</p>
          </li>
          <li>
            <span>03</span>
            <strong>Generate with guardrails</strong>
            <p>The model answers with medical safety constraints and cites the supporting context.</p>
          </li>
          <li>
            <span>04</span>
            <strong>Evaluate grounding</strong>
            <p>RAG quality metrics make the output reviewable for applied AI workflows.</p>
          </li>
        </ol>
      </section>

      <footer className="landing-footer">
        <AlertCircle size={16} />
        <span>For informational purposes only. Not a substitute for professional medical advice.</span>
      </footer>
    </main>
  )
}

function AuthPage({ mode, onModeChange, onAuthenticated, onCancel }) {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)
  const isRegister = mode === 'register'

  const handleSubmit = async (event) => {
    event.preventDefault()
    setError('')
    setIsSubmitting(true)
    try {
      const response = await fetch(`${API_BASE_URL}/auth/${isRegister ? 'register' : 'login'}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      })
      const payload = await response.json()
      if (!response.ok) {
        throw new Error(payload.detail || 'Authentication failed.')
      }
      const auth = { token: payload.access_token, user: payload.user }
      storeAuth(auth)
      onAuthenticated(auth)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Authentication failed.')
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <main className="auth-page">
      <section className="auth-card">
        <div className="auth-mark">
          {isRegister ? <UserPlus size={22} /> : <LockKeyhole size={22} />}
        </div>
        <p className="eyebrow">Secure workspace</p>
        <h1>{isRegister ? 'Create your CureX account' : 'Sign in to CureX'}</h1>
        <p className="auth-copy">
          Access your private CureX workspace and continue healthcare research sessions securely.
        </p>

        <form className="auth-form" onSubmit={handleSubmit}>
          <label>
            Username or email
            <input
              value={username}
              onChange={(event) => setUsername(event.target.value)}
              autoComplete="username"
              placeholder="clinician@example.com"
              required
            />
          </label>
          <label>
            Password
            <input
              type="password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              autoComplete={isRegister ? 'new-password' : 'current-password'}
              placeholder="At least 8 characters"
              required
            />
          </label>
          {error && <p className="auth-error">{error}</p>}
          <button type="submit" className="primary-landing-action" disabled={isSubmitting}>
            {isSubmitting ? 'Please wait...' : isRegister ? 'Create account' : 'Sign in'}
            <ArrowRight size={18} />
          </button>
        </form>

        <div className="auth-switch">
          <button type="button" onClick={() => onModeChange(isRegister ? 'login' : 'register')}>
            {isRegister ? 'Already have an account? Sign in' : 'Need an account? Create one'}
          </button>
          <button type="button" onClick={onCancel}>
            Back to overview
          </button>
        </div>
      </section>
    </main>
  )
}

function App() {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [isUploadingDocument, setIsUploadingDocument] = useState(false)
  const [uploadStatus, setUploadStatus] = useState(null)
  const [activeSessionId, setActiveSessionId] = useState(null)
  const [selectedCitationId, setSelectedCitationId] = useState(null)
  const [messages, setMessages] = useState([])
  const [sessions, setSessions] = useState([])
  const [metrics, setMetrics] = useState(DEFAULT_METRICS)
  const [citationDetails, setCitationDetails] = useState({})
  const [isAgentTraceVisible, setIsAgentTraceVisible] = useState(false)
  const [auth, setAuth] = useState(() => readStoredAuth())
  const [authMode, setAuthMode] = useState(null)
  const chatEndRef = useRef(null)
  const fileInputRef = useRef(null)

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  useEffect(() => {
    if (!auth?.token) return undefined
    let cancelled = false
    fetch(`${API_BASE_URL}/eval/metrics`, { headers: authHeaders() })
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
  }, [auth?.token])

  const authHeaders = (headers = {}) => ({
    ...headers,
    ...(auth?.token ? { Authorization: `Bearer ${auth.token}` } : {}),
  })

  const createSession = async () => {
    if (!auth?.token) {
      setAuthMode('login')
      throw new Error('Authentication required.')
    }
    let sessionId = `session-${Date.now()}`
    try {
      const response = await fetch(`${API_BASE_URL}/session/new`, {
        method: 'POST',
        headers: authHeaders(),
      })
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
    if (!auth?.token) {
      setAuthMode('login')
      return
    }
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
      const response = await fetch(`${API_BASE_URL}/session/${id}/history`, { headers: authHeaders() })
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
              agentTrace: createAgentTrace({
                status: 'complete',
                citations,
                faithfulnessScore: payload.faithfulness_score ?? null,
              }),
            }
          : message,
      ),
    )
  }

  const sendViaApi = async (sessionId, content, assistantId) => {
    const response = await fetch(`${API_BASE_URL}/chat`, {
      method: 'POST',
      headers: authHeaders({ 'Content-Type': 'application/json' }),
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
    if (!auth?.token) {
      setAuthMode('login')
      return
    }
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
      agentTrace: createAgentTrace({ status: 'pending' }),
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
                agentTrace: createAgentTrace({
                  status: 'complete',
                  error: 'Backend API unavailable',
                }),
              }
            : message,
        ),
      )
    } finally {
      setIsLoading(false)
    }
  }

  const handleDocumentUpload = async (event) => {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file) return
    if (!auth?.token) {
      setAuthMode('login')
      return
    }

    const formData = new FormData()
    formData.append('file', file)
    setIsUploadingDocument(true)
    setUploadStatus({ type: 'pending', message: `Indexing ${file.name}...` })

    try {
      const response = await fetch(`${API_BASE_URL}/documents/upload`, {
        method: 'POST',
        headers: authHeaders(),
        body: formData,
      })
      const payload = await response.json()
      if (!response.ok) {
        throw new Error(payload.detail || 'Document upload failed.')
      }
      setUploadStatus({
        type: 'success',
        message: `${payload.chunks_indexed} chunk${payload.chunks_indexed === 1 ? '' : 's'} indexed from ${file.name}.`,
      })
    } catch (err) {
      setUploadStatus({
        type: 'error',
        message: err instanceof Error ? err.message : 'Document upload failed.',
      })
    } finally {
      setIsUploadingDocument(false)
    }
  }

  const selectedCitation = selectedCitationId ? citationDetails[selectedCitationId] : null

  const handleAuthenticated = (nextAuth) => {
    setAuth(nextAuth)
    setAuthMode(null)
  }

  const handleLogout = () => {
    clearStoredAuth()
    setAuth(null)
    setAuthMode(null)
    setActiveSessionId(null)
    setMessages([])
    setSessions([])
    setCitationDetails({})
    setSelectedCitationId(null)
    setUploadStatus(null)
  }

  if (!auth?.token && authMode) {
    return (
      <AuthPage
        mode={authMode}
        onModeChange={setAuthMode}
        onAuthenticated={handleAuthenticated}
        onCancel={() => setAuthMode(null)}
      />
    )
  }

  if (!activeSessionId) {
    return (
      <LandingPage
        metrics={metrics}
        onStart={handleNewSession}
        onSignIn={() => setAuthMode('login')}
        isAuthenticated={Boolean(auth?.token)}
      />
    )
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
          user={auth?.user}
          onLogout={handleLogout}
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
            <div className="chat-toolbar">
              <div className="document-upload">
                <button
                  type="button"
                  className="document-upload-button"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={isUploadingDocument}
                >
                  {isUploadingDocument ? <UploadCloud className="spin" size={16} /> : <FileText size={16} />}
                  {isUploadingDocument ? 'Indexing document' : 'Add document'}
                </button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".pdf,.txt,.md,application/pdf,text/plain,text/markdown"
                  className="document-upload-input"
                  onChange={handleDocumentUpload}
                />
                {uploadStatus && (
                  <span className={`document-upload-status document-upload-${uploadStatus.type}`}>
                    {uploadStatus.message}
                  </span>
                )}
              </div>
              <button
                type="button"
                className={`agent-toggle ${isAgentTraceVisible ? 'agent-toggle-active' : ''}`}
                onClick={() => setIsAgentTraceVisible((visible) => !visible)}
                aria-pressed={isAgentTraceVisible}
              >
                {isAgentTraceVisible ? <EyeOff size={16} /> : <Eye size={16} />}
                {isAgentTraceVisible ? 'Hide agent flow' : 'Show agent flow'}
              </button>
            </div>
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
                    showAgentTrace={isAgentTraceVisible}
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
