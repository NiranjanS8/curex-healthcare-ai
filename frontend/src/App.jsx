import { useEffect, useRef, useState } from 'react'
import { AlertCircle, Menu, X } from 'lucide-react'
import { ChatInput } from './components/ChatInput'
import { ChatMessage } from './components/ChatMessage'
import { CitationCard } from './components/CitationCard'
import { Sidebar } from './components/Sidebar'
import './App.css'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const INITIAL_MESSAGES = [
  {
    id: '1',
    role: 'user',
    content: 'What are the current guidelines for managing type 2 diabetes in adults?',
    timestamp: '10:23 AM',
  },
  {
    id: '2',
    role: 'assistant',
    content:
      'Based on the latest clinical guidelines, the management of type 2 diabetes in adults follows a comprehensive approach:\n\n1. Lifestyle Modifications: Diet and exercise remain the cornerstone of treatment. A Mediterranean-style diet with emphasis on whole grains, vegetables, and lean proteins is recommended.\n\n2. Metformin: First-line pharmacological therapy for most patients, unless contraindicated. Starting dose is typically 500mg once or twice daily, titrated based on tolerance and glycemic control.\n\n3. Additional Medications: If HbA1c targets are not met after 3 months, consider adding:\n   - GLP-1 receptor agonists for patients with cardiovascular disease\n   - SGLT2 inhibitors for those with heart failure or chronic kidney disease\n   - DPP-4 inhibitors as alternatives\n\n4. Monitoring: Regular HbA1c testing every 3-6 months, with target typically <7% for most adults, though individualized based on patient factors.\n\nThese recommendations are based on evidence from multiple large-scale trials and expert consensus.',
    citations: [
      {
        id: 'cite-1',
        source: 'Clinical Guideline',
        docType: 'Clinical Guideline',
        chunkId: 'abc123',
      },
      {
        id: 'cite-2',
        source: 'Research Paper',
        docType: 'Research Paper',
        chunkId: 'def456',
      },
    ],
    faithfulnessScore: 0.92,
    timestamp: '10:23 AM',
  },
]

const INITIAL_SESSIONS = [
  { id: 'session-1', title: 'Type 2 Diabetes Management', timestamp: 'Today, 10:23 AM' },
  { id: 'session-2', title: 'Hypertension Treatment Options', timestamp: 'Yesterday, 3:45 PM' },
  { id: 'session-3', title: 'Asthma Guidelines Update', timestamp: 'May 9, 2:15 PM' },
  { id: 'session-4', title: 'Antibiotic Resistance Patterns', timestamp: 'May 8, 11:30 AM' },
]

const DEFAULT_METRICS = [
  { name: 'Faithfulness', value: 0.92, level: 'high' },
  { name: 'Answer Relevancy', value: 0.88, level: 'high' },
  { name: 'Context Precision', value: 0.75, level: 'medium' },
  { name: 'Context Recall', value: 0.81, level: 'high' },
]

const INITIAL_CITATIONS = {
  'cite-1': {
    id: 'cite-1',
    title: 'Standards of Medical Care in Diabetes 2026',
    docType: 'Clinical Guideline',
    date: 'Jan 2026',
    specialty: 'Endocrinology',
    excerpt:
      'Metformin, if not contraindicated and if tolerated, is the preferred initial pharmacological agent for type 2 diabetes. For patients with established atherosclerotic cardiovascular disease, heart failure, or chronic kidney disease, early addition of agents proven to reduce major adverse cardiovascular events or CKD progression should be considered independent of baseline HbA1c or individualized HbA1c target.',
    source: 'American Diabetes Association',
    metadata: {
      authors: 'ADA Professional Practice Committee',
      journal: 'Diabetes Care, Volume 49, Supplement 1',
      doi: '10.2337/dc26-S001',
    },
  },
  'cite-2': {
    id: 'cite-2',
    title: 'Efficacy and Safety of GLP-1 Receptor Agonists in Type 2 Diabetes',
    docType: 'Research Paper',
    date: 'Dec 2025',
    specialty: 'Endocrinology',
    excerpt:
      'GLP-1 receptor agonists demonstrated significant reductions in HbA1c and body weight compared to placebo. Cardiovascular outcomes were favorable, with reductions in major adverse cardiovascular events.',
    source: 'The Lancet',
    metadata: {
      authors: 'Chen M, Wang Y, Liu X, et al.',
      journal: 'The Lancet Diabetes & Endocrinology',
      doi: '10.1016/S2213-8587(25)00234-5',
    },
  },
}

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

function App() {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [activeSessionId, setActiveSessionId] = useState('session-1')
  const [selectedCitationId, setSelectedCitationId] = useState(null)
  const [messages, setMessages] = useState(INITIAL_MESSAGES)
  const [sessions, setSessions] = useState(INITIAL_SESSIONS)
  const [metrics, setMetrics] = useState(DEFAULT_METRICS)
  const [citationDetails, setCitationDetails] = useState(INITIAL_CITATIONS)
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
          const value = payload.metrics[key] ?? metric.value
          return { ...metric, value, level: metricLevel(value) }
        })
        setMetrics(nextMetrics)
      })
      .catch(() => {})
    return () => {
      cancelled = true
    }
  }, [])

  const handleNewSession = async () => {
    let sessionId = `session-new-${Date.now()}`
    try {
      const response = await fetch(`${API_BASE_URL}/session/new`, { method: 'POST' })
      if (response.ok) {
        const payload = await response.json()
        sessionId = payload.session_id || sessionId
      }
    } catch {
      // Local design preview still works when the API is not running.
    }
    setMessages([])
    setActiveSessionId(sessionId)
    setSelectedCitationId(null)
    setSessions((current) => [
      { id: sessionId, title: 'New Healthcare Conversation', timestamp: 'Just now' },
      ...current,
    ])
    setIsMobileMenuOpen(false)
  }

  const handleSelectSession = (id) => {
    setActiveSessionId(id)
    setIsMobileMenuOpen(false)
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

  const sendViaApi = async (content, assistantId) => {
    const response = await fetch(`${API_BASE_URL}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: activeSessionId, message: content }),
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

  const sendFallbackResponse = async (assistantId) => {
    await new Promise((resolve) => setTimeout(resolve, 700))
    setMessages((current) =>
      current.map((message) =>
        message.id === assistantId
          ? {
              ...message,
              content:
                "Thank you for your question. I'm analyzing the latest medical literature and clinical guidelines to provide evidence-based information. This is a local preview response because the backend API is not currently reachable.",
              citations: [
                {
                  id: 'cite-1',
                  source: 'Clinical Guideline',
                  docType: 'Clinical Guideline',
                  chunkId: 'abc123',
                },
              ],
              faithfulnessScore: 0.87,
            }
          : message,
      ),
    )
  }

  const handleSendMessage = async (content) => {
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
      await sendViaApi(content, assistantId)
    } catch {
      await sendFallbackResponse(assistantId)
    } finally {
      setIsLoading(false)
    }
  }

  const selectedCitation = selectedCitationId ? citationDetails[selectedCitationId] : null

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
