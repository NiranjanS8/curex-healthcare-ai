import { AlertTriangle, Bot, ClipboardCheck, Flag, ThumbsUp, User } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import { CitationChip } from './CitationChip'

function getFaithfulnessClass(score) {
  if (score >= 0.8) return 'faithfulness-high'
  if (score >= 0.6) return 'faithfulness-medium'
  return 'faithfulness-low'
}

function AgentTrace({ steps }) {
  if (!steps?.length) return null

  return (
    <div className="agent-trace" aria-label="Visible agent workflow">
      <div className="agent-trace-header">
        <span>Agent flow</span>
        <small>Visible workflow, not hidden reasoning</small>
      </div>
      <ol>
        {steps.map((step) => (
          <li key={step.label} className={`agent-step agent-step-${step.status}`}>
            <span className="agent-step-dot" />
            <div>
              <strong>{step.label}</strong>
              <small>{step.detail}</small>
            </div>
          </li>
        ))}
      </ol>
    </div>
  )
}

const REVIEW_OPTIONS = [
  { value: 'helpful', label: 'Helpful', icon: ThumbsUp },
  { value: 'unsupported', label: 'Unsupported', icon: Flag },
  { value: 'unsafe', label: 'Unsafe', icon: AlertTriangle },
  { value: 'needs_review', label: 'Needs review', icon: ClipboardCheck },
]

export function ChatMessage({ message, onCitationClick, onReview, showAgentTrace = false }) {
  if (message.role === 'user') {
    return (
      <article className="message-row user-row">
        <div className="message-group user-group">
          <div className="user-bubble">
            <p>{message.content}</p>
          </div>
          <div className="user-avatar">
            <User size={16} />
          </div>
        </div>
      </article>
    )
  }

  return (
    <article className="message-row assistant-row">
      <div className="message-group assistant-group">
        <div className="assistant-avatar">
          <Bot size={16} />
        </div>
        <div className="assistant-card">
          <div className="assistant-copy">
            {message.content ? <ReactMarkdown>{message.content}</ReactMarkdown> : <p />}
          </div>

          {message.citations && message.citations.length > 0 && (
            <div className="citation-chip-list">
              {message.citations.map((citation) => (
                <CitationChip
                  key={citation.id}
                  citation={citation}
                  onClick={() => onCitationClick(citation.id)}
                />
              ))}
            </div>
          )}

          {message.faithfulnessScore !== undefined && (
            <div className="faithfulness-row">
              <span>Faithfulness:</span>
              <strong className={getFaithfulnessClass(message.faithfulnessScore)}>
                {(message.faithfulnessScore * 100).toFixed(0)}%
              </strong>
            </div>
          )}

          {showAgentTrace && <AgentTrace steps={message.agentTrace} />}

          {message.content && (
            <div className="review-row" aria-label="Review assistant response">
              {REVIEW_OPTIONS.map((option) => {
                const Icon = option.icon
                const isSelected = message.reviewRating === option.value
                return (
                  <button
                    key={option.value}
                    type="button"
                    className={`review-button ${isSelected ? 'review-button-active' : ''}`}
                    onClick={() => onReview?.(message.id, option.value)}
                    disabled={message.reviewPending}
                    aria-pressed={isSelected}
                  >
                    <Icon size={14} />
                    {option.label}
                  </button>
                )
              })}
              {message.reviewError && <span className="review-error">{message.reviewError}</span>}
            </div>
          )}
        </div>
      </div>
    </article>
  )
}
