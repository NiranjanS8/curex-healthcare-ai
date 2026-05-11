import { useRef, useState } from 'react'
import { Loader2, Send } from 'lucide-react'

export function ChatInput({ onSend, isLoading = false, disabled = false }) {
  const [message, setMessage] = useState('')
  const textareaRef = useRef(null)

  const handleSubmit = () => {
    if (!message.trim() || isLoading || disabled) return
    onSend(message.trim())
    setMessage('')
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }

  const handleChange = (event) => {
    setMessage(event.target.value)
    event.target.style.height = 'auto'
    event.target.style.height = `${Math.min(event.target.scrollHeight, 200)}px`
  }

  const handleKeyDown = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      handleSubmit()
    }
  }

  return (
    <div className="chat-input-shell">
      <div className="chat-input-wrap">
        <div className="chat-input-row">
          <textarea
            ref={textareaRef}
            value={message}
            onChange={handleChange}
            onKeyDown={handleKeyDown}
            disabled={disabled || isLoading}
            placeholder="Ask a healthcare question..."
            rows={1}
            className="chat-textarea"
          />
          <button
            type="button"
            onClick={handleSubmit}
            disabled={!message.trim() || isLoading || disabled}
            className="send-button"
            aria-label="Send message"
          >
            {isLoading ? <Loader2 size={18} className="spin" /> : <Send size={18} />}
          </button>
        </div>
        {isLoading && (
          <div className="typing-indicator">
            <span className="typing-dots">
              <i />
              <i />
              <i />
            </span>
            <span>Analyzing medical sources...</span>
          </div>
        )}
      </div>
    </div>
  )
}
