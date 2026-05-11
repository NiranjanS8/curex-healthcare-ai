import { Activity, MessageSquare, Plus } from 'lucide-react'

function getMetricClass(level) {
  if (level === 'high') return 'metric-high'
  if (level === 'medium') return 'metric-medium'
  return 'metric-low'
}

export function Sidebar({ sessions, activeSessionId, metrics, onNewSession, onSelectSession }) {
  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <h1>CureX</h1>
        <button type="button" className="new-session-button" onClick={onNewSession}>
          <Plus size={18} />
          <span>New Session</span>
        </button>
      </div>

      <div className="session-list-wrap">
        <div className="session-list-inner">
          <h3>Recent Sessions</h3>
          <div className="session-list">
            {sessions.map((session) => (
              <button
                type="button"
                key={session.id}
                onClick={() => onSelectSession(session.id)}
                className={`session-button ${activeSessionId === session.id ? 'session-active' : ''}`}
              >
                <MessageSquare size={16} />
                <span>
                  <strong>{session.title}</strong>
                  <small>{session.timestamp}</small>
                </span>
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="sidebar-metrics">
        <div className="metrics-heading">
          <Activity size={14} />
          <h3>Evaluation Metrics</h3>
        </div>
        <div className="metrics-list">
          {metrics.map((metric) => (
            <div key={metric.name} className="metric-row">
              <span>{metric.name}</span>
              <strong className={getMetricClass(metric.level)}>
                {(metric.value * 100).toFixed(0)}%
              </strong>
            </div>
          ))}
        </div>

        <div className="system-status">
          <span />
          <small>System Operational</small>
        </div>
      </div>
    </aside>
  )
}
