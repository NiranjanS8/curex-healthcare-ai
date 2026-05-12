import { Activity, BarChart3, LogOut, MessageSquare, Plus } from 'lucide-react'

function getMetricClass(level) {
  if (level === 'high') return 'metric-high'
  if (level === 'medium') return 'metric-medium'
  if (level === 'low') return 'metric-low'
  return 'metric-low'
}

export function Sidebar({
  sessions,
  activeSessionId,
  metrics,
  onNewSession,
  onSelectSession,
  user,
  onLogout,
  activeView,
  onShowDashboard,
  onShowChat,
}) {
  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <h1>CureX</h1>
        <button type="button" className="new-session-button" onClick={onNewSession}>
          <Plus size={18} />
          <span>New Session</span>
        </button>
        {user && (
          <div className="sidebar-user">
            <span>
              Signed in as <strong>{user.username}</strong>
            </span>
            <button type="button" onClick={onLogout} aria-label="Log out">
              <LogOut size={15} />
            </button>
          </div>
        )}
      </div>

      <div className="session-list-wrap">
        <div className="session-list-inner">
          <h3>Workspace</h3>
          <div className="workspace-nav">
            <button
              type="button"
              className={`workspace-nav-button ${activeView === 'chat' ? 'workspace-nav-active' : ''}`}
              onClick={onShowChat}
            >
              <MessageSquare size={16} />
              Chat
            </button>
            <button
              type="button"
              className={`workspace-nav-button ${activeView === 'dashboard' ? 'workspace-nav-active' : ''}`}
              onClick={onShowDashboard}
            >
              <BarChart3 size={16} />
              Eval dashboard
            </button>
          </div>
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
            {sessions.length === 0 && (
              <p className="empty-session-copy">Sessions will appear after you start a chat.</p>
            )}
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
                {metric.value === null ? '--' : `${(metric.value * 100).toFixed(0)}%`}
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
