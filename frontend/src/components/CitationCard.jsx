import { useState } from 'react'
import { Calendar, ChevronDown, ChevronUp, Tag } from 'lucide-react'

function getDocTypeClass(type) {
  if (type === 'Clinical Guideline') return 'doc-clinical'
  if (type === 'Research Paper') return 'doc-research'
  if (type === 'Review Article') return 'doc-review'
  if (type === 'Case Study') return 'doc-case'
  return 'doc-default'
}

export function CitationCard({ citation }) {
  const [isExpanded, setIsExpanded] = useState(false)

  return (
    <div className="citation-card">
      <button type="button" onClick={() => setIsExpanded(!isExpanded)} className="citation-card-head">
        <span className="citation-card-title">
          <strong>{citation.title}</strong>
          <span className="citation-card-meta">
            <small className={`doc-badge ${getDocTypeClass(citation.docType)}`}>{citation.docType}</small>
            <small>
              <Calendar size={12} />
              {citation.date}
            </small>
            <small>
              <Tag size={12} />
              {citation.specialty}
            </small>
          </span>
        </span>
        {isExpanded ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
      </button>

      {isExpanded && (
        <div className="citation-card-body">
          <section>
            <h5>Excerpt</h5>
            <p>{citation.excerpt}</p>
          </section>

          <section>
            <h5>Metadata</h5>
            <dl>
              {citation.metadata?.authors && (
                <div>
                  <dt>Authors:</dt>
                  <dd>{citation.metadata.authors}</dd>
                </div>
              )}
              {citation.metadata?.journal && (
                <div>
                  <dt>Journal:</dt>
                  <dd>{citation.metadata.journal}</dd>
                </div>
              )}
              {citation.metadata?.doi && (
                <div>
                  <dt>DOI:</dt>
                  <dd className="mono">{citation.metadata.doi}</dd>
                </div>
              )}
              <div>
                <dt>Source:</dt>
                <dd>{citation.source}</dd>
              </div>
            </dl>
          </section>
        </div>
      )}
    </div>
  )
}
