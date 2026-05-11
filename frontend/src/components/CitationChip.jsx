import { FileText } from 'lucide-react'

export function CitationChip({ citation, onClick }) {
  return (
    <button type="button" onClick={onClick} className="citation-chip">
      <FileText size={12} />
      <span>Source: {citation.source}</span>
      <small>{citation.chunkId}</small>
    </button>
  )
}
