import { useEffect, useState } from 'react'
import './RiskCard.css'

const API_BASE = 'http://localhost:8000'

interface Prediction {
  SK_ID_CURR: number
  default_probability: number
  prediction: number
}

interface RiskTier {
  label: string
  varName: string
}

function riskTier(probability: number): RiskTier {
  if (probability >= 0.5) return { label: 'High risk', varName: '--risk-high' }
  if (probability >= 0.15) return { label: 'Moderate risk', varName: '--risk-moderate' }
  return { label: 'Low risk', varName: '--risk-low' }
}

interface RiskCardProps {
  currentId: number | undefined
  loadingIds: boolean
  idsError: string | null
  retry: () => void
  canGoPrev: boolean
  canGoNext: boolean
  goPrev: () => void
  goNext: () => void
}

export default function RiskCard({
  currentId,
  loadingIds,
  idsError,
  retry,
  canGoPrev,
  canGoNext,
  goPrev,
  goNext,
}: RiskCardProps) {
  const [prediction, setPrediction] = useState<Prediction | null>(null)
  const [loadingPrediction, setLoadingPrediction] = useState(false)
  const [predictionError, setPredictionError] = useState<string | null>(null)

  useEffect(() => {
    if (currentId === undefined) {
      setPrediction(null)
      return
    }
    const controller = new AbortController()
    setLoadingPrediction(true)
    setPredictionError(null)
    fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ SK_ID_CURR: currentId }),
      signal: controller.signal,
    })
      .then((res) => {
        if (!res.ok) throw new Error(`${res.status}`)
        return res.json()
      })
      .then((data: Prediction) => setPrediction(data))
      .catch((err) => {
        if (controller.signal.aborted) return
        setPrediction(null)
        setPredictionError(
          err instanceof TypeError
            ? 'Underwriting service unreachable.'
            : `No file found for applicant #${currentId}.`,
        )
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoadingPrediction(false)
      })
    return () => controller.abort()
  }, [currentId])

  if (idsError) {
    return (
      <div className="risk-card risk-card--offline">
        <span className="risk-card__eyebrow">Underwriting file</span>
        <h1 className="risk-card__offline-title">Service offline</h1>
        <p className="risk-card__offline-body">{idsError}</p>
        <button type="button" className="risk-card__button" onClick={retry}>
          Retry
        </button>
      </div>
    )
  }

  const tier = prediction ? riskTier(prediction.default_probability) : null

  return (
    <div className="risk-card">
      <div className="risk-card__header">
        <span className="risk-card__eyebrow">Underwriting file</span>
        <span className="risk-card__ref">
          {loadingIds ? '# —' : currentId !== undefined ? `# ${currentId}` : '# —'}
        </span>
      </div>

      <div className="risk-card__body">
        {predictionError ? (
          <p className="risk-card__inline-error">{predictionError}</p>
        ) : (
          <>
            <div className={`risk-card__figure ${loadingPrediction ? 'is-loading' : ''}`}>
              {prediction ? (prediction.default_probability * 100).toFixed(1) : '—'}
              <span className="risk-card__percent">%</span>
            </div>
            <span className="risk-card__figure-label">Probability of default</span>
          </>
        )}
      </div>

      {tier && (
        <div className="risk-card__stamp" style={{ ['--stamp-color' as string]: `var(${tier.varName})` }}>
          {tier.label}
        </div>
      )}

      <div className="risk-card__footer">
        <button
          type="button"
          className="risk-card__nav"
          onClick={goPrev}
          disabled={!canGoPrev || loadingIds}
          aria-label="Previous applicant"
        >
          ‹ Prev
        </button>
        <span className="risk-card__caption">XGBoost · Home Credit Default Risk</span>
        <button
          type="button"
          className="risk-card__nav"
          onClick={goNext}
          disabled={!canGoNext || loadingIds}
          aria-label="Next applicant"
        >
          Next ›
        </button>
      </div>
    </div>
  )
}
