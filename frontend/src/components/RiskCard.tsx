import { useCallback, useEffect, useState } from 'react'
import './RiskCard.css'

const API_BASE = 'http://localhost:8000'
const PAGE_SIZE = 25

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

export default function RiskCard() {
  const [ids, setIds] = useState<number[]>([])
  const [offset, setOffset] = useState(0)
  const [index, setIndex] = useState(0)
  const [prediction, setPrediction] = useState<Prediction | null>(null)
  const [loadingIds, setLoadingIds] = useState(true)
  const [loadingPrediction, setLoadingPrediction] = useState(false)
  const [idsError, setIdsError] = useState<string | null>(null)
  const [predictionError, setPredictionError] = useState<string | null>(null)
  const [retryTick, setRetryTick] = useState(0)

  useEffect(() => {
    const controller = new AbortController()
    setLoadingIds(true)
    setIdsError(null)
    fetch(`${API_BASE}/applicants?limit=${PAGE_SIZE}&offset=${offset}`, {
      signal: controller.signal,
    })
      .then((res) => {
        if (!res.ok) throw new Error(`${res.status}`)
        return res.json()
      })
      .then((data: { applicant_ids: number[] }) => {
        setIds(data.applicant_ids)
        setIndex(0)
      })
      .catch((err) => {
        if (controller.signal.aborted) return
        setIdsError(
          err instanceof TypeError
            ? 'Underwriting service unreachable. Start it with python ml/app.py, then retry.'
            : 'The applicant file list could not be loaded.',
        )
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoadingIds(false)
      })
    return () => controller.abort()
  }, [offset, retryTick])

  const currentId = ids[index]

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

  const retry = useCallback(() => setRetryTick((n) => n + 1), [])

  const canGoPrev = index > 0 || offset > 0
  const canGoNext = index < ids.length - 1 || ids.length === PAGE_SIZE

  const goPrev = () => {
    if (index > 0) {
      setIndex((i) => i - 1)
    } else if (offset > 0) {
      setOffset((o) => Math.max(0, o - PAGE_SIZE))
      setIndex(PAGE_SIZE - 1)
    }
  }

  const goNext = () => {
    if (index < ids.length - 1) {
      setIndex((i) => i + 1)
    } else if (ids.length === PAGE_SIZE) {
      setOffset((o) => o + PAGE_SIZE)
    }
  }

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
