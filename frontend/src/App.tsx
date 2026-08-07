import { useCallback, useEffect, useState } from 'react'
import RiskCard from './components/RiskCard'
import ProfileCard from './components/ProfileCard'
import { API_BASE } from './config'
import './App.css'

const PAGE_SIZE = 25

function App() {
  const [ids, setIds] = useState<number[]>([])
  const [offset, setOffset] = useState(0)
  const [index, setIndex] = useState(0)
  const [loadingIds, setLoadingIds] = useState(true)
  const [idsError, setIdsError] = useState<string | null>(null)
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

  return (
    <>
      <div className="app-background" aria-hidden="true" />
      <div className="dashboard-layout">
        <RiskCard
          currentId={currentId}
          loadingIds={loadingIds}
          idsError={idsError}
          retry={retry}
          canGoPrev={canGoPrev}
          canGoNext={canGoNext}
          goPrev={goPrev}
          goNext={goNext}
        />
        {!idsError && <ProfileCard currentId={currentId}/>}
      </div>
    </>
  )
}

export default App
