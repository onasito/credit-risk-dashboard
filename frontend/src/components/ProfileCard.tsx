import { useEffect, useState } from 'react'
import './ProfileCard.css'

const API_BASE = 'http://localhost:8000'

interface Profile {
  SK_ID_CURR: number,
  age_years: number | null
  gender: string | null,
  income_total: number | null,
  employment_years: number | null,
  education: string | null,
  family_status: string,
  occupation: string | null,
  housing_type: string | null,
  own_car: boolean,
  own_realty: boolean,
  children: number | null,
  credit_amount: number | null,
  annuity: number | null
  goods_price: number | null
  ext_source_avg: number | null
  bureau_active_loans: number | null
  bureau_total_debt: number | null
  bureau_max_days_overdue: number | null
  prev_app_refused_count: number | null
  prev_app_approved_count: number | null
}

interface ProfileCardProps {
  currentId: number | undefined
}

function formatCurrency(value: number | null) {
  if (value === null) return '—'
  return `$${value.toLocaleString('en-US', { maximumFractionDigits: 0 })}`
}

export default function Profile({currentId}: ProfileCardProps) {
  const [profile, setProfile] = useState<Profile | null>(null)
  const [loadingProfile, setLoadingProfile] = useState(false)
  const [profileError, setProfileError] = useState<string | null>(null)
  

  useEffect(() => {
    if (currentId === undefined) {
      setProfile(null)
      return
    }
    const controller = new AbortController()
    setLoadingProfile(true)
    setProfileError(null)

    fetch(`${API_BASE}/applicants/${currentId}/profile`, {
      signal: controller.signal
    })
      .then((res) => {
        if (!res.ok) throw new Error(`${res.status}`)
        return res.json()
      })
      .then((data: Profile) => setProfile(data))
      .catch((err) => {
        if (controller.signal.aborted) return
        setProfile(null)
        setProfileError(
          err instanceof TypeError
            ? 'Underwriting service unreachable. Start it with python ml/app.py, then retry.'
            : `No profile data found for applicant #${currentId}.`
        )
      })
      .finally(() => {
        if (!controller.signal.aborted) return setLoadingProfile(false)
      })
    return () => controller.abort()

  }, [currentId])

  if (profileError){
    return (
      <div className='profile-card profile-card--offline'>
        <span className="profile-card__eyebrow">Applicant profile</span>
        <p className="profile-card__offline-body">{profileError}</p>
      </div>
    ) 
  }

  if (loadingProfile || !profile) {
  return (
    <div className="profile-card">
      <span className="profile-card__eyebrow">Applicant profile</span>
      <p className="profile-card__offline-body">Loading…</p>
    </div>
  )
}

const p = profile


  return (
    <div className="profile-card">
      <div className="profile-card__header">
        <span className="profile-card__eyebrow">Applicant profile</span>
        <span className="profile-card__ref"># {p.SK_ID_CURR}</span>
      </div>

      <div className="profile-card__section">
        <span className="profile-card__section-title">Profile</span>
        <div className="profile-card__grid">
          <div className="profile-card__stat">
            <span className="profile-card__label">Age</span>
            <span className="profile-card__value">{p.age_years ?? '—'}</span>
          </div>
          <div className="profile-card__stat">
            <span className="profile-card__label">Gender</span>
            <span className="profile-card__value">{p.gender ?? '—'}</span>
          </div>
          <div className="profile-card__stat">
            <span className="profile-card__label">Income</span>
            <span className="profile-card__value">{formatCurrency(p.income_total)}</span>
          </div>
          <div className="profile-card__stat">
            <span className="profile-card__label">Employment</span>
            <span className="profile-card__value">
              {p.employment_years !== null ? `${p.employment_years} yrs` : '—'}
            </span>
          </div>
          <div className="profile-card__stat">
            <span className="profile-card__label">Education</span>
            <span className="profile-card__value">{p.education ?? '—'}</span>
          </div>
          <div className="profile-card__stat">
            <span className="profile-card__label">Family status</span>
            <span className="profile-card__value">{p.family_status ?? '—'}</span>
          </div>
          <div className="profile-card__stat">
            <span className="profile-card__label">Occupation</span>
            <span className="profile-card__value">{p.occupation ?? '—'}</span>
          </div>
          <div className="profile-card__stat">
            <span className="profile-card__label">Housing</span>
            <span className="profile-card__value">{p.housing_type ?? '—'}</span>
          </div>
        </div>
      </div>

      <div className="profile-card__section">
        <span className="profile-card__section-title">Loan requested</span>
        <div className="profile-card__grid">
          <div className="profile-card__stat">
            <span className="profile-card__label">Credit amount</span>
            <span className="profile-card__value">{formatCurrency(p.credit_amount)}</span>
          </div>
          <div className="profile-card__stat">
            <span className="profile-card__label">Annuity</span>
            <span className="profile-card__value">{formatCurrency(p.annuity)}</span>
          </div>
          <div className="profile-card__stat">
            <span className="profile-card__label">Goods price</span>
            <span className="profile-card__value">{formatCurrency(p.goods_price)}</span>
          </div>
        </div>
      </div>

      <div className="profile-card__section">
        <span className="profile-card__section-title">External credit score</span>
        <div className="profile-card__grid">
          <div className="profile-card__stat profile-card__stat--wide">
            <span className="profile-card__label">EXT_SOURCE avg</span>
            <span className="profile-card__value">
              {p.ext_source_avg !== null ? p.ext_source_avg.toFixed(3) : '—'}
            </span>
          </div>
        </div>
      </div>

      <div className="profile-card__section">
        <span className="profile-card__section-title">Bureau history</span>
        <div className="profile-card__grid">
          <div className="profile-card__stat">
            <span className="profile-card__label">Active loans</span>
            <span className="profile-card__value">{p.bureau_active_loans ?? '—'}</span>
          </div>
          <div className="profile-card__stat">
            <span className="profile-card__label">Total debt</span>
            <span className="profile-card__value">{formatCurrency(p.bureau_total_debt)}</span>
          </div>
          <div className="profile-card__stat">
            <span className="profile-card__label">Max days overdue</span>
            <span className="profile-card__value">{p.bureau_max_days_overdue ?? '—'}</span>
          </div>
          <div className="profile-card__stat">
            <span className="profile-card__label">Prior apps refused</span>
            <span className="profile-card__value">{p.prev_app_refused_count ?? '—'}</span>
          </div>
        </div>
      </div>

      <div className="profile-card__footer">
        <span className="profile-card__caption">Underlying application &amp; bureau data</span>
      </div>
    </div>
  )
}

  

