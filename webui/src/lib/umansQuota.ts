/** Umans /v1/usage fields attached to credential stats as upstream_quota. */

export interface UmansUpstreamQuota {
  plan?: string | null
  requests_soft_limit?: number
  requests_hard_cap?: number
  requests_used?: number
  requests_remaining?: number
  in_burst_band?: boolean
  deprioritized?: boolean
  priority_low?: boolean
  boxed_until?: string | null
  boxed_until_ts?: number | null
  throttle_reason?: string | null
  concurrency_limit?: number
  concurrency_hard_cap?: number
  concurrent_sessions?: number
  window_resets_at?: string | null
  status?: string
  error?: string | null
}

export function formatUmansRequestQuotaLine(uq: UmansUpstreamQuota): string {
  const used = uq.requests_used ?? 0
  const soft = uq.requests_soft_limit ?? 0
  const hard = uq.requests_hard_cap ?? 0
  if (soft > 0 && hard > soft) {
    return `${used} / ${soft} plan (+${hard - soft} burst headroom → ${hard} hard cap)`
  }
  if (soft > 0) {
    return `${used} / ${soft}`
  }
  return `${used}`
}

export function umansDeprioritizedTooltip(uq: UmansUpstreamQuota): string {
  const parts = [
    "Umans deprioritized this account (usage.priority.low). Requests may be slower until normal priority returns.",
  ]
  if (uq.throttle_reason) parts.push(`Reason: ${uq.throttle_reason}`)
  if (uq.boxed_until) parts.push(`Until: ${uq.boxed_until}`)
  return parts.join(" ")
}