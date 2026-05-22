import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatNumber(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`
  return n.toLocaleString()
}

export function formatUptime(seconds: number): string {
  const d = Math.floor(seconds / 86400)
  const h = Math.floor((seconds % 86400) / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const parts: string[] = []
  if (d > 0) parts.push(`${d}d`)
  if (h > 0) parts.push(`${h}h`)
  parts.push(`${m}m`)
  return parts.join(" ")
}

export function formatCost(cost: number | null | undefined): string {
  if (cost == null) return "$0.00"
  if (cost < 0.01) return `$${cost.toFixed(4)}`
  return `$${cost.toFixed(2)}`
}

export function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`
  return `${(ms / 1000).toFixed(1)}s`
}

export function timeAgo(timestamp: string | number): string {
  let ts = timestamp
  if (typeof ts === "string" && !ts.endsWith("Z") && !ts.includes("+")) {
    ts = ts + "Z"
  }
  const seconds = Math.floor((Date.now() - new Date(ts).getTime()) / 1000)
  if (seconds < 0) return "just now"
  if (seconds < 60) return `${seconds}s ago`
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`
  return `${Math.floor(seconds / 86400)}d ago`
}

export function getStatusColor(status: string): string {
  switch (status) {
    case "active": return "text-success"
    case "cooldown": return "text-warning"
    case "exhausted": return "text-destructive"
    default: return "text-muted-foreground"
  }
}

export function getQuotaColor(pct: number): string {
  if (pct <= 10) return "bg-destructive"
  if (pct <= 30) return "bg-warning"
  return "bg-success"
}

export function formatWindowLabel(label: string): string {
  const hourMatch = label.match(/^(\d+)h$/)
  if (hourMatch) {
    const hours = parseInt(hourMatch[1])
    if (hours >= 24) {
      const days = Math.floor(hours / 24)
      const rem = hours % 24
      return rem > 0 ? `${days}d ${rem}h` : `${days}d`
    }
  }
  return label
}
