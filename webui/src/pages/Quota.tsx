import { useState, useCallback, useMemo } from "react"
import { RefreshCw, ChevronDown, ChevronRight, ArrowLeft, ArrowUpDown } from "lucide-react"
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Table, TableHeader, TableBody, TableRow, TableHead, TableCell } from "@/components/ui/table"
import { usePolling } from "@/hooks/usePolling"
import {
  getQuotaStats,
  reloadQuotaStats,
  forceRefreshQuota,
  type QuotaStatsResponse,
  type ProviderStats,
  type CredentialStats,
  type QuotaGroup,
  type WindowInfo,
  type ModelUsageEntry,
} from "@/api/quota"
import { formatNumber, formatCost, getQuotaColor, formatWindowLabel } from "@/lib/utils"

export function Quota() {
  const [viewMode, setViewMode] = useState<"current" | "global">("current")
  const [selectedProvider, setSelectedProvider] = useState<string | null>(null)
  const [refreshing, setRefreshing] = useState(false)

  const { data, loading, refresh } = usePolling<QuotaStatsResponse>({
    fetcher: () => getQuotaStats(),
    interval: 10000,
  })

  const handleReload = useCallback(async (scope: "all" | "provider", provider?: string) => {
    setRefreshing(true)
    try {
      await reloadQuotaStats(scope, provider)
      await refresh()
    } finally {
      setRefreshing(false)
    }
  }, [refresh])

  const handleForceRefresh = useCallback(async (scope: "all" | "provider" | "credential", provider?: string, credential?: string) => {
    setRefreshing(true)
    try {
      await forceRefreshQuota(scope, provider, credential)
      await refresh()
    } finally {
      setRefreshing(false)
    }
  }, [refresh])

  const [sortCol, setSortCol] = useState<string>("provider")
  const [sortDir, setSortDir] = useState<"asc" | "desc">("asc")

  const toggleSort = useCallback((col: string) => {
    if (sortCol === col) {
      setSortDir(d => d === "asc" ? "desc" : "asc")
    } else {
      setSortCol(col)
      setSortDir(col === "provider" ? "asc" : "desc")
    }
  }, [sortCol])

  const providerEntries = useMemo(() => {
    const raw = data?.providers ? Object.entries(data.providers) : []
    const hasQuota = (p: ProviderStats) =>
      p.quota_groups && Object.keys(p.quota_groups).length > 0
    return raw.sort(([aName, a], [bName, b]) => {
      const aQ = hasQuota(a) ? 0 : 1
      const bQ = hasQuota(b) ? 0 : 1
      if (aQ !== bQ) return aQ - bQ
      const getStat = (p: ProviderStats) => {
        const s = viewMode === "current" && p.current_period ? p.current_period : null
        return {
          requests: s?.total_requests ?? p.total_requests ?? 0,
          tokensIn: (s?.tokens?.input_uncached ?? p.tokens?.input_uncached ?? 0) + (s?.tokens?.input_cached ?? p.tokens?.input_cached ?? 0),
          tokensOut: s?.tokens?.output ?? p.tokens?.output ?? 0,
          cost: s?.approx_cost ?? p.approx_cost ?? 0,
        }
      }
      const sa = getStat(a), sb = getStat(b)
      let cmp = 0
      switch (sortCol) {
        case "provider": cmp = aName.localeCompare(bName); break
        case "credentials": cmp = a.credential_count - b.credential_count; break
        case "requests": cmp = sa.requests - sb.requests; break
        case "tokens_in": cmp = sa.tokensIn - sb.tokensIn; break
        case "tokens_out": cmp = sa.tokensOut - sb.tokensOut; break
        case "cost": cmp = sa.cost - sb.cost; break
        default: cmp = 0
      }
      return sortDir === "asc" ? cmp : -cmp
    })
  }, [data, sortCol, sortDir, viewMode])

  if (selectedProvider && data?.providers) {
    const provider = data.providers[selectedProvider]
    if (provider) {
      return (
        <ProviderDetail
          providerName={selectedProvider}
          provider={provider}
          viewMode={viewMode}
          setViewMode={setViewMode}
          onBack={() => setSelectedProvider(null)}
          onReload={() => handleReload("provider", selectedProvider)}
          onForceRefresh={(credential) =>
            handleForceRefresh(credential ? "credential" : "provider", selectedProvider, credential)
          }
          refreshing={refreshing}
        />
      )
    }
  }

  const summary = viewMode === "global" && data?.global_summary ? data.global_summary : data?.summary

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Quota Statistics</h1>
          {data && (
            <p className="text-sm text-muted-foreground">
              Last updated: {new Date(data.timestamp * 1000).toLocaleTimeString()}
            </p>
          )}
        </div>
        <div className="flex items-center gap-2">
          <Tabs value={viewMode} onValueChange={(v: string) => setViewMode(v as "current" | "global")}>
            <TabsList>
              <TabsTrigger value="current">Current</TabsTrigger>
              <TabsTrigger value="global">Global</TabsTrigger>
            </TabsList>
          </Tabs>
          <Button variant="outline" size="sm" onClick={() => handleReload("all")} disabled={refreshing || loading}>
            <RefreshCw className={`h-4 w-4 mr-2 ${refreshing ? "animate-spin" : ""}`} />
            Reload
          </Button>
          <Button variant="outline" size="sm" onClick={() => handleForceRefresh("all")} disabled={refreshing || loading}>
            Force Refresh
          </Button>
        </div>
      </div>

      {summary && (
        <div className="grid gap-4 sm:grid-cols-4">
          <SummaryCard label="Credentials" value={summary.total_credentials} />
          <SummaryCard label="Requests" value={formatNumber(summary.total_requests)} />
          <SummaryCard
            label="Tokens"
            value={formatNumber(
              (summary.tokens?.input_uncached ?? 0) + (summary.tokens?.input_cached ?? 0) + (summary.tokens?.output ?? 0)
            )}
          />
          <SummaryCard label="Cost" value={formatCost(summary.approx_total_cost)} />
        </div>
      )}

      <Card>
        <CardContent className="p-0">
          <Table>
            <TableHeader>
              <TableRow>
                <SortableHead col="provider" label="Provider" current={sortCol} dir={sortDir} onClick={toggleSort} />
                <SortableHead col="credentials" label="Credentials" current={sortCol} dir={sortDir} onClick={toggleSort} className="text-center" />
                <TableHead>Quota</TableHead>
                <SortableHead col="requests" label="Requests" current={sortCol} dir={sortDir} onClick={toggleSort} className="text-right" />
                <SortableHead col="tokens_in" label="Tokens In" current={sortCol} dir={sortDir} onClick={toggleSort} className="text-right" />
                <SortableHead col="tokens_out" label="Tokens Out" current={sortCol} dir={sortDir} onClick={toggleSort} className="text-right" />
                <SortableHead col="cost" label="Cost" current={sortCol} dir={sortDir} onClick={toggleSort} className="text-right" />
              </TableRow>
            </TableHeader>
            <TableBody>
              {providerEntries.map(([name, p]) => {
                const stats = viewMode === "current" && p.current_period ? p.current_period : null
                const requests = stats?.total_requests ?? p.total_requests ?? 0
                const tokensIn = (stats?.tokens?.input_uncached ?? p.tokens?.input_uncached ?? 0) + (stats?.tokens?.input_cached ?? p.tokens?.input_cached ?? 0)
                const tokensOut = stats?.tokens?.output ?? p.tokens?.output ?? 0
                const cachePct = stats?.tokens?.input_cache_pct ?? p.tokens?.input_cache_pct ?? 0
                const cost = stats?.approx_cost ?? p.approx_cost ?? 0
                return (
                  <TableRow
                    key={name}
                    className="cursor-pointer"
                    onClick={() => setSelectedProvider(name)}
                  >
                    <TableCell className="font-medium">
                      <div className="flex items-center gap-2">
                        {name}
                        <Badge variant="secondary" className="text-[10px]">{p.rotation_mode}</Badge>
                      </div>
                    </TableCell>
                    <TableCell className="text-center">
                      <div className="flex items-center justify-center gap-1">
                        <span>{p.credential_count}</span>
                        {p.exhausted_count > 0 && (
                          <Badge variant="destructive" className="text-[10px]">{p.exhausted_count} exh</Badge>
                        )}
                      </div>
                    </TableCell>
                    <TableCell>
                      <QuotaSummaryBars quotaGroups={p.quota_groups} />
                    </TableCell>
                    <TableCell className="text-right">{formatNumber(requests)}</TableCell>
                    <TableCell className="text-right">
                      {formatNumber(tokensIn)}
                      {cachePct > 0 && (
                        <span className="text-xs text-muted-foreground ml-1">
                          ({cachePct.toFixed(0)}% cached)
                        </span>
                      )}
                    </TableCell>
                    <TableCell className="text-right">{formatNumber(tokensOut)}</TableCell>
                    <TableCell className="text-right">{formatCost(cost)}</TableCell>
                  </TableRow>
                )
              })}
              {!providerEntries.length && (
                <TableRow>
                  <TableCell colSpan={7} className="text-center text-muted-foreground py-8">
                    {loading ? "Loading..." : "No providers found"}
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  )
}

function SummaryCard({ label, value }: { label: string; value: string | number }) {
  return (
    <Card>
      <CardContent className="p-4">
        <p className="text-sm text-muted-foreground">{label}</p>
        <p className="text-xl font-bold">{value}</p>
      </CardContent>
    </Card>
  )
}

function QuotaSummaryBars({ quotaGroups }: { quotaGroups?: Record<string, QuotaGroup> }) {
  if (!quotaGroups) return <span className="text-muted-foreground text-xs">No quota</span>

  const windows: { label: string; key: string; pct: number; remaining: number; max: number }[] = []
  const multiGroup = Object.keys(quotaGroups).length > 1
  for (const [groupName, group] of Object.entries(quotaGroups)) {
    for (const [windowName, win] of Object.entries(group.windows)) {
      const label = multiGroup ? `${groupName}/${formatWindowLabel(windowName)}` : formatWindowLabel(windowName)
      windows.push({ label, key: `${groupName}-${windowName}`, pct: win.remaining_pct ?? 0, remaining: win.total_remaining ?? 0, max: win.total_max ?? 0 })
    }
  }

  if (!windows.length) return <span className="text-muted-foreground text-xs">No windows</span>

  return (
    <div className="space-y-1.5 max-w-[250px]">
      {windows.slice(0, 3).map((w) => (
        <div key={w.key}>
          <div className="flex justify-between text-[10px] text-muted-foreground mb-0.5">
            <span className="truncate">{w.label}</span>
            <span className="whitespace-nowrap ml-1">{formatNumber(w.remaining)}/{formatNumber(w.max)}</span>
          </div>
          <div className="flex items-center gap-1.5">
            <Progress
              value={w.pct}
              className="h-1.5 flex-1"
              indicatorClassName={getQuotaColor(w.pct)}
            />
            <span className="text-[10px] text-muted-foreground w-8 text-right">{w.pct.toFixed(0)}%</span>
          </div>
        </div>
      ))}
      {windows.length > 3 && (
        <span className="text-[10px] text-muted-foreground">+{windows.length - 3} more</span>
      )}
    </div>
  )
}

function ProviderDetail({
  providerName,
  provider,
  viewMode,
  setViewMode,
  onBack,
  onReload,
  onForceRefresh,
  refreshing,
}: {
  providerName: string
  provider: ProviderStats
  viewMode: "current" | "global"
  setViewMode: (v: "current" | "global") => void
  onBack: () => void
  onReload: () => void
  onForceRefresh: (credential?: string) => void
  refreshing: boolean
}) {
  const [expandedModels, setExpandedModels] = useState<Set<string>>(new Set())

  function toggleModels(credId: string) {
    setExpandedModels((prev) => {
      const next = new Set(prev)
      if (next.has(credId)) next.delete(credId)
      else next.add(credId)
      return next
    })
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-3">
          <Button variant="ghost" size="icon" onClick={onBack}>
            <ArrowLeft className="h-4 w-4" />
          </Button>
          <div>
            <h1 className="text-2xl font-bold tracking-tight">{providerName}</h1>
            <p className="text-sm text-muted-foreground">
              {provider.credential_count} credentials &middot; {provider.rotation_mode} rotation
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Tabs value={viewMode} onValueChange={(v: string) => setViewMode(v as "current" | "global")}>
            <TabsList>
              <TabsTrigger value="current">Current</TabsTrigger>
              <TabsTrigger value="global">Global</TabsTrigger>
            </TabsList>
          </Tabs>
          <Button variant="outline" size="sm" onClick={onReload} disabled={refreshing}>
            <RefreshCw className={`h-4 w-4 mr-2 ${refreshing ? "animate-spin" : ""}`} />
            Reload
          </Button>
          <Button variant="outline" size="sm" onClick={() => onForceRefresh()} disabled={refreshing}>
            Force Refresh All
          </Button>
        </div>
      </div>

      {provider.quota_groups && Object.keys(provider.quota_groups).length > 0 && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Quota Groups</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {(Object.entries(provider.quota_groups) as [string, QuotaGroup][]).map(([groupName, group]) => (
                <div key={groupName}>
                  <h4 className="text-sm font-medium mb-2">{groupName}</h4>
                  <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                    {(Object.entries(group.windows) as [string, WindowInfo][]).map(([windowName, win]) => (
                      <div key={windowName} className="space-y-1">
                        <div className="flex justify-between text-xs">
                          <span>{formatWindowLabel(windowName)}</span>
                          <span>
                            {formatNumber(win.total_remaining)}/{formatNumber(win.total_max)}
                          </span>
                        </div>
                        <Progress
                          value={win.remaining_pct}
                          className="h-2"
                          indicatorClassName={getQuotaColor(win.remaining_pct)}
                        />
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      <div className="space-y-4">
        <h2 className="text-lg font-semibold">Credentials</h2>
        {Object.entries(provider.credentials).map(([credId, cred]: [string, CredentialStats]) => (
          <CredentialCard
            key={credId}
            cred={cred}
            viewMode={viewMode}
            showModels={expandedModels.has(credId)}
            onToggleModels={() => toggleModels(credId)}
            onForceRefresh={() => onForceRefresh(cred.full_path || credId)}
            refreshing={refreshing}
          />
        ))}
      </div>
    </div>
  )
}

function resolveModelUsage(entry: ModelUsageEntry): { request_count: number; approx_cost: number } {
  if (entry.totals) {
    return { request_count: entry.totals.request_count ?? 0, approx_cost: entry.totals.approx_cost ?? 0 }
  }
  return { request_count: entry.request_count ?? 0, approx_cost: entry.approx_cost ?? 0 }
}

function CredentialCard({
  cred,
  viewMode,
  showModels,
  onToggleModels,
  onForceRefresh,
  refreshing,
}: {
  cred: CredentialStats
  viewMode: "current" | "global"
  showModels: boolean
  onToggleModels: () => void
  onForceRefresh: () => void
  refreshing: boolean
}) {
  const statusVariant = cred.status === "active" ? "success"
    : cred.status === "cooldown" ? "warning"
    : cred.status === "needs_reauth" || cred.status === "error" ? "destructive"
    : cred.status === "exhausted" ? "destructive"
    : "secondary"

  const statusTooltips: Record<string, string> = {
    mixed: "Some quota windows are active while others are exhausted or on cooldown",
    needs_reauth: "OAuth token expired — re-authenticate with --add-credential",
    cooldown: "Temporarily rate-limited, will recover automatically",
    exhausted: "All quota windows exhausted for this credential",
  }

  const usePeriod = viewMode === "current" && cred.current_period
  const requestCount = usePeriod ? cred.current_period!.request_count : cred.totals.request_count
  const tokensIn = usePeriod ? cred.current_period!.prompt_tokens : cred.totals.prompt_tokens
  const tokensOut = usePeriod ? cred.current_period!.output_tokens : cred.totals.completion_tokens
  const cost = usePeriod ? cred.current_period!.approx_cost : cred.totals.approx_cost

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div className="flex items-center gap-2">
            <CardTitle className="text-sm font-mono">{cred.accessor_masked}</CardTitle>
            <span title={statusTooltips[cred.status] || ""}>
              <Badge variant={statusVariant} className={statusTooltips[cred.status] ? "cursor-help" : ""}>{cred.status}</Badge>
            </span>
            {cred.email && <span className="text-xs text-muted-foreground">{cred.email}</span>}
            {cred.tier && <Badge variant="outline" className="text-[10px]">{cred.tier}</Badge>}
          </div>
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="sm" onClick={onForceRefresh} disabled={refreshing}>
              <RefreshCw className="h-3 w-3 mr-1" /> Refresh
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="grid gap-4 sm:grid-cols-4 text-sm mb-3">
          <div>
            <span className="text-muted-foreground">Requests</span>
            <p className="font-medium">{formatNumber(requestCount)}</p>
          </div>
          <div>
            <span className="text-muted-foreground">Tokens In</span>
            <p className="font-medium">{formatNumber(tokensIn)}</p>
          </div>
          <div>
            <span className="text-muted-foreground">Tokens Out</span>
            <p className="font-medium">{formatNumber(tokensOut)}</p>
          </div>
          <div>
            <span className="text-muted-foreground">Cost</span>
            <p className="font-medium">{formatCost(cost)}</p>
          </div>
        </div>

        {cred.group_usage && Object.keys(cred.group_usage).length > 0 && (
          <div className="mb-3">
            <h4 className="text-xs font-medium mb-2">Quota Usage</h4>
            <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
              {Object.entries(cred.group_usage).map(([groupName, group]) =>
                Object.entries(group.windows).map(([windowName, win]) => {
                  const pct = win.limit > 0 ? ((win.remaining / win.limit) * 100) : 0
                  return (
                    <div key={`${groupName}-${windowName}`} className="space-y-1">
                      <div className="flex justify-between text-[11px]">
                        <span className="truncate">{groupName}/{formatWindowLabel(windowName)}</span>
                        <span>{formatNumber(win.remaining)}/{formatNumber(win.limit)}</span>
                      </div>
                      <Progress
                        value={pct}
                        className="h-1.5"
                        indicatorClassName={getQuotaColor(pct)}
                      />
                    </div>
                  )
                })
              )}
            </div>
          </div>
        )}

        {cred.model_usage && Object.keys(cred.model_usage).length > 0 && (
          <div>
            <button
              className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
              onClick={onToggleModels}
            >
              {showModels ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
              Model usage ({Object.keys(cred.model_usage).length} models)
            </button>
            {showModels && (
              <div className="mt-2 border rounded-md overflow-hidden">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="text-xs">Model</TableHead>
                      <TableHead className="text-xs text-right">Requests</TableHead>
                      <TableHead className="text-xs text-right">Cost</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {Object.entries(cred.model_usage).map(([model, usage]) => {
                      const stats = resolveModelUsage(usage)
                      return (
                        <TableRow key={model}>
                          <TableCell className="text-xs font-mono">{model}</TableCell>
                          <TableCell className="text-xs text-right">{stats.request_count}</TableCell>
                          <TableCell className="text-xs text-right">{formatCost(stats.approx_cost)}</TableCell>
                        </TableRow>
                      )
                    })}
                  </TableBody>
                </Table>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}

function SortableHead({ col, label, current, dir, onClick, className }: {
  col: string; label: string; current: string; dir: "asc" | "desc"
  onClick: (col: string) => void; className?: string
}) {
  const active = current === col
  return (
    <TableHead className={`cursor-pointer select-none hover:text-foreground ${className ?? ""}`} onClick={() => onClick(col)}>
      <span className="inline-flex items-center gap-1">
        {label}
        <ArrowUpDown className={`h-3 w-3 ${active ? "text-foreground" : "text-muted-foreground/50"}`} />
        {active && <span className="text-[10px]">{dir === "asc" ? "\u25b2" : "\u25bc"}</span>}
      </span>
    </TableHead>
  )
}
