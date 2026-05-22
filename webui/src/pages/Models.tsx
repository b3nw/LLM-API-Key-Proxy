import { useState, useMemo, useCallback } from "react"
import { Search, RefreshCw } from "lucide-react"
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Table, TableHeader, TableBody, TableRow, TableHead, TableCell } from "@/components/ui/table"
import { usePolling } from "@/hooks/usePolling"
import { getModels, type ModelList, type ModelCard } from "@/api/models"

export function Models() {
  const { data, loading, refresh } = usePolling<ModelList>({
    fetcher: getModels,
    interval: 60000,
  })
  const [search, setSearch] = useState("")
  const [activeProviders, setActiveProviders] = useState<Set<string>>(new Set())

  const providers = useMemo(() => {
    if (!data?.data) return []
    const counts: Record<string, number> = {}
    for (const m of data.data) {
      counts[m.owned_by] = (counts[m.owned_by] || 0) + 1
    }
    return Object.entries(counts).sort(([a], [b]) => a.localeCompare(b))
  }, [data])

  const toggleProvider = useCallback((provider: string) => {
    setActiveProviders(prev => {
      const next = new Set(prev)
      if (next.has(provider)) {
        next.delete(provider)
      } else {
        next.add(provider)
      }
      return next
    })
  }, [])

  const filteredModels = useMemo(() => {
    if (!data?.data) return []
    return data.data.filter((m: ModelCard) => {
      if (activeProviders.size > 0 && !activeProviders.has(m.owned_by)) return false
      if (search && !m.id.toLowerCase().includes(search.toLowerCase())) return false
      return true
    })
  }, [data, search, activeProviders])

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Models</h1>
          <p className="text-muted-foreground">
            {data?.data.length ?? 0} models available
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={refresh} disabled={loading}>
          <RefreshCw className={`h-4 w-4 mr-2 ${loading ? "animate-spin" : ""}`} />
          Refresh
        </Button>
      </div>

      <div className="space-y-3">
        <div className="relative">
          <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search models..."
            className="pl-8"
            value={search}
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => setSearch(e.target.value)}
          />
        </div>
        <div className="flex flex-wrap gap-1.5">
          {providers.map(([p, count]: [string, number]) => (
            <Button
              key={p}
              variant={activeProviders.has(p) ? "default" : "outline"}
              size="sm"
              className="h-7 text-xs"
              onClick={() => toggleProvider(p)}
            >
              {p}
              <Badge variant="secondary" className="ml-1.5 text-[10px] h-4 px-1">
                {count}
              </Badge>
            </Button>
          ))}
          {activeProviders.size > 0 && (
            <Button
              variant="ghost"
              size="sm"
              className="h-7 text-xs text-muted-foreground"
              onClick={() => setActiveProviders(new Set())}
            >
              Clear
            </Button>
          )}
        </div>
      </div>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base">
            {filteredModels.length} model{filteredModels.length !== 1 ? "s" : ""}
            {search || activeProviders.size > 0 ? " (filtered)" : ""}
          </CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Model ID</TableHead>
                <TableHead>Provider</TableHead>
                <TableHead className="text-right">Context</TableHead>
                <TableHead className="text-right">Input $/M</TableHead>
                <TableHead className="text-right">Output $/M</TableHead>
                <TableHead>Source</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredModels.map((m: ModelCard, idx: number) => (
                <TableRow key={`${m.id}-${m.owned_by}-${idx}`}>
                  <TableCell className="font-mono text-sm">{m.id}</TableCell>
                  <TableCell>
                    <Badge variant="secondary">{m.owned_by}</Badge>
                  </TableCell>
                  <TableCell className="text-right text-sm text-muted-foreground">
                    {m.context_length ? `${(m.context_length / 1000).toFixed(0)}K` : "-"}
                  </TableCell>
                  <TableCell className="text-right text-sm text-muted-foreground">
                    {m.input_cost_per_token != null ? `$${(m.input_cost_per_token * 1_000_000).toFixed(2)}` : "-"}
                  </TableCell>
                  <TableCell className="text-right text-sm text-muted-foreground">
                    {m.output_cost_per_token != null ? `$${(m.output_cost_per_token * 1_000_000).toFixed(2)}` : "-"}
                  </TableCell>
                  <TableCell className="text-xs text-muted-foreground">
                    {m._sources?.join(", ") ?? m._match_type ?? ""}
                  </TableCell>
                </TableRow>
              ))}
              {!filteredModels.length && (
                <TableRow>
                  <TableCell colSpan={6} className="text-center text-muted-foreground py-8">
                    {loading ? "Loading..." : "No models found"}
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
