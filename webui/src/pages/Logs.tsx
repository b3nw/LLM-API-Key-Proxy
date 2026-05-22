import { useState, useCallback, useEffect, useMemo } from "react"
import {
  Search,
  RefreshCw,
  ChevronLeft,
  ChevronRight,
  FileJson,
  AlertTriangle,
  CheckCircle,
  XCircle,
} from "lucide-react"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"
import {
  getTransactions,
  getTransactionDetail,
  getTransactionFile,
  getFailures,
  type TransactionSummary,
  type TransactionDetail,
  type FailureEntry,
} from "@/api/logs"
import { formatDuration, timeAgo } from "@/lib/utils"

export function Logs() {
  const [tab, setTab] = useState("transactions")

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Log Explorer</h1>
        <p className="text-muted-foreground">Browse transaction logs and failure records</p>
      </div>

      <Tabs value={tab} onValueChange={setTab}>
        <TabsList>
          <TabsTrigger value="transactions">Transactions</TabsTrigger>
          <TabsTrigger value="failures">Failures</TabsTrigger>
        </TabsList>
        <TabsContent value="transactions">
          <TransactionBrowser />
        </TabsContent>
        <TabsContent value="failures">
          <FailureBrowser />
        </TabsContent>
      </Tabs>
    </div>
  )
}

function TransactionBrowser() {
  const [transactions, setTransactions] = useState<TransactionSummary[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [loading, setLoading] = useState(true)
  const [search, setSearch] = useState("")
  const [providerFilter, setProviderFilter] = useState<string | null>(null)
  const [statusFilter, setStatusFilter] = useState<string | null>(null)
  const [expanded, setExpanded] = useState<string | null>(null)
  const [expandedDetail, setExpandedDetail] = useState<TransactionDetail | null>(null)
  const [fileContent, setFileContent] = useState<{ name: string; content: string } | null>(null)
  const pageSize = 20

  const fetchData = useCallback(async () => {
    setLoading(true)
    try {
      const data = await getTransactions({
        page,
        page_size: pageSize,
        search: search || undefined,
        provider: providerFilter ?? undefined,
        status: statusFilter ?? undefined,
      })
      setTransactions(data.transactions)
      setTotal(data.total)
    } catch {
      // handled by empty state
    } finally {
      setLoading(false)
    }
  }, [page, search, providerFilter, statusFilter])

  useEffect(() => { fetchData() }, [fetchData])

  const toggleExpand = useCallback(async (requestId: string) => {
    if (expanded === requestId) {
      setExpanded(null)
      setExpandedDetail(null)
      setFileContent(null)
      return
    }
    setExpanded(requestId)
    setFileContent(null)
    try {
      const detail = await getTransactionDetail(requestId)
      setExpandedDetail(detail)
    } catch {
      setExpandedDetail(null)
    }
  }, [expanded])

  async function viewFile(requestId: string, filename: string) {
    try {
      const content = await getTransactionFile(requestId, filename)
      setFileContent({ name: filename, content: JSON.stringify(content, null, 2) })
    } catch {
      setFileContent({ name: filename, content: "Error loading file" })
    }
  }

  const totalPages = Math.ceil(total / pageSize)

  const logProviders = useMemo(() => {
    const set = new Set(transactions.map(tx => tx.provider))
    return [...set].sort()
  }, [transactions])

  return (
    <div className="space-y-4">
      <div className="space-y-3">
        <div className="flex items-center gap-2">
          <div className="relative flex-1">
            <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
            <Input
              placeholder="Search by request ID..."
              className="pl-8"
              value={search}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => { setSearch(e.target.value); setPage(1) }}
            />
          </div>
          <Button variant="outline" size="sm" onClick={fetchData} disabled={loading}>
            <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
          </Button>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {logProviders.map((p: string) => (
            <Button
              key={p}
              variant={providerFilter === p ? "default" : "outline"}
              size="sm"
              className="h-7 text-xs"
              onClick={() => { setProviderFilter(providerFilter === p ? null : p); setPage(1) }}
            >
              {p}
            </Button>
          ))}
          <span className="mx-1 border-r" />
          <Button
            variant={statusFilter === "success" ? "default" : "outline"}
            size="sm"
            className="h-7 text-xs"
            onClick={() => { setStatusFilter(statusFilter === "success" ? null : "success"); setPage(1) }}
          >
            <CheckCircle className="h-3 w-3 mr-1" />Success
          </Button>
          <Button
            variant={statusFilter === "error" ? "default" : "outline"}
            size="sm"
            className="h-7 text-xs"
            onClick={() => { setStatusFilter(statusFilter === "error" ? null : "error"); setPage(1) }}
          >
            <XCircle className="h-3 w-3 mr-1" />Error
          </Button>
          {(providerFilter || statusFilter) && (
            <Button
              variant="ghost"
              size="sm"
              className="h-7 text-xs text-muted-foreground"
              onClick={() => { setProviderFilter(null); setStatusFilter(null); setPage(1) }}
            >
              Clear
            </Button>
          )}
        </div>
      </div>

      <div className="space-y-2">
        {transactions.map((tx) => {
          const shortPrompt = tx.prompt_preview && tx.prompt_preview.length <= 50
          return (
          <Card key={tx.request_id} className="cursor-pointer" onClick={() => toggleExpand(tx.request_id)}>
            <CardContent className="p-3">
              <div className="flex items-center justify-between gap-3">
                <div className="flex items-center gap-2 min-w-0 flex-1">
                  <StatusBadge status={tx.status} />
                  <span className="font-medium text-sm shrink-0">{tx.provider}</span>
                  <span className="text-xs font-mono text-muted-foreground truncate">{tx.model}</span>
                  {shortPrompt && (
                    <span className="text-xs text-muted-foreground truncate hidden sm:inline">&mdash; {tx.prompt_preview}</span>
                  )}
                </div>
                <div className="flex items-center gap-3 shrink-0 text-xs text-muted-foreground">
                  <span>{tx.tokens_in}/{tx.tokens_out}</span>
                  <span>{formatDuration(tx.duration_ms)}</span>
                  <span className="whitespace-nowrap">{timeAgo(tx.timestamp)}</span>
                </div>
              </div>
              {!shortPrompt && tx.prompt_preview && (
                <p className="text-xs text-muted-foreground mt-1 truncate">{tx.prompt_preview}</p>
              )}

              {expanded === tx.request_id && (
                <div className="mt-3 space-y-3 border-t pt-3" onClick={(e) => e.stopPropagation()}>
                  <div className="grid gap-3 sm:grid-cols-4 text-xs">
                    <div>
                      <span className="text-muted-foreground">Request ID</span>
                      <p className="font-mono">{tx.request_id}</p>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Timestamp</span>
                      <p>{new Date(tx.timestamp.endsWith("Z") ? tx.timestamp : tx.timestamp + "Z").toLocaleString()}</p>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Tokens (in/out)</span>
                      <p>{tx.tokens_in} / {tx.tokens_out}</p>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Duration</span>
                      <p>{formatDuration(tx.duration_ms)}</p>
                    </div>
                  </div>

                  {expandedDetail?.files && expandedDetail.files.length > 0 && (
                    <div>
                      <p className="text-xs font-medium mb-2">Files</p>
                      <div className="flex flex-wrap gap-2">
                        {expandedDetail.files.map((f: string) => (
                          <Button
                            key={f}
                            variant={fileContent?.name === f ? "default" : "outline"}
                            size="sm"
                            onClick={() => viewFile(tx.request_id, f)}
                          >
                            <FileJson className="h-3 w-3 mr-1" />
                            {f}
                          </Button>
                        ))}
                      </div>
                    </div>
                  )}

                  {fileContent && (
                    <div>
                      <p className="text-xs font-medium mb-1">{fileContent.name}</p>
                      <pre className="bg-muted rounded-md p-3 overflow-x-auto text-xs max-h-[400px] overflow-y-auto">
                        <code>{fileContent.content}</code>
                      </pre>
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
          )
        })}
        {!transactions.length && (
          <Card>
            <CardContent className="p-8 text-center text-muted-foreground">
              {loading ? "Loading..." : "No transactions found"}
            </CardContent>
          </Card>
        )}
      </div>

      {totalPages > 1 && (
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">{total} total transactions</span>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" disabled={page <= 1} onClick={() => setPage(page - 1)}>
              <ChevronLeft className="h-4 w-4" />
            </Button>
            <span className="text-sm">{page} / {totalPages}</span>
            <Button variant="outline" size="sm" disabled={page >= totalPages} onClick={() => setPage(page + 1)}>
              <ChevronRight className="h-4 w-4" />
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}

function FailureBrowser() {
  const [failures, setFailures] = useState<FailureEntry[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [loading, setLoading] = useState(true)
  const [expanded, setExpanded] = useState<number | null>(null)
  const pageSize = 20

  const fetchData = useCallback(async () => {
    setLoading(true)
    try {
      const data = await getFailures({ page, page_size: pageSize })
      setFailures(data.failures)
      setTotal(data.total)
    } catch {
      // handled by empty state
    } finally {
      setLoading(false)
    }
  }, [page])

  useEffect(() => { fetchData() }, [fetchData])

  const totalPages = Math.ceil(total / pageSize)

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-end">
        <Button variant="outline" size="sm" onClick={fetchData} disabled={loading}>
          <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
        </Button>
      </div>

      <div className="space-y-2">
        {failures.map((f, i) => (
          <Card key={i} className="cursor-pointer" onClick={() => setExpanded(expanded === i ? null : i)}>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <AlertTriangle className="h-4 w-4 text-destructive" />
                  <span className="font-medium text-sm">{f.error_type}</span>
                  <Badge variant="outline" className="text-[10px]">{f.model}</Badge>
                </div>
                <span className="text-xs text-muted-foreground">{timeAgo(f.timestamp)}</span>
              </div>
              <p className="text-xs text-muted-foreground mt-1 truncate">{f.error_message}</p>

              {expanded === i && (
                <div className="mt-3 space-y-2 border-t pt-3">
                  <div className="text-xs space-y-1">
                    <p><span className="font-medium">Timestamp:</span> {new Date(f.timestamp.endsWith("Z") ? f.timestamp : f.timestamp + "Z").toLocaleString()}</p>
                    <p><span className="font-medium">Model:</span> {f.model}</p>
                    {f.attempt_number && <p><span className="font-medium">Attempt:</span> {f.attempt_number}</p>}
                    {f.api_key_ending && <p><span className="font-medium">Key ending:</span> ...{f.api_key_ending}</p>}
                  </div>
                  <div>
                    <p className="text-xs font-medium mb-1">Error Message</p>
                    <pre className="bg-muted rounded p-2 text-xs overflow-x-auto whitespace-pre-wrap">{f.error_message}</pre>
                  </div>
                  {f.error_chain && f.error_chain.length > 0 && (
                    <div>
                      <p className="text-xs font-medium mb-1">Error Chain</p>
                      <div className="space-y-1">
                        {f.error_chain.map((err: string, j: number) => (
                          <pre key={j} className="bg-muted rounded p-2 text-xs overflow-x-auto">{err}</pre>
                        ))}
                      </div>
                    </div>
                  )}
                  {f.raw_response && (
                    <div>
                      <p className="text-xs font-medium mb-1">Raw Response</p>
                      <pre className="bg-muted rounded p-2 text-xs overflow-x-auto max-h-[200px] overflow-y-auto">{f.raw_response}</pre>
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        ))}
        {!failures.length && (
          <Card>
            <CardContent className="p-8 text-center text-muted-foreground">
              {loading ? "Loading..." : "No failures recorded"}
            </CardContent>
          </Card>
        )}
      </div>

      {totalPages > 1 && (
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">{total} total failures</span>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" disabled={page <= 1} onClick={() => setPage(page - 1)}>
              <ChevronLeft className="h-4 w-4" />
            </Button>
            <span className="text-sm">{page} / {totalPages}</span>
            <Button variant="outline" size="sm" disabled={page >= totalPages} onClick={() => setPage(page + 1)}>
              <ChevronRight className="h-4 w-4" />
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}

function StatusBadge({ status }: { status: string }) {
  if (status === "success" || status === "200") {
    return <Badge variant="success"><CheckCircle className="h-3 w-3 mr-1" />Success</Badge>
  }
  if (status === "error" || parseInt(status) >= 400) {
    return <Badge variant="destructive"><XCircle className="h-3 w-3 mr-1" />Error</Badge>
  }
  return <Badge variant="secondary">{status}</Badge>
}
