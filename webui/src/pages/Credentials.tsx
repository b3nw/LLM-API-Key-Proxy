import { useState, useCallback } from "react"
import { KeyRound, Plus, Trash2, RefreshCw, Globe, Terminal, AlertCircle } from "lucide-react"
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog"
import { usePolling } from "@/hooks/usePolling"
import {
  getCredentials,
  addApiKey,
  deleteApiKey,
  deleteOAuthCredential,
  addCustomProvider,
  type CredentialSummary,
  type ApiKeyInfo,
  type OAuthInfo,
} from "@/api/config"

type CredFilter = "all" | "api_key" | "oauth"

export function Credentials() {
  const { data, loading, refresh } = usePolling<CredentialSummary>({
    fetcher: getCredentials,
    interval: 30000,
  })
  const [addKeyOpen, setAddKeyOpen] = useState(false)
  const [addCustomOpen, setAddCustomOpen] = useState(false)
  const [deleting, setDeleting] = useState<string | null>(null)
  const [filter, setFilter] = useState<CredFilter>("all")

  const handleDeleteApiKey = useCallback(async (provider: string, keyName: string) => {
    if (!confirm(`Delete API key ${keyName} for ${provider}?`)) return
    setDeleting(keyName)
    try {
      await deleteApiKey(provider, keyName)
      await refresh()
    } finally {
      setDeleting(null)
    }
  }, [refresh])

  const handleDeleteOAuth = useCallback(async (provider: string, filename: string) => {
    if (!confirm(`Delete OAuth credential ${filename}?`)) return
    setDeleting(filename)
    try {
      await deleteOAuthCredential(provider, filename)
      await refresh()
    } finally {
      setDeleting(null)
    }
  }, [refresh])

  const apiKeyCount = Object.values(data?.api_keys ?? {}).reduce((sum, keys) => sum + keys.length, 0)
  const oauthCount = Object.values(data?.oauth ?? {}).reduce((sum, creds) => sum + creds.length, 0)

  const allProviders = [...new Set([
    ...Object.keys(data?.api_keys ?? {}),
    ...Object.keys(data?.oauth ?? {}),
  ])].sort().filter((provider) => {
    if (filter === "api_key") return (data?.api_keys[provider]?.length ?? 0) > 0
    if (filter === "oauth") return (data?.oauth[provider]?.length ?? 0) > 0
    return true
  })

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Credentials</h1>
          <p className="text-muted-foreground">Manage API keys and OAuth credentials</p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={refresh} disabled={loading}>
            <RefreshCw className={`h-4 w-4 mr-2 ${loading ? "animate-spin" : ""}`} />
            Refresh
          </Button>
          <Button size="sm" onClick={() => setAddKeyOpen(true)}>
            <Plus className="h-4 w-4 mr-2" />
            Add API Key
          </Button>
          <Button variant="outline" size="sm" onClick={() => setAddCustomOpen(true)}>
            <Globe className="h-4 w-4 mr-2" />
            Custom Provider
          </Button>
        </div>
      </div>

      <div className="flex flex-wrap gap-1.5">
        <Button
          variant={filter === "all" ? "default" : "outline"}
          size="sm"
          className="h-7 text-xs"
          onClick={() => setFilter("all")}
        >
          All
          <Badge variant="secondary" className="ml-1.5 text-[10px] h-4 px-1">{apiKeyCount + oauthCount}</Badge>
        </Button>
        <Button
          variant={filter === "api_key" ? "default" : "outline"}
          size="sm"
          className="h-7 text-xs"
          onClick={() => setFilter("api_key")}
        >
          <KeyRound className="h-3 w-3 mr-1" />API Keys
          <Badge variant="secondary" className="ml-1.5 text-[10px] h-4 px-1">{apiKeyCount}</Badge>
        </Button>
        <Button
          variant={filter === "oauth" ? "default" : "outline"}
          size="sm"
          className="h-7 text-xs"
          onClick={() => setFilter("oauth")}
        >
          OAuth
          <Badge variant="secondary" className="ml-1.5 text-[10px] h-4 px-1">{oauthCount}</Badge>
        </Button>
      </div>

      {allProviders.map((provider) => {
        const apiKeys = filter !== "oauth" ? (data?.api_keys[provider] ?? []) : []
        const oauthCreds = filter !== "api_key" ? (data?.oauth[provider] ?? []) : []
        if (apiKeys.length === 0 && oauthCreds.length === 0) return null
        return (
          <Card key={provider}>
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <CardTitle className="text-base">{provider}</CardTitle>
                <div className="flex gap-1">
                  {apiKeys.length > 0 && (
                    <Badge variant="secondary">{apiKeys.length} API key{apiKeys.length !== 1 ? "s" : ""}</Badge>
                  )}
                  {oauthCreds.length > 0 && (
                    <Badge variant="secondary">{oauthCreds.length} OAuth</Badge>
                  )}
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {apiKeys.map((key: ApiKeyInfo) => (
                  <div key={key.key_name} className="flex items-center justify-between py-1.5 px-3 rounded-md bg-muted/50">
                    <div className="flex items-center gap-2">
                      <KeyRound className="h-3.5 w-3.5 text-muted-foreground" />
                      <span className="text-sm font-mono">{key.key_name}</span>
                      <span className="text-xs text-muted-foreground">{key.masked_value}</span>
                    </div>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-7 w-7"
                      onClick={() => handleDeleteApiKey(provider, key.key_name)}
                      disabled={deleting === key.key_name}
                    >
                      <Trash2 className="h-3.5 w-3.5 text-destructive" />
                    </Button>
                  </div>
                ))}
                {oauthCreds.map((cred: OAuthInfo) => {
                  const isError = cred.status === "needs_reauth" || cred.status === "error"
                  const isWarning = cred.status === "cooldown" || cred.status === "exhausted"
                  const isInvalid = isError || isWarning
                  const statusTooltip: Record<string, string> = {
                    mixed: "Some quota windows are active, others are exhausted or on cooldown",
                    needs_reauth: "OAuth token expired — re-authenticate with --add-credential",
                    cooldown: "Temporarily rate-limited, will recover automatically",
                    exhausted: "All quota windows exhausted for this credential",
                    error: "Credential encountered an error",
                  }
                  const badgeVariant = isError ? "destructive" : isWarning ? "warning" : "secondary"
                  return (
                    <div
                      key={cred.filename}
                      className={`flex items-center justify-between py-1.5 px-3 rounded-md ${
                        isError ? "bg-destructive/10 border border-destructive/30" :
                        isWarning ? "bg-warning/10 border border-warning/30" : "bg-muted/50"
                      }`}
                    >
                      <div className="flex items-center gap-2">
                        {isInvalid && <AlertCircle className={`h-3.5 w-3.5 shrink-0 ${isError ? "text-destructive" : "text-warning"}`} />}
                        <Badge variant="outline" className="text-[10px]">OAuth</Badge>
                        {cred.number != null && (
                          <span className="text-xs font-mono text-muted-foreground">#{cred.number}</span>
                        )}
                        <span className="text-sm">{cred.email || cred.filename}</span>
                        {cred.tier && <Badge variant="secondary" className="text-[10px]">{cred.tier}</Badge>}
                        {cred.status && cred.status !== "unknown" && cred.status !== "active" && (
                          <span className="relative group">
                            <Badge
                              variant={badgeVariant}
                              className="text-[10px] cursor-help"
                            >
                              {cred.status}
                            </Badge>
                            {statusTooltip[cred.status] && (
                              <span className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 px-2 py-1 rounded bg-popover text-popover-foreground text-[10px] shadow-md border whitespace-nowrap opacity-0 group-hover:opacity-100 pointer-events-none transition-opacity z-50">
                                {statusTooltip[cred.status]}
                              </span>
                            )}
                          </span>
                        )}
                        {cred.status === "active" && (
                          <Badge variant="success" className="text-[10px]">active</Badge>
                        )}
                      </div>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-7 w-7"
                        onClick={() => handleDeleteOAuth(provider, cred.filename)}
                        disabled={deleting === cred.filename}
                      >
                        <Trash2 className="h-3.5 w-3.5 text-destructive" />
                      </Button>
                    </div>
                  )
                })}
                {apiKeys.length === 0 && oauthCreds.length === 0 && (
                  <p className="text-sm text-muted-foreground">No credentials configured</p>
                )}
              </div>
            </CardContent>
          </Card>
        )
      })}

      {allProviders.length === 0 && !loading && (
        <Card>
          <CardContent className="p-8 text-center text-muted-foreground">
            No credentials configured. Add an API key or OAuth credential to get started.
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader className="pb-2">
          <div className="flex items-center gap-2">
            <Terminal className="h-4 w-4 text-muted-foreground" />
            <CardTitle className="text-base">OAuth Credentials</CardTitle>
          </div>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground mb-2">
            OAuth credentials (Gemini CLI, Codex, Anthropic, Copilot) require an interactive
            terminal for the device authorization flow.
          </p>
          <pre className="bg-muted rounded-md p-3 text-xs overflow-x-auto whitespace-pre-wrap">
            <code>uv run python3 src/proxy_app/main.py --add-credential</code>
          </pre>
          <p className="text-xs text-muted-foreground mt-2">
            Run this from the project directory. In Docker, use{" "}
            <code className="bg-muted px-1 rounded">docker exec -it llm-proxy</code>{" "}
            before the command. After adding credentials, click Refresh above.
          </p>
        </CardContent>
      </Card>

      <AddApiKeyDialog open={addKeyOpen} onOpenChange={setAddKeyOpen} onSuccess={refresh} />
      <AddCustomProviderDialog open={addCustomOpen} onOpenChange={setAddCustomOpen} onSuccess={refresh} />
    </div>
  )
}

function AddApiKeyDialog({
  open,
  onOpenChange,
  onSuccess,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  onSuccess: () => void
}) {
  const [provider, setProvider] = useState("")
  const [key, setKey] = useState("")
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState("")

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setSubmitting(true)
    setError("")
    try {
      await addApiKey(provider, key)
      setProvider("")
      setKey("")
      onOpenChange(false)
      onSuccess()
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to add key")
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent onClose={() => onOpenChange(false)}>
        <DialogHeader>
          <DialogTitle>Add API Key</DialogTitle>
          <DialogDescription>Add a new API key for an LLM provider</DialogDescription>
        </DialogHeader>
        <form onSubmit={handleSubmit} className="space-y-4 mt-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">Provider</label>
            <Input
              placeholder="e.g. openai, anthropic, gemini_cli"
              value={provider}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setProvider(e.target.value)}
              required
            />
          </div>
          <div className="space-y-2">
            <label className="text-sm font-medium">API Key</label>
            <Input
              type="password"
              placeholder="sk-..."
              value={key}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => setKey(e.target.value)}
              required
            />
          </div>
          {error && <p className="text-sm text-destructive">{error}</p>}
          <div className="flex justify-end gap-2">
            <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>Cancel</Button>
            <Button type="submit" disabled={submitting}>
              {submitting ? "Adding..." : "Add Key"}
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  )
}

function AddCustomProviderDialog({
  open,
  onOpenChange,
  onSuccess,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  onSuccess: () => void
}) {
  const [name, setName] = useState("")
  const [baseUrl, setBaseUrl] = useState("")
  const [apiKey, setApiKey] = useState("")
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState("")

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    setSubmitting(true)
    setError("")
    try {
      await addCustomProvider(name, baseUrl, apiKey)
      setName("")
      setBaseUrl("")
      setApiKey("")
      onOpenChange(false)
      onSuccess()
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to add provider")
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent onClose={() => onOpenChange(false)}>
        <DialogHeader>
          <DialogTitle>Add Custom Provider</DialogTitle>
          <DialogDescription>Add an OpenAI-compatible provider with a custom API base</DialogDescription>
        </DialogHeader>
        <form onSubmit={handleSubmit} className="space-y-4 mt-4">
          <div className="space-y-2">
            <label className="text-sm font-medium">Provider Name</label>
            <Input placeholder="my_provider" value={name} onChange={(e: React.ChangeEvent<HTMLInputElement>) => setName(e.target.value)} required />
          </div>
          <div className="space-y-2">
            <label className="text-sm font-medium">Base URL</label>
            <Input placeholder="https://api.example.com/v1" value={baseUrl} onChange={(e: React.ChangeEvent<HTMLInputElement>) => setBaseUrl(e.target.value)} required />
          </div>
          <div className="space-y-2">
            <label className="text-sm font-medium">API Key</label>
            <Input type="password" placeholder="sk-..." value={apiKey} onChange={(e: React.ChangeEvent<HTMLInputElement>) => setApiKey(e.target.value)} required />
          </div>
          {error && <p className="text-sm text-destructive">{error}</p>}
          <div className="flex justify-end gap-2">
            <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>Cancel</Button>
            <Button type="submit" disabled={submitting}>
              {submitting ? "Adding..." : "Add Provider"}
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  )
}
