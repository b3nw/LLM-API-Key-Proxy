import { useState, useEffect, useCallback, useRef } from "react"

interface UsePollingOptions<T> {
  fetcher: () => Promise<T>
  interval?: number
  enabled?: boolean
}

export function usePolling<T>({ fetcher, interval = 10000, enabled = true }: UsePollingOptions<T>) {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<Error | null>(null)
  const [loading, setLoading] = useState(true)
  const fetcherRef = useRef(fetcher)
  fetcherRef.current = fetcher

  const refresh = useCallback(async () => {
    try {
      setLoading(true)
      const result = await fetcherRef.current()
      setData(result)
      setError(null)
    } catch (e) {
      setError(e instanceof Error ? e : new Error(String(e)))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (!enabled) return
    refresh()
    const id = setInterval(refresh, interval)
    return () => clearInterval(id)
  }, [enabled, interval, refresh])

  return { data, error, loading, refresh }
}
