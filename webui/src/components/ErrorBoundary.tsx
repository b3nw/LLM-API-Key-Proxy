import { Component, type ReactNode } from "react"

interface Props {
  children: ReactNode
}

interface State {
  hasError: boolean
  error: Error | null
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false, error: null }
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error }
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen flex items-center justify-center p-8">
          <div className="max-w-lg w-full space-y-4">
            <h1 className="text-2xl font-bold text-destructive">Something went wrong</h1>
            <p className="text-muted-foreground">
              The application encountered an unexpected error. Try refreshing the page.
            </p>
            <pre className="bg-muted rounded-md p-4 text-xs overflow-auto max-h-[200px]">
              {this.state.error?.message}
            </pre>
            <button
              onClick={() => {
                this.setState({ hasError: false, error: null })
                window.location.reload()
              }}
              className="inline-flex items-center justify-center rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90"
            >
              Reload Page
            </button>
          </div>
        </div>
      )
    }

    return this.props.children
  }
}
