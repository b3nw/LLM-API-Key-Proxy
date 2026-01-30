# src/proxy_app/log_viewer.py
"""
Log Viewer TUI for reviewing transaction and failure logs.

Provides an interactive interface for:
- Browsing recent API transactions
- Viewing failure logs with error details
- Filtering by provider, model, date range
- Searching by request ID
"""

import json
import fnmatch
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text
from rich.syntax import Syntax


def _get_logs_dir() -> Path:
    """Get the logs directory (local implementation to avoid heavy imports)."""
    import sys
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).parent
    else:
        base = Path.cwd()
    logs_dir = base / "logs"
    logs_dir.mkdir(exist_ok=True)
    return logs_dir


@dataclass
class TransactionEntry:
    """Represents a parsed transaction log entry."""
    dir_path: Path
    dir_name: str
    timestamp: datetime
    api_format: str
    provider: str
    model: str
    request_id: str
    # Lazy-loaded from metadata.json
    status_code: Optional[int] = None
    duration_ms: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    has_provider_logs: bool = False
    # File availability info (lazy-loaded)
    has_request: bool = False
    has_response: bool = False
    has_streaming: bool = False
    _metadata_loaded: bool = field(default=False, repr=False)

    def load_metadata(self) -> None:
        """Load metadata from metadata.json if available."""
        if self._metadata_loaded:
            return

        metadata_path = self.dir_path / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.status_code = data.get("status_code")
                self.duration_ms = data.get("duration_ms")
                usage = data.get("usage", {})
                self.prompt_tokens = usage.get("prompt_tokens")
                self.completion_tokens = usage.get("completion_tokens")
                self.has_provider_logs = data.get("has_provider_logs", False)
            except (json.JSONDecodeError, IOError):
                pass

        # Check for file availability based on API format
        if self.api_format == "ant":
            # Anthropic format: files at root and in openai/ subdirectory
            self.has_request = (self.dir_path / "anthropic_request.json").exists()
            self.has_response = (self.dir_path / "anthropic_response.json").exists()
            openai_dir = self.dir_path / "openai"
            self.has_streaming = (openai_dir / "streaming_chunks.jsonl").exists() if openai_dir.exists() else False
            if not self.has_provider_logs and openai_dir.exists():
                provider_dir = openai_dir / "provider"
                self.has_provider_logs = provider_dir.exists() and any(provider_dir.iterdir())
        else:
            # OAI format: files at root
            self.has_request = (self.dir_path / "request.json").exists()
            self.has_response = (self.dir_path / "response.json").exists()
            self.has_streaming = (self.dir_path / "streaming_chunks.jsonl").exists()
            if not self.has_provider_logs:
                provider_dir = self.dir_path / "provider"
                self.has_provider_logs = provider_dir.exists() and any(provider_dir.iterdir())

        self._metadata_loaded = True

    def get_log_level_indicator(self) -> str:
        """Get an indicator showing the level of logging available.

        Returns:
            A string indicator:
            - "📄" = metadata only
            - "📋" = has request/response
            - "📦" = has provider logs (full logging)
        """
        if self.has_provider_logs:
            return "📦"  # Full logging with provider details
        elif self.has_request or self.has_response:
            return "📋"  # Has request/response
        else:
            return "📄"  # Metadata only


@dataclass
class FailureEntry:
    """Represents a parsed failure log entry."""
    timestamp: datetime
    model: str
    error_type: str
    error_message: str
    raw_response: str
    request_headers: Dict[str, Any]
    error_chain: List[Dict[str, str]]
    api_key_ending: str
    attempt_number: int


@dataclass
class FilterState:
    """Current filter settings."""
    providers: Optional[List[str]] = None  # None = all providers
    model_pattern: Optional[str] = None
    date_start: Optional[datetime] = None
    date_end: Optional[datetime] = None
    status_filter: Optional[str] = None  # "success", "errors", None

    def is_active(self) -> bool:
        """Check if any filters are active."""
        return any([
            self.providers is not None,
            self.model_pattern is not None,
            self.date_start is not None,
            self.date_end is not None,
            self.status_filter is not None,
        ])

    def describe(self) -> str:
        """Get human-readable description of active filters."""
        if not self.is_active():
            return "None"
        parts = []
        if self.providers:
            parts.append(f"Providers: {', '.join(self.providers)}")
        if self.model_pattern:
            parts.append(f"Model: {self.model_pattern}")
        if self.date_start or self.date_end:
            start = self.date_start.strftime("%m-%d") if self.date_start else "..."
            end = self.date_end.strftime("%m-%d") if self.date_end else "..."
            parts.append(f"Date: {start} to {end}")
        if self.status_filter:
            parts.append(f"Status: {self.status_filter}")
        return ", ".join(parts)


class LogViewer:
    """Main Log Viewer TUI component."""

    def __init__(self, console: Console):
        self.console = console
        self.logs_dir = _get_logs_dir()
        self.transactions_dir = self.logs_dir / "transactions"
        self.failures_log = self.logs_dir / "failures.log"
        self.filters = FilterState()
        self.page_size = 20

    def _clear_screen(self, subtitle: str = "") -> None:
        """Clear screen and show header."""
        import os
        os.system("cls" if os.name == "nt" else "clear")
        if subtitle:
            self.console.print(
                Panel(
                    f"[bold cyan]{subtitle}[/bold cyan]",
                    title="--- Log Viewer ---",
                )
            )

    def show_menu(self) -> None:
        """Display the main Log Viewer menu."""
        while True:
            self._clear_screen("📋 Log Viewer")
            
            self.console.print()
            self.console.print("[bold]📋 Log Viewer Menu[/bold]")
            self.console.print("━" * 50)
            self.console.print()
            self.console.print("   1. 📜 Recent Transactions")
            self.console.print("   2. ❌ View Failures")
            self.console.print("   3. 🔍 Search by Request ID")
            self.console.print("   4. 🔎 Filter & View Transactions")
            self.console.print("   5. ↩️  Back to Main Menu")
            self.console.print()
            
            if self.filters.is_active():
                self.console.print(f"[dim]Active Filters: {self.filters.describe()}[/dim]")
                self.console.print()

            choice = Prompt.ask(
                "Select option",
                choices=["1", "2", "3", "4", "5"],
                show_choices=False,
            )

            if choice == "1":
                self.list_transactions()
            elif choice == "2":
                self.list_failures()
            elif choice == "3":
                self.search_by_request_id()
            elif choice == "4":
                # Open filter menu; show transactions if user chose 'See Results'
                result = self.filter_menu()
                if result == "results":
                    self.list_transactions()
            elif choice == "5":
                break

    # ==================== Transaction Listing ====================

    def _parse_transaction_dir(self, dir_path: Path) -> Optional[TransactionEntry]:
        """Parse a transaction directory name into a TransactionEntry."""
        dir_name = dir_path.name
        parts = dir_name.split("_")

        # Expected format: MMDD_HHMMSS_{api_format}_{provider}_{model...}_{request_id}
        # Or older format: MMDD_HHMMSS_{provider}_{model...}_{request_id}
        # Note: model may contain underscores (from sanitized slashes like provider/lab/name)
        # The request_id is always exactly 8 characters at the end
        if len(parts) < 5:
            return None

        try:
            date_str = parts[0]  # MMDD
            time_str = parts[1]  # HHMMSS

            # Request ID is always the last part (8 chars from uuid4)
            request_id = parts[-1]

            # Handle both old and new format by detecting api_format
            # New format has api_format like "oai" or "ant" at parts[2]
            # Note: This assumes old-format logs don't have providers literally named "ant" or "oai"
            if len(parts) >= 6 and parts[2] in ("oai", "ant"):
                api_format = parts[2]
                provider = parts[3]
                # Model is everything between provider and request_id
                # Join parts[4:-1] to handle model names with underscores
                model = "_".join(parts[4:-1])
            else:
                api_format = "oai"  # Default for old format
                provider = parts[2]
                # Model is everything between provider and request_id
                model = "_".join(parts[3:-1])
            
            # Parse timestamp from metadata.json for accuracy (has full year)
            full_timestamp = None
            metadata_path = dir_path / "metadata.json"
            if metadata_path.exists():
                try:
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if "timestamp_utc" in data:
                        full_timestamp = datetime.fromisoformat(data["timestamp_utc"].replace("Z", "+00:00")).replace(tzinfo=None)
                except (json.JSONDecodeError, IOError, ValueError):
                    pass

            if full_timestamp is None:
                # Fallback to parsing from directory name
                now = datetime.now()
                month = int(date_str[:2])
                day = int(date_str[2:])
                hour = int(time_str[:2])
                minute = int(time_str[2:4])
                second = int(time_str[4:6]) if len(time_str) >= 6 else 0

                # Handle year rollover: if parsed date is in future, use previous year
                tentative = datetime(now.year, month, day, hour, minute, second)
                if tentative > now + timedelta(days=1):
                    full_timestamp = datetime(now.year - 1, month, day, hour, minute, second)
                else:
                    full_timestamp = tentative

            return TransactionEntry(
                dir_path=dir_path,
                dir_name=dir_name,
                timestamp=full_timestamp,
                api_format=api_format,
                provider=provider,
                model=model,
                request_id=request_id,
            )
        except (ValueError, IndexError):
            return None

    def _get_transactions(self) -> List[TransactionEntry]:
        """Get all transaction entries, sorted by timestamp (newest first)."""
        if not self.transactions_dir.exists():
            return []
        
        entries = []
        for dir_path in self.transactions_dir.iterdir():
            if dir_path.is_dir():
                entry = self._parse_transaction_dir(dir_path)
                if entry:
                    entries.append(entry)
        
        # Sort by timestamp, newest first
        entries.sort(key=lambda e: e.timestamp, reverse=True)
        return entries

    def _apply_filters(self, entries: List[TransactionEntry]) -> List[TransactionEntry]:
        """Apply current filters to transaction entries."""
        filtered = entries

        # Handle empty provider list as "show nothing" vs None as "no filter"
        if self.filters.providers is not None:
            filtered = [e for e in filtered if e.provider in self.filters.providers]

        if self.filters.model_pattern:
            filtered = [e for e in filtered if fnmatch.fnmatch(e.model, self.filters.model_pattern)]

        if self.filters.date_start:
            filtered = [e for e in filtered if e.timestamp >= self.filters.date_start]

        if self.filters.date_end:
            # Only add 1 day if time is at midnight (date-only filter)
            # If time is already set (e.g., 23:59:59), use as-is
            if self.filters.date_end.hour == 0 and self.filters.date_end.minute == 0:
                end = self.filters.date_end + timedelta(days=1)
            else:
                end = self.filters.date_end + timedelta(seconds=1)
            filtered = [e for e in filtered if e.timestamp < end]
        
        if self.filters.status_filter:
            # Need to load metadata for status filtering
            for entry in filtered:
                entry.load_metadata()
            if self.filters.status_filter == "success":
                filtered = [e for e in filtered if e.status_code == 200]
            elif self.filters.status_filter == "errors":
                filtered = [e for e in filtered if e.status_code and e.status_code != 200]
        
        return filtered

    def _format_tokens(self, prompt: Optional[int], completion: Optional[int]) -> str:
        """Format token counts as 'in/out'."""
        if prompt is None and completion is None:
            return "-/-"
        
        def fmt(n: Optional[int]) -> str:
            if n is None:
                return "-"
            if n >= 1000:
                return f"{n/1000:.1f}k"
            return str(n)
        
        return f"{fmt(prompt)}/{fmt(completion)}"

    def _format_duration(self, ms: Optional[int]) -> str:
        """Format duration in ms to human-readable string."""
        if ms is None:
            return "-"
        if ms >= 1000:
            return f"{ms/1000:.1f}s"
        return f"{ms}ms"

    def list_transactions(self, page: int = 0) -> None:
        """Display paginated list of transactions."""
        entries = self._get_transactions()
        entries = self._apply_filters(entries)
        
        total = len(entries)
        total_pages = max(1, (total + self.page_size - 1) // self.page_size)
        page = max(0, min(page, total_pages - 1))
        
        start_idx = page * self.page_size
        end_idx = min(start_idx + self.page_size, total)
        page_entries = entries[start_idx:end_idx]
        
        # Load metadata for displayed entries
        for entry in page_entries:
            entry.load_metadata()
        
        while True:
            self._clear_screen(f"📜 Recent Transactions ({total} total)")
            
            # Show prominent filter status bar when filters are active
            if self.filters.is_active():
                all_entries = self._get_transactions()
                unfiltered_count = len(all_entries)
                self.console.print()
                self.console.print(
                    Panel(
                        f"[bold yellow]🔎 FILTERS ACTIVE[/bold yellow]: {self.filters.describe()}\n"
                        f"[dim]Showing {total} of {unfiltered_count} transactions • Press [C] to clear filters[/dim]",
                        border_style="yellow",
                    )
                )
            
            if not entries:
                self.console.print()
                self.console.print("[dim]No transactions found.[/dim]")
                if self.filters.is_active():
                    self.console.print("[dim]Try clearing filters with [C] or [F] to modify.[/dim]")
                self.console.print()
                Prompt.ask("Press Enter to go back", default="")
                return
            
            # Build table
            table = Table(show_header=True, header_style="bold", box=None)
            table.add_column("#", style="dim", width=4)
            table.add_column("Timestamp", width=17)
            table.add_column("Provider", width=12)
            table.add_column("Model", width=25, overflow="ellipsis")
            table.add_column("Status", width=6, justify="center")
            table.add_column("Tokens", width=10, justify="right")
            table.add_column("Duration", width=8, justify="right")
            table.add_column("Logs", width=4, justify="center")

            for i, entry in enumerate(page_entries):
                row_num = str(start_idx + i + 1)
                ts = entry.timestamp.strftime("%m-%d %H:%M:%S")

                # Color-code status
                status = str(entry.status_code) if entry.status_code else "-"
                if entry.status_code == 200:
                    status = f"[green]{status}[/green]"
                elif entry.status_code and 400 <= entry.status_code < 500:
                    status = f"[yellow]{status}[/yellow]"
                elif entry.status_code and entry.status_code >= 500:
                    status = f"[red]{status}[/red]"

                tokens = self._format_tokens(entry.prompt_tokens, entry.completion_tokens)
                duration = self._format_duration(entry.duration_ms)
                log_indicator = entry.get_log_level_indicator()

                table.add_row(
                    row_num,
                    ts,
                    entry.provider,
                    entry.model,
                    status,
                    tokens,
                    duration,
                    log_indicator,
                )

            self.console.print()
            self.console.print(table)
            self.console.print()
            self.console.print("[dim]Logs: 📄=metadata only  📋=req/resp  📦=full (provider logs)[/dim]")
            self.console.print(f"Page {page + 1}/{total_pages}")
            self.console.print()

            # Show different options based on filter state
            if self.filters.is_active():
                self.console.print("[dim][N] Next  [P] Prev  [1-N] View Details  [F] Filter  [C] Clear Filters  [B] Back[/dim]")
            else:
                self.console.print("[dim][N] Next  [P] Prev  [1-N] View Details  [F] Filter  [B] Back[/dim]")
            
            choice = Prompt.ask("Select", default="b").lower()
            
            if choice == "b":
                return
            elif choice == "n" and page < total_pages - 1:
                page += 1
                start_idx = page * self.page_size
                end_idx = min(start_idx + self.page_size, total)
                page_entries = entries[start_idx:end_idx]
                for entry in page_entries:
                    entry.load_metadata()
            elif choice == "p" and page > 0:
                page -= 1
                start_idx = page * self.page_size
                end_idx = min(start_idx + self.page_size, total)
                page_entries = entries[start_idx:end_idx]
                for entry in page_entries:
                    entry.load_metadata()
            elif choice == "f":
                self.filter_menu()
                # Reload with new filters
                entries = self._get_transactions()
                entries = self._apply_filters(entries)
                total = len(entries)
                total_pages = max(1, (total + self.page_size - 1) // self.page_size)
                page = 0
                start_idx = 0
                end_idx = min(self.page_size, total)
                page_entries = entries[start_idx:end_idx]
                for entry in page_entries:
                    entry.load_metadata()
            elif choice == "c":
                # Clear all filters and reload
                self.filters = FilterState()
                # Reload all transactions (no filter needed since filters are now empty)
                entries = self._get_transactions()
                total = len(entries)
                total_pages = max(1, (total + self.page_size - 1) // self.page_size)
                page = 0
                start_idx = 0
                end_idx = min(self.page_size, total)
                page_entries = entries[start_idx:end_idx]
                for entry in page_entries:
                    entry.load_metadata()
                continue  # Force immediate screen refresh
            elif choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < total:
                    self.view_transaction(entries[idx])

    def view_transaction(self, entry: TransactionEntry) -> None:
        """Display detailed view of a transaction."""
        entry.load_metadata()
        
        while True:
            self._clear_screen(f"📄 Transaction: {entry.request_id}")
            
            self.console.print()
            self.console.print(f"[dim]Directory: {entry.dir_name}[/dim]")
            self.console.print()
            
            # Metadata section
            self.console.print("[bold]📊 Metadata[/bold]")
            self.console.print("━" * 50)
            self.console.print(f"   Request ID:      {entry.request_id}")
            self.console.print(f"   Timestamp:       {entry.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            self.console.print(f"   Provider:        {entry.provider}")
            self.console.print(f"   Model:           {entry.model}")
            
            status_str = str(entry.status_code) if entry.status_code else "N/A"
            if entry.status_code == 200:
                status_str = f"[green]{status_str} ✅[/green]"
            elif entry.status_code and entry.status_code >= 400:
                status_str = f"[red]{status_str} ❌[/red]"
            self.console.print(f"   Status:          {status_str}")
            
            self.console.print(f"   Duration:        {self._format_duration(entry.duration_ms)}")
            self.console.print()
            
            # Token usage
            if entry.prompt_tokens or entry.completion_tokens:
                self.console.print("[bold]📈 Token Usage[/bold]")
                self.console.print("━" * 50)
                self.console.print(f"   Prompt:          {entry.prompt_tokens or 'N/A'} tokens")
                self.console.print(f"   Completion:      {entry.completion_tokens or 'N/A'} tokens")
                total = (entry.prompt_tokens or 0) + (entry.completion_tokens or 0)
                self.console.print(f"   Total:           {total} tokens")
                self.console.print()
            
            # Available files
            self.console.print("[bold]📁 Available Files[/bold]")
            self.console.print("━" * 50)

            files = []  # List of (display_name, actual_path) tuples

            # Handle different API formats
            if entry.api_format == "ant":
                # Anthropic format: anthropic_request.json, anthropic_response.json at root
                # Plus openai/ subdirectory with translated request/response
                self.console.print("[dim]API Format: Anthropic[/dim]")
                self.console.print()

                # Check for Anthropic-native files
                ant_files = [
                    ("anthropic_request.json", "Anthropic-native request"),
                    ("anthropic_response.json", "Anthropic-native response"),
                    ("metadata.json", "Transaction metadata"),
                ]
                for filename, description in ant_files:
                    path = entry.dir_path / filename
                    if path.exists():
                        files.append((filename, path))
                        self.console.print(f"   {len(files)}. [green]✓[/green] {filename} [dim]({description})[/dim]")
                    else:
                        self.console.print(f"      [dim]✗ {filename}[/dim]")

                # Check for OpenAI translation subdirectory
                openai_dir = entry.dir_path / "openai"
                if openai_dir.exists():
                    self.console.print()
                    self.console.print("[dim]OpenAI Translation Layer:[/dim]")
                    oai_files = [
                        ("request.json", "OpenAI-compatible request"),
                        ("response.json", "OpenAI-compatible response"),
                        ("streaming_chunks.jsonl", "Streaming chunks"),
                    ]
                    for filename, description in oai_files:
                        path = openai_dir / filename
                        if path.exists():
                            files.append((f"openai/{filename}", path))
                            self.console.print(f"   {len(files)}. [green]✓[/green] openai/{filename} [dim]({description})[/dim]")
                        else:
                            self.console.print(f"      [dim]✗ openai/{filename}[/dim]")

                    # Check for provider subdirectory in openai/
                    oai_provider_dir = openai_dir / "provider"
                    if oai_provider_dir.exists() and any(oai_provider_dir.iterdir()):
                        files.append(("openai/provider/", oai_provider_dir))
                        self.console.print(f"   {len(files)}. [green]✓[/green] openai/provider/ [cyan](provider-level logs)[/cyan]")
            else:
                # OAI format: request.json, response.json at root
                self.console.print("[dim]API Format: OpenAI[/dim]")
                self.console.print()

                expected_files = [
                    ("request.json", "OpenAI-compatible request"),
                    ("response.json", "OpenAI-compatible response"),
                    ("metadata.json", "Transaction metadata"),
                    ("streaming_chunks.jsonl", "Streaming chunks (if streaming)"),
                ]

                for filename, description in expected_files:
                    path = entry.dir_path / filename
                    if path.exists():
                        files.append((filename, path))
                        self.console.print(f"   {len(files)}. [green]✓[/green] {filename} [dim]({description})[/dim]")
                    else:
                        self.console.print(f"      [dim]✗ {filename}[/dim]")

                provider_dir = entry.dir_path / "provider"
                if provider_dir.exists() and any(provider_dir.iterdir()):
                    files.append(("provider/", provider_dir))
                    self.console.print(f"   {len(files)}. [green]✓[/green] provider/ [cyan](provider-level logs)[/cyan]")
                else:
                    self.console.print(f"      [dim]✗ provider/ (no provider logs)[/dim]")
            
            self.console.print()
            self.console.print("[dim][1-N] View File  [B] Back[/dim]")

            choice = Prompt.ask("Select", default="b").lower()

            if choice == "b":
                return
            elif choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(files):
                    display_name, path = files[idx]
                    if display_name.endswith("/"):
                        # It's a directory (provider logs)
                        self._view_provider_logs_dir(path)
                    else:
                        self._view_json_file(path)

    def _view_json_file(self, file_path: Path) -> None:
        """Display JSON file with syntax highlighting."""
        self._clear_screen(f"📄 {file_path.name}")
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Pretty print JSON
            try:
                data = json.loads(content)
                content = json.dumps(data, indent=2, ensure_ascii=False)
            except json.JSONDecodeError:
                pass
            
            syntax = Syntax(content, "json", theme="monokai", line_numbers=True)
            self.console.print(syntax)
            
        except IOError as e:
            self.console.print(f"[red]Error reading file: {e}[/red]")
        
        self.console.print()
        Prompt.ask("Press Enter to go back", default="")

    def _view_provider_logs(self, entry: TransactionEntry) -> None:
        """View provider-level logs (legacy wrapper)."""
        self._view_provider_logs_dir(entry.dir_path / "provider")

    def _view_provider_logs_dir(self, provider_dir: Path) -> None:
        """View provider-level logs from a directory path."""
        while True:
            self._clear_screen("📂 Provider Logs")

            files = sorted(provider_dir.iterdir(), key=lambda x: x.name) if provider_dir.exists() else []
            if not files:
                self.console.print("[dim]No provider logs found.[/dim]")
                Prompt.ask("Press Enter to go back", default="")
                return

            self.console.print()
            for i, f in enumerate(files, 1):
                if f.is_dir():
                    self.console.print(f"   {i}. 📁 {f.name}/")
                else:
                    size = f.stat().st_size
                    size_str = f"{size/1024:.1f}KB" if size >= 1024 else f"{size}B"
                    self.console.print(f"   {i}. {f.name} ({size_str})")

            self.console.print()
            self.console.print("[dim][1-N] View File  [B] Back[/dim]")

            choice = Prompt.ask("Select", default="b").lower()

            if choice == "b":
                return
            elif choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(files):
                    selected = files[idx]
                    if selected.is_dir():
                        self._view_provider_logs_dir(selected)
                    else:
                        self._view_json_file(selected)

    # ==================== Failure Log ====================

    def _parse_failures(self) -> List[FailureEntry]:
        """Parse the failures.log file."""
        if not self.failures_log.exists():
            return []
        
        entries = []
        try:
            with open(self.failures_log, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        timestamp = datetime.fromisoformat(data["timestamp"])
                        entries.append(FailureEntry(
                            timestamp=timestamp,
                            model=data.get("model", "N/A"),
                            error_type=data.get("error_type", "Unknown"),
                            error_message=data.get("error_message", ""),
                            raw_response=data.get("raw_response", ""),
                            request_headers=data.get("request_headers", {}),
                            error_chain=data.get("error_chain") or [],
                            api_key_ending=data.get("api_key_ending", ""),
                            attempt_number=data.get("attempt_number", 1),
                        ))
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
        except IOError:
            pass
        
        # Sort by timestamp, newest first
        entries.sort(key=lambda e: e.timestamp, reverse=True)
        return entries

    def list_failures(self, page: int = 0) -> None:
        """Display paginated list of failures."""
        entries = self._parse_failures()
        
        total = len(entries)
        total_pages = max(1, (total + self.page_size - 1) // self.page_size)
        page = max(0, min(page, total_pages - 1))
        
        start_idx = page * self.page_size
        end_idx = min(start_idx + self.page_size, total)
        page_entries = entries[start_idx:end_idx]
        
        while True:
            self._clear_screen(f"❌ Failure Log ({total} entries)")
            
            if not entries:
                self.console.print()
                self.console.print("[dim]No failure entries found.[/dim]")
                self.console.print()
                Prompt.ask("Press Enter to go back", default="")
                return
            
            # Build table
            table = Table(show_header=True, header_style="bold", box=None)
            table.add_column("#", style="dim", width=4)
            table.add_column("Timestamp", width=17)
            table.add_column("Model", width=28, overflow="ellipsis")
            table.add_column("Error Type", width=18, overflow="ellipsis")
            table.add_column("Message", width=35, overflow="ellipsis")
            
            for i, entry in enumerate(page_entries):
                row_num = str(start_idx + i + 1)
                ts = entry.timestamp.strftime("%m-%d %H:%M:%S")
                msg = entry.error_message[:35] + "..." if len(entry.error_message) > 35 else entry.error_message
                
                table.add_row(
                    row_num,
                    ts,
                    entry.model,
                    f"[red]{entry.error_type}[/red]",
                    msg,
                )
            
            self.console.print()
            self.console.print(table)
            self.console.print()
            self.console.print(f"Page {page + 1}/{total_pages}")
            self.console.print()
            self.console.print("[dim][N] Next  [P] Prev  [1-N] View Details  [B] Back[/dim]")
            
            choice = Prompt.ask("Select", default="b").lower()
            
            if choice == "b":
                return
            elif choice == "n" and page < total_pages - 1:
                page += 1
                start_idx = page * self.page_size
                end_idx = min(start_idx + self.page_size, total)
                page_entries = entries[start_idx:end_idx]
            elif choice == "p" and page > 0:
                page -= 1
                start_idx = page * self.page_size
                end_idx = min(start_idx + self.page_size, total)
                page_entries = entries[start_idx:end_idx]
            elif choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < total:
                    self.view_failure(entries[idx])

    def view_failure(self, entry: FailureEntry) -> None:
        """Display detailed view of a failure."""
        while True:
            self._clear_screen("❌ Failure Details")
            
            self.console.print()
            self.console.print(f"   Timestamp:   {entry.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
            self.console.print(f"   Model:       {entry.model}")
            self.console.print(f"   Attempt:     {entry.attempt_number}")
            self.console.print(f"   Credential:  {entry.api_key_ending}")
            self.console.print()
            
            # Error type
            self.console.print(f"[bold red]🔴 Error Type: {entry.error_type}[/bold red]")
            self.console.print("━" * 50)
            self.console.print()
            self.console.print(entry.error_message)
            self.console.print()
            
            # Error chain
            if entry.error_chain:
                self.console.print(f"[bold]🔗 Error Chain ({len(entry.error_chain)} errors)[/bold]")
                self.console.print("━" * 50)
                for i, err in enumerate(entry.error_chain, 1):
                    self.console.print(f"   {i}. {err.get('type', 'Unknown')}")
                    msg = err.get('message', '')
                    if len(msg) > 60:
                        msg = msg[:60] + "..."
                    self.console.print(f"      └─ {msg}")
                    self.console.print()
            
            self.console.print("[dim][R] View Raw Response  [H] View Headers  [B] Back[/dim]")
            
            choice = Prompt.ask("Select", default="b").lower()
            
            if choice == "b":
                return
            elif choice == "r":
                self._view_raw_response(entry)
            elif choice == "h":
                self._view_headers(entry)

    def _view_raw_response(self, entry: FailureEntry) -> None:
        """Display raw response from failure."""
        self._clear_screen("📋 Raw Response")
        
        self.console.print()
        
        # Try to pretty-print if it's JSON
        try:
            data = json.loads(entry.raw_response)
            content = json.dumps(data, indent=2, ensure_ascii=False)
            syntax = Syntax(content, "json", theme="monokai", line_numbers=True)
            self.console.print(syntax)
        except json.JSONDecodeError:
            self.console.print(entry.raw_response)
        
        self.console.print()
        Prompt.ask("Press Enter to go back", default="")

    def _view_headers(self, entry: FailureEntry) -> None:
        """Display request headers from failure."""
        self._clear_screen("📨 Request Headers")
        
        self.console.print()
        
        table = Table(show_header=True, header_style="bold", box=None)
        table.add_column("Header", width=30)
        table.add_column("Value", overflow="ellipsis")
        
        for key, value in entry.request_headers.items():
            # Mask sensitive data - show only last 4 chars
            key_lower = key.lower()
            if "key" in key_lower or "auth" in key_lower or "token" in key_lower or "secret" in key_lower:
                val_str = str(value)
                if len(val_str) > 4:
                    value = "****" + val_str[-4:]
                else:
                    value = "****"
            table.add_row(key, str(value))
        
        self.console.print(table)
        self.console.print()
        Prompt.ask("Press Enter to go back", default="")

    # ==================== Search & Filter ====================

    def search_by_request_id(self) -> None:
        """Search for a transaction by request ID."""
        self._clear_screen("🔍 Search by Request ID")
        
        self.console.print()
        self.console.print("Enter a full or partial request ID (8 characters):")
        self.console.print()
        
        search_term = Prompt.ask("Request ID", default="").strip().lower()
        
        if not search_term:
            return
        
        entries = self._get_transactions()
        matches = [e for e in entries if search_term in e.request_id.lower()]
        
        if not matches:
            self.console.print()
            self.console.print(f"[yellow]No transactions found matching '{search_term}'[/yellow]")
            Prompt.ask("Press Enter to continue", default="")
            return
        
        if len(matches) == 1:
            self.view_transaction(matches[0])
        else:
            # Show list of matches (limit to 20)
            display_limit = min(20, len(matches))
            self.console.print()
            self.console.print(f"Found {len(matches)} matches{' (showing first 20)' if len(matches) > 20 else ''}:")
            self.console.print()

            for i, entry in enumerate(matches[:display_limit], 1):
                ts = entry.timestamp.strftime("%m-%d %H:%M:%S")
                self.console.print(f"   {i}. [{entry.request_id}] {ts} - {entry.provider}/{entry.model}")

            self.console.print()
            choice = Prompt.ask("Select transaction (or B to go back)", default="b").lower()

            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < display_limit:
                    self.view_transaction(matches[idx])

    def filter_menu(self) -> None:
        """Display filter options menu."""
        while True:
            self._clear_screen("🔎 Filter Transactions")
            
            self.console.print()
            self.console.print(f"[bold]Current Filters:[/bold] {self.filters.describe()}")
            self.console.print()
            self.console.print("━" * 50)
            self.console.print()
            self.console.print("[bold]Quick Filters:[/bold]")
            self.console.print("   1. 📅 Today only")
            self.console.print("   2. 🕐 Last hour")
            self.console.print("   3. ❌ Errors only (non-200 status)")
            self.console.print("   4. ✅ Successful only (200 status)")
            self.console.print()
            self.console.print("[bold]Custom Filters:[/bold]")
            self.console.print("   5. 🏢 By Provider")
            self.console.print("   6. 🤖 By Model")
            self.console.print("   7. 📆 By Date Range")
            self.console.print()
            self.console.print("   8. 🧹 Clear All Filters")
            self.console.print("   9. ↩️  Back to Filter Menu")
            self.console.print()
            
            choice = Prompt.ask(
                "Select option",
                choices=["1", "2", "3", "4", "5", "6", "7", "8", "9"],
                show_choices=False,
            )
            
            now = datetime.now()
            
            if choice == "1":  # Today
                self.filters.date_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
                self.filters.date_end = now
                if self._prompt_see_results("Today only"):
                    return "results"
            elif choice == "2":  # Last hour
                self.filters.date_start = now - timedelta(hours=1)
                self.filters.date_end = now
                if self._prompt_see_results("Last hour"):
                    return "results"
            elif choice == "3":  # Errors only
                self.filters.status_filter = "errors"
                if self._prompt_see_results("Errors only"):
                    return "results"
            elif choice == "4":  # Successful only
                self.filters.status_filter = "success"
                if self._prompt_see_results("Successful only"):
                    return "results"
            elif choice == "5":  # By Provider
                result = self._filter_by_provider()
                if result == "results":
                    return "results"
            elif choice == "6":  # By Model
                result = self._filter_by_model()
                if result == "results":
                    return "results"
            elif choice == "7":  # By Date Range
                result = self._filter_by_date_range()
                if result == "results":
                    return "results"
            elif choice == "8":  # Clear all
                self.filters = FilterState()
                self.console.print("[green]✅ All filters cleared[/green]")
            elif choice == "9":  # Back
                return "menu"
        
        return "menu"

    def _prompt_see_results(self, filter_name: str) -> bool:
        """Prompt user to see results immediately after setting a filter."""
        self.console.print(f"[green]✅ Filter set: {filter_name}[/green]")
        self.console.print()
        self.console.print("   1. 👁️  See results now")
        self.console.print("   2. 🔎 Add more filters")
        self.console.print()
        choice = Prompt.ask("What next?", choices=["1", "2"], default="1")
        return choice == "1"

    def _filter_by_provider(self) -> str:
        """Filter by provider submenu. Returns 'results' to go directly to results, 'menu' otherwise."""
        # Discover available providers from transactions
        entries = self._get_transactions()
        providers = sorted(set(e.provider for e in entries))
        
        if not providers:
            self.console.print("[yellow]No transactions found to filter.[/yellow]")
            Prompt.ask("Press Enter to continue", default="")
            return "menu"
        
        # Initialize selection (all selected by default, preserve empty selection if set)
        if self.filters.providers is not None:
            selected = set(self.filters.providers)
        else:
            selected = set(providers)
        
        while True:
            self._clear_screen("🏢 Filter by Provider")
            
            self.console.print()
            self.console.print("Select providers to include (toggle with number):")
            self.console.print()
            
            for i, provider in enumerate(providers, 1):
                check = "✅" if provider in selected else "  "
                self.console.print(f"   [{check}] {i}. {provider}")
            
            self.console.print()
            self.console.print("   A. Select All")
            self.console.print("   N. Select None")
            self.console.print("   S. Save & See Results")
            self.console.print("   B. Back (save selection)")
            self.console.print()
            
            choice = Prompt.ask("Toggle", default="b").lower()
            
            if choice == "b":
                # None = no filter (all), empty list = filter to nothing, list = specific providers
                self.filters.providers = None if selected == set(providers) else list(selected)
                return "menu"
            elif choice == "s":
                self.filters.providers = None if selected == set(providers) else list(selected)
                return "results"
            elif choice == "a":
                selected = set(providers)
            elif choice == "n":
                selected = set()
            elif choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(providers):
                    provider = providers[idx]
                    if provider in selected:
                        selected.discard(provider)
                    else:
                        selected.add(provider)

    def _filter_by_model(self) -> str:
        """Filter by model pattern. Returns 'results' to go directly to results, 'menu' otherwise."""
        self._clear_screen("🤖 Filter by Model")
        
        self.console.print()
        self.console.print("Enter model filter pattern (supports wildcards):")
        self.console.print()
        self.console.print("[dim]Examples:[/dim]")
        self.console.print("  • claude-*        (all Claude models)")
        self.console.print("  • gemini-2.5-*    (all Gemini 2.5 models)")
        self.console.print("  • *-opus-*        (any Opus variant)")
        self.console.print()
        self.console.print(f"[dim]Current filter: {self.filters.model_pattern or '<none>'}[/dim]")
        self.console.print()
        
        pattern = Prompt.ask("Pattern (or empty to clear)", default="").strip()
        
        self.filters.model_pattern = pattern if pattern else None
        
        if pattern:
            self.console.print(f"[green]✅ Model filter set: {pattern}[/green]")
            self.console.print()
            self.console.print("   1. 👁️  See results now")
            self.console.print("   2. 🔎 Add more filters")
            self.console.print()
            choice = Prompt.ask(
                "What next?",
                choices=["1", "2"],
                default="1",
            )
            if choice == "1":
                return "results"
        else:
            self.console.print("[green]✅ Model filter cleared[/green]")
        
        return "menu"

    def _filter_by_date_range(self) -> str:
        """Filter by date range submenu. Returns 'results' to go directly to results, 'menu' otherwise."""
        now = datetime.now()
        
        while True:
            self._clear_screen("📆 Filter by Date Range")
            
            current = "All time"
            if self.filters.date_start or self.filters.date_end:
                start = self.filters.date_start.strftime("%b %d") if self.filters.date_start else "..."
                end = self.filters.date_end.strftime("%b %d") if self.filters.date_end else "..."
                current = f"{start} to {end}"
            
            self.console.print()
            self.console.print(f"[bold]Current range:[/bold] {current}")
            self.console.print()
            self.console.print("[bold]Presets:[/bold]")
            self.console.print(f"   1. Today ({now.strftime('%b %d')})")
            self.console.print(f"   2. Yesterday ({(now - timedelta(days=1)).strftime('%b %d')})")
            self.console.print("   3. Last 7 days")
            self.console.print("   4. Last 30 days")
            self.console.print(f"   5. This month ({now.strftime('%B')})")
            self.console.print()
            self.console.print("[bold]Custom:[/bold]")
            self.console.print("   6. Enter custom date range")
            self.console.print()
            self.console.print("   7. Clear date filter")
            self.console.print("   B. Back")
            self.console.print()
            
            choice = Prompt.ask("Select", default="b").lower()
            
            if choice == "b":
                return "menu"
            elif choice == "1":  # Today
                self.filters.date_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
                self.filters.date_end = now
                if self._prompt_see_results("Today"):
                    return "results"
            elif choice == "2":  # Yesterday
                yesterday = now - timedelta(days=1)
                self.filters.date_start = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
                self.filters.date_end = yesterday.replace(hour=23, minute=59, second=59)
                if self._prompt_see_results("Yesterday"):
                    return "results"
            elif choice == "3":  # Last 7 days
                self.filters.date_start = now - timedelta(days=7)
                self.filters.date_end = now
                if self._prompt_see_results("Last 7 days"):
                    return "results"
            elif choice == "4":  # Last 30 days
                self.filters.date_start = now - timedelta(days=30)
                self.filters.date_end = now
                if self._prompt_see_results("Last 30 days"):
                    return "results"
            elif choice == "5":  # This month
                self.filters.date_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                self.filters.date_end = now
                if self._prompt_see_results(f"This month ({now.strftime('%B')})"):
                    return "results"
            elif choice == "6":  # Custom
                if self._enter_custom_date_range():
                    return "results"
            elif choice == "7":  # Clear
                self.filters.date_start = None
                self.filters.date_end = None
                self.console.print("[green]✅ Date filter cleared[/green]")

    def _enter_custom_date_range(self) -> bool:
        """Enter custom date range. Returns True if user wants to see results immediately."""
        self.console.print()
        self.console.print("Enter dates in YYYY-MM-DD format:")
        self.console.print()
        
        start_str = Prompt.ask("Start date", default="")
        end_str = Prompt.ask("End date", default="")
        
        try:
            if start_str:
                self.filters.date_start = datetime.strptime(start_str, "%Y-%m-%d")
            if end_str:
                self.filters.date_end = datetime.strptime(end_str, "%Y-%m-%d")
            
            if start_str or end_str:
                return self._prompt_see_results("Custom date range")
        except ValueError:
            self.console.print()
            self.console.print("[red]Invalid date format. Please use YYYY-MM-DD.[/red]")
            Prompt.ask("Press Enter to continue", default="")
        
        return False
