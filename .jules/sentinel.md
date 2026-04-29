## 2025-05-18 - Prevent Command Injection in A2A Sidecar Manager
**Vulnerability:** Command injection was possible in `_restart_sidecar` within `A2ASidecarManager` because `asyncio.create_subprocess_shell` was used to execute a shell command constructed with unsanitized environment variables (e.g., `A2A_COMPOSE_PROJECT`).
**Learning:** Even internal management operations that seem safe because they use configuration values are vulnerable if those values originate from user-controlled inputs like environment variables.
**Prevention:** Avoid shell=True or shell-based command execution (`create_subprocess_shell`). Use the array-based execution (`create_subprocess_exec`) which passes arguments directly to the program without shell interpretation.
