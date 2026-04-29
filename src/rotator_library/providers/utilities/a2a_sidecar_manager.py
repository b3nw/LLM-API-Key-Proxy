# SPDX-License-Identifier: LGPL-3.0-only
# Manages the A2A sidecar container lifecycle and credential rotation.

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import List, Optional

lib_logger = logging.getLogger("rotator_library")


class A2ASidecarManager:
    """
    Manages the A2A sidecar container and credential rotation.

    In production (sidecar mode), the A2A server runs as a separate Docker container.
    Credential rotation requires restarting the sidecar with a new
    GOOGLE_APPLICATION_CREDENTIALS environment.

    In development (local mode), the A2A server can be started as a local subprocess.

    The manager supports two backends:
        - "sidecar": Container is managed externally via docker compose.
          Rotation updates an env file and restarts the container.
        - "local": A2A server runs as a child node process (for dev/testing).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        credential_paths: Optional[List[str]] = None,
        backend: str = "local",
    ):
        """
        Args:
            base_url: URL where the A2A server is accessible.
            credential_paths: Ordered list of OAuth credential file paths.
            backend: "sidecar" for Docker compose or "local" for subprocess.
        """
        self.base_url = base_url.rstrip("/")
        self._credential_paths = credential_paths or []
        self._current_index = 0
        self._backend = backend
        self._healthy = False

        # Local subprocess management
        self._process: Optional[asyncio.subprocess.Process] = None
        self._port = int(os.getenv("A2A_PORT", "8080"))

        # Sidecar Docker config
        self._compose_project = os.getenv("A2A_COMPOSE_PROJECT", "llm-proxy-new")
        self._compose_service = os.getenv("A2A_COMPOSE_SERVICE", "a2a-server")
        self._compose_dir = os.getenv("A2A_COMPOSE_DIR", "/opt/llm-proxy/env")

    @property
    def current_credential(self) -> Optional[str]:
        """Path to the currently active credential file."""
        if not self._credential_paths:
            return None
        return self._credential_paths[self._current_index % len(self._credential_paths)]

    @property
    def credential_count(self) -> int:
        """Total number of available credentials."""
        return len(self._credential_paths)

    def set_credentials(self, paths: List[str]):
        """Set the list of available credential paths."""
        self._credential_paths = paths
        self._current_index = 0

    async def start(self) -> bool:
        """
        Start the A2A server (local backend only).

        For sidecar backend, the container is assumed to be already running
        via docker compose. Call wait_for_ready() instead.

        Returns:
            True if server started successfully.
        """
        if self._backend == "sidecar":
            lib_logger.info("[A2A Sidecar] Sidecar mode - container managed externally")
            return await self.wait_for_ready(timeout=30.0)

        if not self._credential_paths:
            lib_logger.error("[A2A Local] No credential paths configured")
            return False

        await self._start_local_process()
        return await self.wait_for_ready(timeout=15.0)

    async def _start_local_process(self):
        """Start the A2A server as a local node subprocess."""
        if self._process and self._process.returncode is None:
            lib_logger.warning("[A2A Local] Process already running, stopping first")
            await self.stop()

        credential_path = self.current_credential
        if not credential_path:
            raise RuntimeError("No credential path available")

        # Resolve the credential path to absolute
        abs_cred_path = str(Path(credential_path).resolve())

        env = {
            **os.environ,
            "GOOGLE_APPLICATION_CREDENTIALS": abs_cred_path,
            "USE_CCPA": "true",
            "GEMINI_CLI_USE_COMPUTE_ADC": "true",
            "CODER_AGENT_PORT": str(self._port),
            "NODE_ENV": "production",
        }

        lib_logger.info(
            f"[A2A Local] Starting A2A server on port {self._port} "
            f"with credential: {Path(credential_path).name}"
        )

        self._process = await asyncio.create_subprocess_exec(
            "node",
            "-e",
            "import('@google/gemini-cli-a2a-server').then(m => m.main())",
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # Start background tasks to drain stdout/stderr
        asyncio.create_task(self._drain_output(self._process.stdout, "stdout"))
        asyncio.create_task(self._drain_output(self._process.stderr, "stderr"))

    async def _drain_output(self, stream, name: str):
        """Drain subprocess output and log it."""
        try:
            async for line in stream:
                text = line.decode("utf-8", errors="replace").strip()
                if text:
                    lib_logger.debug(f"[A2A {name}] {text}")
        except Exception:
            pass

    async def stop(self):
        """Stop the A2A server."""
        if self._backend == "local" and self._process:
            if self._process.returncode is None:
                lib_logger.info("[A2A Local] Stopping A2A server process")
                try:
                    self._process.terminate()
                    try:
                        await asyncio.wait_for(self._process.wait(), timeout=5.0)
                    except asyncio.TimeoutError:
                        lib_logger.warning("[A2A Local] Process didn't stop, killing")
                        self._process.kill()
                        await self._process.wait()
                except ProcessLookupError:
                    pass
            self._process = None

        self._healthy = False

    async def rotate_credential(self) -> str:
        """
        Rotate to the next credential and restart the server.

        Returns:
            The new credential path being used.

        Raises:
            RuntimeError: If no credentials available or restart fails.
        """
        if not self._credential_paths:
            raise RuntimeError("No credential paths configured for rotation")

        old_index = self._current_index
        self._current_index = (self._current_index + 1) % len(self._credential_paths)
        new_credential = self.current_credential

        lib_logger.info(
            f"[A2A] Rotating credential: index {old_index} → {self._current_index} "
            f"({Path(new_credential).name})"
        )

        self._healthy = False

        if self._backend == "local":
            await self.stop()
            await self._start_local_process()
        elif self._backend == "sidecar":
            await self._restart_sidecar(new_credential)

        if not await self.wait_for_ready(timeout=30.0):
            raise RuntimeError(
                f"A2A server failed to become ready after credential rotation to {new_credential}"
            )

        return new_credential

    async def _restart_sidecar(self, credential_path: str):
        """
        Restart the sidecar container with a new credential.

        Updates the GOOGLE_APPLICATION_CREDENTIALS env var and restarts.
        """
        abs_cred_path = str(Path(credential_path).resolve())

        # For sidecar mode, we need to update the env and restart the container
        # This uses docker compose to recreate with new environment
        env = {
            **os.environ,
            "GOOGLE_APPLICATION_CREDENTIALS": abs_cred_path,
        }

        lib_logger.info(f"[A2A Sidecar] Restarting container with new credential")
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "compose", "-p", self._compose_project, "restart", self._compose_service,
                cwd=self._compose_dir,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30.0)

            if proc.returncode != 0:
                err = stderr.decode("utf-8", errors="replace")
                lib_logger.error(f"[A2A Sidecar] Restart failed: {err}")
                raise RuntimeError(f"Sidecar restart failed: {err}")

        except asyncio.TimeoutError:
            lib_logger.error("[A2A Sidecar] Restart timed out after 30s")
            raise RuntimeError("Sidecar restart timed out")

    async def wait_for_ready(self, timeout: float = 30.0) -> bool:
        """
        Wait for the A2A server to become healthy.

        Polls the agent card endpoint until the server responds.

        Args:
            timeout: Maximum seconds to wait.

        Returns:
            True if server became ready within timeout.
        """
        import httpx

        start = time.time()
        poll_interval = 0.5

        while (time.time() - start) < timeout:
            try:
                async with httpx.AsyncClient() as client:
                    resp = await client.get(
                        f"{self.base_url}/.well-known/agent-card.json",
                        timeout=3.0,
                    )
                    if resp.status_code == 200:
                        self._healthy = True
                        elapsed = time.time() - start
                        lib_logger.info(
                            f"[A2A] Server ready after {elapsed:.1f}s"
                        )
                        return True
            except Exception:
                pass

            await asyncio.sleep(poll_interval)

        lib_logger.error(f"[A2A] Server not ready after {timeout}s")
        return False

    @property
    def is_healthy(self) -> bool:
        """Whether the server is known to be healthy."""
        return self._healthy

    async def health_check(self) -> bool:
        """Perform an active health check."""
        import httpx

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"{self.base_url}/.well-known/agent-card.json",
                    timeout=5.0,
                )
                self._healthy = resp.status_code == 200
                return self._healthy
        except Exception:
            self._healthy = False
            return False
