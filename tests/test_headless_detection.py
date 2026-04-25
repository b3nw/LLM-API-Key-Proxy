import os
import sys
import pytest
from unittest.mock import patch

from rotator_library.utils.headless_detection import is_headless_environment


@pytest.fixture
def clean_env(monkeypatch):
    """Fixture to ensure a clean environment for each test."""
    # Clear variables we care about
    for var in [
        "DISPLAY", "SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY", "SESSIONNAME",
        "CI", "GITHUB_ACTIONS", "GITLAB_CI", "JENKINS_URL", "CIRCLECI",
        "TRAVIS", "BUILDKITE", "DRONE", "TEAMCITY_VERSION", "TF_BUILD", "CODEBUILD_BUILD_ID"
    ]:
        monkeypatch.delenv(var, raising=False)

    # Default mock: not a container
    with patch("os.path.exists", return_value=False):
        yield monkeypatch


def test_linux_gui(clean_env):
    """Test Linux with DISPLAY set (GUI environment)."""
    with patch("os.name", "posix"), patch("sys.platform", "linux"):
        clean_env.setenv("DISPLAY", ":0.0")
        assert is_headless_environment() is False


def test_linux_headless_no_display(clean_env):
    """Test Linux without DISPLAY set (headless)."""
    with patch("os.name", "posix"), patch("sys.platform", "linux"):
        assert is_headless_environment() is True


def test_linux_headless_empty_display(clean_env):
    """Test Linux with empty DISPLAY set (headless)."""
    with patch("os.name", "posix"), patch("sys.platform", "linux"):
        clean_env.setenv("DISPLAY", "")
        assert is_headless_environment() is True


def test_mac_gui(clean_env):
    """Test macOS ignores DISPLAY and returns False (GUI)."""
    with patch("os.name", "posix"), patch("sys.platform", "darwin"):
        assert is_headless_environment() is False


def test_windows_gui(clean_env):
    """Test Windows ignores DISPLAY and returns False (GUI)."""
    with patch("os.name", "nt"), patch("sys.platform", "win32"):
        assert is_headless_environment() is False


def test_windows_headless_services(clean_env):
    """Test Windows with SESSIONNAME=services (headless)."""
    with patch("os.name", "nt"), patch("sys.platform", "win32"):
        clean_env.setenv("SESSIONNAME", "services")
        assert is_headless_environment() is True


def test_windows_headless_rdp(clean_env):
    """Test Windows with SESSIONNAME=rdp-tcp (headless)."""
    with patch("os.name", "nt"), patch("sys.platform", "win32"):
        clean_env.setenv("SESSIONNAME", "rdp-tcp")
        assert is_headless_environment() is True


def test_windows_headless_caps_sessionname(clean_env):
    """Test Windows with capitalized SESSIONNAME=Services (headless)."""
    with patch("os.name", "nt"), patch("sys.platform", "win32"):
        clean_env.setenv("SESSIONNAME", "Services")
        assert is_headless_environment() is True


@pytest.mark.parametrize("ssh_var", ["SSH_CONNECTION", "SSH_CLIENT", "SSH_TTY"])
def test_ssh_detection(clean_env, ssh_var):
    """Test SSH connection detection."""
    with patch("os.name", "posix"), patch("sys.platform", "darwin"):  # use darwin so DISPLAY check doesn't trigger
        clean_env.setenv(ssh_var, "1")
        assert is_headless_environment() is True


@pytest.mark.parametrize("ci_var", [
    "CI", "GITHUB_ACTIONS", "GITLAB_CI", "JENKINS_URL", "CIRCLECI",
    "TRAVIS", "BUILDKITE", "DRONE", "TEAMCITY_VERSION", "TF_BUILD", "CODEBUILD_BUILD_ID"
])
def test_ci_environments(clean_env, ci_var):
    """Test CI environment detection."""
    with patch("os.name", "posix"), patch("sys.platform", "darwin"):
        clean_env.setenv(ci_var, "true")
        assert is_headless_environment() is True


@pytest.mark.parametrize("container_path", ["/.dockerenv", "/run/.containerenv"])
def test_container_detection(clean_env, container_path):
    """Test container environment detection."""
    def mock_exists(path):
        return path == container_path

    with patch("os.name", "posix"), patch("sys.platform", "darwin"):
        with patch("os.path.exists", side_effect=mock_exists):
            assert is_headless_environment() is True
