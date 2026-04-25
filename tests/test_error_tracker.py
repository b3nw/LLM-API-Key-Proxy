import threading
import pytest
import importlib.util
import sys

# Load error_tracker bypassing package-level imports
spec = importlib.util.spec_from_file_location("error_tracker", "src/rotator_library/error_tracker.py")
error_tracker_module = importlib.util.module_from_spec(spec)
sys.modules["error_tracker"] = error_tracker_module
spec.loader.exec_module(error_tracker_module)

ErrorTracker = error_tracker_module.ErrorTracker
get_error_tracker = error_tracker_module.get_error_tracker

@pytest.fixture(autouse=True)
def reset_error_tracker():
    """Reset the global _error_tracker to None before and after each test."""
    original_tracker = getattr(error_tracker_module, "_error_tracker", None)
    error_tracker_module._error_tracker = None
    yield
    error_tracker_module._error_tracker = original_tracker

def test_get_error_tracker_returns_instance():
    """Verify that get_error_tracker returns an ErrorTracker instance."""
    tracker = get_error_tracker()
    assert isinstance(tracker, ErrorTracker)

def test_get_error_tracker_singleton():
    """Verify that get_error_tracker returns the same instance on multiple calls."""
    tracker1 = get_error_tracker()
    tracker2 = get_error_tracker()
    assert tracker1 is tracker2

def test_get_error_tracker_thread_safe_initialization():
    """Verify thread-safe lazy initialization of the error tracker singleton."""
    num_threads = 10
    barrier = threading.Barrier(num_threads)
    results = []

    def worker():
        barrier.wait()
        results.append(get_error_tracker())

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    assert len(results) == num_threads

    # All threads should have received the exact same instance
    first_instance = results[0]
    for instance in results[1:]:
        assert instance is first_instance
