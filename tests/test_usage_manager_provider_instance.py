"""UsageManager must use RotatingClient's shared provider instance for upstream quota."""

from rotator_library.usage.manager import UsageManager


def test_get_provider_plugin_instance_uses_injected_callback():
    sentinel = object()
    manager = UsageManager(
        provider="umans",
        get_provider_instance=lambda name: sentinel if name == "umans" else None,
    )
    assert manager._get_provider_plugin_instance() is sentinel


def test_get_provider_plugin_instance_falls_back_without_callback():
    manager = UsageManager(provider="umans", provider_plugins={})
    assert manager._get_provider_plugin_instance() is None