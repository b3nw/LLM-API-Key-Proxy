import os
import sys
from pathlib import Path

# Add src/ to path so we can import rotator_library
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rotator_library.client.models import ModelResolver
from rotator_library.providers import PROVIDER_PLUGINS

# Mock environment variables for test
os.environ["IGNORE_MODELS_COMMAND"] = "*"
os.environ["WHITELIST_MODELS_COMMAND"] = "*deepseek*,*Qwen*,*qwen*,*Kimi*,*kimi*,*GLM*,*glm*,*MiniMax*,*minimax*,*Step*,*step*,*mimo*"

# Parse ignore list
ignore_models = {}
for key, value in os.environ.items():
    if key.startswith("IGNORE_MODELS_"):
        provider = key.replace("IGNORE_MODELS_", "").lower()
        ignore_models[provider] = [m.strip() for m in value.split(",") if m.strip()]

# Parse whitelist
whitelist_models = {}
for key, value in os.environ.items():
    if key.startswith("WHITELIST_MODELS_"):
        provider = key.replace("WHITELIST_MODELS_", "").lower()
        whitelist_models[provider] = [m.strip() for m in value.split(",") if m.strip()]

# Instantiate ModelResolver
resolver = ModelResolver(
    provider_plugins=PROVIDER_PLUGINS,
    model_definitions=None,
    ignore_models=ignore_models,
    whitelist_models=whitelist_models
)

# List of models to test
test_models = [
    # Should be whitelisted (open-source)
    "command/deepseek-v4-pro",
    "command/deepseek-v4-flash",
    "command/qwen-3.7-max",
    "command/qwen-3.6-plus",
    "command/qwen-3.6-max-preview",
    "command/kimi-k2.6",
    "command/kimi-k2.5",
    "command/glm-5.1",
    "command/glm-5",
    "command/minimax-m3",
    "command/minimax-m2.7",
    "command/minimax-m2.5",
    "command/step-3.7-flash",
    "command/step-3.5-flash",
    "command/mimo-v2.5-pro",
    "command/mimo-v2.5",
    
    # Should be blocked (commercial/closed-source or not matching whitelist)
    "command/gemini-3.5-flash",
    "command/gemini-3.1-flash-lite",
    "command/claude-3-5-sonnet",
    "command/gpt-4o"
]

print("=== Command Code Model Filter Verification ===")
print(f"IGNORE_MODELS_COMMAND: {os.environ['IGNORE_MODELS_COMMAND']}")
print(f"WHITELIST_MODELS_COMMAND: {os.environ['WHITELIST_MODELS_COMMAND']}\n")

passed = True
for model in test_models:
    allowed = resolver.is_model_allowed(model, "command")
    is_commercial = "gemini" in model or "claude" in model or "gpt" in model
    expected = not is_commercial
    status = "Allowed" if allowed else "Blocked"
    expected_status = "Allowed" if expected else "Blocked"
    
    if allowed == expected:
        print(f"✅ {model}: {status} (Expected: {expected_status})")
    else:
        print(f"❌ {model}: {status} (Expected: {expected_status})")
        passed = False

if passed:
    print("\n🎉 ALL TESTS PASSED! Filtering works 100% correctly and respects whitelists/blacklists statically.")
else:
    print("\n❌ SOME TESTS FAILED. Filtering logic is not matching expected behavior.")
