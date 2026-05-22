"""Admin API for OAuth provider information and status."""

from fastapi import APIRouter

from rotator_library.provider_factory import get_available_providers

router = APIRouter(prefix="/v1/admin", tags=["admin-oauth"])


@router.get("/oauth/providers")
async def list_oauth_providers():
    """List available OAuth providers and their setup requirements."""
    providers = get_available_providers()

    provider_info = {
        "gemini_cli": {
            "name": "Gemini CLI",
            "flow": "device_code",
            "description": "Google Gemini via CLI OAuth. Requires browser sign-in.",
            "setup_command": "python src/proxy_app/main.py --add-credential",
        },
        "codex": {
            "name": "Codex (OpenAI)",
            "flow": "device_code",
            "description": "OpenAI Codex via OAuth. Requires browser sign-in.",
            "setup_command": "python src/proxy_app/main.py --add-credential",
        },
        "anthropic": {
            "name": "Anthropic",
            "flow": "device_code",
            "description": "Anthropic via OAuth. Requires browser sign-in.",
            "setup_command": "python src/proxy_app/main.py --add-credential",
        },
        "copilot": {
            "name": "GitHub Copilot",
            "flow": "device_code",
            "description": "GitHub Copilot via device OAuth flow.",
            "setup_command": "python src/proxy_app/main.py --add-credential",
        },
    }

    result = []
    for p in providers:
        info = provider_info.get(p, {
            "name": p,
            "flow": "unknown",
            "description": f"OAuth provider: {p}",
            "setup_command": "python src/proxy_app/main.py --add-credential",
        })
        info["provider_id"] = p
        result.append(info)

    return {
        "providers": result,
        "note": "OAuth credential setup requires interactive terminal access. "
                "Use 'docker exec -it <container> python src/proxy_app/main.py --add-credential' "
                "or run the credential tool locally.",
    }
