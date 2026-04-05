import json
import httpx
import asyncio
from rotator_library.providers.command_provider import CommandProvider

async def main():
    # Load cookies & session token
    provider = CommandProvider()
    cookies_list, session_token, user_agent = provider._load_session_credentials()

    headers = {
        "Content-Type": "application/json",
        "User-Agent": user_agent or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "x-command-code-version": "0.51.202",
        "x-cli-environment": "production",
    }
    if session_token:
        headers["Authorization"] = f"Bearer {session_token}"
    if cookies_list:
        headers["Cookie"] = "; ".join(cookies_list)

    # Load failed request messages
    import subprocess
    cmd = "ssh docker-test 'cat /opt/llm-proxy/logs/transactions/0601_214359_oai_command_deepseek-v4-pro_549e7e8b/request.json'"
    res = subprocess.check_output(cmd, shell=True).decode("utf-8")
    req_json = json.loads(res)
    original_messages = req_json["data"]["messages"]

    cleaned_messages = provider._clean_messages(original_messages)

    payload = {
        "config": {
            "editor": "vscode",
            "shell": "bash",
            "version": "0.51.202",
            "workingDir": "/home/b3nw/projects/core/LLM-API-Key-Proxy",
            "date": "2026-06-01T21:43:59.000Z",
            "environment": "production",
            "structure": [],
            "isGitRepo": False,
            "currentBranch": "",
            "mainBranch": "",
            "gitStatus": "",
            "recentCommits": []
        },
        "memory": "",
        "taste": "",
        "skills": "",
        "params": {
            "messages": cleaned_messages,
            "model": "deepseek-v4-pro",
            "stream": True,
            "max_tokens": 8192
        },
        "threadId": "test-thread-id"
    }

    url = "https://api.commandcode.ai/alpha/generate"
    async with httpx.AsyncClient(timeout=30.0) as client:
        print("Sending request to upstream Command Code API...")
        async with client.stream("POST", url, headers=headers, json=payload) as response:
            print("Status Code:", response.status_code)
            body = await response.aread()
            print("Response:", body.decode("utf-8", errors="ignore"))

if __name__ == "__main__":
    asyncio.run(main())
