import json
import httpx
import asyncio
import uuid

async def test_payload(messages_payload, tools_payload, label):
    api_key = "user_5FViYmamkbF7nvMF3WBV76vvtSZGn3GeRA5uEYyD7iPgH8CNSsTjxUEEJUTkjmoX2j2ppNHaAyz5BZ7xeDfZPAW"
    
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "x-command-code-version": "0.51.202",
        "x-cli-environment": "production",
        "Authorization": f"Bearer {api_key}"
    }

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
            "messages": messages_payload,
            "model": "deepseek/deepseek-v4-pro",
            "stream": True,
            "max_tokens": 8192
        },
        "threadId": str(uuid.uuid4())
    }
    
    if tools_payload is not None:
        payload["params"]["tools"] = tools_payload

    url = "https://api.commandcode.ai/alpha/generate"
    async with httpx.AsyncClient(timeout=30.0) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as response:
            body = await response.aread()
            resp_str = body.decode("utf-8", errors="ignore")
            print(f"[{label}] Status: {response.status_code}, Response: {resp_str[:300]}")

async def main():
    tools = [
        {
            "name": "Shell",
            "description": "Run a shell command",
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"}
                },
                "required": ["command"]
            }
        }
    ]

    base_messages = [
        {
            "role": "user",
            "content": "System Prompt:\nYou are an AI coding assistant, powered by command/deepseek-v4-pro.\n\nHello! Please run a shell command to list files."
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool-call",
                    "toolCallId": "call_abc123",
                    "toolName": "Shell",
                    "input": {"command": "ls"}
                }
            ]
        }
    ]

    # Test 22: tool-call with input inside assistant message, tool-result with output in tool message
    t22 = base_messages + [
        {
            "role": "tool",
            "content": [
                {
                    "type": "tool-result",
                    "toolCallId": "call_abc123",
                    "toolName": "Shell",
                    "output": {
                        "type": "text",
                        "value": "file1.py\nfile2.py"
                    }
                }
            ]
        }
    ]
    await test_payload(t22, tools, "Test 22: tool-call with input + tool-result with output")

if __name__ == "__main__":
    asyncio.run(main())
