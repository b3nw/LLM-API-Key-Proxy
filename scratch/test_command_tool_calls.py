import os
import sys
import json
import uuid
import datetime
import httpx
from pathlib import Path
from dotenv import load_dotenv

# Load env variables from root .env
load_dotenv(Path(__file__).parent.parent / ".env")

api_key = os.getenv("COMMAND_API_KEY_1")
if not api_key:
    print("Error: COMMAND_API_KEY_1 not found in .env")
    sys.exit(1)

print(f"Using API Key: {api_key[:15]}...")

headers = {
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36",
    "x-command-code-version": "0.31.0",
    "x-cli-environment": "production",
    "Authorization": f"Bearer {api_key}"
}

# Anthropic style tool schema
tools = [
    {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"]
                }
            },
            "required": ["location"]
        }
    }
]

payload = {
    "config": {
        "editor": "vscode",
        "shell": "bash",
        "version": "0.31.0",
        "workingDir": "/home/b3nw/projects/core/LLM-API-Key-Proxy",
        "date": datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z"),
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
        "messages": [
            {
                "role": "user",
                "content": "What is the weather in Paris, France right now? Use the get_current_weather tool to find out."
            }
        ],
        "model": "deepseek/deepseek-v4-pro",
        "stream": True,
        "max_tokens": 4096,
        "tools": tools,
        "tool_choice": {"type": "auto"}
    },
    "threadId": str(uuid.uuid4())
}

url = "https://api.commandcode.ai/alpha/generate"

async def main():
    async with httpx.AsyncClient(timeout=30.0) as client:
        print("Sending request to alpha/generate...")
        async with client.stream("POST", url, headers=headers, json=payload) as response:
            print(f"Status Code: {response.status_code}")
            if response.status_code >= 400:
                body = await response.aread()
                print(f"Error: {body.decode('utf-8')}")
                return

            print("--- Event Stream Output ---")
            async for line in response.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                print(line)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
