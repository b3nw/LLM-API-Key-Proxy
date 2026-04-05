import os
import sys
import json
import httpx

# We will query our live local proxy running on port 9220
url = "http://docker.local.ben.io:9220/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer sk-proxy-afeymLqkQRNC6NTdied4Tp9m3vRTpNjO"
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

# Load the real failing history
with open("scratch/last_request.json") as f:
    last_req = json.load(f)
req_data = last_req["data"]

payload = {
    "model": "command/deepseek-v4-pro",
    "messages": req_data["messages"],
    "stream": True,
    "tools": req_data["tools"]
}

async def main():
    async with httpx.AsyncClient(timeout=30.0) as client:
        print("Sending request to local llm-proxy...")
        async with client.stream("POST", url, headers=headers, json=payload) as response:
            print(f"Status Code: {response.status_code}")
            if response.status_code >= 400:
                body = await response.aread()
                print(f"Error: {body.decode('utf-8')}")
                return

            print("--- Received SSE Chunks from Proxy ---")
            async for line in response.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("data:"):
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        print("[DONE]")
                        continue
                    try:
                        chunk = json.loads(data_str)
                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            if "tool_calls" in delta:
                                print(f"Tool Call Chunk: {json.dumps(delta['tool_calls'])}")
                            elif "content" in delta and delta["content"]:
                                print(f"Text Content Chunk: {delta['content']}", end="", flush=True)
                            elif "reasoning_content" in delta and delta["reasoning_content"]:
                                print(f"Reasoning Chunk: {delta['reasoning_content']}", end="", flush=True)
                    except Exception as e:
                        print(f"\nFailed to parse line: {line} - Error: {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
