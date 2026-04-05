import asyncio
import json
import httpx

url = "http://docker.local.ben.io:9220/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer sk-proxy-afeymLqkQRNC6NTdied4Tp9m3vRTpNjO"
}

payload = {
    "model": "codex/gpt-5.3-codex",
    "messages": [
        {"role": "user", "content": "Write a short poem about antigravity in exactly 3 lines."}
    ],
    "stream": True,
    "max_tokens": 15
}

async def main():
    async with httpx.AsyncClient(timeout=30.0) as client:
        print(f"Sending request to local llm-proxy for model {payload['model']} with max_tokens={payload['max_tokens']}...")
        async with client.stream("POST", url, headers=headers, json=payload) as response:
            print(f"Status Code: {response.status_code}")
            if response.status_code >= 400:
                body = await response.aread()
                print(f"Error: {body.decode('utf-8')}")
                return

            print("--- Received SSE Chunks from Proxy ---")
            text_received = ""
            async for line in response.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("data:"):
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        print("\n[DONE]")
                        continue
                    try:
                        chunk = json.loads(data_str)
                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            if "content" in delta and delta["content"]:
                                content = delta["content"]
                                text_received += content
                                print(content, end="", flush=True)
                    except Exception as e:
                        print(f"\nFailed to parse line: {line} - Error: {e}")
            print(f"\nTotal characters received: {len(text_received)}")

if __name__ == "__main__":
    asyncio.run(main())
