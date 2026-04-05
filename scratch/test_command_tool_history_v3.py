import json
import httpx
import asyncio
import uuid

def clean_messages_new(messages):
    system_prompts = []
    translated_messages = []
    
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        
        if role == "system":
            if isinstance(content, str):
                system_prompts.append(content)
            elif isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        text_parts.append(block)
                system_prompts.append(" ".join(text_parts))
            continue
            
        if role == "user":
            translated_messages.append({
                "role": "user",
                "content": content if content else ""
            })
            continue
            
        if role == "assistant":
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                content_parts = []
                if isinstance(content, str) and content:
                    content_parts.append({"type": "text", "text": content})
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            content_parts.append({"type": "text", "text": part.get("text", "")})
                
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        func = tc.get("function", {})
                        arguments = func.get("arguments", "{}")
                        if isinstance(arguments, dict):
                            input_data = arguments
                        else:
                            try:
                                input_data = json.loads(arguments)
                            except Exception:
                                input_data = {}
                        
                        content_parts.append({
                            "type": "tool-call",
                            "toolCallId": tc.get("id", ""),
                            "toolName": func.get("name", ""),
                            "input": input_data  # Changed from args to input
                        })
                translated_messages.append({
                    "role": "assistant",
                    "content": content_parts
                })
            else:
                translated_messages.append({
                    "role": "assistant",
                    "content": content if content else ""
                })
            continue
            
        if role == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            tool_name = msg.get("name", "")
            tool_content = content
            
            # Extract plain text string value from tool_content
            text_value = ""
            if isinstance(tool_content, str):
                text_value = tool_content
            elif isinstance(tool_content, list):
                text_parts = []
                for block in tool_content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif block.get("type") == "tool-result":
                            output = block.get("output")
                            if isinstance(output, dict) and "value" in output:
                                text_parts.append(output["value"])
                            else:
                                text_parts.append(str(block.get("result", "")))
                    elif isinstance(block, str):
                        text_parts.append(block)
                text_value = "\n".join(text_parts)
            else:
                text_value = str(tool_content)

            translated_messages.append({
                "role": "tool",
                "content": [{
                    "type": "tool-result",
                    "toolCallId": tool_call_id,
                    "toolName": tool_name,
                    "output": {
                        "type": "text",
                        "value": text_value
                    }
                }]
            })
            continue
            
        # Unknown role
        translated_messages.append(dict(msg))

    # Now group consecutive messages of the same role
    grouped_messages = []
    for msg in translated_messages:
        if not grouped_messages:
            grouped_messages.append(dict(msg))
            continue
        
        last_msg = grouped_messages[-1]
        if last_msg["role"] == msg["role"]:
            # Merge
            if msg["role"] == "tool":
                last_msg["content"] = (last_msg.get("content") or []) + (msg.get("content") or [])
            elif msg["role"] in ("user", "assistant"):
                last_content = last_msg.get("content")
                new_content = msg.get("content")
                
                if isinstance(last_content, str) and isinstance(new_content, str):
                    last_msg["content"] = (last_content + "\n\n" + new_content).strip()
                else:
                    if isinstance(last_content, str):
                        last_blocks = [{"type": "text", "text": last_content}] if last_content else []
                    else:
                        last_blocks = list(last_content) if last_content else []
                        
                    if isinstance(new_content, str):
                        new_blocks = [{"type": "text", "text": new_content}] if new_content else []
                    else:
                        new_blocks = list(new_content) if new_content else []
                        
                    last_msg["content"] = last_blocks + new_blocks
        else:
            grouped_messages.append(dict(msg))

    # Prepend system prompt to the first user message
    if system_prompts:
        system_text = "\n".join(system_prompts)
        user_msg_idx = -1
        for idx, msg in enumerate(grouped_messages):
            if msg.get("role") == "user":
                user_msg_idx = idx
                break
        
        if user_msg_idx != -1:
            orig_content = grouped_messages[user_msg_idx].get("content") or ""
            if isinstance(orig_content, str):
                grouped_messages[user_msg_idx]["content"] = f"System Prompt:\n{system_text}\n\n{orig_content}"
            elif isinstance(orig_content, list):
                grouped_messages[user_msg_idx]["content"] = [
                    {"type": "text", "text": f"System Prompt:\n{system_text}\n\n"}
                ] + orig_content
        else:
            grouped_messages.insert(0, {
                "role": "user",
                "content": f"System Prompt:\n{system_text}"
            })
            
    return grouped_messages

async def main():
    api_key = "user_5FViYmamkbF7nvMF3WBV76vvtSZGn3GeRA5uEYyD7iPgH8CNSsTjxUEEJUTkjmoX2j2ppNHaAyz5BZ7xeDfZPAW"
    
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "x-command-code-version": "0.51.202",
        "x-cli-environment": "production",
        "Authorization": f"Bearer {api_key}"
    }

    # Load failed request messages
    with open("scratch/last_request.json") as f:
        req_json = json.load(f)
    original_messages = req_json["data"]["messages"]
    original_tools = req_json["data"]["tools"]

    from rotator_library.providers.command_provider import CommandProvider
    provider = CommandProvider()
    translated_tools = provider._translate_tools_to_anthropic(original_tools)

    cleaned_messages = clean_messages_new(original_messages)

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
            "model": "Qwen/Qwen3.7-Max",
            "stream": True,
            "max_tokens": 8192,
            "tools": translated_tools
        },
        "threadId": str(uuid.uuid4())
    }

    url = "https://api.commandcode.ai/alpha/generate"
    async with httpx.AsyncClient(timeout=30.0) as client:
        print("Sending full history request to upstream Command Code API...")
        async with client.stream("POST", url, headers=headers, json=payload) as response:
            print("Status Code:", response.status_code)
            body = await response.aread()
            print("Response:", body.decode("utf-8", errors="ignore")[:300])

if __name__ == "__main__":
    asyncio.run(main())
