import json
from rotator_library.providers.command_provider import CommandProvider

import subprocess
cmd = "ssh docker-test 'cat /opt/llm-proxy/logs/transactions/0601_214359_oai_command_deepseek-v4-pro_549e7e8b/request.json'"
res = subprocess.check_output(cmd, shell=True).decode("utf-8")
req_json = json.loads(res)
messages = req_json["data"]["messages"]

provider = CommandProvider()
cleaned = provider._clean_messages(messages)

for idx, m in enumerate(cleaned):
    print("--- Msg %d ---" % idx)
    print("role:", m["role"])
    content = m["content"]
    if isinstance(content, list):
        print("content (list):")
        for b in content:
            # print block type and some details
            print("  block type:", b.get("type"))
            if b.get("type") == "tool-call":
                print("    toolCallId:", b.get("toolCallId"))
                print("    toolName:", b.get("toolName"))
                print("    args:", b.get("args"))
            elif b.get("type") == "tool-result":
                print("    toolCallId:", b.get("toolCallId"))
                print("    toolName:", b.get("toolName"))
                print("    result keys/type:", type(b.get("result")), str(b.get("result"))[:100])
            else:
                print("    text len:", len(b.get("text", "")))
    else:
        print("content (str):", len(content), repr(content[:100]))
