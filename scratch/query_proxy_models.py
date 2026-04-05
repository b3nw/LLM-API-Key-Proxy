import httpx
import json
import sys

url = "http://docker.local.ben.io:9220/v1/models"
headers = {
    "Authorization": "Bearer sk-proxy-afeymLqkQRNC6NTdied4Tp9m3vRTpNjO"
}

print(f"Querying running proxy models endpoint: {url}...")
try:
    resp = httpx.get(url, headers=headers, timeout=10.0)
    print(f"Status Code: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json().get("data", [])
        print(f"Total models returned: {len(data)}\n")
        
        command_models = [m for m in data if m.get("id", "").startswith("command/")]
        print(f"=== Command Provider Models ({len(command_models)}) ===")
        for m in sorted(command_models, key=lambda x: x["id"]):
            print(f"- {m['id']}")
            
        other_models = [m for m in data if not m.get("id", "").startswith("command/")]
        if other_models:
            print(f"\n=== Other Provider Models ({len(other_models)}) ===")
            for m in sorted(other_models, key=lambda x: x["id"])[:10]:
                print(f"- {m['id']}")
            if len(other_models) > 10:
                print(f"... and {len(other_models) - 10} more")
    else:
        print(f"Error Response:\n{resp.text}")
except Exception as e:
    print(f"Failed to query endpoint: {e}")
