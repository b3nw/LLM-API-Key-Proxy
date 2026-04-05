import httpx
import json

url = "http://docker.local.ben.io:9220/v1/quota-stats?provider=command"
headers = {
    "Authorization": "Bearer sk-proxy-afeymLqkQRNC6NTdied4Tp9m3vRTpNjO"
}

print(f"Querying running proxy quota stats: {url}...")
try:
    resp = httpx.get(url, headers=headers, timeout=10.0)
    print(f"Status Code: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print("Raw Quota Stats Response:")
        print(json.dumps(data, indent=2))
    else:
        print(f"Error Response:\n{resp.text}")
except Exception as e:
    print(f"Failed to query quota stats: {e}")
