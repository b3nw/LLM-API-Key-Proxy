import httpx
import json

api_key = "user_5FViYmamkbF7nvMF3WBV76vvtSZGn3GeRA5uEYyD7iPgH8CNSsTjxUEEJUTkjmoX2j2ppNHaAyz5BZ7xeDfZPAW"
url = "https://api.commandcode.ai/alpha/billing/credits"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36",
    "x-command-code-version": "0.30.2",
    "x-cli-environment": "production",
    "Authorization": f"Bearer {api_key}"
}

print(f"Querying Command Code credits API: {url}...")
try:
    resp = httpx.get(url, headers=headers, timeout=10.0)
    print(f"Status Code: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        print("Raw JSON Response:")
        print(json.dumps(data, indent=2))
    else:
        print(f"Error Response:\n{resp.text}")
except Exception as e:
    print(f"Failed to query credits API: {e}")
