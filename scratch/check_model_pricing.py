import httpx
import json

url = "http://docker.local.ben.io:9220/v1/models"
headers = {
    "Authorization": "Bearer sk-proxy-afeymLqkQRNC6NTdied4Tp9m3vRTpNjO"
}

try:
    resp = httpx.get(url, headers=headers, timeout=10.0)
    if resp.status_code == 200:
        models = resp.json().get("data", [])
        print(f"Fetched {len(models)} models.")
        # Print first few models with non-zero pricing, or just some model structures
        printed = 0
        for m in models:
            # Look for pricing or cost keys
            keys = list(m.keys())
            pricing_keys = [k for k in keys if "cost" in k.lower() or "price" in k.lower() or "pricing" in k.lower()]
            if pricing_keys or printed < 5:
                print(f"Model ID: {m.get('id')}")
                for pk in pricing_keys:
                    print(f"  {pk}: {m[pk]}")
                # print other keys related to cost
                for k in ["input_cost_per_token", "output_cost_per_token", "pricing"]:
                    if k in m:
                        print(f"  {k}: {m[k]}")
                printed += 1
                if printed >= 10:
                    break
    else:
        print(f"Error {resp.status_code}: {resp.text}")
except Exception as e:
    print(f"Failed: {e}")
