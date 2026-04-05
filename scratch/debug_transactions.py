import httpx
import json
from datetime import datetime

url_base = "http://docker.local.ben.io:9220"
headers = {
    "Authorization": "Bearer sk-proxy-afeymLqkQRNC6NTdied4Tp9m3vRTpNjO"
}

print("Querying transactions...")
all_txs = []
page = 1
while page <= 15:
    url = f"{url_base}/v1/admin/transactions?page_size=100&page={page}"
    try:
        resp = httpx.get(url, headers=headers, timeout=10.0)
        if resp.status_code != 200:
            print(f"Page {page} returned status {resp.status_code}")
            break
        data = resp.json()
        txs = data.get("transactions", [])
        if not txs:
            print(f"Page {page} is empty.")
            break
        print(f"Page {page}: fetched {len(txs)} transactions. Newest: {txs[0]['timestamp']}, Oldest: {txs[-1]['timestamp']}")
        all_txs.extend(txs)
        page += 1
    except Exception as e:
        print(f"Error on page {page}: {e}")
        break

print(f"Total transactions fetched: {len(all_txs)}")
if all_txs:
    print(f"Newest overall: {all_txs[0]['timestamp']}")
    print(f"Oldest overall: {all_txs[-1]['timestamp']}")

# Count by hour
hourly_counts = {}
for tx in all_txs:
    ts_str = tx["timestamp"]
    # Extract hour
    dt = datetime.fromisoformat(ts_str)
    hour_key = dt.strftime("%Y-%m-%d %H:00")
    hourly_counts[hour_key] = hourly_counts.get(hour_key, 0) + 1

print("\nTransactions by hour from API:")
for hk in sorted(hourly_counts.keys()):
    print(f"  {hk}: {hourly_counts[hk]} transactions")
