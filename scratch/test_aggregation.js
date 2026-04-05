const httpx = require("http");

// Simple helper to fetch API json
function apiGet(url) {
  return new Promise((resolve, reject) => {
    httpx.get(url, {
      headers: { "Authorization": "Bearer sk-proxy-afeymLqkQRNC6NTdied4Tp9m3vRTpNjO" }
    }, (res) => {
      let data = "";
      res.on("data", (chunk) => data += chunk);
      res.on("end", () => {
        try {
          resolve(JSON.parse(data));
        } catch (e) {
          reject(e);
        }
      });
    }).on("error", reject);
  });
}

async function run() {
  const urlBase = "http://docker.local.ben.io:9220";
  const txs = [];
  let page = 1;
  const maxPages = 15;
  let keepFetching = true;
  const cutoffTime = Date.now() - 24 * 60 * 60 * 1000;

  console.log("Fetching pages...");
  while (keepFetching && page <= maxPages) {
    try {
      const data = await apiGet(`${urlBase}/v1/admin/transactions?page_size=100&page=${page}`);
      const pageTxs = data?.transactions || [];
      if (pageTxs.length === 0) {
        break;
      }
      txs.push(...pageTxs);

      const oldestTx = pageTxs[pageTxs.length - 1];
      if (oldestTx && oldestTx.timestamp) {
        const oldestTxTimeStr = oldestTx.timestamp.endsWith("Z") ? oldestTx.timestamp : oldestTx.timestamp + "Z";
        const oldestTxTime = new Date(oldestTxTimeStr).getTime();
        console.log(`Page ${page} oldest raw: ${oldestTx.timestamp} -> parsed: ${new Date(oldestTxTimeStr).toISOString()} (ms: ${oldestTxTime}), cutoff: ${new Date(cutoffTime).toISOString()}`);
        if (oldestTxTime < cutoffTime) {
          console.log(`Stopping fetch because oldest ${oldestTxTimeStr} is older than cutoff`);
          keepFetching = false;
        }
      }
      page++;
    } catch (e) {
      console.error("Error fetching", e);
      break;
    }
  }

  console.log(`Fetched ${txs.length} transactions total.`);

  // Group transactions into 24 one-hour buckets in UTC
  const buckets = {};
  const nowMs = Date.now();
  const oneHourMs = 60 * 60 * 1000;

  for (let i = 23; i >= 0; i--) {
    const bucketTime = new Date(nowMs - i * oneHourMs);
    const bucketStartHour = Date.UTC(
      bucketTime.getUTCFullYear(),
      bucketTime.getUTCMonth(),
      bucketTime.getUTCDate(),
      bucketTime.getUTCHours()
    );
    const label = new Date(bucketStartHour).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    });
    buckets[bucketStartHour] = {
      timeLabel: label,
      timestamp: bucketStartHour / 1000,
      count: 0,
    };
  }

  let matched = 0;
  let discarded = 0;
  txs.forEach((tx) => {
    if (!tx.timestamp) return;
    const txTimeStr = tx.timestamp.endsWith("Z") ? tx.timestamp : tx.timestamp + "Z";
    const txDate = new Date(txTimeStr);
    const txHourStart = Date.UTC(
      txDate.getUTCFullYear(),
      txDate.getUTCMonth(),
      txDate.getUTCDate(),
      txDate.getUTCHours()
    );

    const bucket = buckets[txHourStart];
    if (bucket) {
      bucket.count++;
      matched++;
    } else {
      discarded++;
    }
  });

  console.log(`Matched to buckets: ${matched}, discarded (outside last 24h buckets): ${discarded}`);
  console.log("\nBucket Counts:");
  Object.keys(buckets).sort().forEach((k) => {
    console.log(`  ${new Date(Number(k)).toISOString()} (${buckets[k].timeLabel}): ${buckets[k].count}`);
  });
}

run();
