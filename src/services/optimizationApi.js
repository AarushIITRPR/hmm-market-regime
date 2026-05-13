const API_BASE_URL =
  window.__APP_CONFIG__?.API_BASE_URL ??
  import.meta.env.VITE_API_BASE_URL ??
  "http://127.0.0.1:8000";

export async function runOptimization(payload, signal) {
  const response = await fetch(`${API_BASE_URL}/optimize`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
    signal,
  });

  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(resolveApiError(data, response.status));
  }
  return data;
}

function resolveApiError(payload, status) {
  if (typeof payload.detail === "string") {
    return payload.detail;
  }

  if (Array.isArray(payload.detail) && payload.detail.length > 0) {
    return payload.detail.map((item) => item.msg || "Invalid request").join(" ");
  }

  return `Optimization request failed with status ${status}.`;
}
