export const regimePalette = [
  "#0f766e",
  "#ca8a04",
  "#dc2626",
  "#2563eb",
  "#7c3aed",
  "#db2777",
  "#475569",
  "#ea580c",
  "#16a34a",
  "#0891b2",
];

export const baseLayout = {
  autosize: true,
  margin: { l: 48, r: 24, t: 18, b: 42 },
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: {
    family: "Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
    color: "#172033",
  },
  hovermode: "x unified",
  xaxis: {
    gridcolor: "rgba(148, 163, 184, 0.18)",
    zeroline: false,
  },
  yaxis: {
    gridcolor: "rgba(148, 163, 184, 0.18)",
    zeroline: false,
  },
};

export const plotConfig = {
  displaylogo: false,
  responsive: true,
  modeBarButtonsToRemove: ["lasso2d", "select2d"],
};

export function colorForState(state) {
  return regimePalette[Math.abs(Number(state)) % regimePalette.length];
}
