export const regimePalette = [
  "#19f5c8",
  "#f7c948",
  "#ff5c7a",
  "#6ee7ff",
  "#a78bfa",
  "#fb7185",
  "#94a3b8",
  "#f97316",
  "#22c55e",
  "#38bdf8",
];

export const baseLayout = {
  autosize: true,
  margin: { l: 48, r: 24, t: 18, b: 42 },
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: {
    family: "Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
    color: "#d6f7ff",
  },
  hovermode: "x unified",
  xaxis: {
    gridcolor: "rgba(110, 231, 255, 0.12)",
    zeroline: false,
  },
  yaxis: {
    gridcolor: "rgba(110, 231, 255, 0.12)",
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
