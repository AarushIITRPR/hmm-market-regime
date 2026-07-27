export const regimePalette = [
  "#6e91bd",
  "#c46565",
  "#7ea6d8",
  "#5f9446",
  "#d4a24f",
  "#8b7bbd",
  "#9ca3af",
  "#cf7c54",
  "#4e9f73",
  "#6a9fbf",
];

export const baseLayout = {
  autosize: true,
  margin: { l: 48, r: 24, t: 18, b: 42 },
  paper_bgcolor: "#ffffff",
  plot_bgcolor: "#ffffff",
  font: {
    family: "Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
    color: "#374151",
  },
  hovermode: "x unified",
  xaxis: {
    gridcolor: "rgba(226, 232, 240, 0.8)",
    zeroline: false,
  },
  yaxis: {
    gridcolor: "rgba(226, 232, 240, 0.8)",
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
