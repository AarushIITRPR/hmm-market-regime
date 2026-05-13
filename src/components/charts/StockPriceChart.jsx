import ChartPanel from "./ChartPanel.jsx";
import Plot from "./Plot.jsx";
import { baseLayout, plotConfig } from "./chartConfig.js";

export default function StockPriceChart({ result }) {
  return (
    <ChartPanel title="Stock Price History">
      <Plot
        data={[
          {
            x: result.dates,
            y: result.close_prices,
            type: "scatter",
            mode: "lines",
            name: "Close",
            line: { color: "#1d4ed8", width: 2.4 },
          },
        ]}
        layout={{
          ...baseLayout,
          yaxis: { ...baseLayout.yaxis, title: "Close" },
          height: 340,
        }}
        config={plotConfig}
        useResizeHandler
        className="plot"
      />
    </ChartPanel>
  );
}
