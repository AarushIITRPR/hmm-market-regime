export default function TickerForm({ values, isLoading, onChange, onSubmit }) {
  function updateField(field, value) {
    onChange((current) => ({
      ...current,
      [field]: value,
    }));
  }

  return (
    <form className="ticker-form" onSubmit={onSubmit}>
      <label>
        <span>Ticker</span>
        <input
          value={values.ticker}
          onChange={(event) => updateField("ticker", event.target.value)}
          required
          maxLength={20}
          placeholder="SPY"
        />
      </label>

      <label>
        <span>Start Date</span>
        <input
          type="date"
          value={values.startDate}
          onChange={(event) => updateField("startDate", event.target.value)}
          required
        />
      </label>

      <label>
        <span>End Date</span>
        <input
          type="date"
          value={values.endDate}
          onChange={(event) => updateField("endDate", event.target.value)}
          required
        />
      </label>

      <button type="submit" disabled={isLoading}>
        {isLoading ? "Running" : "Predict"}
      </button>
    </form>
  );
}
