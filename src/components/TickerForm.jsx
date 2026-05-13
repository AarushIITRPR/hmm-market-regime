import { useEffect, useState } from "react";

export default function TickerForm({ values, isLoading, onSubmit }) {
  const [draftValues, setDraftValues] = useState(values);

  useEffect(() => {
    setDraftValues(values);
  }, [values]);

  function updateField(field, value) {
    setDraftValues((current) => ({
      ...current,
      [field]: value,
    }));
  }

  function handleSubmit(event) {
    event.preventDefault();
    onSubmit(draftValues);
  }

  return (
    <form className="ticker-form" onSubmit={handleSubmit}>
      <label>
        <span>Ticker</span>
        <input
          value={draftValues.ticker}
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
          value={draftValues.startDate}
          onChange={(event) => updateField("startDate", event.target.value)}
          required
        />
      </label>

      <label>
        <span>End Date</span>
        <input
          type="date"
          value={draftValues.endDate}
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
