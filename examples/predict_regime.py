"""Example CLI-style usage of the reusable market_regime package."""

from market_regime.data_loading import load_market_data
from market_regime.inference import predict_market_regime, prediction_to_frame
from market_regime.preprocessing import prepare_market_features


def main() -> None:
    data = load_market_data("SPY", start="2018-01-01")
    features = prepare_market_features(data)
    prediction = predict_market_regime(features)
    result = prediction_to_frame(features, prediction)

    print("Transition probabilities:")
    print(prediction.transition_probabilities.round(3))
    print(result[["Returns", "HiddenState", "RegimeLabel"]].tail())


if __name__ == "__main__":
    main()
