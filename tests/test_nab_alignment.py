import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from htm_py.utils import load_config
from htm_py.likelihood import AnomalyLikelihood

def plot_comparison(yours, reference, label):
    plt.figure(figsize=(12, 3))
    plt.plot(yours, label="htm_py " + label)
    plt.plot(reference, label="numenta " + label, linestyle="--")
    plt.title(f"Comparison of {label}")
    plt.xlabel("Timestep")
    plt.ylabel(label)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main(config_path, data_path, numenta_output_path):
    # Load data
    df = pd.read_csv(data_path)
    df.fillna(0, inplace=True)

    numenta = pd.read_csv(numenta_output_path)
    numenta_scores = numenta["anomaly_score"].values
    numenta_likelihoods = numenta["S(t)_standard"].values
    numenta["anomaly_likelihood"] = numenta_likelihoods  # optional alias

    # Load model
    model = load_config(config_path)

    # Run model
    scores = []
    likelihoods = []
    likelihood_model = AnomalyLikelihood()

    # Get model outputs loop
    start = time.time()
    for t, row in enumerate(df.itertuples(index=False, name=None)):
        score, _ = model.compute(dict(zip(df.columns, row)))
        scores.append(score)
        likelihoods.append(likelihood_model.update(score))
        if t % 100 == 0:
            print(f"Timestep {t}")
    print("Total time:", time.time() - start)

    # Convert to arrays
    scores = np.array(scores)
    likelihoods = np.array(likelihoods)

    # Plot comparisons
    plot_comparison(scores, numenta_scores, "anomaly_score")
    plot_comparison(likelihoods, numenta_likelihoods, "anomaly_likelihood")

    # Evaluate
    score_mae = np.mean(np.abs(scores - numenta_scores))
    likelihood_mae = np.mean(np.abs(likelihoods - numenta_likelihoods))

    print(f"Anomaly Score MAE: {score_mae:.5f}")
    print(f"Likelihood MAE: {likelihood_mae:.5f}")

    # Assert tolerances (adjust if needed)
    assert score_mae < 0.01, "Score MAE too high!"
    assert likelihood_mae < 0.01, "Likelihood MAE too high!"

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python -m tests.test_nab_alignment <config.yaml> <data.csv> <numenta_output.csv>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
