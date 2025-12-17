# compute_metrics.py
import json

# Mock metrics — you can tweak the numbers if you want
metrics = {
    "accuracy": 0.92,
    "precision": 0.90,
    "recall": 0.88,
    "auc": 0.94,
    "inference_time_ms": 120  # average inference time per image in milliseconds
}

# Save metrics to metrics.json in the current folder
METRICS_FILE = "metrics.json"
with open(METRICS_FILE, "w") as f:
    json.dump(metrics, f, indent=4)

print(f"{METRICS_FILE} created successfully!")
