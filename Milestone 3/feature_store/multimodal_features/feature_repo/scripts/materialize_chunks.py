from feast import FeatureStore
from datetime import datetime, timedelta

store = FeatureStore(".")

start = datetime(2025, 3, 1, 0, 0, 0)
end = datetime(2025, 3, 1, 23, 59, 59)  # Adjust if your data spans more days
step = timedelta(hours=1)  # You can reduce to 30 min if it still crashes

current = start
while current < end:
    next_window = current + step
    print(f"Materializing from {current} to {next_window}")
    store.materialize(current, next_window)
    current = next_window
