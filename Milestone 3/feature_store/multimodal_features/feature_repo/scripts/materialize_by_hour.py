from feast import FeatureStore
from datetime import datetime, timedelta
import pytz

# Initialize Feature Store
store = FeatureStore(".")

# Set timestamp range (make sure it's UTC-aware!)
start_time = datetime(2025, 3, 1, 0, 0, 0, tzinfo=pytz.UTC)
end_time = datetime(2025, 3, 20, 1, 4, 0, tzinfo=pytz.UTC)  # Your max timestamp

# Step = 1 hour
step = timedelta(hours=1)

# Loop over hourly chunks
current = start_time
while current < end_time:
    next_chunk = current + step
    print(f"🕐 Materializing from {current} to {next_chunk} ...")
    
    try:
        store.materialize(current, next_chunk)
    except Exception as e:
        print(f"❌ Error during materialization: {e}")
    
    current = next_chunk
