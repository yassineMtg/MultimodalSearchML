from feast import FeatureStore
from datetime import datetime, timedelta
import pytz

# Initialize store
store = FeatureStore(".")

# Define range
start_date = datetime(2025, 3, 1, tzinfo=pytz.UTC)
end_date = datetime(2025, 3, 20, tzinfo=pytz.UTC)

# Materialize day-by-day
current = start_date
while current < end_date:
    next_day = current + timedelta(days=1)
    print(f"\n🟡 Materializing from {current} to {next_day} ...")
    
    store.materialize(current, next_day)

    current = next_day
