from feast import FeatureStore
from datetime import datetime, timedelta
import pytz

store = FeatureStore(".")

start_date = datetime(2025, 3, 1, tzinfo=pytz.UTC)
end_date = datetime(2025, 3, 20, tzinfo=pytz.UTC)

current = start_date

while current < end_date:
    next_day = current + timedelta(days=1)
    print(f"\n🟡 Materializing from {current} to {next_day} ...")

    try:
        store.materialize(current, next_day)
        print(f"✅ Successfully materialized for {current.date()}")
    except Exception as e:
        print(f"❌ Failed on {current.date()}: {e}")

    current = next_day

print("\n🟢 All materialization attempts completed.")

