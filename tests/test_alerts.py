"""Smoke test 5.7: Alert engine."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")

from src.db.schema import init_db
from src.alerts.engine import AlertEngine
import time, json

# Init fresh DB
test_db = "tests/test_alerts.db"
if os.path.exists(test_db):
    os.remove(test_db)
init_db(test_db)

# Temporarily override DB path
import src.db.schema as schema_mod
original_db = schema_mod.DB_PATH
schema_mod.DB_PATH = test_db

engine = AlertEngine()
print(f"[OK] Alert engine initialized with {len(engine._plates)} blacklisted plates")
for p in engine._plates:
    print(f"  Blacklisted: {p}")

# Test exact match
now = time.time()
alerts = engine.check_plate("MH12AB1234", "cam_1", 12.9758, 77.6082, now)
print(f"[OK] Exact match test: {len(alerts)} alert(s)")
assert len(alerts) >= 1, "FAIL: Expected at least 1 alert for exact match"
assert alerts[0]["alert_type"] == "blacklist_exact", f"FAIL: Wrong type: {alerts[0]['alert_type']}"
print(f"  Alert: plate={alerts[0]['plate']}, type={alerts[0]['alert_type']}")

# Test fuzzy match (edit distance 1)
alerts = engine.check_plate("MH12AB1235", "cam_2", 12.9768, 77.6050, now + 1)
print(f"[OK] Fuzzy match test: {len(alerts)} alert(s)")
assert len(alerts) >= 1, "FAIL: Expected at least 1 alert for fuzzy match"
assert alerts[0]["alert_type"] == "blacklist_fuzzy", f"FAIL: Wrong type: {alerts[0]['alert_type']}"
print(f"  Alert: plate={alerts[0]['plate']}, type={alerts[0]['alert_type']}")

# Test no match
alerts = engine.check_plate("XY99ZZ0000", "cam_3", 12.9745, 77.6095, now + 2)
print(f"[OK] No-match test: {len(alerts)} alert(s) (expected 0)")
assert len(alerts) == 0, f"FAIL: Expected 0 alerts, got {len(alerts)}"

# Test get_alerts feed
all_alerts = engine.get_alerts(limit=10)
print(f"[OK] Alert feed: {len(all_alerts)} alerts")
assert len(all_alerts) >= 2, f"FAIL: Expected >=2 alerts in feed, got {len(all_alerts)}"

# Test acknowledge
ack = engine.acknowledge_alert(all_alerts[0]["id"])
print(f"[OK] Acknowledge alert {all_alerts[0]['id']}: {ack}")
assert ack, "FAIL: Acknowledge returned False"

# Clean up
schema_mod.DB_PATH = original_db
os.remove(test_db)
print("[OK] Test database cleaned up")

print("SMOKE TEST 5.7 PASSED")
