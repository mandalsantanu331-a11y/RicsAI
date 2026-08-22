from datetime import datetime, timezone
from threading import Lock

from app.schemas.iot import IoTData


_latest_data: dict[str, dict] = {}
_lock = Lock()


def save_sensor_data(data: IoTData) -> dict:
    record = data.model_dump()
    record["received_at"] = datetime.now(timezone.utc).isoformat()

    with _lock:
        _latest_data[data.device_id] = record

    return record


def get_latest_sensor_data(device_id: str | None = None) -> dict | None:
    with _lock:
        if device_id:
            return _latest_data.get(device_id)

        if not _latest_data:
            return None

        return max(
            _latest_data.values(),
            key=lambda item: item["received_at"],
        )
