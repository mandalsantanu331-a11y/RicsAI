from fastapi import APIRouter, HTTPException

from app.schemas.iot import IoTData, IoTDataResponse
from app.services.iot_service import get_latest_sensor_data, save_sensor_data


router = APIRouter(prefix="/iot", tags=["IoT"])


@router.post("/data", response_model=IoTDataResponse)
def receive_sensor_data(data: IoTData):
    saved = save_sensor_data(data)

    return {
        "success": True,
        "message": "Sensor data received successfully",
        "data": {
            **data.model_dump(),
        },
    }


@router.get("/latest")
def latest_sensor_data(device_id: str | None = None):
    data = get_latest_sensor_data(device_id)

    if data is None:
        raise HTTPException(
            status_code=404,
            detail="No sensor data available.",
        )

    return {
        "success": True,
        "data": data,
    }
