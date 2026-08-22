from pydantic import BaseModel, Field


class IoTData(BaseModel):
    device_id: str = Field(..., min_length=1, max_length=100)
    temperature: float = Field(..., ge=-20, le=80)
    humidity: float = Field(..., ge=0, le=100)
    soil_moisture: float = Field(..., ge=0, le=100)
    rainfall: float = Field(default=0, ge=0)
    light: float | None = Field(default=None, ge=0)


class IoTDataResponse(BaseModel):
    success: bool
    message: str
    data: IoTData
