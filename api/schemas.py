# request/response shapes for the API

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        description="User message, e.g. ingredients like 'Egg, Onion'",
        min_length=1,
        max_length=2000,
    )


class ChatResponse(BaseModel):
    response: str = Field(..., description="Recipe suggestion or reply")


class HealthResponse(BaseModel):
    status: str = Field(...)
    model: str = Field(...)
