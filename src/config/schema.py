from __future__ import annotations

from pydantic import BaseModel, Field


class LossConfig(BaseModel):
    name: str
    params: dict = Field(default_factory=dict)
