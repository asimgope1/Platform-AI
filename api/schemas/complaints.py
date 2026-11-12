from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class ComplaintCreateSchema(BaseModel):
    description: str = Field(..., min_length=5, max_length=1000)
    category: Optional[str] = Field(default=None, max_length=64)
    subject: Optional[str] = Field(default=None, max_length=255)
    order_id: Optional[str] = Field(default=None, max_length=128)
    customer_id: Optional[str] = Field(default=None, max_length=128)
    caller_phone: Optional[str] = Field(default=None, max_length=64)
    priority: Optional[str] = Field(default=None, max_length=32)
    call_uuid: Optional[str] = Field(default=None, max_length=150)


class ComplaintUpdateSchema(BaseModel):
    status: Optional[str] = Field(default=None, max_length=32)
    priority: Optional[str] = Field(default=None, max_length=32)
    resolution_notes: Optional[str] = Field(default=None, max_length=1000)

