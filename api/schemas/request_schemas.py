"""
api/schemas/request_schemas.py
Pydantic request models for all API endpoints.
"""
from __future__ import annotations
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request body for the query endpoint (Phase 7+)."""
    query: str = Field(..., min_length=3, description="The user's question.")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of passages to retrieve.")
    use_eval_model: bool = Field(default=False, description="Use evaluation-quality LLM.")
