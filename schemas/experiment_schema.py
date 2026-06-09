"""
Experiment Schema — Pydantic models for experiment output structure.

Defines the metadata and wrapper for each experiment run's output JSON.

Implementation: Phase 3
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class ExperimentMetadata(BaseModel):
    """Metadata for a single experiment run."""

    method: str = Field(..., description="Retrieval method (llm_only, vanilla_rag, graph_rag)")
    model: str = Field(..., description="LLM model name")
    timestamp: str = Field(..., description="ISO format timestamp of the run")
    dataset_size: int = Field(..., ge=0, description="Number of events processed")
    dev_mode: bool = Field(False, description="Whether run was in dev mode")


class EventResult(BaseModel):
    """Result for a single processed event."""

    event_id: str = Field(..., description="Original event ID from XML")
    file_source: str = Field(..., description="Source XML filename")
    extraction: dict = Field(default_factory=dict, description="Validated extraction data")
    processing_time_seconds: Optional[float] = Field(None, description="Time taken to process")
    status: str = Field("success", description="Processing status: success, error, skipped")
    error_message: Optional[str] = Field(None, description="Error details if status is error")


class ExperimentOutput(BaseModel):
    """Top-level output structure for an experiment."""

    experiment_metadata: ExperimentMetadata
    results: List[EventResult] = Field(default_factory=list)
