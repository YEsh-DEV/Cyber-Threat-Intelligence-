"""
Extraction Schema — Pydantic models for CTI extraction validation.

Defines the structure for entities, relations, and the combined
extraction result. All LLM outputs must validate against these models
before being persisted.

Implementation: Phase 3
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class Entity(BaseModel):
    """A single extracted CTI entity."""

    text: str = Field(..., description="The raw text of the entity as found in source")
    type: str = Field(..., description="Entity type (e.g., malware, threat_actor, ip_address)")
    canonical_name: str = Field(..., description="Normalized/canonical form of the entity")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score between 0 and 1"
    )


class Relation(BaseModel):
    """A single extracted CTI relationship."""

    head: str = Field(..., description="Source entity of the relation")
    relation: str = Field(..., description="Relation type (e.g., uses, targets, attributed_to)")
    tail: str = Field(..., description="Target entity of the relation")
    time: Optional[str] = Field(None, description="Temporal context of the relation")
    evidence: Optional[str] = Field(None, description="Supporting text from the source")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score between 0 and 1"
    )


class ExtractionResult(BaseModel):
    """Combined extraction output for a single event."""

    entities: List[Entity] = Field(default_factory=list, description="Extracted entities")
    relations: List[Relation] = Field(default_factory=list, description="Extracted relations")
