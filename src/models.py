"""
Shared Pydantic schemas used across the ETL pipeline and storage layer.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class ExtractedEntity(BaseModel):
    name: str = Field(
        description="Canonical normalised name, e.g. 'John Smith', 'ACC-1234', 'Acme Corp Ltd'"
    )
    entity_type: str = Field(
        description="One of: Person, Company, Account, Location, Jurisdiction, Document, Event"
    )


class ExtractedRelation(BaseModel):
    source_entity: str = Field(description="Subject entity name")
    relation_type: str = Field(
        description=(
            "Typed edge label, e.g. BENEFICIAL_OWNER, DIRECTOR_OF, HOLDS_ACCOUNT, "
            "TRANSFERRED_TO, MENTIONED_IN, LOCATED_IN, ALIAS_OF"
        )
    )
    target_entity: str = Field(description="Object entity name")
    context: Optional[str] = Field(
        default=None,
        description="Short verbatim snippet from the source that supports this relation"
    )


class DocumentExtractionResult(BaseModel):
    """Structured output from the LLM extraction step for one document."""
    doc_id: str
    entities: List[ExtractedEntity]
    relations: List[ExtractedRelation]


class CommsBodyExtraction(BaseModel):
    """
    Lightweight LLM extraction from a single communication body.

    Deliberately narrower than DocumentExtractionResult — structural fields
    (sender, recipients, timestamp, channel) are parsed deterministically.
    The LLM only handles what rules cannot: entity mentions buried in free text
    and intent signals that require reading the message.
    """
    event_id: str
    entity_mentions: List[str] = Field(
        description=(
            "Canonical names of people, companies, or accounts explicitly or implicitly "
            "mentioned in the message body. Use the same normalised form as other sources "
            "(e.g. 'John Smith', 'Acme Corp Ltd', 'ACC-1234')."
        )
    )
    sender_alias: Optional[str] = Field(
        default=None,
        description=(
            "Resolved canonical name of the sender if it can be inferred from the message "
            "content or email address (e.g. 'j.smith@acme.io' → 'John Smith'). "
            "Null if uncertain."
        )
    )
    intent_signal: Optional[str] = Field(
        default=None,
        description=(
            "Short label for the communication's apparent purpose, e.g. "
            "'instruct_wire_transfer', 'coordinate_layering', 'compliance_deflection', "
            "'logistics_coordination'. Null if benign or unclear."
        )
    )
