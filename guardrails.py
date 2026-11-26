"""Typed, strict contracts for generated claims and verification decisions."""
from pydantic import BaseModel, ConfigDict, Field, StrictBool, ValidationError

from provider import ProviderError


class Claim(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    answer: str = Field(min_length=1, max_length=6000)
    source: str = Field(pattern=r"^P[1-9][0-9]*(?:T[1-9][0-9]*|V)$")
    evidence: str = Field(min_length=1, max_length=6000)


class Draft(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    claims: list[Claim] = Field(max_length=12)
    relevant_sources: list[str]


class Verdict(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    supported: StrictBool
    claim_support: list[StrictBool]
    addresses_question: StrictBool


def validate_output(schema, value, code):
    try:
        return schema.model_validate(value).model_dump()
    except ValidationError:
        # Validation errors include model text, so expose only a diagnostic code.
        raise ProviderError(code) from None
