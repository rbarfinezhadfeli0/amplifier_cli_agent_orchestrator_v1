"""MCP server models."""

from pydantic import BaseModel
from pydantic import Field


class HandoffResult(BaseModel):
    """Result of a handoff operation."""

    success: bool = Field(description="Whether the handoff was successful")
    message: str = Field(description="A message describing the result of the handoff")
    output: str | None = Field(None, description="The output from the target agent")
    terminal_id: str | None = Field(None, description="The terminal ID used for the handoff")
