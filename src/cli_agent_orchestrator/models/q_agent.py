"""Q CLI agent configuration model."""

from typing import Any

from pydantic import BaseModel
from pydantic import Field


class QAgentConfig(BaseModel):
    """Q CLI agent configuration."""

    name: str
    description: str
    tools: list[str] = Field(default_factory=lambda: ["*"])
    allowedTools: list[str] = Field(default_factory=list)
    useLegacyMcpJson: bool = False
    resources: list[str] = Field(default_factory=list)

    # Optional pass-through fields
    prompt: str | None = None
    mcpServers: dict[str, Any] | None = None
    toolAliases: dict[str, str] | None = None
    toolsSettings: dict[str, Any] | None = None
    hooks: dict[str, Any] | None = None
    model: str | None = None

    class Config:
        # Exclude None values when serializing to JSON
        exclude_none = True
