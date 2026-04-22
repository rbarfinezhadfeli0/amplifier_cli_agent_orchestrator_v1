"""Agent profile models."""

from typing import Any

from pydantic import BaseModel
from pydantic import Field


class McpServer(BaseModel):
    """MCP server configuration."""

    type: str | None = None
    command: str
    args: list[str] | None = None
    env: dict[str, str] | None = None
    timeout: int | None = None


class AgentProfile(BaseModel):
    """Agent profile configuration with Q CLI agent fields."""

    name: str
    description: str
    provider: str | None = None  # Provider override (e.g. "claude_code", "kiro_cli")
    system_prompt: str | None = None  # The markdown content
    role: str | None = None  # "supervisor", "developer", "reviewer"

    # Q CLI agent fields (all optional, will be passed through to JSON)
    prompt: str | None = None
    mcpServers: dict[str, Any] | None = None
    tools: list[str] | None = Field(default=None)
    toolAliases: dict[str, str] | None = None
    allowedTools: list[str] | None = None
    toolsSettings: dict[str, Any] | None = None
    resources: list[str] | None = None
    hooks: dict[str, Any] | None = None
    useLegacyMcpJson: bool | None = None
    model: str | None = None
