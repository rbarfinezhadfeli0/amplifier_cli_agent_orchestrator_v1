"""Public API for the CAO plugin system."""

from cli_agent_orchestrator.plugins.base import CaoPlugin
from cli_agent_orchestrator.plugins.base import hook
from cli_agent_orchestrator.plugins.events import CaoEvent
from cli_agent_orchestrator.plugins.events import PostCreateSessionEvent
from cli_agent_orchestrator.plugins.events import PostCreateTerminalEvent
from cli_agent_orchestrator.plugins.events import PostKillSessionEvent
from cli_agent_orchestrator.plugins.events import PostKillTerminalEvent
from cli_agent_orchestrator.plugins.events import PostSendMessageEvent
from cli_agent_orchestrator.plugins.registry import PluginRegistry

__all__ = [
    "CaoPlugin",
    "hook",
    "CaoEvent",
    "PostSendMessageEvent",
    "PostCreateSessionEvent",
    "PostKillSessionEvent",
    "PostCreateTerminalEvent",
    "PostKillTerminalEvent",
    "PluginRegistry",
]
