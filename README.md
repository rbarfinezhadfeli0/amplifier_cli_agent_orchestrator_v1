# CLI Agent Orchestrator

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/awslabs/cli-agent-orchestrator)

CLI Agent Orchestrator(CAO, pronounced as "kay-oh"), is a lightweight orchestration system for managing multiple AI agent sessions in tmux terminals. Enables Multi-agent collaboration via MCP server.

## Hierarchical Multi-Agent System

CLI Agent Orchestrator (CAO) implements a hierarchical multi-agent system that enables complex problem-solving through specialized division of CLI Developer Agents.

![CAO Architecture](./docs/assets/cao_architecture.png)

### Key Features

* **Hierarchical orchestration** – CAO's supervisor agent coordinates workflow management and task delegation to specialized worker agents. The supervisor maintains overall project context while agents focus on their domains of expertise.
* **Session-based isolation** – Each agent operates in isolated tmux sessions, ensuring proper context separation while enabling seamless communication through Model Context Protocol (MCP) servers. This provides both coordination and parallel processing capabilities.
* **Intelligent task delegation** – CAO automatically routes tasks to appropriate specialists based on project requirements, expertise matching, and workflow dependencies. The system adapts between individual agent work and coordinated team efforts through three orchestration patterns:
    - **Handoff** - Synchronous task transfer with wait-for-completion
    - **Assign** - Asynchronous task spawning for parallel execution  
    - **Send Message** - Direct communication with existing agents
* **Flexible workflow patterns** – CAO supports both sequential coordination for dependent tasks and parallel processing for independent work streams. This allows optimization of both development speed and quality assurance processes.
* **Flow - Scheduled runs** – Automated execution of workflows at specified intervals using cron-like scheduling, enabling routine tasks and monitoring workflows to run unattended.
* **Context preservation** – The supervisor agent provides only necessary context to each worker agent, avoiding context pollution while maintaining workflow coherence.
* **Direct worker interaction and steering** – Users can interact directly with worker agents to provide additional steering, distinguishing from sub-agents features by allowing real-time guidance and course correction.
* **Tool restrictions** – Control what each agent can do through `role` and `allowedTools`. Built-in roles (`supervisor`, `developer`, `reviewer`) provide sensible defaults, while `allowedTools` gives fine-grained control. CAO translates restrictions to each provider's native enforcement mechanism. See [Tool Restrictions](#tool-restrictions-allowedtools).
* **Advanced CLI integration** – CAO agents have full access to advanced features of the developer CLI, such as the [sub-agents](https://docs.claude.com/en/docs/claude-code/sub-agents) feature of Claude Code, [Custom Agent](https://docs.aws.amazon.com/amazonq/latest/qdeveloper-ug/command-line-custom-agents.html) of Amazon Q Developer for CLI and so on.

For detailed project structure and architecture, see [CODEBASE.md](CODEBASE.md).

## Installation

### Requirements

- **curl** and **git** — For downloading installers and cloning the repo
- **Python 3.10 or higher** — CAO requires Python >=3.10 (see [pyproject.toml](pyproject.toml))
- **tmux 3.3+** — Used for agent session isolation
- **[uv](https://docs.astral.sh/uv/)** — Fast Python package installer and virtual environment manager

### 1. Install Python 3.10+

If you don't have Python 3.10+ installed, use your platform's package manager:

```bash
# macOS (Homebrew)
brew install python@3.12

# Ubuntu/Debian
sudo apt update && sudo apt install python3.12 python3.12-venv

# Amazon Linux 2023 / Fedora
sudo dnf install python3.12
```

Verify your Python version:

```bash
python3 --version   # Should be 3.10 or higher
```

> **Note:** We recommend using [uv](https://docs.astral.sh/uv/) to manage Python environments instead of system-wide installations like Anaconda. `uv` automatically handles virtual environments and Python version resolution per-project.

### 2. Install tmux (version 3.3 or higher required)

```bash
bash <(curl -s https://raw.githubusercontent.com/awslabs/cli-agent-orchestrator/refs/heads/main/tmux-install.sh)
```

### 3. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env   # Add uv to PATH (or restart your shell)
```

### 4. Install CLI Agent Orchestrator

```bash
uv tool install git+https://github.com/awslabs/cli-agent-orchestrator.git@main --upgrade
```

### Development Setup

For local development, clone the repo and install with `uv sync`:

```bash
git clone https://github.com/awslabs/cli-agent-orchestrator.git
cd cli-agent-orchestrator/
uv sync          # Creates .venv/ and installs all dependencies
uv run cao --help  # Verify installation
```

For development workflow, testing, code quality checks, and project structure, see [DEVELOPMENT.md](DEVELOPMENT.md).

## Prerequisites

Before using CAO, install at least one supported CLI agent tool:

| Provider | Documentation | Authentication |
|----------|---------------|----------------|
| **Kiro CLI** (default) | [Provider docs](docs/kiro-cli.md) · [Installation](https://kiro.dev/docs/kiro-cli) | AWS credentials |
| **Claude Code** | [Provider docs](docs/claude-code.md) · [Installation](https://docs.anthropic.com/en/docs/claude-code/getting-started) | Anthropic API key |
| **Codex CLI** | [Provider docs](docs/codex-cli.md) · [Installation](https://github.com/openai/codex) | OpenAI API key |
| **Gemini CLI** | [Provider docs](docs/gemini-cli.md) · [Installation](https://github.com/google-gemini/gemini-cli) | Google AI API key |
| **Kimi CLI** | [Provider docs](docs/kimi-cli.md) · [Installation](https://platform.moonshot.cn/docs/kimi-cli) | Moonshot API key |
| **GitHub Copilot CLI** | [Provider docs](docs/copilot-cli.md) · [Installation](https://github.com/features/copilot/cli) | GitHub auth |
| **Q CLI** | [Installation](https://docs.aws.amazon.com/amazonq/latest/qdeveloper-ug/command-line.html) | AWS credentials |

## Quick Start

### 1. Install Agent Profiles

Install the supervisor agent (the orchestrator that delegates to other agents):

```bash
cao install code_supervisor
```

Optionally install additional worker agents:

```bash
cao install developer
cao install reviewer
```

You can also install agents from local files or URLs:

```bash
cao install ./my-custom-agent.md
cao install https://example.com/agents/custom-agent.md
```

For details on creating custom agent profiles, see [docs/agent-profile.md](docs/agent-profile.md).

### 2. Start the Server

```bash
cao-server
```

### 3. Launch the Supervisor

In another terminal, launch the supervisor agent:

```bash
cao launch --agents code_supervisor

# Or specify a provider
cao launch --agents code_supervisor --provider kiro_cli
cao launch --agents code_supervisor --provider claude_code
cao launch --agents code_supervisor --provider codex
cao launch --agents code_supervisor --provider gemini_cli
cao launch --agents code_supervisor --provider kimi_cli
cao launch --agents code_supervisor --provider copilot_cli
# Unrestricted access + skip confirmation (DANGEROUS)
cao launch --agents code_supervisor --yolo
```

The supervisor will coordinate and delegate tasks to worker agents (developer, reviewer, etc.) as needed using the orchestration patterns.

### 4. Shutdown

```bash
# Shutdown all cao sessions
cao shutdown --all

# Shutdown specific session
cao shutdown --session cao-my-session
```

### Working with tmux Sessions

All agent sessions run in tmux. Useful commands:

```bash
# List all sessions
tmux list-sessions

# Attach to a session
tmux attach -t <session-name>

# Detach from session (inside tmux)
Ctrl+b, then d

# Switch between windows (inside tmux)
Ctrl+b, then n          # Next window
Ctrl+b, then p          # Previous window
Ctrl+b, then <number>   # Go to window number (0-9)
Ctrl+b, then w          # List all windows (interactive selector)

# Delete a session
cao shutdown --session <session-name>
```

**List all windows (Ctrl+b, w):**

![Tmux Window Selector](./docs/assets/tmux_all_windows.png)

## Web UI

CAO includes a web dashboard for managing agents, terminals, and flows from the browser.

![CAO Web UI](https://github.com/user-attachments/assets/e7db9261-62b1-4422-b9f5-6fe5f65bdea4)

### Additional Requirements

- **Node.js 18+** — Required for the frontend dev server and Codex CLI

```bash
# macOS (Homebrew)
brew install node

# Ubuntu/Debian
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo bash -
sudo apt-get install -y nodejs

# Amazon Linux 2023 / Fedora
sudo dnf install nodejs20

# Verify
node --version   # Should be 18 or higher
```

### Starting the Web UI

**Option A: Development mode** (hot-reload, two terminals needed)

```bash
# Terminal 1 — start the backend server
cao-server

# Terminal 2 — start the frontend dev server
cd web/
npm install        # First time only
npm run dev        # Starts on http://localhost:5173
```

Open http://localhost:5173 in your browser.

**Option B: Production mode** (single server, no Vite needed)

The built Web UI is bundled into the CAO wheel, so a plain `uv tool install` ships everything you need. Just start the server:

```bash
cao-server
```

To rebuild the frontend from source:

```bash
cd web/
npm install && npm run build   # Outputs to src/cli_agent_orchestrator/web_ui/
uv tool install . --reinstall
```

Open http://localhost:9889 in your browser.

> **Custom host/port:** `cao-server --host 0.0.0.0 --port 9889` exposes the server to the network — see Security note below.

**Remote machine access** — If you're running CAO on a remote host (e.g. dev desktop), set up an SSH tunnel:

```bash
# Dev mode (proxy both frontend and backend)
ssh -L 5173:localhost:5173 -L 9889:localhost:9889 your-remote-host

# Production mode (backend serves UI directly)
ssh -L 9889:localhost:9889 your-remote-host
```

Then open the same URLs (localhost:5173 or localhost:9889) in your local browser.

### Features

Manage sessions, spawn agents, create scheduled flows, configure agent directories, and interact with live terminals — all from the browser. Includes live status badges, an inbox for agent-to-agent messaging, output viewer, and provider auto-detection.

For frontend architecture and component details, see [web/README.md](web/README.md). For agent directory configuration, see [docs/settings.md](docs/settings.md).

## MCP Server Tools and Orchestration Modes

CAO provides a local HTTP server that processes orchestration requests. CLI agents can interact with this server through MCP tools to coordinate multi-agent workflows.

### How It Works

Each agent terminal is assigned a unique `CAO_TERMINAL_ID` environment variable. The server uses this ID to:

- Route messages between agents
- Track terminal status (IDLE, PROCESSING, COMPLETED, ERROR)
- Manage terminal-to-terminal communication via inbox
- Coordinate orchestration operations

When an agent calls an MCP tool, the server identifies the caller by their `CAO_TERMINAL_ID` and orchestrates accordingly.

### Orchestration Modes

CAO supports three orchestration patterns:

> **Note:** All orchestration modes support optional `working_directory` parameter when enabled via `CAO_ENABLE_WORKING_DIRECTORY=true`. See [Working Directory Support](#working-directory-support) for details.

**1. Handoff** - Transfer control to another agent and wait for completion

- Creates a new terminal with the specified agent profile
- Sends the task message and waits for the agent to finish
- Returns the agent's output to the caller
- Automatically exits the agent after completion
- Use when you need **synchronous** task execution with results

Example: Sequential code review workflow

![Handoff Workflow](./docs/assets/handoff-workflow.png)

**2. Assign** - Spawn an agent to work independently (async)

- Creates a new terminal with the specified agent profile
- Sends the task message with callback instructions
- Returns immediately with the terminal ID
- Agent continues working in the background
- Assigned agent sends results back to supervisor via `send_message` when complete
- Messages are queued for delivery if the supervisor is busy (common in parallel workflows)
- Use for **asynchronous** task execution or fire-and-forget operations

Example: A supervisor assigns parallel data analysis tasks to multiple analysts while using handoff to sequentially generate a report template, then combines all results.

See [examples/assign](examples/assign) for the complete working example.

![Parallel Data Analysis](./docs/assets/parallel-data-analysis.png)

**3. Send Message** - Communicate with an existing agent

- Sends a message to a specific terminal's inbox
- Messages are queued and delivered when the terminal is idle
- Enables ongoing collaboration between agents
- Common for **swarm** operations where multiple agents coordinate dynamically
- Use for iterative feedback or multi-turn conversations

Example: Multi-role feature development

![Multi-role Feature Development](./docs/assets/multi-role-feature-development.png)

### Custom Orchestration

The `cao-server` runs on `http://localhost:9889` by default and exposes REST APIs for session management, terminal control, and messaging. The CLI commands (`cao launch`, `cao shutdown`) and MCP server tools (`handoff`, `assign`, `send_message`) are just examples of how these APIs can be packaged together.

You can combine the three orchestration modes above into custom workflows, or create entirely new orchestration patterns using the underlying APIs to fit your specific needs.

For complete API documentation, see [docs/api.md](docs/api.md).

## Flows - Scheduled Agent Sessions

Flows allow you to schedule agent sessions to run automatically based on cron expressions.

### Prerequisites

Install the agent profile you want to use:

```bash
cao install developer
```

### Quick Start

The example flow asks a simple world trivia question every morning at 7:30 AM.

```bash
# 1. Start the cao server
cao-server

# 2. In another terminal, add a flow
cao flow add examples/flow/morning-trivia.md

# 3. List flows to see schedule and status
cao flow list

# 4. Manually run a flow (optional - for testing)
cao flow run morning-trivia

# 5. View flow execution (after it runs)
tmux list-sessions
tmux attach -t <session-name>

# 6. Cleanup session when done
cao shutdown --session <session-name>
```

**IMPORTANT:** The `cao-server` must be running for flows to execute on schedule.

### Example 1: Simple Scheduled Task

A flow that runs at regular intervals with a static prompt (no script needed):

**File: `daily-standup.md`**

```yaml
---
name: daily-standup
schedule: "0 9 * * 1-5"  # 9am weekdays
agent_profile: developer
provider: kiro_cli  # Optional, defaults to kiro_cli
---

Review yesterday's commits and create a standup summary.
```

### Example 2: Conditional Execution with Health Check

A flow that monitors a service and only executes when there's an issue:

**File: `monitor-service.md`**

```yaml
---
name: monitor-service
schedule: "*/5 * * * *"  # Every 5 minutes
agent_profile: developer
script: ./health-check.sh
---

The service at [[url]] is down (status: [[status_code]]).
Please investigate and triage the issue:
1. Check recent deployments
2. Review error logs
3. Identify root cause
4. Suggest remediation steps
```

**Script: `health-check.sh`**

```bash
#!/bin/bash
URL="https://api.example.com/health"
STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$URL")

if [ "$STATUS" != "200" ]; then
  # Service is down - execute flow
  echo "{\"execute\": true, \"output\": {\"url\": \"$URL\", \"status_code\": \"$STATUS\"}}"
else
  # Service is healthy - skip execution
  echo "{\"execute\": false, \"output\": {}}"
fi
```

### Flow Commands

```bash
# Add a flow
cao flow add daily-standup.md

# List all flows (shows schedule, next run time, enabled status)
cao flow list

# Enable/disable a flow
cao flow enable daily-standup
cao flow disable daily-standup

# Manually run a flow (ignores schedule)
cao flow run daily-standup

# Remove a flow
cao flow remove daily-standup
```

## Working Directory Support

CAO supports specifying working directories for agent handoff/delegation operations. By default this is disabled to prevent agents from hallucinating directory paths.

All paths are canonicalized via `realpath` and validated against a security policy:

- **Allowed:** any real directory that is not a blocked system path — including `~/`, external volumes (e.g., `/Volumes/workplace`), and custom paths like `/opt/projects`
- **Blocked:** system directories (`/`, `/etc`, `/var`, `/tmp`, `/proc`, `/sys`, `/root`, `/boot`, `/bin`, `/sbin`, `/usr/bin`, `/usr/sbin`, `/lib`, `/lib64`, `/dev`)

For configuration and usage details, see [docs/working-directory.md](docs/working-directory.md).

## Cross-Provider Orchestration

By default, worker agents inherit the provider of the terminal that spawned them. To run specific agents on different providers, add a `provider` key to the agent profile frontmatter:

```markdown
---
name: developer
description: Developer Agent
provider: claude_code
---
```

Valid values: `kiro_cli`, `claude_code`, `codex`, `q_cli`, `gemini_cli`, `kimi_cli`, `copilot_cli`.

When a supervisor calls `assign` or `handoff`, CAO reads the worker's agent profile and uses the declared provider if present. If the key is missing or invalid, the worker falls back to the supervisor's provider.

The `cao launch --provider` flag always takes precedence — it is treated as an explicit override and the profile's `provider` key is not consulted for the initial session.

For ready-to-use examples, see [`examples/cross-provider/`](examples/cross-provider/).

## Tool Restrictions

CAO controls what tools each agent can use through `role` in the agent profile. Built-in roles (`supervisor`, `developer`, `reviewer`) map to sensible defaults, and `allowedTools` provides fine-grained override when needed. CAO translates restrictions to each provider's native enforcement mechanism — 5 of 7 providers support hard enforcement.

```yaml
---
name: my_agent
role: supervisor  # @cao-mcp-server, fs_read, fs_list
---
```

```bash
cao launch --agents code_supervisor                  # Uses role defaults (confirmation prompt shown)
cao launch --agents code_supervisor --auto-approve   # Skip prompt (restrictions still enforced)
cao launch --agents code_supervisor --yolo           # Unrestricted access (WARNING shown)
```

For the full reference — roles, tool vocabulary, custom roles, launch prompts, provider enforcement, and known limitations — see [docs/tool-restrictions.md](docs/tool-restrictions.md).

## Skills

Skills are portable, structured guides (following the universal [SKILL.md](https://github.com/anthropics/skills) format) that encode domain knowledge for AI agents. They work across AI coding assistants (Claude Code, Kiro CLI, Gemini CLI, Codex CLI, Kimi CLI, GitHub Copilot, Cursor, OpenCode, LobeHub), agent frameworks ([Strands Agents SDK](https://strandsagents.com/docs/user-guide/concepts/plugins/skills/), [Microsoft Agent Framework](https://devblogs.microsoft.com/agent-framework/give-your-agents-domain-expertise-with-agent-skills-in-microsoft-agent-framework/)), and other tools that support the SKILL.md format — allowing any agent to follow the same expert playbook regardless of provider.

CAO includes the following built-in skills:

| Skill | Description |
|-------|-------------|
| **[cao-provider](skills/cao-provider/SKILL.md)** | Scaffold a new CLI agent provider for CAO. Guides through the full implementation: ProviderType enum, provider class with regex patterns and status detection, ProviderManager registration, tool restriction wiring, unit/e2e tests, and documentation. Includes 20 lessons learnt from building 7 existing providers. |

### Loading Skills

Each AI coding tool loads skills from a different location. Copy or symlink the skill directory to the appropriate path for your tool:

| Tool | Skill Location | Command |
|------|---------------|---------|
| **Claude Code** | `.claude/skills/` | `cp -r skills/cao-provider .claude/skills/` |
| **Kiro CLI** | `.kiro/skills/` | `cp -r skills/cao-provider .kiro/skills/` |
| **Amazon Q CLI** | `.amazonq/skills/` | `cp -r skills/cao-provider .amazonq/skills/` |
| **Other tools** | Check your tool's docs for skill/prompt loading conventions |

Then ask your AI coding assistant to create a new provider:

```
> I want to add support for Aider CLI as a new CAO provider
```

The assistant will follow the skill's step-by-step guide, reference the provider template, and apply lessons learnt from existing providers.

### Managed Skills

CAO also manages skills that are shared across all agent sessions. Builtin skills (`cao-supervisor-protocols`, `cao-worker-protocols`) are auto-seeded when the `cao-server` starts — no `cao init` required.

```bash
# List installed skills
cao skills list

# Install a custom skill from a local folder
cao skills add ./my-coding-standards

# Update an existing skill (overwrite)
cao skills add ./my-coding-standards --force

# Remove a skill
cao skills remove my-coding-standards
```

Skills are delivered to each provider automatically:

| Provider | Delivery Method |
|----------|----------------|
| Kiro CLI | Native `skill://` resources (progressive loading) |
| Claude Code, Codex, Gemini CLI, Kimi CLI | Runtime prompt injection (every terminal creation) |
| Copilot CLI | Baked into `.agent.md` at install time |

When you add or remove a skill, all providers pick up the change automatically. Copilot agent files are refreshed immediately; other providers pick up changes on the next terminal creation.

**Updating skills:** Use `cao skills add ./my-skill --force` to overwrite an existing skill. Without `--force`, the command errors if the skill already exists. Builtin skills are auto-seeded on server startup but are never overwritten — to update a builtin after a CAO upgrade, remove it first with `cao skills remove` then restart the server.

For full details, see [docs/skills.md](docs/skills.md).

## Plugins

Plugins are observer-only extensions that react to server-side events inside `cao-server` — session and terminal lifecycle changes, and message delivery between agents. Typical uses include forwarding inter-agent messages to external chat (Discord, Slack), audit logging, and observability/metrics export.

Plugins are standard Python packages discovered automatically via the `cao.plugins` entry-point group at server startup. Install a plugin into the same environment as `cao-server`, configure it, and restart the server — no registration step required.

- **Installation, events, and troubleshooting:** [docs/plugins.md](docs/plugins.md)
- **Ready-to-run example:** [examples/plugins/cao-discord/](examples/plugins/cao-discord/)
- **Author your own plugin:** use the [cao-plugin skill](skills/cao-plugin/SKILL.md)

## Security

The server is designed for **localhost-only use**. The WebSocket terminal endpoint (`/terminals/{id}/ws`) provides full PTY access and will reject connections from non-loopback addresses. Do not expose the server to untrusted networks without adding authentication.

### DNS Rebinding Protection

The CAO server validates HTTP `Host` headers to prevent [DNS rebinding attacks](https://owasp.org/www-community/attacks/DNS_Rebinding). Only `localhost` and `127.0.0.1` are accepted by default — requests with other hostnames are rejected with `400 Bad Request`.

**Note:** If you need to expose the server on a network (not recommended for development use), be aware that the Host header validation will reject requests unless the hostname matches the allowed list.

See [SECURITY.md](SECURITY.md) for vulnerability reporting, security scanning, and best practices.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to this project.

## License

This project is licensed under the Apache-2.0 License.


---

# Appendix: Microsoft Amplifier (merged from `amplifier-amplifier-claude/`)

> The content below was merged in from a separate project (Microsoft Amplifier, Claude Code edition).
> It is preserved here for reference. It describes a different tool and is not required to run CAO.


> [!NOTE]
> **This is the Claude Code-based version of Amplifier.** The new modular Amplifier is now in the [`main`](https://github.com/microsoft/amplifier) branch.
> 
> ### About the New Modular Amplifier
> 
> The [`main`](https://github.com/microsoft/amplifier) branch contains the new modular Amplifier, which offers:
> 
> - **Provider Independence**: Works with Anthropic Claude, OpenAI, Azure OpenAI, Ollama, and more
> - **Modular Architecture**: Swap AI providers, tools, and behaviors like LEGO bricks
> - **True Extensibility**: Build your own modules, tools, and interfaces
> - **Profile-Based Configuration**: Pre-configured capability sets for different scenarios
> - **Cross-Platform**: Works on macOS, Linux, and WSL
> 
> **Install the modular Amplifier:**
> 
> ```bash
> uv tool install git+https://github.com/microsoft/amplifier
> amplifier init
> ```
> 
> **Learn more:** [github.com/microsoft/amplifier](https://github.com/microsoft/amplifier)

> _"Automate complex workflows by describing how you think through them."_

> [!CAUTION]
> This project is a research demonstrator. It is in early development and may change significantly. Using permissive AI tools in your repository requires careful attention to security considerations and careful human supervision, and even then things can still go wrong. Use it with caution, and at your own risk. See [Disclaimer](#disclaimer).

---

Amplifier is a coordinated and accelerated development system that turns your expertise into reusable AI tools without requiring code. Describe the step-by-step thinking process for handling a task—a "metacognitive recipe"—and Amplifier builds a tool that executes it reliably. As you create more tools, they combine and build on each other, transforming individual solutions into a compounding automation system.

## 🚀 QuickStart

### Prerequisites Guide

<details>
<summary>Click to expand prerequisite instructions</summary>

1. Check if prerequisites are already met.

   - ```bash
     python3 --version  # Need 3.11+
     ```
   - ```bash
     uv --version       # Need any version
     ```
   - ```bash
     node --version     # Need any version
     ```
   - ```bash
     pnpm --version     # Need any version
     ```
   - ```bash
     git --version      # Need any version
     ```

2. Install what is missing.

   **Mac**

   ```bash
   brew install python3 node git pnpm uv
   ```

   **Ubuntu/Debian/WSL**

   ```bash
   # System packages
   sudo apt update && sudo apt install -y python3 python3-pip nodejs npm git

   # pnpm
   npm install -g pnpm
   pnpm setup && source ~/.bashrc

   # uv (Python package manager)
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   **Windows**

   1. Install [WSL2](https://learn.microsoft.com/windows/wsl/install)
   2. Run Ubuntu commands above inside WSL

   **Manual Downloads**

   - [Python](https://python.org/downloads) (3.11 or newer)
   - [Node.js](https://nodejs.org) (any recent version)
   - [pnpm](https://pnpm.io/installation) (package manager)
   - [Git](https://git-scm.com) (any version)
   - [uv](https://docs.astral.sh/uv/getting-started/installation/) (Python package manager)

> **Platform Note**: Development and testing has primarily been done in Windows WSL2. macOS and Linux should work but have received less testing. Your mileage may vary.

</details>

### Setup

```bash
# Clone Amplifier repository
git clone https://github.com/microsoft/amplifier.git amplifier
cd amplifier

# Install dependencies
make install

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac/WSL
# .venv\Scripts\Activate.ps1  # Windows PowerShell
```

### Get Started

```bash
# Start Claude Code
claude
```

**Create your first tool in 5 steps:**

1. **Identify a task** you want to automate (e.g., "weekly learning digest")

   Need ideas? Try This:

   ```
   /ultrathink-task I'm new to "metacognitive recipes". What are some useful
   tools I could create with Amplifier that show how recipes can self-evaluate
   and improve via feedback loops? Just brainstorm ideas, don't build them yet.
   ```

2. **Describe the thinking process** - How would an expert handle it step-by-step?

   Need help? Try This:

   ```
   /ultrathink-task This is my idea: <your idea here>. Can you help me describe the
   thinking process to handle it step-by-step?
   ```

   Example of a metacognitive recipe:

   ```markdown
   I want to create a tool called "Research Synthesizer". Goal: help me research a topic by finding sources, extracting key themes, then asking me to choose which themes to explore in depth, and finally producing a summarized report.

   Steps:

   1. Do a preliminary web research on the topic and collect notes.
   2. Extract the broad themes from the notes.
   3. Present me the list of themes and highlight the top 2-3 you recommend focusing on (with reasons).
   4. Allow me to refine or add to that theme list.
   5. Do in-depth research on the refined list of themes.
   6. Draft a report based on the deep research, ensuring the report stays within my requested length and style.
   7. Offer the draft for my review and incorporate any feedback.
   ```

3. **Generate with `/ultrathink-task`** - Let Amplifier build the tool

   ```
   /ultrathink-task <your metacognitive recipe here>
   ```

4. **Refine through feedback** - "Make connections more insightful"

   ```
   Let's see how it works. Run <your generated tool>.
   ```

   Then:

   - Observe and note issues.
   - Provide feedback in context.
   - Iterate until satisfied.

**Learn more** with [Create Your Own Tools](docs/CREATE_YOUR_OWN_TOOLS.md) - Deep dive into the process.

---

## 📖 How to Use Amplifier

### Setup Your Project

```bash
# Clone Amplifier repository
git clone https://github.com/microsoft/amplifier.git amplifier
```

1. For existing GitHub projects

   ```bash
   # Add your project as a submodule
   cd amplifier
   git submodule add https://github.com/<your-username>/<your-project-name>.git my-project
   ```

2. For new projects

   ```bash
   # Create a new GitHub repository

   # Option 1: gh CLI
   gh repo create <your-username>/<your-project-name> --private

   # Option 2: Go to https://github.com/new
   ```

   ```bash
   # Initialize your new project
   git init my-project
   cd my-project/
   git remote add origin https://github.com/<your-username>/<your-project-name>.git
   echo "# My Project" > README.md
   git add .
   git commit -m "Initial commit"
   git push -u origin main

   # 2. Add as submodule
   cd ../amplifier
   git submodule add https://github.com/<your-username>/<your-project-name>.git my-project
   ```

```bash
# Install dependencies
make install

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac/WSL
# .venv\Scripts\Activate.ps1  # Windows PowerShell

# Set up project context & start Claude
echo "# Project-specific AI guidance" > my-project/AGENTS.md
claude
```

_Tell Claude Code:_

```
I'm working on @my-project/ with Amplifier.
Read @my-project/AGENTS.md for project context.
Let's use /ddd:1-plan to design the architecture.
```

> [!NOTE]
>
> **Why use this?** Clean git history per component, independent Amplifier updates, persistent context across sessions, scalable to multiple projects. See [Workspace Pattern for Serious Projects](#workspace-pattern-for-serious-projects) below for full details.

---

## ✨ Features To Try

### 🔧 Create Amplifier-powered Tools for Scenarios

Amplifier is designed so **you can create new AI-powered tools** just by describing how they should think. See the [Create Your Own Tools](docs/CREATE_YOUR_OWN_TOOLS.md) guide for more information.

- _Tell Claude Code:_ `Walk me through creating my own scenario tool`

- _View the documentation:_ [Scenario Creation Guide](docs/CREATE_YOUR_OWN_TOOLS.md)

### 🎨 Design Intelligence

Amplifier includes comprehensive design intelligence with 7 specialist agents, evidence-based design knowledge, and orchestrated design workflows:

- _Tell Claude Code:_

  `/designer create a button component with hover states and accessibility`

  `Use the art-director agent to establish visual direction for my app`

  `Deploy component-designer to create a reusable card component`

- _Available Design Specialists:_

  - **animation-choreographer** - Motion design and transitions
  - **art-director** - Aesthetic strategy and visual direction
  - **component-designer** - Component design and creation
  - **design-system-architect** - Design system architecture
  - **layout-architect** - Information architecture and layout
  - **responsive-strategist** - Device adaptation and responsive design
  - **voice-strategist** - Voice & tone for UI copy

- _Design Framework:_

  - **9 Dimensions** - Purpose, hierarchy, color, typography, spacing, responsive, accessibility, motion, voice
  - **4 Layers** - Foundational, structural, behavioral, experiential
  - **Evidence-based** - WCAG 2.1, color theory, animation principles, accessibility standards

- _View the documentation:_ [Design Intelligence](docs/design/README.md)

### 🤖 Explore Amplifier's agents on your code

Try out one of the specialized experts:

- _Tell Claude Code:_

  `Use the zen-architect agent to design my application's caching layer`

  `Deploy bug-hunter to find why my login system is failing`

  `Have security-guardian review my API implementation for vulnerabilities`

- _View the files:_ [Agents](.claude/agents/)

### 📝 Document-Driven Development

**Why use this?** Eliminate doc drift and context poisoning. When docs lead and code follows, your specifications stay perfectly in sync with reality.

Execute a complete feature workflow with numbered slash commands:

```bash
/ddd:1-plan         # Design the feature
/ddd:2-docs         # Update all docs (iterate until approved)
/ddd:3-code-plan    # Plan code changes
/ddd:4-code         # Implement and test (iterate until working)
/ddd:5-finish       # Clean up and finalize
```

Each phase creates artifacts the next phase reads. You control all git operations with explicit authorization at every step. The workflow prevents expensive mistakes by catching design flaws before implementation.

- _Tell Claude Code:_ `/ddd:0-help`

- _View the documentation:_ [Document-Driven Development Guide](docs/document_driven_development/)

### 🌳 Parallel Development

**Why use this?** Stop wondering "what if" — build multiple solutions simultaneously and pick the winner.

```bash
# Try different approaches in parallel
make worktree feature-jwt     # JWT authentication approach
make worktree feature-oauth   # OAuth approach in parallel

# Compare and choose
make worktree-list            # See all experiments
make worktree-rm feature-jwt  # Remove the one you don't want
```

Each worktree is completely isolated with its own branch, environment, and context.

See the [Worktree Guide](docs/WORKTREE_GUIDE.md) for advanced features, such as hiding worktrees from VSCode when not in use, adopting branches from other machines, and more.

- _Tell Claude Code:_ `What make worktree commands are available to me?`

- _View the documentation:_ [Worktree Guide](docs/WORKTREE_GUIDE.md)

### 📊 Enhanced Status Line

See costs, model, and session info at a glance:

**Example**: `~/repos/amplifier (main → origin) Opus 4.1 💰$4.67 ⏱18m`

Shows:

- Current directory and git branch/status
- Model name with cost-tier coloring (red=high, yellow=medium, blue=low)
- Running session cost and duration

Enable with:

```
/statusline use the script at .claude/tools/statusline-example.sh
```

### 💬 Conversation Transcripts

**Never lose context again.** Amplifier automatically exports your entire conversation before compaction, preserving all the details that would otherwise be lost. When Claude Code compacts your conversation to stay within token limits, you can instantly restore the full history.

**Automatic Export**: A PreCompact hook captures your conversation before any compaction event:

- Saves complete transcript with all content types (messages, tool usage, thinking blocks)
- Timestamps and organizes transcripts in `.data/transcripts/`
- Works for both manual (`/compact`) and auto-compact events

**Easy Restoration**: Use the `/transcripts` command in Claude Code to restore your full conversation:

```
/transcripts  # Restores entire conversation history
```

The transcript system helps you:

- **Continue complex work** after compaction without losing details
- **Review past decisions** with full context
- **Search through conversations** to find specific discussions
- **Export conversations** for sharing or documentation

**Transcript Commands** (via Makefile):

```bash
make transcript-list                # List available transcripts
make transcript-search TERM="auth"  # Search past conversations
make transcript-restore             # Restore full lineage (for CLI use)
```

### 🏗️ Workspace Pattern for Serious Projects

**For long-term development**, consider using the workspace pattern where Amplifier hosts your project as a git submodule. This architectural approach provides:

- **Clean boundaries** - Project files stay in project directory, Amplifier stays pristine and updatable
- **Version control isolation** - Each component maintains independent git history
- **Context persistence** - AGENTS.md preserves project guidance across sessions
- **Scalability** - Work on multiple projects simultaneously without interference
- **Philosophy alignment** - Project-specific decision filters and architectural principles

Perfect for:

- Projects that will live for months or years
- Codebases with their own git repository
- Teams collaborating on shared projects
- When you want to update Amplifier without affecting your projects
- Working on multiple projects that need isolation

The pattern inverts the typical relationship: instead of your project containing Amplifier, Amplifier becomes a dedicated workspace that hosts your projects. Each project gets persistent context through AGENTS.md (AI guidance), philosophy documents (decision filters), and clear namespace boundaries using `@project-name/` syntax.

- _Tell Claude Code:_ `What are the recommended workspace patterns for serious projects?`

- _View the documentation:_ [Workspace Pattern Guide](docs/WORKSPACE_PATTERN.md) - complete setup, usage patterns, and migration from `ai_working/`.

### 💡 Best Practices & Tips

**Want to get the most out of Amplifier?** Check out [The Amplifier Way](docs/THIS_IS_THE_WAY.md) for battle-tested strategies including:

- Understanding capability vs. context
- Decomposition strategies for complex tasks
- Using transcript tools to capture and improve workflows
- Demo-driven development patterns
- Practical tips for effective AI-assisted development

- _Tell Claude Code:_ `What are the best practices to get the MOST out of Amplifier?`

- _View the documentation:_ [The Amplifier Way](docs/THIS_IS_THE_WAY.md)

### ⚙️ Development Commands

```bash
make check            # Format, lint, type-check
make test             # Run tests
make ai-context-files # Rebuild AI context
```

### 🧪 Testing & Benchmarks

Testing and benchmarking are critical to ensuring that any product leveraging AI, including Amplifier, is quantitatively measured for performance and reliability.
Currently, we leverage [terminal-bench](https://github.com/laude-institute/terminal-bench) to reproducibly benchmark Amplifier against other agents.
Further details on how to run the benchmark can be found in [tests/terminal_bench/README.md](tests/terminal_bench/README.md).

---

## Disclaimer

> [!IMPORTANT] > **This is an experimental system. _We break things frequently_.**

- Not accepting contributions yet (but we plan to!)
- No stability guarantees
- Pin commits if you need consistency
- This is a learning resource, not production software
- **No support provided** - See [SUPPORT.md](SUPPORT.md)

## Contributing

> [!NOTE]
> This project is not currently accepting external contributions, but we're actively working toward opening this up. We value community input and look forward to collaborating in the future. For now, feel free to fork and experiment!

Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [Contributor License Agreements](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
