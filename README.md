# DABench AgentBeats Implementation

A complete A2A (Agent-to-Agent) compatible implementation of the [DABench](https://github.com/InfiAgent/InfiAgent/tree/main/examples/DA-Agent/data) benchmark, following the [AgentBeats](https://rdi.berkeley.edu/agentx-agentbeats) methodology with Green Agent (evaluator) and Purple Agent (test subject) architecture.

The Data Agent Benchmark (DABench) is designed to measure and push the state-of-the-art in **Data Analysis tasks** for AI agents.

## Overview

![](image.png)

This project implements the [DABench](https://github.com/InfiAgent/InfiAgent/tree/main/examples/DA-Agent/data) benchmark as an A2A-compatible evaluation system where:

- **Green Agent** (Evaluator): Manages DABench assessments and evaluates other agents
- **Purple Agent** (Test Subject): The agent being evaluated, with embedded Jupyter MCP capabilities
- **Launcher**: One-command execution script for easy setup and evaluation

## Features

- ✅ **A2A Protocol Compatible**: Full compatibility with Agent-to-Agent standard using [Pydantic FastA2A](https://github.com/pydantic/fasta2a)
- ✅ **AgentBeats Architecture**: Proper green/purple agent separation as per [AgentBeats](https://rdi.berkeley.edu/agentx-agentbeats) guidelines
- ✅ **DABench Scoring**: [DABench](https://github.com/InfiAgent/InfiAgent/tree/main/examples/DA-Agent/data) benchmark dataset 
- ✅ **PydanticA AI Agent and Evaluation**: Utilizes [Pydantic AI](https://ai.pydantic.dev/evals/evaluators/llm-judge/) for agent and evaluation
- ✅ **Embedded MCP Tools**: Purple agent includes embedded [jupyter-mcp-server](https://github.com/datalayer/jupyter-mcp-server) for autonomous code execution

## Quick Start

### Configuration

The system uses environment variables for configuration. Copy `.env.template` to `.env` and configure:

```bash
# LLM Configuration - Pydantic AI supports multiple providers
LLM_API_KEY=your_api_key_here

# Model Configuration (specify provider explicitly using Pydantic AI format)
GREEN_AGENT_MODEL=openai:gpt-4o             # OpenAI format: openai:model_name
PURPLE_AGENT_MODEL=openai:gpt-4o             # OpenAI format: openai:model_name

# Alternative provider examples:
# GREEN_AGENT_MODEL=azure:gpt-4             # Azure OpenAI (requires endpoint below)
# GREEN_AGENT_MODEL=anthropic:claude-3-sonnet # Anthropic Claude
# GREEN_AGENT_MODEL=gemini:gemini-pro       # Google Gemini
# GREEN_AGENT_MODEL=groq:llama3-70b         # Groq

# Azure OpenAI Configuration (when using azure: models)
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-06-01
```

### Option 1: All-in-One Launcher Script

```bash
# Install dependencies
pip install -r requirements.txt

# Full dataset evaluation
python launcher.py --evaluate --full

# Quick sample evaluation (3 tasks)
python launcher.py --evaluate --quick-sample 3
```

### Option 2: Separate Services with Makefile

For better control and monitoring, use the 3-terminal workflow:

```bash
# Install dependencies
pip install -r requirements.txt

# Terminal 1: Start Purple Agent (Test Subject with embedded MCP)  
make start-purple

# Terminal 2: Start Green Agent (Evaluator)
make start-green

# Terminal 3: Run Evaluation
make run-eval-monitor           # Full evaluation with real-time monitoring
make run-eval-quick-monitor     # Quick 3-task evaluation with real-time monitoring
```

### Option 3: Docker Development

For containerized development, use the Docker workflow:

```bash
# Build all Docker images (run this first or after code changes)
make docker-build-all
```

#### macOS/Windows Docker Workflow

```bash
# Start services in separate terminals:
# Terminal 1: Start Purple Agent with embedded MCP (Test Subject)
make docker-start-purple

# Terminal 2: Start Green Agent (Evaluator)
make docker-start-green

# Terminal 3: Run Evaluation (on Docker containers)
make docker-run-eval-quick-monitor     # Quick 3-task evaluation with monitoring
make docker-run-eval-monitor           # Full dataset evaluation with monitoring
```

#### Linux Docker Workflow

```bash
# Start services in separate terminals:
# Terminal 1: Start Purple Agent with embedded MCP (Test Subject)
make docker-start-purple-linux

# Terminal 2: Start Green Agent (Evaluator)
make docker-start-green-linux

# Terminal 4: Run Evaluation (on Docker containers)
make docker-run-eval-quick-monitor-linux     # Quick 3-task evaluation with monitoring
make docker-run-eval-monitor-linux           # Full dataset evaluation with monitoring
```

### Evaluation Results

After evaluation, Pydantic AI generates a detailed report in the `results/` directory, including:
- Overall scores and metrics
- Task-by-task performance breakdown
- Green Agent reason for scores

## Embedded Jupyter MCP Integration

The Purple Agent includes an embedded [**Jupyter MCP (Model Context Protocol) Server**](https://github.com/datalayer/jupyter-mcp-server) for enhanced data analysis and code execution capabilities. This embedded server provides the agent with powerful computational tools for autonomous problem-solving without requiring a separate container.

### Embedded MCP Features

#### Automatic Startup
The Purple Agent automatically starts its own Jupyter MCP server:

```python
# Purple Agent starts embedded Jupyter MCP server on initialization
await self._start_embedded_jupyter_mcp()

# Automatically generates tokens and manages server lifecycle
# No external MCP server setup required
```

#### Network Configuration
- **Agent Port**: 9019 (Purple Agent A2A endpoint)
- **Embedded MCP Port**: 8888 (JupyterLab standard)
- **MCP Endpoints**: `http://localhost:8888/mcp/*`
- **Health Check**: `http://localhost:8888/mcp/healthz`
- **Tools List**: `http://localhost:8888/mcp/tools/list`
- **Tool Execution**: `http://localhost:8888/mcp/tools/call`

## Data Context Files & DABench Tasks

The evaluation process involves two data components:

- **Data Files**: 68 diverse CSV datasets in `src/purple/agent-workings/data/` available for the Purple Agent to analyze through the Jupyter MCP Server
- **Task Distribution**: Green Agent sends tasks one-by-one from the 257 DABench evaluation tasks stored in `data-dabench/`
