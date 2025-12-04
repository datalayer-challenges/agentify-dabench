# DABench AgentBeats Implementation

A complete A2A (Agent-to-Agent) compatible implementation of the [DABench](https://github.com/InfiAgent/InfiAgent/tree/main/examples/DA-Agent/data) benchmark, following the [AgentBeats](https://rdi.berkeley.edu/agentx-agentbeats) methodology with Green Agent (evaluator) and White Agent (test subject) architecture.

## Overview

![](image.png)

This project implements the [DABench](https://github.com/InfiAgent/InfiAgent/tree/main/examples/DA-Agent/data) benchmark as an A2A-compatible evaluation system where:

- **Green Agent** (Evaluator): Manages DABench assessments and evaluates other agents
- **White Agent** (Test Subject): The agent being evaluated, with MCP tool capabilities
- **Launcher**: One-command execution script for easy setup and evaluation

The Data Agent Benchmark (DABench) is designed to measure and push the state-of-the-art in Data Analysis by LLMs.

## Features

- ✅ **A2A Protocol Compatible**: Full compatibility with Agent-to-Agent standard using [Pydantic FastA2A](https://github.com/pydantic/fasta2a)
- ✅ **[AgentBeats](https://rdi.berkeley.edu/agentx-agentbeats) Architecture**: Proper green/white agent separation
- ✅ **DABench Scoring**: [DABench](https://github.com/InfiAgent/InfiAgent/tree/main/examples/DA-Agent/data) benchmark dataset 
- ✅ **PydanticA AI Agent and Evaluation**: Utilizes [Pydantic AI](https://ai.pydantic.dev/evals/evaluators/llm-judge/) for agent and evaluation
- ✅ **MCP Tools Integration**: White agent supports [jupyter-mcp-server](https://github.com/datalayer/jupyter-mcp-server) tools

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys and model preferences
```

### Option 1: All-in-One Launcher Script

```bash
# Start both green and white agents
python launcher.py

# Full dataset evaluation
python launcher.py --evaluate --full

# Quick sample evaluation (3 tasks)
python launcher.py --evaluate --quick-sample 3
```

### Option 2: Separate Services with Makefile

For better control and monitoring, use the 4-terminal workflow:

```bash
# Terminal 1: Start MCP Server
make start-mcp

# Terminal 2: Start White Agent (Test Subject)  
make start-white

# Terminal 3: Start Green Agent (Evaluator)
make start-green

# Terminal 4: Run Evaluation
make run-eval-monitor           # Full evaluation with real-time monitoring
make run-eval-quick-monitor     # Quick 3-task evaluation with real-time monitoring
```

### Option 3: Docker Development

For containerized development, use the Docker workflow:

```bash
# Build all Docker images (run this first or after code changes)
make docker-build-all

# Start services in separate terminals:
# Terminal 1: Start Jupyter MCP Server
make docker-start-jupyter

# Terminal 2: Start White Agent (Test Subject)
make docker-start-white

# Terminal 3: Start Green Agent (Evaluator)
make docker-start-green

# Terminal 4: Run Evaluation (on Docker containers)
make docker-run-eval-quick-monitor     # Quick 3-task evaluation with monitoring
make docker-run-eval-monitor           # Full dataset evaluation with monitoring  
```

#### Platform-Specific Docker Configuration

**macOS/Windows**: The Docker configuration uses `host.docker.internal` for container-to-host communication (default setup).

**Linux**: Linux Docker doesn't support `host.docker.internal`. Add `--network=host` to these Makefile commands:
- Line ~270: `docker-start-jupyter` command - add `--network=host \` after `@$(DOCKER_RUN) --rm -it \`
- Line ~278: `docker-start-white` command - add `--network=host \` after `@$(DOCKER_RUN) --rm -it \`  
- Line ~287: `docker-start-green` command - add `--network=host \` after `@$(DOCKER_RUN) --rm -it \`

## Configuration

### Environment Variables

The system uses environment variables for configuration. Copy `.env.example` to `.env` and configure:

```bash
# Agent Settings

# LLM Configuration (LiteLLM supports 100+ providers)
LLM_API_KEY=your_api_key_here
GREEN_AGENT_MODEL=gpt-3.5-turbo     # Any LiteLLM-supported model
WHITE_AGENT_MODEL=gpt-4             # Any LiteLLM-supported model

# For Azure OpenAI (additional config)
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

## Jupyter MCP Server Integration

The White Agent is integrated with [**Jupyter MCP (Model Context Protocol) Server**](https://github.com/datalayer/jupyter-mcp-server) for enhanced data analysis and code execution capabilities. This integration provides the agent with powerful computational tools for autonomous problem-solving.


### Authentication & Security

#### Token-Based Authentication
The MCP server uses Bearer token authentication for security:

```python
# White Agent automatically uses token from environment
headers = {"Authorization": f"Bearer {self.jupyter_token}"}

# Automatic fallback to no-auth if token fails
# (for development environments)
```

#### Network Configuration
- **Default Port**: 8888 (JupyterLab standard)
- **MCP Endpoints**: `http://localhost:8888/mcp/*`
- **Health Check**: `http://localhost:8888/mcp/healthz`
- **Tools List**: `http://localhost:8888/mcp/tools/list`
- **Tool Execution**: `http://localhost:8888/mcp/tools/call`

### Data Context Files

The White Agent has access to DABench data files in `agent-workings/data/`.
