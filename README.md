# DABSTEP AgentBeats Implementation

A complete A2A (Agent-to-Agent) compatible implementation of the DABSTEP benchmark, following the AgentBeats methodology with Green Agent (evaluator) and White Agent (test subject) architecture.

## Overview

This project implements the DABSTEP (Difficult Algorithmic and Benchmark Tasks for Evaluating the Performance of Language Models) benchmark as an A2A-compatible evaluation system where:

- **Green Agent** (Evaluator): Manages DABSTEP assessments and evaluates other agents
- **White Agent** (Test Subject): The agent being evaluated, with MCP tool capabilities
- **Launcher**: One-command execution script for easy setup and evaluation

## Features

- ✅ **A2A Protocol Compatible**: Full compatibility with Agent-to-Agent communication standard
- ✅ **DABSTEP Scoring**: Original DABSTEP benchmark scoring methodology
- ✅ **AgentBeats Architecture**: Proper green/white agent separation
- ✅ **LLM Provider Integration**: 100+ LLM providers via LiteLLM (OpenAI, Anthropic, Google, Cohere, Ollama, etc.)
- ✅ **MCP Tools Integration**: White agent supports jupyter-mcp-server tools
- ✅ **One-Command Launch**: Simple launcher script for complete setup
- ✅ **Interactive Mode**: Real-time evaluation and monitoring capabilities
- ✅ **Environment Configuration**: Flexible configuration via .env files

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys and model preferences
```

### Launch Both Agents

```bash
# Start both green and white agents
python launcher.py

# Quick evaluation (3 tasks)
python launcher.py --evaluate

# Development evaluation (10 tasks)
python launcher.py --evaluate --sample-mode dev

# Full dataset evaluation (450 tasks)
python launcher.py --evaluate --full-dataset

# Interactive mode with commands
python launcher.py --interactive
```

### Individual Agent Launch

```bash
# Start only green agent (evaluator)
python launcher.py --green-only

# Start only white agent (test subject)
python launcher.py --white-only
```

## Configuration

### Environment Variables

The system uses environment variables for configuration. Copy `.env.example` to `.env` and configure:

```bash
# Agent Settings

# LLM Configuration (LiteLLM supports 100+ providers)
LLM_API_KEY=your_api_key_here
LLM_MODEL=gpt-3.5-turbo               # Any LiteLLM-supported model

# For Azure OpenAI (additional config)
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

### Supported LLM Providers (via LiteLLM)

- **OpenAI**: `gpt-3.5-turbo`, `gpt-4`, `gpt-4-turbo`
- **Azure OpenAI**: `azure/your-deployment-name`
- **Anthropic**: `claude-3-haiku-20240307`, `claude-3-sonnet-20240229`, `claude-3-opus-20240229`
- **Google**: `gemini-pro`, `gemini-1.5-pro`
- **Cohere**: `command-r`, `command-r-plus`
- **Ollama**: `ollama/llama2`, `ollama/codellama`
- **100+ more providers** supported via LiteLLM

## JupyterLab MCP Server Integration

The White Agent is integrated with **JupyterLab MCP (Model Context Protocol) Server** for enhanced data analysis and code execution capabilities. This integration provides the agent with powerful computational tools for autonomous problem-solving.

### MCP Server Setup

#### 1. Install JupyterLab MCP Extension

```bash
# Install JupyterLab with MCP extension
pip install jupyterlab==4.4.1
pip install jupyter-mcp-server

# Start JupyterLab with MCP extension and token authentication
jupyter lab --port=8888 --ip=0.0.0.0 --no-browser \
    --IdentityProvider.token=your_secure_token_here \
    --ServerApp.disable_check_xsrf=True \
    --ServerApp.allow_remote_access=True
```

#### 2. Environment Configuration

Add to your `.env` file:

```bash
# JupyterLab MCP Server Configuration
JUPYTER_TOKEN=your_secure_token_here
JUPYTER_PORT=8888
JUPYTER_HOST=localhost

# The launcher will automatically set this for the White Agent
# when starting JupyterLab with MCP extension
```

#### 3. Automatic Launcher Integration

The launcher script (`launcher.py`) automatically handles JupyterLab MCP server startup:

```bash
# Starts JupyterLab with MCP extension automatically
python launcher.py --evaluate

# Manual JupyterLab startup (if needed)
python launcher.py --start-jupyter
```

### MCP Tools Available

The JupyterLab MCP server provides **14 powerful tools** for the White Agent:

#### 📊 **Data Analysis Tools**
- `mcp_datalayerjupy_execute_ipython` - Execute Python code with full pandas/numpy support
- `mcp_datalayerjupy_list_files` - Explore available data files and directory structure
- `mcp_datalayerjupy_read_cell` - Read specific notebook cells for context

#### 🔄 **Notebook Management Tools**
- `mcp_datalayerjupy_use_notebook` - Connect to or create notebooks
- `mcp_datalayerjupy_list_notebooks` - View available notebooks
- `mcp_datalayerjupy_insert_cell` - Add new code/markdown cells
- `mcp_datalayerjupy_execute_cell_simple_timeout` - Run cells with timeout
- `mcp_datalayerjupy_overwrite_cell_source` - Modify existing cells

#### ⚙️ **System and Kernel Tools**
- `mcp_datalayerjupy_list_kernels` - View available Python kernels
- `mcp_datalayerjupy_restart_notebook` - Restart notebook kernels
- `mcp_datalayerjupy_assign_kernel_to_notebook` - Connect kernels to notebooks
- `mcp_datalayerjupy_execute_cell_streaming` - Long-running cell execution with progress
- `mcp_datalayerjupy_execute_cell_with_progress` - Execute with progress monitoring
- `mcp_datalayerjupy_delete_cell` - Remove cells from notebooks

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

The White Agent has access to DABSTEP data files in `/Users/eleonorecharles/Desktop/dabstep/data/context/`:

- **`payments.csv`** - Transaction data for financial analysis
- **`acquirer_countries.csv`** - Country mapping data
- **`merchant_category_codes.csv`** - Business category classifications
- **`merchant_data.json`** - Merchant information database
- **`fees.json`** - Fee structure data
- **`manual.md`** - Documentation and context information

### AI-Driven Tool Usage

The White Agent uses **autonomous AI reasoning** to decide which MCP tools to use:

```python
# Example: AI generates analysis code automatically
prompt = f"""Generate Python code to answer: {question}
Available tools: {available_mcp_tools}
Data files: payments.csv, merchant_data.json, etc.
"""

# LLM generates intelligent exploration code
ai_code = await llm.generate_code(prompt)

# Execute via MCP server
result = await mcp_client.call_tool(
    "mcp_datalayerjupy_execute_ipython", 
    {"code": ai_code}
)
```

### Usage Examples

#### Autonomous Data Analysis
```python
# White Agent receives: "What country has the highest number of transactions?"

# AI generates and executes:
import pandas as pd
df = pd.read_csv('/path/to/payments.csv')
country_counts = df['ip_country'].value_counts()
print(f"FINAL_ANSWER: {country_counts.index[0]}")
```

#### Complex Multi-Step Analysis
```python
# White Agent receives: "Analyze fraud patterns by merchant category"

# AI generates comprehensive analysis:
import pandas as pd
import json

# Load data
payments = pd.read_csv('payments.csv')
merchant_data = json.load(open('merchant_data.json'))

# Merge and analyze
merged = payments.merge(merchant_data, on='merchant_id')
fraud_by_category = merged.groupby('category')['is_fraud'].mean()
print(f"FINAL_ANSWER: {fraud_by_category.to_dict()}")
```

### Performance & Monitoring

#### Health Monitoring
The launcher continuously monitors MCP server health:

```bash
# Health check output
✅ JupyterLab MCP server: http://localhost:8888
✅ 14 MCP tools available for AI agent
🔑 White Agent token: xhRr-sGX...
```

#### Execution Timeouts
- **Default timeout**: 60 seconds for code execution
- **Streaming execution**: For long-running analysis
- **Progress monitoring**: Real-time feedback for complex tasks

### Troubleshooting MCP Integration

#### Common Issues and Solutions

**1. MCP Server Not Starting**
```bash
# Check JupyterLab installation
pip install jupyterlab==4.4.1
pip install jupyter-mcp-server

# Verify port availability
netstat -an | grep 8888
```

**2. Authentication Errors**
```bash
# Set token in environment
export JUPYTER_TOKEN=your_secure_token

# Verify token in launcher output
🔑 White Agent token: your_token...
```

**3. Tool Discovery Failures**
```bash
# Test MCP endpoints manually
curl http://localhost:8888/mcp/healthz
curl -H "Authorization: Bearer your_token" \
     http://localhost:8888/mcp/tools/list
```

**4. Code Execution Errors**
- Check data file paths in `/data/context/`
- Verify Python environment has required packages
- Review execution logs in launcher output

### Integration Benefits

The JupyterLab MCP integration provides the White Agent with:

- ✅ **Autonomous Code Generation** - AI creates analysis code
- ✅ **Dynamic Tool Discovery** - Runtime detection of available tools
- ✅ **Secure Execution Environment** - Token-based authentication
- ✅ **Rich Data Analysis** - Full pandas/numpy/matplotlib support
- ✅ **Interactive Notebooks** - Persistent computation context
- ✅ **Progress Monitoring** - Real-time execution feedback
- ✅ **Error Handling** - Graceful fallbacks and error recovery

This makes the White Agent truly autonomous, capable of handling complex data analysis tasks without hardcoded rules or patterns.

## Architecture

### Green Agent (Port 8000)
- **Role**: DABSTEP evaluator and assessment coordinator
- **Location**: `src/green_agent/agent.py`
- **Capabilities**:
  - Receives evaluation requests with tasks and target agent URL
  - Communicates with white agent via A2A protocol
  - Applies DABSTEP scoring methodology
  - **LLM-Enhanced Analysis**: Provides detailed reasoning analysis and improvement suggestions
  - Returns comprehensive evaluation results with confidence scores

### White Agent (Port 8001)
- **Role**: Agent under test with enhanced capabilities
- **Location**: `src/white_agent/agent.py`
- **Capabilities**:
  - Processes evaluation tasks from green agent
  - **LLM-Powered Reasoning**: Uses configured LLM for advanced problem-solving
  - Accesses MCP tools (jupyter-mcp-server integration ready)
  - Demonstrates problem-solving across various domains
  - Provides responses for DABSTEP evaluation

### Launcher Script
- **Location**: `launcher.py`
- **Features**:
  - One-command setup for both agents
  - Health checking and monitoring
  - Sample evaluation execution
  - Interactive mode for custom evaluations

## Usage Examples

### Dataset Options

The system supports three evaluation modes:

- **Quick Mode** (`--sample-mode quick`): 3 tasks for fast testing
- **Dev Mode** (`--sample-mode dev`): 10 tasks from development set
- **Full Mode** (`--full-dataset`): 450 tasks from complete DABSTEP dataset

### Running a DABSTEP Evaluation

```bash
# Quick test with 3 tasks
python launcher.py --evaluate

# Development test with 10 tasks
python launcher.py --evaluate --sample-mode dev

# Full benchmark with 450 tasks
python launcher.py --evaluate --full-dataset

# Custom quick sample size
python launcher.py --evaluate --quick-sample 5
```

### Real DABSTEP Data

The system loads tasks from `data/tasks/`:
- `dev.jsonl`: 10 development tasks
- `all.jsonl`: 450 complete dataset tasks

Tasks include real DABSTEP questions like:
- Financial analysis and fraud detection
- Data processing and statistics
- Complex reasoning with context files

### Custom Evaluation via A2A Client

```python
from fasta2a.client import A2AClient
from fasta2a.schema import Message, TextPart, DataPart

# Create evaluation request
eval_request = {
    "white_agent_url": "http://localhost:8001",
    "tasks": your_tasks
}

# Send to green agent
client = A2AClient(base_url="http://localhost:8000")
message = Message(
    role='user',
    parts=[
        TextPart(text="Please evaluate the white agent using DABSTEP tasks.", kind='text'),
        DataPart(data=eval_request, kind='data')
    ],
    kind='message',
    message_id="eval_001"
)

response = await client.send_message(message)
```

### Interactive Commands

When running `python launcher.py --interactive`:

- `eval` - Run sample DABSTEP evaluation
- `status` - Check health of both agents
- `quit` - Exit interactive mode

## Agent Endpoints

### Green Agent (http://localhost:8000)
- Agent Card: `/.well-known/agent-card.json`
- A2A Endpoint: `/`
- Skills: DABSTEP benchmark evaluation

### White Agent (http://localhost:8001) 
- Agent Card: `/.well-known/agent-card.json`
- A2A Endpoint: `/`
- Skills: Problem solving with MCP tools

## Development

### Project Structure

```
dabstep/
├── src/
│   ├── green_agent/
│   │   ├── agent.py          # Green agent implementation
│   │   └── scorer.py         # DABSTEP scoring logic (ESSENTIAL)
│   └── white_agent/
│       └── agent.py          # White agent implementation
├── launcher.py               # Main launcher script
├── requirements.txt          # Dependencies
├── .env.example             # Environment configuration template
└── README.md                 # This file
```

### Extending the White Agent

To add MCP tool capabilities:

1. Install jupyter-mcp-server
2. Modify white agent's `_process_question` method
3. Add tool-specific handling in the agent logic

### Adding Custom Evaluation Tasks

```python
# Define your tasks following DABSTEP format
custom_tasks = [
    {
        "task_id": "custom_1",
        "question": "Your question here",
        "correct_answer": "Expected answer",
        "level": "difficulty_level"
    }
]

# Send via launcher or A2A client
```

## Troubleshooting

### Port Conflicts
- Green agent uses port 8000
- White agent uses port 8001
- Modify ports in agent files if needed

### Agent Health Checks
```bash
# Check green agent
curl http://localhost:8000/.well-known/agent-card.json

# Check white agent  
curl http://localhost:8001/.well-known/agent-card.json
```

### Logs and Debugging
- Agents output logs to stdout
- Use `--interactive` mode for real-time monitoring
- Check agent processes with launcher status commands

## DABSTEP Benchmark Details

The DABSTEP benchmark tests agents on:
- Mathematical calculations
- Logical reasoning
- Data analysis tasks
- Code execution problems
- Multi-step problem solving

Scoring uses the original DABSTEP methodology with exact match comparison and numerical tolerance for mathematical answers.

## Contributing

This implementation follows the AgentBeats methodology and A2A protocol standards. Contributions should maintain compatibility with both frameworks.

## License

This project builds upon the original DABSTEP benchmark and implements the A2A protocol for agent evaluation.