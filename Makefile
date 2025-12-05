.PHONY: help start-mcp start-white start-green run-eval-monitor run-eval-quick-monitor docker-build-all docker-start-jupyter docker-start-white docker-start-green docker-run-eval-monitor docker-run-eval-quick-monitor docker-start-jupyter-linux docker-start-white-linux docker-start-green-linux docker-run-eval-monitor-linux docker-run-eval-quick-monitor-linux

# Default Python executable
PYTHON := python

# Directories
PROJECT_ROOT := $(shell pwd)
WHITE_DIR := src/white_agent
GREEN_DIR := src/green_agent
AGENT_WORKINGS_DIR := agent-workings
LOGS_DIR := logs

# Ports (matching launcher.py)
JUPYTER_PORT := 8888
GREEN_PORT := 8000
WHITE_PORT := 8001

# Log directory with timestamp
LOG_DIR := $(LOGS_DIR)/run_$(shell date +%Y%m%d_%H%M%S)

# Colors for output
RED := \033[31m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m
CYAN := \033[36m
RESET := \033[0m

help: ## Show this help message
	@echo "$(CYAN)AgentBeats Makefile - DABench A2A Implementation$(RESET)"
	@echo ""
	@echo "$(YELLOW)Available commands:$(RESET)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-30s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# ============================================================================
# Local Development Commands
# ============================================================================

start-mcp: ## Start JupyterLab MCP server (foreground - keeps terminal busy)
	@echo "$(BLUE)📊 Starting JupyterLab with MCP Server...$(RESET)"
	@mkdir -p $(LOG_DIR)
	@JUPYTER_TOKEN=$${JUPYTER_TOKEN:-$$(openssl rand -hex 16)} && \
	echo "$(CYAN)🔑 Using Jupyter token: $${JUPYTER_TOKEN:0:8}...$(RESET)" && \
	echo "$(CYAN)🔗 JupyterLab URL: http://localhost:$(JUPYTER_PORT)/lab?token=$${JUPYTER_TOKEN}$(RESET)" && \
	echo "$(YELLOW)📝 Logs will also be saved to: $(LOG_DIR)/jupyter_mcp.log$(RESET)" && \
	echo "$(YELLOW)🚀 Starting JupyterLab... (Press Ctrl+C to stop)$(RESET)" && \
	echo "$(BLUE)────────────────────────────────────────$(RESET)" && \
	export JUPYTER_TOKEN=$${JUPYTER_TOKEN} && \
	cd $(AGENT_WORKINGS_DIR) && \
	jupyter lab \
		--port=$(JUPYTER_PORT) \
		--no-browser \
		--IdentityProvider.token=$${JUPYTER_TOKEN} \
	| tee ../$(LOG_DIR)/jupyter_mcp.log

start-white: ## Start white agent (foreground - keeps terminal busy)
	@echo "$(BLUE)⚪ Starting White Agent (Test Subject)...$(RESET)"
	@mkdir -p $(LOG_DIR)
	@echo "$(CYAN)🔗 White agent endpoint: http://localhost:$(WHITE_PORT)$(RESET)"
	@echo "$(YELLOW)🚀 Starting White Agent... (Press Ctrl+C to stop)$(RESET)"
	@echo "$(BLUE)────────────────────────────────────────$(RESET)"
	@cd $(WHITE_DIR) && $(PYTHON) agent.py | tee ../../$(LOG_DIR)/white_agent.log

start-green: ## Start green agent (foreground - keeps terminal busy)
	@echo "$(BLUE)🟢 Starting Green Agent (Evaluator)...$(RESET)"
	@mkdir -p $(LOG_DIR)
	@echo "$(CYAN)🔗 Green agent endpoint: http://localhost:$(GREEN_PORT)$(RESET)"
	@echo "$(YELLOW)🚀 Starting Green Agent... (Press Ctrl+C to stop)$(RESET)"
	@echo "$(BLUE)────────────────────────────────────────$(RESET)"
	@cd $(GREEN_DIR) && $(PYTHON) agent.py | tee ../../$(LOG_DIR)/green_agent.log

run-eval-monitor: ## Send evaluation request and monitor progress (keeps terminal busy)
	@echo "$(BLUE)🔍 Sending full evaluation request with monitoring...$(RESET)"
	@$(PYTHON) send_evaluation.py --tasks 0 --monitor

run-eval-quick-monitor: ## Send quick evaluation request and monitor (keeps terminal busy)
	@echo "$(BLUE)⚡ Sending quick evaluation with monitoring...$(RESET)"
	@$(PYTHON) send_evaluation.py --tasks 3 --monitor

# ============================================================================
# Docker Commands
# ============================================================================

DOCKER_BUILD := docker build
DOCKER_RUN := docker run

docker-build-all: ## Build all Docker images
	@echo "$(BLUE)🐳 Building all Docker images...$(RESET)"
	@$(DOCKER_BUILD) -f Dockerfile.jupyter -t agentbeats-jupyter:latest .
	@$(DOCKER_BUILD) -f Dockerfile.white -t agentbeats-white:latest .
	@$(DOCKER_BUILD) -f Dockerfile.green -t agentbeats-green:latest .
	@echo "$(GREEN)✅ All Docker images built successfully$(RESET)"

# ============================================================================
# macOS Docker Commands (default - use host.docker.internal)
# ============================================================================

docker-start-jupyter: ## Start Jupyter MCP in foreground (macOS - keeps terminal busy)
	@if [ ! -f .env ]; then echo "$(RED)❌ .env file not found$(RESET)"; exit 1; fi
	@echo "$(BLUE)🐳 Starting Jupyter MCP in Docker (macOS)...$(RESET)"
	@$(DOCKER_RUN) --rm -it \
		--env-file .env \
		-p 8888:8888 \
		-v $(PWD)/agent-workings:/app/agent-workings \
		-v $(PWD)/logs:/app/logs \
		agentbeats-jupyter:latest

docker-start-white: ## Start White Agent in foreground (macOS - keeps terminal busy)
	@if [ ! -f .env ]; then echo "$(RED)❌ .env file not found$(RESET)"; exit 1; fi
	@echo "$(BLUE)🐳 Starting White Agent in Docker (macOS)...$(RESET)"
	@$(DOCKER_RUN) --rm -it \
		--env-file .env \
		-e JUPYTER_BASE_URL=http://host.docker.internal:8888 \
		-p 8001:8001 \
		-v $(PWD)/logs:/app/logs \
		-v $(PWD)/results:/app/results \
		agentbeats-white:latest

docker-start-green: ## Start Green Agent in foreground (macOS - keeps terminal busy)
	@if [ ! -f .env ]; then echo "$(RED)❌ .env file not found$(RESET)"; exit 1; fi
	@echo "$(BLUE)🐳 Starting Green Agent in Docker (macOS)...$(RESET)"
	@$(DOCKER_RUN) --rm -it \
		--env-file .env \
		-p 8000:8000 \
		-v $(PWD)/logs:/app/logs \
		-v $(PWD)/results:/app/results \
		-v $(PWD)/data-dabench:/app/data-dabench \
		agentbeats-green:latest

docker-run-eval-monitor: ## Send evaluation request to Docker containers with monitoring (macOS)
	@echo "$(BLUE)🐳🔍 Sending full evaluation to Docker containers with monitoring (macOS)...$(RESET)"
	@$(PYTHON) send_evaluation.py --tasks 0 --monitor --green-url http://localhost:8000 --white-url http://host.docker.internal:8001

docker-run-eval-quick-monitor: ## Send quick evaluation request to Docker containers with monitoring (macOS)
	@echo "$(BLUE)🐳⚡ Sending quick evaluation to Docker containers with monitoring (macOS)...$(RESET)"
	@$(PYTHON) send_evaluation.py --tasks 3 --monitor --green-url http://localhost:8000 --white-url http://host.docker.internal:8001

# ============================================================================
# Linux Docker Commands (use --network=host)
# ============================================================================

docker-start-jupyter-linux: ## Start Jupyter MCP in foreground (Linux - keeps terminal busy)
	@if [ ! -f .env ]; then echo "$(RED)❌ .env file not found$(RESET)"; exit 1; fi
	@echo "$(BLUE)🐳 Starting Jupyter MCP in Docker (Linux)...$(RESET)"
	@$(DOCKER_RUN) --rm -it \
		--env-file .env \
		--network=host \
		-v $(PWD)/agent-workings:/app/agent-workings \
		-v $(PWD)/logs:/app/logs \
		agentbeats-jupyter:latest

docker-start-white-linux: ## Start White Agent in foreground (Linux - keeps terminal busy)
	@if [ ! -f .env ]; then echo "$(RED)❌ .env file not found$(RESET)"; exit 1; fi
	@echo "$(BLUE)🐳 Starting White Agent in Docker (Linux)...$(RESET)"
	@$(DOCKER_RUN) --rm -it \
		--env-file .env \
		-e JUPYTER_BASE_URL=http://localhost:8888 \
		--network=host \
		-v $(PWD)/logs:/app/logs \
		-v $(PWD)/results:/app/results \
		agentbeats-white:latest

docker-start-green-linux: ## Start Green Agent in foreground (Linux - keeps terminal busy)
	@if [ ! -f .env ]; then echo "$(RED)❌ .env file not found$(RESET)"; exit 1; fi
	@echo "$(BLUE)🐳 Starting Green Agent in Docker (Linux)...$(RESET)"
	@$(DOCKER_RUN) --rm -it \
		--env-file .env \
		--network=host \
		-v $(PWD)/logs:/app/logs \
		-v $(PWD)/results:/app/results \
		-v $(PWD)/data-dabench:/app/data-dabench \
		agentbeats-green:latest

docker-run-eval-monitor-linux: ## Send evaluation request to Docker containers with monitoring (Linux)
	@echo "$(BLUE)🐳🔍 Sending full evaluation to Docker containers with monitoring (Linux)...$(RESET)"
	@$(PYTHON) send_evaluation.py --tasks 0 --monitor --green-url http://localhost:8000 --white-url http://localhost:8001

docker-run-eval-quick-monitor-linux: ## Send quick evaluation request to Docker containers with monitoring (Linux)
	@echo "$(BLUE)🐳⚡ Sending quick evaluation to Docker containers with monitoring (Linux)...$(RESET)"
	@$(PYTHON) send_evaluation.py --tasks 3 --monitor --green-url http://localhost:8000 --white-url http://localhost:8001