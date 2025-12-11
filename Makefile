.PHONY: help start-purple start-green run-eval-monitor run-eval-quick-monitor docker-build-all docker-build-agentbeats docker-push-agentbeats docker-start-purple docker-start-green docker-run-eval-monitor docker-run-eval-quick-monitor docker-start-purple-linux docker-start-green-linux docker-run-eval-monitor-linux docker-run-eval-quick-monitor-linux

# Default Python executable
PYTHON := python

# Directories
PROJECT_ROOT := $(shell pwd)
PURPLE_DIR := src/purple_agent
GREEN_DIR := src/green_agent
AGENT_WORKINGS_DIR := src/purple_agent/agent-workings
LOGS_DIR := logs

# Ports (matching launcher.py)
JUPYTER_PORT := 8888
GREEN_PORT := 8000
PURPLE_PORT := 8001

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

start-purple: ## Start purple agent with embedded MCP (foreground - keeps terminal busy)
	@echo "$(BLUE)� Starting Purple Agent with embedded Jupyter MCP...$(RESET)"
	@mkdir -p $(LOG_DIR)
	@echo "$(CYAN)� Purple agent endpoint: http://localhost:$(PURPLE_PORT)$(RESET)"
	@echo "$(CYAN)📊 Embedded Jupyter MCP: http://localhost:$(JUPYTER_PORT)$(RESET)"
	@echo "$(YELLOW)🚀 Starting Purple Agent... (Press Ctrl+C to stop)$(RESET)"
	@echo "$(BLUE)────────────────────────────────────────$(RESET)"
	@cd $(PURPLE_DIR) && $(PYTHON) agent.py | tee ../../$(LOG_DIR)/purple_agent.log

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
	@$(DOCKER_BUILD) -f Dockerfile.purple -t agentbeats-purple:latest .
	@$(DOCKER_BUILD) -f Dockerfile.green -t agentbeats-green:latest .
	@echo "$(GREEN)✅ All Docker images built successfully$(RESET)"

docker-build-agentbeats: ## Build Docker images for AgentBeats deployment (linux/amd64)
	@echo "$(BLUE)🐳 Building AgentBeats-compatible Docker images (linux/amd64)...$(RESET)"
	@echo "$(YELLOW)📦 Building Green Agent...$(RESET)"
	@docker build --platform linux/amd64 -f Dockerfile.green -t ghcr.io/$(USER)/agentify-dab-step-green:latest .
	@echo "$(YELLOW)📦 Building Purple Agent (with embedded MCP)...$(RESET)"  
	@docker build --platform linux/amd64 -f Dockerfile.purple -t ghcr.io/$(USER)/agentify-dab-step-purple:latest .
	@echo "$(GREEN)✅ All AgentBeats-compatible images built successfully$(RESET)"
	@echo "$(CYAN)🚀 Ready to push to GitHub Container Registry:$(RESET)"
	@echo "  docker push ghcr.io/$(USER)/agentify-dab-step-green:latest"
	@echo "  docker push ghcr.io/$(USER)/agentify-dab-step-purple:latest"

docker-push-agentbeats: ## Push AgentBeats images to GitHub Container Registry
	@echo "$(BLUE)🐳 Pushing AgentBeats images to GitHub Container Registry...$(RESET)"
	@docker push ghcr.io/$(USER)/agentify-dab-step-green:latest
	@docker push ghcr.io/$(USER)/agentify-dab-step-purple:latest
	@echo "$(GREEN)✅ All images pushed successfully$(RESET)"

# ============================================================================
# macOS Docker Commands (default - embedded MCP in purple agent)
# ============================================================================

docker-start-purple: ## Start Purple Agent with embedded MCP in foreground (macOS - keeps terminal busy)
	@if [ ! -f .env ]; then echo "$(RED)❌ .env file not found$(RESET)"; exit 1; fi
	@echo "$(BLUE)🐳 Starting Purple Agent with embedded Jupyter MCP in Docker (macOS)...$(RESET)"
	@$(DOCKER_RUN) --rm -it \
		--env-file .env \
		-p 8888:8888 \
		-p 8001:8001 \
		-v $(PWD)/src/purple_agent/agent-workings:/app/agent-workings \
		-v $(PWD)/logs:/app/logs \
		-v $(PWD)/results:/app/results \
		agentbeats-purple:latest

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
	@$(PYTHON) send_evaluation.py --tasks 0 --monitor --green-url http://localhost:8000 --purple-url http://host.docker.internal:8001

docker-run-eval-quick-monitor: ## Send quick evaluation request to Docker containers with monitoring (macOS)
	@echo "$(BLUE)🐳⚡ Sending quick evaluation to Docker containers with monitoring (macOS)...$(RESET)"
	@$(PYTHON) send_evaluation.py --tasks 3 --monitor --green-url http://localhost:8000 --purple-url http://host.docker.internal:8001

# ============================================================================
# Linux Docker Commands (embedded MCP in purple agent)
# ============================================================================

docker-start-purple-linux: ## Start Purple Agent with embedded MCP in foreground (Linux - keeps terminal busy)
	@if [ ! -f .env ]; then echo "$(RED)❌ .env file not found$(RESET)"; exit 1; fi
	@echo "$(BLUE)🐳 Starting Purple Agent with embedded Jupyter MCP in Docker (Linux)...$(RESET)"
	@$(DOCKER_RUN) --rm -it \
		--env-file .env \
		--network=host \
		-v $(PWD)/src/purple_agent/agent-workings:/app/agent-workings \
		-v $(PWD)/logs:/app/logs \
		-v $(PWD)/results:/app/results \
		agentbeats-purple:latest

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
	@$(PYTHON) send_evaluation.py --tasks 0 --monitor --green-url http://localhost:8000 --purple-url http://localhost:8001

docker-run-eval-quick-monitor-linux: ## Send quick evaluation request to Docker containers with monitoring (Linux)
	@echo "$(BLUE)🐳⚡ Sending quick evaluation to Docker containers with monitoring (Linux)...$(RESET)"
	@$(PYTHON) send_evaluation.py --tasks 3 --monitor --green-url http://localhost:8000 --purple-url http://localhost:8001