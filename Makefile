.PHONY: help start-mcp start-white start-green start-all stop clean logs evaluate evaluate-quick evaluate-dev evaluate-full check-env

# Default Python executable
PYTHON := python3

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

# Generate timestamp for log directory
TIMESTAMP := $(shell date +%Y%m%d_%H%M%S)
LOG_DIR := $(LOGS_DIR)/run_$(TIMESTAMP)

# Colors for output
RED := \033[31m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m
MAGENTA := \033[35m
CYAN := \033[36m
RESET := \033[0m

help: ## Show this help message
	@echo "$(CYAN)AgentBeats Makefile - DABench A2A Implementation$(RESET)"
	@echo ""
	@echo "$(YELLOW)Available commands:$(RESET)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-20s$(RESET) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(YELLOW)Recommended Workflow:$(RESET)"
	@echo "  $(BLUE)1. Start Services (3 terminals):$(RESET)"
	@echo "    Terminal 1: $(GREEN)make start-mcp$(RESET)        # Start MCP server"
	@echo "    Terminal 2: $(GREEN)make start-white$(RESET)      # Start white agent"  
	@echo "    Terminal 3: $(GREEN)make start-green$(RESET)      # Start green agent"
	@echo ""
	@echo "  $(BLUE)2. Run Evaluation (4th terminal):$(RESET)"
	@echo "    Terminal 4: $(GREEN)make run-eval$(RESET)         # Send evaluation request"
	@echo ""
	@echo "  $(BLUE)Alternative:$(RESET)"
	@echo "    $(GREEN)make evaluate-quick$(RESET)   # All-in-one launcher.py"

check-env: ## Check environment setup
	@echo "$(BLUE)🔧 Checking environment setup...$(RESET)"
	@if [ ! -f .env ]; then \
		echo "$(RED)❌ .env file not found. Please copy .env.example to .env$(RESET)"; \
		exit 1; \
	fi
	@if ! command -v jupyter >/dev/null 2>&1; then \
		echo "$(RED)❌ Jupyter not found. Please install: pip install jupyter$(RESET)"; \
		exit 1; \
	fi
	@if [ ! -f requirements.txt ]; then \
		echo "$(RED)❌ requirements.txt not found$(RESET)"; \
		exit 1; \
	fi
	@echo "$(GREEN)✅ Environment checks passed$(RESET)"

setup-logs: ## Create log directory
	@mkdir -p $(LOG_DIR)
	@echo "$(BLUE)📝 Log directory created: $(LOG_DIR)$(RESET)"

start-mcp: check-env setup-logs ## Start JupyterLab MCP server (foreground - keeps terminal busy)
	@echo "$(BLUE)📊 Starting JupyterLab with MCP Server...$(RESET)"
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
	@echo "$(CYAN)🔗 White agent endpoint: http://localhost:$(WHITE_PORT)$(RESET)"
	@echo "$(YELLOW)🚀 Starting White Agent... (Press Ctrl+C to stop)$(RESET)"
	@echo "$(BLUE)────────────────────────────────────────$(RESET)"
	@cd $(WHITE_DIR) && $(PYTHON) agent.py | tee ../../$(LOG_DIR)/white_agent.log

start-green: ## Start green agent (foreground - keeps terminal busy)
	@echo "$(BLUE)🟢 Starting Green Agent (Evaluator)...$(RESET)"
	@echo "$(CYAN)🔗 Green agent endpoint: http://localhost:$(GREEN_PORT)$(RESET)"
	@echo "$(YELLOW)🚀 Starting Green Agent... (Press Ctrl+C to stop)$(RESET)"
	@echo "$(BLUE)────────────────────────────────────────$(RESET)"
	@cd $(GREEN_DIR) && $(PYTHON) agent.py | tee ../../$(LOG_DIR)/green_agent.log

start-evaluation: ## Send evaluation request to green agent (requires all agents running)
	@echo "$(BLUE)🎯 Sending full evaluation request to green agent...$(RESET)"
	@$(PYTHON) send_evaluation.py --tasks 0

run-eval: start-evaluation ## Alias for start-evaluation (cleaner name)

run-eval-quick: ## Send quick evaluation request (3 tasks) to green agent
	@echo "$(BLUE)⚡ Sending quick evaluation request to green agent...$(RESET)"
	@$(PYTHON) send_evaluation.py --tasks 3

run-eval-monitor: ## Send evaluation request and monitor progress (keeps terminal busy)
	@echo "$(BLUE)🔍 Sending full evaluation request with monitoring...$(RESET)"
	@$(PYTHON) send_evaluation.py --tasks 0 --monitor

run-eval-quick-monitor: ## Send quick evaluation request and monitor (keeps terminal busy)
	@echo "$(BLUE)⚡ Sending quick evaluation with monitoring...$(RESET)"
	@$(PYTHON) send_evaluation.py --tasks 3 --monitor

start-green-eval: setup-logs ## DEPRECATED - Use separate start-green + run-eval
	@echo "$(YELLOW)⚠️  DEPRECATED: Use 'make start-green' then 'make run-eval' instead$(RESET)"
	@echo "$(BLUE)🟢 Starting Green Agent and triggering full evaluation...$(RESET)"
	@echo "$(CYAN)🔗 Green agent endpoint: http://localhost:$(GREEN_PORT)$(RESET)"
	@echo "$(YELLOW)🚀 Starting Green Agent in background...$(RESET)"
	@cd $(GREEN_DIR) && $(PYTHON) agent.py > ../../$(LOG_DIR)/green_agent.log 2>&1 & 
	@echo "$(BLUE)⏳ Waiting 5s for green agent startup...$(RESET)"
	@sleep 5
	@echo "$(BLUE)🎯 Sending full evaluation request and monitoring progress...$(RESET)"
	@$(PYTHON) send_evaluation.py --tasks 0 --monitor

start-all: ## Use launcher.py for all-in-one startup (background processes)
	@echo "$(BLUE)🚀 Starting all services using launcher.py...$(RESET)"
	@$(PYTHON) launcher.py

stop: ## Stop all services by killing processes on ports
	@echo "$(BLUE)🛑 Stopping all services...$(RESET)"
	@echo "$(BLUE)� Stopping JupyterLab (port $(JUPYTER_PORT))...$(RESET)"
	@lsof -ti:$(JUPYTER_PORT) | xargs -r kill -9 2>/dev/null || true
	@echo "$(BLUE)🟢 Stopping green agent (port $(GREEN_PORT))...$(RESET)"
	@lsof -ti:$(GREEN_PORT) | xargs -r kill -9 2>/dev/null || true
	@echo "$(BLUE)⚪ Stopping white agent (port $(WHITE_PORT))...$(RESET)"
	@lsof -ti:$(WHITE_PORT) | xargs -r kill -9 2>/dev/null || true
	@echo "$(GREEN)✅ All services stopped$(RESET)"

clean: stop ## Stop all services and clean up
	@echo "$(BLUE)🧹 Cleaning up...$(RESET)"
	@echo "$(GREEN)✅ Cleanup complete$(RESET)"

status: ## Check status of all services by checking ports
	@echo "$(BLUE)📊 Service Status:$(RESET)"
	@echo ""
	@if lsof -Pi:$(JUPYTER_PORT) -sTCP:LISTEN -t >/dev/null 2>&1; then \
		echo "$(GREEN)✅ JupyterLab MCP Server: Running (Port: $(JUPYTER_PORT))$(RESET)"; \
		curl -s http://localhost:$(JUPYTER_PORT)/mcp/healthz >/dev/null 2>&1 && \
			echo "   $(GREEN)🔌 MCP endpoints: Available$(RESET)" || \
			echo "   $(YELLOW)⚠️  MCP endpoints: Check needed$(RESET)"; \
	else \
		echo "$(RED)❌ JupyterLab MCP Server: Not running$(RESET)"; \
	fi
	@if lsof -Pi:$(WHITE_PORT) -sTCP:LISTEN -t >/dev/null 2>&1; then \
		echo "$(GREEN)✅ White Agent: Running (Port: $(WHITE_PORT))$(RESET)"; \
		curl -s http://localhost:$(WHITE_PORT)/.well-known/agent-card.json >/dev/null 2>&1 && \
			echo "   $(GREEN)🤖 A2A endpoint: Available$(RESET)" || \
			echo "   $(YELLOW)⚠️  A2A endpoint: Check needed$(RESET)"; \
	else \
		echo "$(RED)❌ White Agent: Not running$(RESET)"; \
	fi
	@if lsof -Pi:$(GREEN_PORT) -sTCP:LISTEN -t >/dev/null 2>&1; then \
		echo "$(GREEN)✅ Green Agent: Running (Port: $(GREEN_PORT))$(RESET)"; \
		curl -s http://localhost:$(GREEN_PORT)/.well-known/agent-card.json >/dev/null 2>&1 && \
			echo "   $(GREEN)🎯 Evaluator endpoint: Available$(RESET)" || \
			echo "   $(YELLOW)⚠️  Evaluator endpoint: Check needed$(RESET)"; \
	else \
		echo "$(RED)❌ Green Agent: Not running$(RESET)"; \
	fi

logs: ## Show recent logs from all services
	@echo "$(BLUE)📝 Recent logs from all services:$(RESET)"
	@echo ""
	@if [ -d "$(shell ls -1dt $(LOGS_DIR)/run_* 2>/dev/null | head -n1)" ]; then \
		LATEST_LOG_DIR=$$(ls -1dt $(LOGS_DIR)/run_* | head -n1); \
		echo "$(CYAN)📊 JupyterLab MCP Server logs:$(RESET)"; \
		tail -n 10 $$LATEST_LOG_DIR/jupyter_mcp.log 2>/dev/null || echo "$(YELLOW)No logs found$(RESET)"; \
		echo ""; \
		echo "$(CYAN)⚪ White Agent logs:$(RESET)"; \
		tail -n 10 $$LATEST_LOG_DIR/white_agent.log 2>/dev/null || echo "$(YELLOW)No logs found$(RESET)"; \
		echo ""; \
		echo "$(CYAN)🟢 Green Agent logs:$(RESET)"; \
		tail -n 10 $$LATEST_LOG_DIR/green_agent.log 2>/dev/null || echo "$(YELLOW)No logs found$(RESET)"; \
	else \
		echo "$(YELLOW)No log directories found$(RESET)"; \
	fi

evaluate-quick: ## Run quick evaluation (3 tasks) using launcher.py
	@echo "$(BLUE)⚡ Running quick evaluation...$(RESET)"
	@$(PYTHON) launcher.py --evaluate --quick-sample 3

evaluate-dev: ## Run development evaluation (~10 tasks) using launcher.py
	@echo "$(BLUE)🔧 Running development evaluation...$(RESET)"
	@$(PYTHON) launcher.py --evaluate --sample-mode dev

evaluate-full: ## Run full dataset evaluation (~450 tasks) using launcher.py
	@echo "$(BLUE)🚀 Running full evaluation...$(RESET)"
	@$(PYTHON) launcher.py --evaluate --full-dataset

evaluate: evaluate-quick ## Alias for evaluate-quick

# Development and debugging targets  
debug-white: setup-logs ## Start white agent with verbose logging (foreground)
	@echo "$(BLUE)🐛 Starting white agent in debug mode...$(RESET)"
	@cd $(WHITE_DIR) && $(PYTHON) agent.py

debug-green: setup-logs ## Start green agent with verbose logging (foreground)
	@echo "$(BLUE)🐛 Starting green agent in debug mode...$(RESET)"
	@cd $(GREEN_DIR) && $(PYTHON) agent.py

install: ## Install dependencies
	@echo "$(BLUE)📦 Installing dependencies...$(RESET)"
	@pip install -r requirements.txt
	@echo "$(GREEN)✅ Dependencies installed$(RESET)"

setup: install ## Setup environment (install deps + copy .env.example)
	@if [ ! -f .env ]; then \
		if [ -f .env.example ]; then \
			cp .env.example .env; \
			echo "$(GREEN)✅ .env file created from .env.example$(RESET)"; \
			echo "$(YELLOW)⚠️  Please edit .env with your API keys and preferences$(RESET)"; \
		else \
			echo "$(RED)❌ .env.example not found$(RESET)"; \
		fi; \
	else \
		echo "$(GREEN)✅ .env file already exists$(RESET)"; \
	fi

# Watch logs in real-time
watch-logs: ## Watch logs in real-time
	@if [ -d "$(shell ls -1dt $(LOGS_DIR)/run_* 2>/dev/null | head -n1)" ]; then \
		LATEST_LOG_DIR=$$(ls -1dt $(LOGS_DIR)/run_* | head -n1); \
		echo "$(BLUE)👀 Watching logs from: $$LATEST_LOG_DIR$(RESET)"; \
		tail -f $$LATEST_LOG_DIR/*.log; \
	else \
		echo "$(YELLOW)No log directories found. Start services first.$(RESET)"; \
	fi