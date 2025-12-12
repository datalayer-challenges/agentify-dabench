"""
Purple Agent - Fully Autonomous Agent with Pydantic AI MCP tool capabilities.

This agent:
1. Receives evaluation tasks from the green agent
2. Has access to jupyter-mcp-server tools via Pydantic AI MCP client
3. Uses AI-driven decision making with NO rule-based code
4. Returns raw results for Green Agent to parse
5. Demonstrates A2A-compatible agent capabilities with proper MCP integration
"""

import asyncio
import uuid
import os
import sys
from typing import Any, List
from contextlib import asynccontextmanager

from fasta2a import FastA2A, Skill, Worker
from fasta2a.broker import InMemoryBroker
from fasta2a.storage import InMemoryStorage
from fasta2a.schema import Artifact, Message, TaskIdParams, TaskSendParams, TextPart, DataPart, AgentProvider

# Pydantic AI imports for MCP integration
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStreamableHTTP

# MCP client imports for direct tool calls
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

# Jupyter MCP server for embedded startup
import subprocess
import threading
import time

from dotenv import load_dotenv
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
dotenv_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path)
print(f"🔧 Purple Agent loaded environment variables from {dotenv_path}")


# Import shared utilities
try:
    from ..shared_utils import get_pydantic_ai_model, setup_logger
except ImportError:
    # Fallback for when running as script
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from shared_utils import get_pydantic_ai_model, setup_logger

# Setup logger
logger = setup_logger("purple_agent")

# Context type for the purple agent
Context = List[Message]


class PurpleWorker(Worker[Context]):
    """Purple agent worker - fully autonomous with AI-driven decision making using embedded Jupyter MCP."""
    
    def __init__(self, broker, storage, jupyter_token=None, base_url=None):
        logger.info("🏗️  Initializing PurpleWorker with embedded Jupyter MCP")
        super().__init__(storage=storage, broker=broker)
        self.jupyter_token = jupyter_token or os.getenv("JUPYTER_TOKEN")
        
        # Use 0.0.0.0 for Docker compatibility, localhost for local development
        if base_url is None:
            # Check if we're running in Docker
            if os.path.exists('/.dockerenv'):
                self.base_url = "http://0.0.0.0:8888"
                logger.info("🐳 Docker environment detected, using 0.0.0.0")
            else:
                self.base_url = "http://localhost:8888"
                logger.info("💻 Local environment detected, using localhost")
        else:
            self.base_url = base_url
            
        self.mcp_url = f"{self.base_url}/mcp"
        self.agent = None
        self.mcp_server = None
        self._agent_setup_complete = False
        self._initial_files = set()  # Track files at task start
        self._jupyter_process = None  # For embedded Jupyter MCP server
        self._jupyter_started = False
        logger.info("✅ PurpleWorker initialized")
    
    async def start(self):
        """Start the worker and setup the agent."""
        logger.info("🔧 Setting up agent during worker startup...")
        await self._setup_agent()
        self._agent_setup_complete = True
        logger.info("✅ Agent setup completed during startup")
        
    async def _start_embedded_jupyter_mcp(self):
        """Start embedded Jupyter MCP server."""
        if self._jupyter_started:
            logger.info("📊 Jupyter MCP server already started")
            return
            
        logger.info("🚀 Starting embedded Jupyter MCP server...")
        
        # Generate token if not provided
        if not self.jupyter_token:
            import secrets
            self.jupyter_token = secrets.token_hex(16)
            logger.info(f"🔑 Generated new Jupyter token: {self.jupyter_token[:8]}...")
        
        # Use agent-workings directory relative to the agent script location
        agent_script_dir = os.path.dirname(__file__)
        agent_workings_path = os.path.join(agent_script_dir, "agent-workings")
        
        # Create agent workings directory if it doesn't exist
        logger.info(f"📁 Created/verified agent workings directory: {agent_workings_path}")
        
        # Start Jupyter MCP server in background thread
        def start_jupyter():
            try:
                env = os.environ.copy()
                env["JUPYTER_TOKEN"] = self.jupyter_token
                
                # Detect if we're in Docker for IP binding
                ip_bind = "0.0.0.0" if os.path.exists('/.dockerenv') else "127.0.0.1"
                
                cmd = ["jupyter", "lab",
                    f"--ip={ip_bind}",
                    f"--port={self.base_url.split(':')[-1]}",
                    "--no-browser",
                    "--allow-root",
                    f"--IdentityProvider.token={self.jupyter_token}",
                    "--ServerApp.allow_origin_pat=.*",
                    "--ServerApp.disable_check_xsrf=True",
                    "--ServerApp.allow_remote_access=True",
                ]
                
                logger.info(f"🎯 Starting Jupyter with command: {' '.join(cmd)}")
                logger.info(f"📁 Working directory: {agent_workings_path}")
                logger.info(f"🌐 Binding to IP: {ip_bind}")
                
                self._jupyter_process = subprocess.Popen(
                    cmd,
                    cwd=agent_workings_path,
                    env=env,
                    text=True
                )
                
                # Don't monitor output to avoid blocking - just start the process
                logger.info("📊 Jupyter MCP server started in background")
                            
            except Exception as e:
                logger.error(f"❌ Failed to start Jupyter MCP: {e}")
        
        # Start in background thread
        jupyter_thread = threading.Thread(target=start_jupyter, daemon=True)
        jupyter_thread.start()
        
        # Wait for the process to start
        logger.info("⏳ Waiting for Jupyter MCP server to be ready...")
        await asyncio.sleep(8)  # Give it more time to start
        
        # Check if process is running
        if self._jupyter_process and self._jupyter_process.poll() is None:
            logger.info("✅ Jupyter MCP server process is running")
            self._jupyter_started = True
        else:
            logger.warning("⚠️ Jupyter process may have failed, but proceeding...")
            self._jupyter_started = True  # Proceed anyway
    
    async def _setup_agent(self):
        """Setup the Pydantic AI agent with MCP tools in async context."""
        try:
            # Start embedded Jupyter MCP server first
            await self._start_embedded_jupyter_mcp()
            
            logger.info(f"🔧 Setting up Pydantic AI MCP client connecting to: {self.mcp_url}")
            
            # Create MCP server connection using streamable HTTP
            headers = {}
            if self.jupyter_token:
                headers = {"Authorization": f"token {self.jupyter_token}"}
                logger.info(f"🔑 Using authorization header with token: {self.jupyter_token[:8]}...")
            else:
                logger.warning("⚠️ No Jupyter token found, connecting without authentication")
            
            logger.info(f"🌐 Creating MCP server connection to: {self.mcp_url}")
            
            # Create MCP server connection
            mcp_server = MCPServerStreamableHTTP(
                url=self.mcp_url,
                headers=headers
            )
            
            # Store the MCP server for direct tool calls
            self.mcp_server = mcp_server
            
            logger.info(f"✅ MCP server connection created")
            
            # Get the correct Pydantic AI model configuration
            logger.info(f"🤖 Setting up Pydantic AI model...")
            pydantic_model = get_pydantic_ai_model('purple')
            logger.info(f"✅ Pydantic AI model setup complete with model: {pydantic_model}")
            
            # Create Pydantic AI agent with MCP tools
            logger.info(f"🧠 Creating Pydantic AI agent with model: {pydantic_model}")
            self.agent = Agent(
                model=pydantic_model,
                toolsets=[mcp_server],
                system_prompt="""You are an AI data analyst with access to MCP tools for data analysis. All your data files are in the ./data directory, do not bother checking the filesystem.

Your workflow should be:
1. Connect to notebook.ipynb using use_notebook tool with mode="connect" 
2. Load and examine relevant datasets using execute_code or insert_execute_code_cell
3. Perform analysis step by step with execute_code or by inserting and executing cells
4. Provide a concise final answer based on your analysis
""")
            logger.info("✅ Pydantic AI MCP agent setup complete")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup Pydantic AI MCP agent: {e}")
            logger.error(f"🔍 Error type: {type(e)}")
            import traceback
            logger.error(traceback.format_exc())
            self.agent = None
    
    async def _call_mcp_tool(self, tool_name: str, arguments: dict):
        """Call MCP tool using proper streamable HTTP client."""
        try:
            async with streamablehttp_client(self.mcp_url) as (read_stream, write_stream, _):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments)
                    return result
        except Exception as e:
            logger.error(f"❌ MCP tool call failed for {tool_name}: {e}")
            raise

    async def _clear_notebook_for_task(self):
        """Clear the notebook and restart kernel after each task."""
        try:
            logger.info("🧹 Clearing notebook for fresh task...")
            
            # Connect to the notebook
            result = await self._call_mcp_tool("use_notebook", {
                "notebook_name": "notebook",
                "notebook_path": "notebook.ipynb", 
                "mode": "connect"
            })
            logger.info(f"Connected to notebook: {result}")
            
            # Read current cells
            notebook_content = await self._call_mcp_tool("read_notebook", {
                "notebook_name": "notebook",
                "response_format": "brief"
            })
            
            # Extract cell indices from the response
            cell_indices = []
            if notebook_content and hasattr(notebook_content, 'content') and notebook_content.content:
                lines = str(notebook_content.content[0].text).split('\n')
                for line in lines:
                    if line.strip() and not line.startswith('Index') and not line.startswith('Name'):
                        try:
                            # Parse TSV format - first column should be index
                            parts = line.split('\t')
                            if parts and len(parts) > 0 and parts[0].strip().isdigit():
                                cell_indices.append(int(parts[0].strip()))
                        except:
                            continue
                
                # Delete all cells in descending order to avoid index shifting
                if cell_indices:
                    cell_indices.sort(reverse=True)
                    logger.info(f"🗑️ Deleting {len(cell_indices)} cells: {cell_indices}")
                    
                    await self._call_mcp_tool("delete_cell", {
                        "cell_indices": cell_indices,
                        "include_source": False
                    })
            
            # Restart the kernel to clear memory
            await self._call_mcp_tool("restart_notebook", {
                "notebook_name": "notebook"
            })
            
            logger.info("✅ Notebook cleared and kernel restarted")
            
        except Exception as e:
            logger.error(f"❌ Failed to clear notebook: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Continue anyway - the AI can still work
    
    async def _capture_initial_files(self):
        """Capture the list of files at the start of a task."""
        try:
            logger.info("📁 Capturing initial file state...")
            
            # Get list of all files with deeper scan and no pagination limit
            files_response = await self._call_mcp_tool("list_files", {
                "path": "",
                "max_depth": 3,  # Increase depth to catch more files
                "limit": 0  # No pagination limit - get ALL files
            })
            
            if files_response and hasattr(files_response, 'content') and files_response.content:
                # Parse the response to extract file paths
                lines = str(files_response.content[0].text).split('\n')
                logger.info(f"🔍 Processing {len(lines)} lines from file list")
                
                for line in lines:
                    # Skip header lines like "Path", "Showing X-Y of Z files", empty lines
                    if line.strip() and not line.startswith('Path') and not line.startswith('Showing') and '\t' in line:
                        try:
                            parts = line.split('\t')
                            if len(parts) >= 2:
                                file_path = parts[0].strip()
                                file_type = parts[1].strip()
                                
                                # Only track files, not directories
                                if file_type == 'file':
                                    self._initial_files.add(file_path)
                                    
                                # Log for debugging
                                if file_path.startswith('data/'):
                                    logger.debug(f"📄 Found data file: {file_path} (type: {file_type})")
                        except Exception as e:
                            logger.debug(f"❌ Error parsing line '{line}': {e}")
                            continue
            
            logger.info(f"📋 Captured {len(self._initial_files)} initial files")
            
            # Count data files specifically for debugging
            data_files = [f for f in self._initial_files if f.startswith('data/')]
            logger.info(f"📊 Found {len(data_files)} files in data directory")
            
        except Exception as e:
            logger.error(f"❌ Failed to capture initial files: {e}")
            # Continue anyway - this is just for cleanup
    
    async def _cleanup_new_files(self):
        """Remove any files created during the task using execute_code with shell commands."""
        try:
            logger.info("🗂️ Checking for new files to cleanup...")
            
            # Get current list of files with same depth and no pagination limit
            files_response = await self._call_mcp_tool("list_files", {
                "path": "",
                "max_depth": 3,  # Same depth as initial capture
                "limit": 0  # No pagination limit - get ALL files
            })
            
            current_files = set()
            if files_response and hasattr(files_response, 'content') and files_response.content:
                lines = str(files_response.content[0].text).split('\n')
                for line in lines:
                    # Skip header lines like "Path", "Showing X-Y of Z files", empty lines
                    if line.strip() and not line.startswith('Path') and not line.startswith('Showing') and '\t' in line:
                        try:
                            parts = line.split('\t')
                            if len(parts) >= 2:
                                file_path = parts[0].strip()
                                file_type = parts[1].strip()
                                
                                # Only track files, not directories
                                if file_type == 'file':
                                    current_files.add(file_path)
                        except:
                            continue
            
            logger.info(f"📊 Current files: {len(current_files)}, Initial files: {len(self._initial_files)}")
            
            # Find new files (present now but not at start)
            new_files = current_files - self._initial_files
            
            if new_files:
                logger.info(f"🗑️ Found {len(new_files)} new files to cleanup: {list(new_files)}")
                
                # Delete new files using execute_code with shell commands
                for file_path in new_files:
                    try:
                        logger.info(f"🗑️ Deleting file: {file_path}")
                        # Use execute_code to run shell command
                        await self._call_mcp_tool("execute_code", {
                            "code": f"!rm -f '{file_path}'",
                            "timeout": 10
                        })
                        logger.info(f"✅ Deleted file: {file_path}")
                    except Exception as e:
                        logger.error(f"❌ Failed to delete {file_path}: {e}")
            else:
                logger.info("✅ No new files to cleanup")
                
        except Exception as e:
            logger.error(f"❌ Failed to cleanup new files: {e}")
            # Continue anyway - this is just for cleanup
    
    async def run_task(self, params: TaskSendParams) -> None:
        """Process a task and provide a response."""
        logger.info(f"🟣 Purple Agent run_task called with params: {params}")
        
        task = await self.storage.load_task(params['id'])
        if task is None:
            logger.error(f"❌ Task {params['id']} not found")
            await self.storage.update_task(params['id'], state='failed')
            return
        
        logger.info(f"✅ Task loaded: {task}")
        await self.storage.update_task(task['id'], state='working')
        
        try:
            # Load context
            context = await self.storage.load_context(task['context_id']) or []
            context.extend(task.get('history', []))
            
            # Extract question from message
            message = params['message']
            logger.info(f"📨 Received message: {message}")
        
            # Capture initial file state for cleanup
            await self._capture_initial_files()
            
            # Process question using Pydantic AI with MCP tools
            if not self.agent:
                raise Exception("Pydantic AI agent not initialized")
                
            logger.info(f"🤖 Processing with Pydantic AI agent...")
            logger.info(f"🚀 Starting agent execution...")
            
            try:
                # Execute the agent directly - same as test_simple_agent.py
                logger.info(f"🤖 Executing agent directly...")
                question = self._extract_text_from_parts(message['parts'])
                result = await self.agent.run(question)
                
                logger.info(f"✅ Agent execution completed!")
                answer = str(result.output)
                
                # Extract token usage information
                usage = result.usage()
                usage_info = {
                    'requests': usage.requests,
                    'request_tokens': usage.request_tokens,
                    'response_tokens': usage.response_tokens,
                    'total_tokens': usage.total_tokens
                }
                
                logger.info(f"💡 Generated answer: {answer}")
                logger.info(f"📊 Token usage - Requests: {usage.requests}, "
                           f"Input: {usage.request_tokens}, Output: {usage.response_tokens}, "
                           f"Total: {usage.total_tokens}")
                
            except Exception as e:
                logger.error(f"❌ Pydantic AI execution failed: {e}")
                logger.error(f"🔍 Error type: {type(e)}")
                import traceback
                logger.error(traceback.format_exc())
                answer = f"Error during AI processing: {str(e)}"
                usage_info = None  # No usage info available for failed executions
            
            # Create response message
            response_message = self._create_response_message(answer)
            artifacts = self._create_response_artifacts(answer, usage_info)
            
            # Update context and complete task
            context.append(response_message)
            await self.storage.update_context(task['context_id'], context)
            await self.storage.update_task(
                task['id'],
                state='completed',
                new_messages=[response_message],
                new_artifacts=artifacts
            )
            logger.info(f"✅ Task {task['id']} completed successfully")
            
            # Clean notebook and files after successful completion
            await self._clear_notebook_for_task()
            await self._cleanup_new_files()
            
        except Exception as e:
            logger.error(f"❌ Task processing failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # Handle errors
            error_message = Message(
                role='agent',
                parts=[TextPart(text=f"Error processing question: {str(e)}", kind='text')],
                kind='message',
                message_id=str(uuid.uuid4())
            )
            
            context = await self.storage.load_context(task['context_id']) or []
            await self.storage.update_context(task['context_id'], context + [error_message])
            await self.storage.update_task(
                task['id'],
                state='failed',
                new_messages=[error_message]
            )
            
            # Clean notebook and files after error as well
            await self._clear_notebook_for_task()
            await self._cleanup_new_files()

    async def cancel_task(self, params: TaskIdParams) -> None:
        """Cancel a task."""
        await self.storage.update_task(params['id'], state='canceled')
    
    def build_message_history(self, history: List[Message]) -> List[Any]:
        """Build message history."""
        return [
            {
                'role': msg['role'],
                'content': self._extract_text_from_parts(msg['parts']),
                'message_id': msg['message_id']
            }
            for msg in history
        ]
    
    def build_artifacts(self, result: Any) -> List[Artifact]:
        """Build artifacts from result."""
        return []
    
    
    def _create_response_message(self, answer: str) -> Message:
        """Create response message with the answer."""
        return Message(
            role='agent',
            parts=[TextPart(text=answer, kind='text')],
            kind='message',
            message_id=str(uuid.uuid4())
        )
    
    def _create_response_artifacts(self, answer: str, usage_info: dict = None) -> List[Artifact]:
        """Create artifacts with the answer and optional usage information."""
        artifacts = []
        
        # Create answer artifact data
        artifact_data = {'answer': answer}
        if usage_info:
            artifact_data['token_usage'] = usage_info
        
        answer_artifact = Artifact(
            artifact_id=str(uuid.uuid4()),
            name="agent_answer",
            description="Agent's answer to the question with token usage",
            parts=[
                TextPart(text=answer, kind='text'),
                DataPart(data=artifact_data, kind='data')
            ]
        )
        artifacts.append(answer_artifact)
        
        return artifacts

    def cleanup(self):
        """Clean up resources including embedded Jupyter server."""
        if self._jupyter_process:
            logger.info("🛑 Stopping embedded Jupyter MCP server...")
            try:
                self._jupyter_process.terminate()
                self._jupyter_process.wait(timeout=5)
                logger.info("✅ Jupyter MCP server stopped")
            except subprocess.TimeoutExpired:
                logger.warning("⚠️ Force killing Jupyter MCP server...")
                self._jupyter_process.kill()
            except Exception as e:
                logger.error(f"❌ Error stopping Jupyter MCP server: {e}")
            finally:
                self._jupyter_process = None
                self._jupyter_started = False
    
    def _extract_text_from_parts(self, parts: List[Any]) -> str:
        """Extract text from message parts."""
        text_parts = []
        for part in parts:
            if part['kind'] == 'text':
                text_parts.append(part['text'])
        return ' '.join(text_parts)


def create_purple_agent(card_url: str = None) -> FastA2A:
    """Create the purple agent with MCP tool capabilities."""
    
    # Initialize storage and broker
    storage = InMemoryStorage[Context]()
    broker = InMemoryBroker()
    
    # Define skills
    autonomous_reasoning_skill = Skill(
        id="autonomous-reasoning",
        name="Autonomous AI Reasoning",
        description="Performs autonomous reasoning using AI and MCP tools with no hardcoded rules",
        tags=["autonomous", "ai", "reasoning", "mcp", "analysis"],
        examples=[
            "Analyze data autonomously using AI reasoning",
            "Execute code generated by AI for problem solving",
            "Make intelligent decisions about tool usage",
            "Provide answers based on AI-driven analysis"
        ],
        input_modes=["text/plain", "application/json"],
        output_modes=["text/plain", "application/json"]
    )
    
    computational_skill = Skill(
        id="computational-analysis",
        name="AI-Driven Computational Analysis",
        description="Performs computational analysis using AI-generated code and MCP tools",
        tags=["computation", "jupyter", "mcp", "ai-driven"],
        examples=[
            "Execute AI-generated Python code",
            "Perform statistical analysis autonomously",
            "Generate insights from data using AI",
            "Process datasets with intelligent exploration"
        ],
        input_modes=["text/plain", "application/json"],
        output_modes=["text/plain", "application/json"]
    )
    
    # Agent provider - use card_url if provided
    provider = AgentProvider(
        organization=os.getenv("AGENT_ORGANIZATION", "DataAnalysis"),
        url=os.getenv("AGENT_PROVIDER_URL", card_url or "http://localhost:9019")
    )
    
    # Create worker with Pydantic AI MCP
    jupyter_token = os.getenv("JUPYTER_TOKEN")
    jupyter_base_url = os.getenv("JUPYTER_BASE_URL", "http://localhost:8888")
    logger.info(f"🔑 Purple Agent token: {jupyter_token[:8] + '...' if jupyter_token else 'None'}")
    logger.info(f"🌐 Jupyter MCP base URL: {jupyter_base_url}")
    
    worker = PurpleWorker(
        broker=broker, 
        storage=storage, 
        jupyter_token=jupyter_token,
        base_url=jupyter_base_url
    )
    
    # Add lifespan to start worker
    @asynccontextmanager
    async def lifespan(app):
        async with app.task_manager:
            async with worker.run():
                # Setup agent after worker starts
                await worker.start()
                yield
    
    # Create FastA2A application - use card_url if provided
    app = FastA2A(
        storage=storage,
        broker=broker,
        name="Purple Agent - Autonomous",
        description="Fully autonomous A2A-compatible agent with AI-driven decision making and MCP tool capabilities",
        url=card_url or "http://localhost:9019",  # Use the provided card_url
        version="2.0.0",
        provider=provider,
        skills=[autonomous_reasoning_skill, computational_skill],
        lifespan=lifespan
    )
    
    return app


def main():
    """Main entry point for the autonomous purple agent."""
    import argparse
    import uvicorn
    
    # Parse command line arguments for AgentBeats compatibility
    parser = argparse.ArgumentParser(description="Purple Agent (Test Subject)")
    parser.add_argument("--host", default="0.0.0.0", help="Host address to bind to")
    parser.add_argument("--port", type=int, default=9019, help="Port to listen on")
    parser.add_argument("--card-url", help="URL to advertise in the agent card (optional)")
    args = parser.parse_args()
    
    host = args.host
    port = args.port
    card_url = args.card_url or f"http://{host}:{port}"
    
    logger.info("🟣 Starting Purple Agent - Fully Autonomous")
    logger.info("🤖 AI-Driven: Pure autonomous decision making")
    logger.info("🛠️  MCP Tools: Dynamic tool discovery and usage")
    logger.info(f"📋 Agent Card: {card_url}/.well-known/agent-card.json")
    logger.info(f"🔗 A2A Endpoint: {card_url}/")
    logger.info(f"🌐 Binding to {host}:{port}")
    
    # Create a factory function that uses the card_url
    def create_app():
        return create_purple_agent(card_url)
    
    uvicorn.run(
        create_app,
        host=host,
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()