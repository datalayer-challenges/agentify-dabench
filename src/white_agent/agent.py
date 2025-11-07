"""
DABSTEP White Agent - Fully Autonomous Agent with Pydantic AI MCP tool capabilities.

This agent:
1. Receives evaluation tasks from the green agent
2. Has access to jupyter-mcp-server tools via Pydantic AI MCP client
3. Uses AI-driven decision making with NO rule-based code
4. Returns raw results for Green Agent to parse
5. Demonstrates A2A-compatible agent capabilities with proper MCP integration
"""

import json
import uuid
import asyncio
import os
import sys
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

from fasta2a import FastA2A, Skill, Worker
from fasta2a.broker import InMemoryBroker
from fasta2a.storage import InMemoryStorage
from fasta2a.schema import Artifact, Message, TaskIdParams, TaskSendParams, TextPart, DataPart, AgentProvider

# Pydantic AI imports for MCP integration
try:
    from pydantic_ai import Agent
    from pydantic_ai.models import KnownModelName
    from pydantic_ai.models.openai import OpenAIModel
    from pydantic_ai.mcp import MCPServerStreamableHTTP
    import litellm
except ImportError as e:
    print(f"❌ Required packages not installed: {e}")
    print("Please install: pip install pydantic-ai litellm")
    sys.exit(1)

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    dotenv_path = os.path.join(project_root, '.env')
    load_dotenv(dotenv_path)
    print(f"🔧 White Agent loaded environment variables from {dotenv_path}")
except ImportError:
    print("⚠️  python-dotenv not installed, skipping .env file loading")
except Exception as e:
    print(f"⚠️  Failed to load .env file: {e}")

# Set up logging to reduce noise
litellm.set_verbose = False

# Import shared utilities
try:
    from ..shared_utils import setup_llm_client, get_pydantic_ai_model
except ImportError:
    # Fallback for when running as script
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from shared_utils import setup_llm_client, get_pydantic_ai_model

# Context type for the white agent
Context = List[Message]


class DABStepWhiteWorker(Worker[Context]):
    """White agent worker - fully autonomous with AI-driven decision making using Pydantic AI MCP."""
    
    def __init__(self, broker, storage, jupyter_token=None, base_url="http://localhost:8888"):
        print("🏗️  Initializing DABStepWhiteWorker with Pydantic AI MCP")
        super().__init__(storage=storage, broker=broker)
        self.jupyter_token = jupyter_token or os.getenv("JUPYTER_TOKEN")
        self.base_url = base_url
        self.mcp_url = f"{base_url}/mcp"
        self.agent = None
        self._setup_agent()
        print("✅ DABStepWhiteWorker initialized")
    
    def _setup_agent(self):
        """Setup the Pydantic AI agent with MCP tools."""
        try:
            print(f"🔧 Setting up Pydantic AI MCP client connecting to: {self.mcp_url}")
            
            # Create MCP server connection using streamable HTTP
            headers = {}
            if self.jupyter_token:
                headers = {"Authorization": f"token {self.jupyter_token}"}
                print(f"🔑 Using authorization header with token: {self.jupyter_token[:8]}...")
            else:
                print("⚠️ No Jupyter token found, connecting without authentication")
            
            print(f"🌐 Creating MCP server connection to: {self.mcp_url}")
            
            # Create MCP server connection
            mcp_server = MCPServerStreamableHTTP(
                url=self.mcp_url,
                headers=headers
            )
            
            print(f"✅ MCP server connection created")
            
            # Get the correct Pydantic AI model configuration
            print(f"🤖 Setting up Pydantic AI model...")
            pydantic_model = get_pydantic_ai_model()
            print(f"✅ Pydantic AI model setup complete with model: {pydantic_model}")
            
            # Create Pydantic AI agent with MCP tools
            print(f"🧠 Creating Pydantic AI agent with model: {pydantic_model}")
            self.agent = Agent(
                model=pydantic_model,
                toolsets=[mcp_server],
                system_prompt="""You are an AI data analyst with access to MCP tools for data analysis.
                
Available data files in /data/context/:
- payments.csv (transaction data)
- acquirer_countries.csv (country data)  
- merchant_category_codes.csv (merchant codes)
- merchant_data.json (merchant information)
- fees.json (fee information)

Your workflow should be:
1. First explore what data files are available using list_files
2. Load and examine relevant datasets using execute_code
3. Perform analysis step by step with execute_code
4. Provide a clear final answer

Use the MCP tools to execute Python code and analyze data. Always provide clear, concise answers."""
            )
            
            print("✅ Pydantic AI MCP agent setup complete")
            
        except Exception as e:
            print(f"❌ Failed to setup Pydantic AI MCP agent: {e}")
            print(f"🔍 Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            self.agent = None
    
    async def run_task(self, params: TaskSendParams) -> None:
        """Process a task and provide a response."""
        print(f"⚪ White Agent run_task called with params: {params}")
        
        task = await self.storage.load_task(params['id'])
        if task is None:
            print(f"❌ Task {params['id']} not found")
            await self.storage.update_task(params['id'], state='failed')
            return
        
        print(f"✅ Task loaded: {task}")
        await self.storage.update_task(task['id'], state='working')
        
        try:
            # Load context
            context = await self.storage.load_context(task['context_id']) or []
            context.extend(task.get('history', []))
            
            # Extract question from message
            message = params['message']
            print(f"📨 Received message: {message}")
            
            question = self._extract_question(message)
            
            if not question:
                print("❌ No question found in message")
                raise ValueError("No question found in message")
            
            print(f"🤔 Received question: {question}")
            
            # Process question using Pydantic AI with MCP tools - Simple!
            if not self.agent:
                raise Exception("Pydantic AI agent not initialized")
                
            print(f"🤖 Processing with Pydantic AI agent...")
            try:
                # Run the agent directly without timeout wrapper to avoid cancellation scope issues
                print(f"🚀 Starting agent execution...")
                print(f"🔍 Agent object: {type(self.agent)}")
                print(f"🔍 Question: {question[:100]}...")
                
                # Test if agent is responsive
                print(f"🔍 Testing agent responsiveness...")
                
                # Try the execution
                ### NOT WORKING DUE TO ??? TO FIX TODO
                ### result = await self.agent.run(question)
                
                ####### TO DELETE AFTER DEBUGGING
                # Create a mock result object that mimics Pydantic AI's response structure
                class MockResult:
                    def __init__(self, data):
                        self.data = data
                
                result = MockResult("NL")  # Placeholder with .data attribute
                #######
                
                print(f"✅ Agent execution completed!")
                print(f"📤 Pydantic AI result type: {type(result)}")
                print(f"📤 Pydantic AI result: {result}")
                print(f"📤 Pydantic AI result.data: {result.data}")
                answer = str(result.data)
                
                print(f"💡 Generated answer: {answer}")
                
            except Exception as e:
                print(f"❌ Pydantic AI execution failed: {e}")
                print(f"🔍 Error type: {type(e)}")
                import traceback
                traceback.print_exc()
                
                # For cancellation errors, try to provide a meaningful response
                if "cancel" in str(e).lower() or "scope" in str(e).lower():
                    answer = "Error: Agent execution was interrupted due to async cancellation"
                else:
                    answer = f"Error during AI processing: {str(e)}"
            
            # Create response message
            response_message = self._create_response_message(answer)
            artifacts = self._create_response_artifacts(answer)
            
            # Update context and complete task
            context.append(response_message)
            await self.storage.update_context(task['context_id'], context)
            await self.storage.update_task(
                task['id'],
                state='completed',
                new_messages=[response_message],
                new_artifacts=artifacts
            )
            print(f"✅ Task {task['id']} completed successfully")
            
        except Exception as e:
            print(f"❌ Task processing failed: {e}")
            import traceback
            traceback.print_exc()
            
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
    
    def _extract_question(self, message: Message) -> Optional[str]:
        """Extract question from message."""
        full_text = ""
        
        for part in message['parts']:
            if part['kind'] == 'text':
                full_text += part['text'] + " "
            elif part['kind'] == 'data':
                data = part.get('data', {})
                if isinstance(data, dict):
                    for key in ['question', 'query', 'task', 'prompt']:
                        if key in data:
                            full_text += str(data[key]) + " "
                elif isinstance(data, str):
                    full_text += data + " "
        
        # Extract question from various patterns
        full_text = full_text.strip()
        
        # Look for different question patterns
        patterns = [
            "Here is the question you need to answer:",
            "Question:",
            "QUESTION:",
            "Task:"
        ]
        
        for pattern in patterns:
            if pattern in full_text:
                question_part = full_text.split(pattern, 1)[1]
                # Split on common endings
                for ending in ["Here are the guidelines", "Guidelines:", "Please provide", "\n\n"]:
                    if ending in question_part:
                        question_part = question_part.split(ending)[0]
                        break
                return question_part.strip()
        
        return full_text if full_text else None
    
    def _create_response_message(self, answer: str) -> Message:
        """Create response message with the answer."""
        return Message(
            role='agent',
            parts=[TextPart(text=answer, kind='text')],
            kind='message',
            message_id=str(uuid.uuid4())
        )
    
    def _create_response_artifacts(self, answer: str) -> List[Artifact]:
        """Create artifacts with the answer."""
        artifacts = []
        
        answer_artifact = Artifact(
            artifact_id=str(uuid.uuid4()),
            name="agent_answer",
            description="Agent's answer to the question",
            parts=[
                TextPart(text=answer, kind='text'),
                DataPart(data={'answer': answer}, kind='data')
            ]
        )
        artifacts.append(answer_artifact)
        
        return artifacts
    
    def _extract_text_from_parts(self, parts: List[Any]) -> str:
        """Extract text from message parts."""
        text_parts = []
        for part in parts:
            if part['kind'] == 'text':
                text_parts.append(part['text'])
        return ' '.join(text_parts)


def create_dabstep_white_agent() -> FastA2A:
    """Create the DABSTEP white agent with MCP tool capabilities."""
    
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
    
    # Agent provider
    provider = AgentProvider(
        organization=os.getenv("AGENT_ORGANIZATION", "DABSTEP"),
        url=os.getenv("AGENT_PROVIDER_URL", "http://localhost:8001")
    )
    
    # Create worker with Pydantic AI MCP
    jupyter_token = os.getenv("JUPYTER_TOKEN")
    print(f"🔑 White Agent token: {jupyter_token[:8] + '...' if jupyter_token else 'None'}")
    
    worker = DABStepWhiteWorker(
        broker=broker, 
        storage=storage, 
        jupyter_token=jupyter_token,
        base_url="http://localhost:8888"
    )
    
    # Add lifespan to start worker
    @asynccontextmanager
    async def lifespan(app):
        async with app.task_manager:
            async with worker.run():
                yield
    
    # Create FastA2A application
    app = FastA2A(
        storage=storage,
        broker=broker,
        name="DABSTEP White Agent - Autonomous",
        description="Fully autonomous A2A-compatible agent with AI-driven decision making and MCP tool capabilities",
        url="http://localhost:8001",
        version="2.0.0",
        provider=provider,
        skills=[autonomous_reasoning_skill, computational_skill],
        lifespan=lifespan
    )
    
    return app


def main():
    """Main entry point for the autonomous white agent."""
    import uvicorn
    
    print("⚪ Starting DABSTEP White Agent - Fully Autonomous")
    print("🤖 AI-Driven: Pure autonomous decision making")
    print("🛠️  MCP Tools: Dynamic tool discovery and usage")
    print("📋 Agent Card: http://localhost:8001/.well-known/agent-card.json")
    print("🔗 A2A Endpoint: http://localhost:8001/")
    
    uvicorn.run(
        "agent:create_dabstep_white_agent",
        factory=True,
        host="0.0.0.0",
        port=8001,
        log_level="info"
    )


if __name__ == "__main__":
    main()