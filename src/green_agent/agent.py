"""
DABSTEP Green Agent - A2A-compatible evaluator agent using Pydantic Eval.

This agent:
1. Receives evaluation requests containing tasks and target agent URL
2. Sends tasks to the white agent (agent under test)
3. Evaluates responses using pydantic-evals with LLM as judge
4. Returns pydantic eval report as results
"""

import json
import uuid
import time
import asyncio
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

from fasta2a import FastA2A, Skill, Worker
from fasta2a.broker import InMemoryBroker
from fasta2a.storage import InMemoryStorage
from fasta2a.schema import Artifact, Message, TaskIdParams, TaskSendParams, TextPart, DataPart, AgentProvider
from fasta2a.client import A2AClient

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge, Contains, EqualsExpected

import sys
import os
import logging

logging.getLogger("pydantic_ai").setLevel(logging.DEBUG)

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Load from project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    dotenv_path = os.path.join(project_root, '.env')
    load_dotenv(dotenv_path)
    print(f"🔧 Green Agent loaded environment variables from {dotenv_path}")
except ImportError:
    print("⚠️  python-dotenv not installed, skipping .env file loading")
except Exception as e:
    print(f"⚠️  Failed to load .env file: {e}")

# Import shared utils
try:
    from ..shared_utils import get_pydantic_ai_model
except ImportError:
    # Fallback for when running as script
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from shared_utils import get_pydantic_ai_model


# Context type for the green agent
Context = List[Message]


class DABStepGreenWorker(Worker[Context]):
    """Green agent worker that evaluates other A2A agents using Pydantic Eval."""
    
    def __init__(self, broker, storage):
        super().__init__(storage=storage, broker=broker)
        self.pydantic_ai_model = get_pydantic_ai_model()
    
    async def run_task(self, params: TaskSendParams) -> None:
        """Execute a DABSTEP evaluation task using Pydantic Eval."""
        task = await self.storage.load_task(params['id'])
        if task is None:
            await self.storage.update_task(params['id'], state='failed')
            return
        
        await self.storage.update_task(task['id'], state='working')
        
        try:
            # Load context
            context = await self.storage.load_context(task['context_id']) or []
            context.extend(task.get('history', []))
            
            # Extract message content
            message = params['message']
            eval_request = self._extract_evaluation_request(message)
            
            if eval_request:
                # Handle evaluation request using Pydantic Eval
                print("🔍 Processing DABSTEP evaluation request with Pydantic Eval...")
                report = await self._evaluate_agent_pydantic(eval_request)
                response_message = self._create_response_message(report)
                artifacts = self._create_response_artifacts(report)
            else:
                # Handle simple message (like "hello")
                print("💬 Processing simple message...")
                response_message = await self._handle_simple_message(message)
                artifacts = []
            
            # Update context and complete task
            context.append(response_message)
            await self.storage.update_context(task['context_id'], context)
            await self.storage.update_task(
                task['id'],
                state='completed',
                new_messages=[response_message],
                new_artifacts=artifacts
            )
            
        except Exception as e:
            # Handle errors
            error_message = Message(
                role='agent',
                parts=[TextPart(text=f"Evaluation failed: {str(e)}", kind='text')],
                kind='message',
                message_id=str(uuid.uuid4())
            )
            
            await self.storage.update_context(task['context_id'], context + [error_message])
            await self.storage.update_task(
                task['id'],
                state='failed',
                new_messages=[error_message]
            )
    
    async def cancel_task(self, params: TaskIdParams) -> None:
        """Cancel an evaluation task."""
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
    
    def _extract_evaluation_request(self, message: Message) -> Optional[Dict[str, Any]]:
        """Extract evaluation request from message."""
        for part in message['parts']:
            if part['kind'] == 'data':
                data = part.get('data', {})
                if 'white_agent_url' in data and 'tasks' in data:
                    return data
            elif part['kind'] == 'text':
                try:
                    data = json.loads(part['text'])
                    if 'white_agent_url' in data and 'tasks' in data:
                        return data
                except (json.JSONDecodeError, ValueError):
                    continue
        return None
    
    async def _handle_simple_message(self, message: Message) -> Message:
        """Handle simple messages like greetings or basic questions."""
        # Extract text from message
        user_text = ""
        for part in message['parts']:
            if part['kind'] == 'text':
                user_text += part['text'] + " "
        user_text = user_text.strip().lower()
        
        # Generate appropriate response based on content
        if any(greeting in user_text for greeting in ['hello', 'hi', 'hey', 'greetings']):
            response_text = "Hello! I'm the DABSTEP Green Agent, an evaluator that can assess other AI agents using DABSTEP benchmark tasks with Pydantic Eval. How can I help you today?"
        elif 'test' in user_text or 'check' in user_text:
            response_text = "I'm working properly! I can evaluate A2A agents using DABSTEP benchmark tasks with Pydantic Eval and LLM as judge. Send me an evaluation request with a white agent URL and tasks to get started."
        elif any(word in user_text for word in ['help', 'what', 'how', 'explain']):
            response_text = """I'm the DABSTEP Green Agent! Here's what I can do:

🎯 **Primary Function**: Evaluate AI agents using DABSTEP benchmark tasks
📋 **Evaluation Process**: Send tasks to target agents and score their responses using Pydantic Eval
📊 **Scoring**: Use LLM as judge with structured evaluation criteria
🔍 **Framework**: Built on pydantic-evals for robust evaluation pipelines

To run an evaluation, send me a message with:
- `white_agent_url`: URL of the agent to test
- `tasks`: Array of DABSTEP tasks with questions and correct answers

Example: "Please evaluate the agent at http://localhost:8001 using these tasks: [...]"
"""
        else:
            response_text = f"I received your message: '{user_text}'. I'm specialized in evaluating AI agents using DABSTEP benchmark tasks with Pydantic Eval. Would you like me to run an evaluation?"
        
        return Message(
            role='agent',
            parts=[TextPart(text=response_text, kind='text')],
            kind='message',
            message_id=str(uuid.uuid4())
        )

    async def _evaluate_agent_pydantic(self, eval_request: dict) -> dict:
        """Evaluate an agent using Pydantic Eval framework."""
        white_agent_url = eval_request['white_agent_url']
        tasks = eval_request['tasks']
        
        print(f"🤖 Starting Pydantic Eval assessment of agent at {white_agent_url}")
        print(f"📝 Evaluating {len(tasks)} tasks")
        
        # Create timeout configuration for white agent communication
        import httpx
        timeout = httpx.Timeout(
            connect=30.0,   # Connection timeout: 30 seconds
            read=600.0,     # Read timeout: 10 minutes (white agent can take time to analyze data)
            write=30.0,     # Write timeout: 30 seconds
            pool=30.0       # Pool timeout: 30 seconds
        )
        
        # Create evaluation function that calls the white agent
        async def evaluate_white_agent(task_input: str) -> str:
            """Function that sends tasks to the white agent and returns the response."""
            try:
                # Parse task input to get question and guidelines
                task_data = json.loads(task_input)
                question = task_data['question']
                guidelines = task_data['guidelines']
                
                # Create task prompt for white agent
                task_prompt = f"""You are an expert data analyst and you will answer the question using the tools at your disposal.

Here is the question you need to answer:
{question}

Here are the guidelines you must follow:
{guidelines}
"""
                
                # Create fresh HTTP client for each task to avoid context accumulation
                fresh_http_client = httpx.AsyncClient(timeout=timeout)
                fresh_client = A2AClient(base_url=white_agent_url, http_client=fresh_http_client)
                
                # Create task message for white agent
                task_message = Message(
                    role='user',
                    parts=[TextPart(
                        text=task_prompt,
                        kind='text'
                    )],
                    kind='message',
                    message_id=str(uuid.uuid4())
                )
                
                print(f"   📤 Sending task to white agent: {question[:50]}{'...' if len(question) > 50 else ''}")
                
                # Send task to white agent
                response = await fresh_client.send_message(task_message)
                                
                if 'result' in response:
                    # Get task result
                    agent_task = response['result']
                    task_id = agent_task['id']
                                        
                    # Wait for completion
                    max_wait_time = 600  # 10 minutes
                    poll_interval = 10   # Check every 10 seconds
                    elapsed_time = 0
                    
                    print(f"   ⏳ Waiting for white agent to complete task...")
                    
                    while elapsed_time < max_wait_time:
                        await asyncio.sleep(poll_interval)
                        elapsed_time += poll_interval
                        
                        try:
                            task_response = await fresh_client.get_task(task_id)
                            
                            if 'result' in task_response:
                                final_task = task_response['result']
                                task_status = final_task.get('status', {}).get('state', 'unknown')
                                
                                print(f"   📊 Task status: {task_status} (elapsed: {elapsed_time}s)")
                                
                                if task_status == 'completed':
                                    print(f"   ✅ White agent completed task after {elapsed_time}s")
                                    break
                                elif task_status == 'failed':
                                    print(f"   ❌ White agent task failed after {elapsed_time}s")
                                    await fresh_http_client.aclose()
                                    return f"[Task failed: {task_status}]"
                                elif task_status in ['working', 'submitted']:
                                    continue  # Keep waiting
                                else:
                                    print(f"   ⚠️ Unknown task status: {task_status}")
                                    break
                            else:
                                print(f"   ❌ Failed to get white agent task status: {task_response}")
                                await fresh_http_client.aclose()
                                return "[Failed to get task status]"
                        except Exception as e:
                            print(f"   ⚠️ Error checking white agent status: {e}")
                            import traceback
                            print(f"   📍 Status check traceback: {traceback.format_exc()}")
                            continue
                    
                    if elapsed_time >= max_wait_time:
                        print(f"   ⏰ White agent task timed out after {max_wait_time}s")
                        await fresh_http_client.aclose()
                        return "[Task timed out]"
                    
                    # Get final task result and extract answer
                    task_response = await fresh_client.get_task(task_id)
                                        
                    if 'result' in task_response:
                        final_task = task_response['result']
                        
                        # Use the outer self reference
                        agent_answer = self._extract_agent_answer(final_task)
                        
                        if agent_answer:
                            print(f"   💬 White agent answered: '{agent_answer[:100]}{'...' if len(agent_answer) > 100 else ''}'")
                            await fresh_http_client.aclose()
                            return agent_answer
                        else:
                            print(f"   ⚠️ Could not extract answer from white agent response")
                            print(f"   🔍 Final task structure: {list(final_task.keys()) if isinstance(final_task, dict) else type(final_task)}")
                            await fresh_http_client.aclose()
                            return "[No answer extracted]"
                    else:
                        error = task_response.get('error', 'Unknown error')
                        print(f"   ❌ White agent task failed: {error}")
                        await fresh_http_client.aclose()
                        return f"[Task failed: {error}]"
                else:
                    error = response.get('error', 'Unknown error')
                    print(f"   ❌ Failed to send message to white agent: {error}")
                    print(f"   🔍 Full response: {response}")
                    await fresh_http_client.aclose()
                    return f"[Message send failed: {error}]"
                        
            except Exception as e:
                print(f"   ❌ Exception during white agent evaluation: {e}")
                print(f"   🔍 Exception type: {type(e).__name__}")
                import traceback
                print(f"   📍 Full traceback: {traceback.format_exc()}")
                # Clean up if we have a fresh_http_client
                try:
                    await fresh_http_client.aclose()
                except:
                    pass
                return f"[Error: {str(e)}]"
        
        # Create pydantic eval cases from DABSTEP tasks
        cases = []
        for i, task in enumerate(tasks):
            task_input_data = {
                'question': task['question'],
                'guidelines': task['guidelines'],
            }
            
            case = Case(
                name=task['task_id'],
                inputs=json.dumps(task_input_data),
                expected_output=task['correct_answer'],
                metadata={
                    'level': task['level'],
                    'task_id': task['task_id'],
                    'question': task['question']
                }
            )
            cases.append(case)
        
        # Create evaluators
        evaluators = []
        
        # Add LLM Judge evaluator

        
        evaluators.append(
            LLMJudge(
                rubric="Response is accurate",
                include_input=True,
                include_expected_output=True,
                model=self.pydantic_ai_model
            )
        )
        
        # Create dataset
        dataset = Dataset(
            cases=cases,
            evaluators=evaluators
        )
        
        print(f"🔍 Running Pydantic Eval with {len(cases)} cases and {len(evaluators)} evaluators...")
        
        # Run evaluation
        start_time = time.time()
        try:
            report = await dataset.evaluate(evaluate_white_agent)
            evaluation_time = time.time() - start_time
            
            print(f"✅ Pydantic Eval completed in {evaluation_time:.2f} seconds")
            report.print(include_reasons=True, include_output=True, include_expected_output=True)
            
            # Return the report object directly
            return report
            
        except Exception as e:
            print(f"❌ Pydantic Eval failed: {e}")
            import traceback
            print(f"📍 Traceback: {traceback.format_exc()}")
            
            return {
                'success': False,
                'error': str(e),
                'evaluation_time': time.time() - start_time,
                'total_cases': len(cases)
            }
    
    def _extract_agent_answer(self, task_result: Dict[str, Any]) -> Optional[str]:
        """Extract agent's answer from task result."""
        # First try artifacts
        if 'artifacts' in task_result:
            for artifact in task_result['artifacts']:
                for part in artifact.get('parts', []):
                    if part.get('kind') == 'text':
                        text = part.get('text', '').strip()
                        if text:
                            return text
                    elif part.get('kind') == 'data':
                        data = part.get('data', {})
                        if isinstance(data, dict):
                            for key in ['answer', 'result', 'response', 'text', 'content']:
                                if key in data:
                                    return str(data[key]).strip()
                        elif isinstance(data, str):
                            return data.strip()
        
        # Then try messages
        if 'history' in task_result:
            for message in reversed(task_result['history']):
                if message.get('role') == 'agent':
                    for part in message.get('parts', []):
                        if part.get('kind') == 'text':
                            text = part.get('text', '').strip()
                            if text:
                                return text
        
        return None
    
    def _create_response_message(self, report) -> Message:
        """Create response message with Pydantic Eval results."""
        if hasattr(report, '__str__'):
            # Use the pydantic eval report's __str__ method
            text = str(report)
        elif isinstance(report, dict) and not report.get('success'):
            text = f"❌ Pydantic Eval assessment failed: {report.get('error', 'Unknown error')}"
        else:
            text = str(report)
                    
        return Message(
            role='agent',
            parts=[TextPart(text=text, kind='text')],
            kind='message',
            message_id=str(uuid.uuid4())
        )
                
    
    def _create_response_artifacts(self, report) -> List[Artifact]:
        """Create artifacts with Pydantic Eval results."""
        artifacts = []
        
        # Convert report to serializable format
        if hasattr(report, 'model_dump'):
            # Pydantic v2 style serialization
            report_data = report.model_dump()
        elif hasattr(report, 'dict'):
            # Pydantic v1 style serialization
            report_data = report.dict()
        elif isinstance(report, dict):
            # Already a dictionary
            report_data = report
        else:
            # Fallback - convert to string then wrap in dict
            report_data = {"report": str(report)}
        
        # Main report artifact
        report_artifact = Artifact(
            artifact_id=str(uuid.uuid4()),
            name="pydantic_eval_report",
            description="Complete Pydantic Eval assessment report",
            parts=[DataPart(data=report_data, kind='data')]
        )
        artifacts.append(report_artifact)
        
        return artifacts
        
    def _extract_text_from_parts(self, parts: List[Any]) -> str:
        """Extract text from message parts."""
        text_parts = []
        for part in parts:
            if part['kind'] == 'text':
                text_parts.append(part['text'])
        return ' '.join(text_parts)


def create_dabstep_green_agent() -> FastA2A:
    """Create the DABSTEP green agent with Pydantic Eval."""
    
    # Initialize storage and broker
    storage = InMemoryStorage[Context]()
    broker = InMemoryBroker()
    
    # Define skills
    evaluation_skill = Skill(
        id="pydantic-eval-assessment",
        name="Pydantic Eval DABSTEP Assessment",
        description="Evaluates A2A agents using DABSTEP benchmark tasks with Pydantic Eval framework and LLM as judge",
        tags=["evaluation", "pydantic-eval", "llm-judge", "benchmark", "scoring", "dabstep"],
        examples=[
            "Evaluate an A2A agent on DABSTEP tasks using Pydantic Eval",
            "Run benchmark assessment with LLM as judge and return structured report"
        ],
        input_modes=["application/json"],
        output_modes=["application/json"]
    )
    
    # Agent provider - configurable via environment
    import os
    provider = AgentProvider(
        organization=os.getenv("AGENT_ORGANIZATION", "Local"),
        url=os.getenv("AGENT_PROVIDER_URL", "http://localhost:8000")
    )
    
    # Create worker
    worker = DABStepGreenWorker(broker=broker, storage=storage)
    
    # Add lifespan to start worker
    @asynccontextmanager
    async def lifespan(app):
        async with app.task_manager:
            async with worker.run():
                yield
    
    # Create app with lifespan
    app = FastA2A(
        storage=storage,
        broker=broker,
        name="DABSTEP Green Agent (Pydantic Eval)",
        description="A2A-compatible green agent that evaluates other agents using DABSTEP benchmark with Pydantic Eval framework",
        url="http://localhost:8000",
        version="1.0.0",
        provider=provider,
        skills=[evaluation_skill],
        lifespan=lifespan
    )
    
    return app


def main():
    """Main entry point for the green agent."""
    import uvicorn
    
    print("🟢 Starting DABSTEP Green Agent with Pydantic Eval...")
    print("📋 Agent Card: http://localhost:8000/.well-known/agent-card.json")
    print("🔗 A2A Endpoint: http://localhost:8000/")
    print("🎯 Evaluation Framework: Pydantic Eval with LLM as Judge")
    
    uvicorn.run(
        "agent:create_dabstep_green_agent",
        factory=True,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )


if __name__ == "__main__":
    main()