"""
DABSTEP Green Agent - A2A-compatible evaluator agent.

This agent:
1. Receives evaluation requests containing tasks and target agent URL
2. Sends tasks to the white agent (agent under test)
3. Collects responses and evaluates them using DABSTEP scoring
4. Returns comprehensive evaluation results
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

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

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

# Import scorer from same directory
try:
    from .scorer import question_scorer
    from ..shared_utils import setup_llm_client
except ImportError:
    # Fallback for when running as script
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from scorer import question_scorer
    from shared_utils import setup_llm_client


# Context type for the green agent
Context = List[Message]


class DABStepGreenWorker(Worker[Context]):
    """Green agent worker that evaluates other A2A agents using DABSTEP."""
    
    def __init__(self, broker, storage, llm_client=None):
        super().__init__(storage=storage, broker=broker)
        self.llm_client = llm_client
        self.llm_client, self.model_name = setup_llm_client()
    
    async def run_task(self, params: TaskSendParams) -> None:
        """Execute a DABSTEP evaluation task."""
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
                # Handle evaluation request
                print("🔍 Processing DABSTEP evaluation request...")
                results = await self._evaluate_agent(eval_request)
                response_message = self._create_response_message(results)
                artifacts = self._create_response_artifacts(results)
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
            response_text = "Hello! I'm the DABSTEP Green Agent, an evaluator that can assess other AI agents using DABSTEP benchmark tasks. How can I help you today?"
        elif 'test' in user_text or 'check' in user_text:
            response_text = "I'm working properly! I can evaluate A2A agents using DABSTEP benchmark tasks. Send me an evaluation request with a white agent URL and tasks to get started."
        elif any(word in user_text for word in ['help', 'what', 'how', 'explain']):
            response_text = """I'm the DABSTEP Green Agent! Here's what I can do:

🎯 **Primary Function**: Evaluate AI agents using DABSTEP benchmark tasks
📋 **Evaluation Process**: Send tasks to target agents and score their responses
📊 **Scoring**: Use DABSTEP methodology to assess accuracy and reasoning

To run an evaluation, send me a message with:
- `white_agent_url`: URL of the agent to test
- `tasks`: Array of DABSTEP tasks with questions and correct answers

Example: "Please evaluate the agent at http://localhost:8001 using these tasks: [...]"
"""
        else:
            response_text = f"I received your message: '{user_text}'. I'm specialized in evaluating AI agents using DABSTEP benchmark tasks. Would you like me to run an evaluation?"
        
        return Message(
            role='agent',
            parts=[TextPart(text=response_text, kind='text')],
            kind='message',
            message_id=str(uuid.uuid4())
        )

    async def _evaluate_agent(self, eval_request: dict) -> dict:
        """Evaluate an agent using DABSTEP tasks."""
        white_agent_url = eval_request['white_agent_url']
        tasks = eval_request['tasks']
        
        print(f"🤖 Starting DABSTEP evaluation of agent at {white_agent_url}")
        print(f"📝 Evaluating {len(tasks)} tasks")
        
        # Create A2A client to communicate with white agent
        client = A2AClient(base_url=white_agent_url)
        
        start_time = time.time()
        task_results = []
        correct_count = 0
        
        try:
            # Check if white agent is available
            try:
                import httpx
                async with httpx.AsyncClient() as http_client:
                    response = await http_client.get(f"{white_agent_url}/.well-known/agent-card.json")
                    if response.status_code == 200:
                        agent_card = response.json()
                        print(f"🏷️  Testing agent: {agent_card.get('name', 'Unknown')}")
                    else:
                        raise Exception(f"White agent not available at {white_agent_url}")
            except ImportError:
                print("⚠️  httpx not available, skipping agent card check")
            except Exception as e:
                raise Exception(f"Failed to connect to white agent: {e}")
            
            # Process each task
            for i, task in enumerate(tasks):
                print(f"📋 Task {i+1}/{len(tasks)}: {task.get('task_id', f'task_{i}')}")
                
                try:
                    # Create task message for white agent with context
                    task_prompt = f"""You are an expert data analyst and you will answer factoid questions by loading and referencing the files/documents listed below.
Don't forget to reference any documentation in the data dir before answering a question.

Here is the question you need to answer:
{task['question']}

Here are the guidelines you must follow when answering the question above:
{task.get('guidelines', 'Answer the question accurately based on the available data.')}
"""
                    
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
                    
                    # Send task to white agent
                    response = await client.send_message(task_message)
                    
                    if 'result' in response:
                        # Get task result
                        agent_task = response['result']
                        task_id = agent_task['id']
                        
                        # Wait for completion with enhanced polling and logging
                        max_wait_time = 600  # Maximum wait time in seconds (10 minutes)
                        poll_interval = 10   # Check every 10 seconds
                        elapsed_time = 0
                        last_status = None
                        
                        print(f"   ⏳ Waiting for white agent to complete task...")
                        print(f"   📊 Max wait time: {max_wait_time}s, polling every {poll_interval}s")
                        
                        while elapsed_time < max_wait_time:
                            await asyncio.sleep(poll_interval)
                            elapsed_time += poll_interval
                            
                            try:
                                task_response = await client.get_task(task_id)
                                
                                if 'result' in task_response:
                                    final_task = task_response['result']
                                    task_status = final_task.get('status', {}).get('state', 'unknown')
                                    
                                    # Only log status changes or every 30 seconds to reduce noise
                                    if task_status != last_status or elapsed_time % 30 == 0:
                                        if task_status == 'working':
                                            print(f"   🔄 White agent working... ({elapsed_time}s elapsed)")
                                        elif task_status == 'submitted':
                                            print(f"   📝 White agent processing... ({elapsed_time}s elapsed)")
                                        else:
                                            print(f"   ⏱️ White agent status: {task_status} ({elapsed_time}s elapsed)")
                                        last_status = task_status
                                    
                                    if task_status == 'completed':
                                        print(f"   ✅ White agent completed task after {elapsed_time}s")
                                        break
                                    elif task_status == 'failed':
                                        print(f"   ❌ White agent task failed after {elapsed_time}s")
                                        break
                                    elif task_status in ['working', 'submitted']:
                                        continue  # Keep waiting
                                    else:
                                        print(f"   ⚠️ Unknown white agent task status: {task_status}")
                                        break
                                else:
                                    print(f"   ❌ Failed to get white agent task status: {task_response.get('error', 'Unknown error')}")
                                    break
                            except Exception as e:
                                print(f"   ⚠️ Error checking white agent status: {e}")
                                # Continue trying instead of breaking
                        
                        if elapsed_time >= max_wait_time:
                            print(f"   ⏰ White agent task timed out after {max_wait_time}s")
                        
                        # Get final task result and extract answer
                        print(f"   📥 Retrieving final result from white agent...")
                        task_response = await client.get_task(task_id)
                        
                        if 'result' in task_response:
                            final_task = task_response['result']
                            
                            # Extract agent's answer with detailed logging
                            print(f"   🔍 Extracting answer from white agent response...")
                            agent_answer = self._extract_agent_answer(final_task)
                            
                            if agent_answer:
                                print(f"   💬 White agent answered: '{agent_answer}'")
                                print(f"   ✓ Expected answer: '{task['correct_answer']}'")
                                
                                # Get baseline DABSTEP score
                                dabstep_score = question_scorer(agent_answer, task['correct_answer'])
                                print(f"   � DABSTEP baseline score: {'✅' if dabstep_score else '❌'}")
                                
                                # Enhanced analysis with LLM if available
                                llm_analysis = None
                                final_score = dabstep_score  # Default to DABSTEP score
                                
                                if self.llm_client:
                                    try:
                                        print(f"   🤖 Running LLM analysis for final scoring...")
                                        llm_analysis = await self._analyze_with_llm(
                                            task['question'], 
                                            agent_answer, 
                                            task['guidelines'],
                                            task['correct_answer']
                                        )
                                        if llm_analysis:
                                            # Use LLM analysis as the final score
                                            final_score = llm_analysis.get('correct', dabstep_score)
                                            print(f"   🧠 LLM final score: {'✅' if final_score else '❌'}")
                                            if final_score != dabstep_score:
                                                print(f"   🔄 LLM overrode DABSTEP score!")
                                        else:
                                            print(f"   🧠 LLM analysis returned None, using DABSTEP score")
                                    except Exception as e:
                                        print(f"   ⚠️ LLM analysis failed: {e}")
                                        llm_analysis = None
                                
                                task_result = {
                                    'task_id': task.get('task_id', f'task_{i}'),
                                    'question': task['question'],
                                    'agent_answer': agent_answer,
                                    'correct_answer': task['correct_answer'],
                                    'score': final_score,  # Use final score (LLM or DABSTEP fallback)
                                    'dabstep_score': dabstep_score,  # Keep DABSTEP for reference
                                    'level': task.get('level', 'unknown'),
                                    'llm_analysis': llm_analysis
                                }
                                
                                task_results.append(task_result)
                                if final_score:
                                    correct_count += 1
                                
                                print(f"   📊 Final Result: {'✅ CORRECT' if final_score else '❌ INCORRECT'}")
                                
                                # Show LLM analysis if available
                                if llm_analysis and isinstance(llm_analysis, dict):
                                    if 'reasoning_analysis' in llm_analysis:
                                        print(f"   🧠 Analysis: {llm_analysis['reasoning_analysis']}")
                                    if 'confidence' in llm_analysis:
                                        print(f"   🧠 Confidence: {llm_analysis['confidence']:.2f}")
                            else:
                                print(f"   ⚠️ Could not extract answer from white agent response")
                                print(f"   🔍 Response structure: {type(final_task).__name__} with keys: {list(final_task.keys()) if isinstance(final_task, dict) else 'N/A'}")
                                task_results.append({
                                    'task_id': task.get('task_id', f'task_{i}'),
                                    'question': task['question'],
                                    'agent_answer': '[No answer extracted]',
                                    'correct_answer': task['correct_answer'],
                                    'score': False,
                                    'level': task.get('level', 'unknown')
                                })
                        else:
                            print(f"   ❌ White agent task failed: {task_response.get('error', 'Unknown error')}")
                            print(f"   🔍 Full response: {task_response}")
                            task_results.append({
                                'task_id': task.get('task_id', f'task_{i}'),
                                'question': task['question'],
                                'agent_answer': '[Task failed]',
                                'correct_answer': task['correct_answer'],
                                'score': False,
                                'level': task.get('level', 'unknown')
                            })
                    else:
                        print(f"   ❌ Failed to send message to white agent: {response.get('error', 'Unknown error')}")
                        print(f"   🔍 Full response: {response}")
                        task_results.append({
                            'task_id': task.get('task_id', f'task_{i}'),
                            'question': task['question'],
                            'agent_answer': '[Message send failed]',
                            'correct_answer': task['correct_answer'],
                            'score': False,
                            'level': task.get('level', 'unknown')
                        })
                        
                except Exception as e:
                    print(f"   ❌ Exception during task execution: {e}")
                    print(f"   🔍 Exception type: {type(e).__name__}")
                    import traceback
                    print(f"   📍 Traceback: {traceback.format_exc()}")
                    task_results.append({
                        'task_id': task.get('task_id', f'task_{i}'),
                        'question': task['question'],
                        'agent_answer': f'[Error: {str(e)}]',
                        'correct_answer': task['correct_answer'],
                        'score': False,
                        'level': task.get('level', 'unknown')
                    })
        
        except Exception as e:
            raise Exception(f"Evaluation failed: {e}")
        
        # Calculate results
        total_time = time.time() - start_time
        total_tasks = len(task_results)
        accuracy = correct_count / total_tasks if total_tasks > 0 else 0.0
        
        results = {
            'success': True,
            'total_tasks': total_tasks,
            'correct_tasks': correct_count,
            'accuracy': accuracy,
            'time_used': total_time,
            'task_results': task_results
        }
        
        print(f"🎯 Evaluation complete! Accuracy: {accuracy * 100:.1f}% ({correct_count}/{total_tasks})")
        
        return results
    
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
    
    def _create_response_message(self, results: Dict[str, Any]) -> Message:
        """Create response message with evaluation results."""
        if results['success']:
            text = f"DABSTEP evaluation completed successfully!\n"
            text += f"Accuracy: {results['accuracy'] * 100:.1f}% ({results['correct_tasks']}/{results['total_tasks']})\n"
            text += f"Time used: {results['time_used']:.2f} seconds"
        else:
            text = f"DABSTEP evaluation failed: {results.get('error', 'Unknown error')}"
        
        return Message(
            role='agent',
            parts=[TextPart(text=text, kind='text')],
            kind='message',
            message_id=str(uuid.uuid4())
        )
    
    def _create_response_artifacts(self, results: Dict[str, Any]) -> List[Artifact]:
        """Create artifacts with detailed results."""
        artifacts = []
        
        # Summary artifact
        summary_artifact = Artifact(
            artifact_id=str(uuid.uuid4()),
            name="dabstep_evaluation_summary",
            description="DABSTEP evaluation summary results",
            parts=[DataPart(data={
                'success': results['success'],
                'accuracy': results['accuracy'],
                'total_tasks': results['total_tasks'],
                'correct_tasks': results['correct_tasks'],
                'time_used': results['time_used']
            }, kind='data')]
        )
        artifacts.append(summary_artifact)
        
        # Detailed results artifact
        if 'task_results' in results:
            details_artifact = Artifact(
                artifact_id=str(uuid.uuid4()),
                name="dabstep_evaluation_details",
                description="Detailed DABSTEP evaluation results",
                parts=[DataPart(data={
                    'task_results': results['task_results']
                }, kind='data')]
            )
            artifacts.append(details_artifact)
        
        return artifacts
    
    async def _analyze_with_llm(self, question: str, agent_answer: str, guidelines: str, correct_answer: str) -> Dict[str, Any]:
        """Use LLM to provide detailed analysis of the agent's performance."""

        analysis_prompt = f"""
Analyze this DABSTEP evaluation result:

Question: {question}
Guidelines: {guidelines}
Agent's Answer: {agent_answer}
Correct Answer: {correct_answer}

Please provide a JSON response with the following structure:
{{
    "correct": true/false,
    "reasoning_analysis": "your detailed analysis here",
    "suggestions": "your suggestions for improvement",
    "confidence": 0.0-1.0
}}

IMPORTANT: Respond ONLY with valid JSON. Do not include any text before or after the JSON.
"""
            
        try:
            # Use LiteLLM's unified interface
            response = await self.llm_client.acompletion(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator. You must respond with valid JSON only, no additional text."},
                    {"role": "user", "content": analysis_prompt}
                ],
                max_tokens=500,
                temperature=0.1  # Lower temperature for more consistent JSON output
            )
            
            # Extract and validate response content
            if not response or not response.choices or not response.choices[0] or not response.choices[0].message:
                print(f"      ⚠️ LLM returned invalid response structure")
                return None
            
            analysis_text = response.choices[0].message.content
            
            if not analysis_text or not analysis_text.strip():
                print(f"      ⚠️ LLM returned empty content")
                return None
            
            analysis_text = analysis_text.strip()
            
            # Try to clean up the response if it has markdown formatting
            if analysis_text.startswith("```json"):
                analysis_text = analysis_text[7:]  # Remove ```json
            if analysis_text.startswith("```"):
                analysis_text = analysis_text[3:]   # Remove ```
            if analysis_text.endswith("```"):
                analysis_text = analysis_text[:-3]  # Remove trailing ```
            
            analysis_text = analysis_text.strip()
            
            # Try to parse as JSON
            analysis_data = json.loads(analysis_text)
            
            # Validate that it's a dictionary
            if not isinstance(analysis_data, dict):
                print(f"      ⚠️ LLM response is not a JSON object: {type(analysis_data)}")
                return None
            
            # Validate required keys exist
            required_keys = ['correct', 'reasoning_analysis', 'suggestions', 'confidence']
            missing_keys = [key for key in required_keys if key not in analysis_data]
            if missing_keys:
                print(f"      ⚠️ LLM response missing required keys: {missing_keys}")
                return None
            
            return analysis_data
                
        except json.JSONDecodeError as e:
            print(f"      ⚠️ Failed to parse LLM response as JSON: {e}")
            print(f"      📝 Raw LLM response: '{analysis_text[:200]}{'...' if len(analysis_text) > 200 else ''}'")
            return None
                
        except Exception as e:
            print(f"      ⚠️ LLM analysis failed: {e}")
            return None
        
    def _extract_text_from_parts(self, parts: List[Any]) -> str:
        """Extract text from message parts."""
        text_parts = []
        for part in parts:
            if part['kind'] == 'text':
                text_parts.append(part['text'])
        return ' '.join(text_parts)


def create_dabstep_green_agent() -> FastA2A:
    """Create the DABSTEP green agent."""
    
    # Initialize storage and broker
    storage = InMemoryStorage[Context]()
    broker = InMemoryBroker()
    
    # Define skills
    evaluation_skill = Skill(
        id="dabstep-evaluation",
        name="DABSTEP Benchmark Evaluation",
        description="Evaluates A2A agents using DABSTEP benchmark tasks and scoring methodology",
        tags=["evaluation", "benchmark", "scoring", "dabstep"],
        examples=[
            "Evaluate an A2A agent on DABSTEP tasks",
            "Run benchmark evaluation and return accuracy scores"
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
        name="DABSTEP Green Agent",
        description="A2A-compatible green agent that evaluates other agents using DABSTEP benchmark",
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
    
    print("🟢 Starting DABSTEP Green Agent...")
    print("📋 Agent Card: http://localhost:8000/.well-known/agent-card.json")
    print("🔗 A2A Endpoint: http://localhost:8000/")
    
    uvicorn.run(
        "agent:create_dabstep_green_agent",
        factory=True,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )


if __name__ == "__main__":
    main()