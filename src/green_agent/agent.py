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
    
    async def _get_context_files_info(self) -> str:
        """Get information about available context files."""
        import os
        context_dir = "/Users/eleonorecharles/Desktop/dabstep/data/context"
        
        if not os.path.exists(context_dir):
            return "No context files available."
        
        files_info = []
        for filename in os.listdir(context_dir):
            filepath = os.path.join(context_dir, filename)
            if os.path.isfile(filepath):
                files_info.append(f"- {filename}: Available in {filepath}")
        
        return "\n".join(files_info) if files_info else "No context files available."

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
            
            # Load context files information
            context_files_info = await self._get_context_files_info()
            
            # Process each task
            for i, task in enumerate(tasks):
                print(f"📋 Task {i+1}/{len(tasks)}: {task.get('task_id', f'task_{i}')}")
                
                try:
                    # Create task message for white agent with context
                    task_prompt = f"""You are an expert data analyst and you will answer factoid questions by loading and referencing the files/documents listed below.
You have these files available:
{context_files_info}
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
                        
                        # Wait for completion
                        await asyncio.sleep(3)  # Give agent time to process
                        
                        task_response = await client.get_task(task_id)
                        
                        if 'result' in task_response:
                            final_task = task_response['result']
                            
                            # Extract agent's answer
                            agent_answer = self._extract_agent_answer(final_task)
                            
                            if agent_answer:
                                # Score using DABSTEP
                                score = question_scorer(agent_answer, task['correct_answer'])
                                
                                # Enhanced analysis with LLM if available
                                llm_analysis = None
                                if self.llm_client:
                                    try:
                                        llm_analysis = await self._analyze_with_llm(
                                            task['question'], 
                                            agent_answer, 
                                            task['correct_answer']
                                        )
                                    except Exception as e:
                                        print(f"   ⚠️  LLM analysis failed: {e}")
                                
                                task_result = {
                                    'task_id': task.get('task_id', f'task_{i}'),
                                    'question': task['question'],
                                    'agent_answer': agent_answer,
                                    'correct_answer': task['correct_answer'],
                                    'score': score,
                                    'level': task.get('level', 'unknown'),
                                    'llm_analysis': llm_analysis
                                }
                                
                                task_results.append(task_result)
                                if score:
                                    correct_count += 1
                                
                                print(f"   Answer: {agent_answer}")
                                print(f"   Score: {'✅ CORRECT' if score else '❌ INCORRECT'}")
                                
                                # Show LLM analysis if available
                                if llm_analysis and isinstance(llm_analysis, dict):
                                    if 'reasoning_analysis' in llm_analysis:
                                        print(f"   🤖 Analysis: {llm_analysis['reasoning_analysis']}")
                                    if 'confidence' in llm_analysis:
                                        print(f"   📊 Confidence: {llm_analysis['confidence']:.2f}")
                            else:
                                print(f"   ⚠️  Could not extract answer")
                                task_results.append({
                                    'task_id': task.get('task_id', f'task_{i}'),
                                    'question': task['question'],
                                    'agent_answer': '[No answer extracted]',
                                    'correct_answer': task['correct_answer'],
                                    'score': False,
                                    'level': task.get('level', 'unknown')
                                })
                        else:
                            print(f"   ❌ Task failed: {task_response.get('error', 'Unknown error')}")
                            task_results.append({
                                'task_id': task.get('task_id', f'task_{i}'),
                                'question': task['question'],
                                'agent_answer': '[Task failed]',
                                'correct_answer': task['correct_answer'],
                                'score': False,
                                'level': task.get('level', 'unknown')
                            })
                    else:
                        print(f"   ❌ Message send failed: {response.get('error', 'Unknown error')}")
                        task_results.append({
                            'task_id': task.get('task_id', f'task_{i}'),
                            'question': task['question'],
                            'agent_answer': '[Message send failed]',
                            'correct_answer': task['correct_answer'],
                            'score': False,
                            'level': task.get('level', 'unknown')
                        })
                        
                except Exception as e:
                    print(f"   ❌ Error: {e}")
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
    
    async def _analyze_with_llm(self, question: str, agent_answer: str, correct_answer: str) -> Dict[str, Any]:
        """Use LLM to provide detailed analysis of the agent's performance."""
        if not self.llm_client:
            return {"analysis": "LLM analysis not available", "confidence": 0.5}
        
        try:
            analysis_prompt = f"""
Analyze this DABSTEP evaluation result:

Question: {question}
Agent's Answer: {agent_answer}
Correct Answer: {correct_answer}

Please provide:
1. Whether the agent's answer is correct (yes/no)
2. Analysis of the reasoning quality
3. Suggestions for improvement
4. Confidence score (0-1)

Format as JSON with keys: correct, reasoning_analysis, suggestions, confidence
"""
            
            # Use LiteLLM's unified interface
            response = await self.llm_client.acompletion(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator analyzing AI agent performance on DABSTEP benchmark tasks. Provide detailed, constructive analysis."},
                    {"role": "user", "content": analysis_prompt}
                ],
                max_tokens=300
            )
            analysis_text = response.choices[0].message.content.strip()
            
            # Try to parse JSON response
            try:
                import json
                analysis_data = json.loads(analysis_text)
                return analysis_data
            except json.JSONDecodeError:
                return {
                    "analysis": analysis_text,
                    "confidence": 0.7
                }
                
        except Exception as e:
            print(f"⚠️  LLM analysis error: {e}")
            return {"analysis": f"Analysis error: {str(e)}", "confidence": 0.3}

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