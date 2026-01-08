"""
Green Agent - A2A-compatible evaluator agent using Pydantic Eval.

This agent:
1. Receives evaluation requests containing tasks and target agent URL
2. Sends tasks to the purple agent (agent under test)
3. Evaluates responses using pydantic-evals with LLM as judge
4. Returns pydantic eval report as results
"""

# Standard library imports
import argparse
import asyncio
import json
import os
import sys
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from datetime import datetime

# Third-party imports
import httpx
import uvicorn
from dotenv import load_dotenv

# FastA2A imports
from fasta2a import FastA2A, Skill, Worker
from fasta2a.broker import InMemoryBroker
from fasta2a.storage import InMemoryStorage
from fasta2a.schema import Artifact, Message, TaskIdParams, TaskSendParams, TextPart, DataPart, AgentProvider
from fasta2a.client import A2AClient

# Pydantic Eval imports
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge
from pydantic_ai.settings import ModelSettings

# Shared utilities
try:
    from ..shared_utils import get_pydantic_ai_model, setup_logger
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from shared_utils import get_pydantic_ai_model, setup_logger

# Load environment variables from .env file
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
dotenv_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path)
print(f"🔧 Green Agent loaded environment variables from {dotenv_path}")

DEFAULT_PORT = 9009

# Setup logger
logger = setup_logger("green_agent")


# Context type for the green agent
Context = List[Message]


class GreenWorker(Worker[Context]):
    """Green agent worker that evaluates other A2A agents using Pydantic Eval."""
    
    def __init__(self, broker, storage):
        super().__init__(storage=storage, broker=broker)
        self.pydantic_ai_model = get_pydantic_ai_model('green')
        self.completed_tasks = []  # Store completed task results for token usage aggregation
    
    async def run_task(self, params: TaskSendParams) -> None:
        """Execute an evaluation task using Pydantic Eval."""
        logger.info(f"🔄 Green Agent received task: {params['id']}")
        
        task = await self.storage.load_task(params['id'])
        if task is None:
            logger.error(f"❌ Task {params['id']} not found in storage")
            await self.storage.update_task(params['id'], state='failed')
            return
        
        logger.info(f"📝 Processing task {params['id']}")
        await self.storage.update_task(task['id'], state='working')
        
        # Send initial status update
        await self.broker.send_stream_event(params['id'], {
            'kind': 'status-update',
            'task_id': params['id'],
            'context_id': task['context_id'],
            'status': {'state': 'working'},
            'final': False,
            'metadata': {'message': 'Starting evaluation task...'}
        })
        
        try:
            # Load context
            context = await self.storage.load_context(task['context_id']) or []
            context.extend(task.get('history', []))
            
            # Extract message content
            message = params['message']
            eval_request = self._extract_evaluation_request(message)
            
            if eval_request:
                # Send evaluation processing status
                await self.broker.send_stream_event(params['id'], {
                    'kind': 'status-update',
                    'task_id': params['id'],
                    'context_id': task['context_id'],
                    'status': {'state': 'working'},
                    'final': False,
                    'metadata': {
                        'message': f'Processing {len(eval_request["tasks"])} evaluation tasks with Pydantic Eval...'
                    }
                })
                
                # Handle evaluation request using Pydantic Eval
                logger.info("🔍 Processing evaluation request with Pydantic Eval...")
                
                report = await self._evaluate_agent_pydantic(eval_request, task['id'])
                response_message = self._create_response_message(report)
                artifacts = self._create_response_artifacts(report)
            else:
                # Send simple message processing status
                await self.broker.send_stream_event(params['id'], {
                    'kind': 'status-update',
                    'task_id': params['id'],
                    'context_id': task['context_id'],
                    'status': {'state': 'working'},
                    'final': False,
                    'metadata': {'message': 'Processing simple message...'}
                })
                
                # Handle simple message (like "hello")
                logger.info("💬 Processing simple message...")
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
            
            # Get stored artifacts for streaming (A2A framework serializes them to dicts)
            stored_task = await self.storage.load_task(task['id'])
            stored_artifacts = stored_task.get('artifacts', []) if stored_task else []
            
            # Emit final completion event with artifact and status
            if stored_artifacts:
                # Send artifact update first
                artifact = stored_artifacts[0]
                report_data = artifact['parts'][0]['data']
                
                await self.broker.send_stream_event(params['id'], {
                    'kind': 'artifact-update',
                    'task_id': params['id'],
                    'context_id': task['context_id'],
                    'artifact': {
                        'artifact_id': artifact['artifact_id'],
                        'name': artifact['name'],
                        'description': artifact['description'],
                        'parts': [{'kind': 'data', 'data': report_data}]
                    }
                })
                
                # Then send final status update
                await self.broker.send_stream_event(params['id'], {
                    'kind': 'status-update',
                    'task_id': params['id'],
                    'context_id': task['context_id'],
                    'status': {'state': 'completed'},
                    'final': True
                })
            else:
                # Just status if no artifacts
                await self.broker.send_stream_event(params['id'], {
                    'kind': 'status-update',
                    'task_id': params['id'],
                    'context_id': task['context_id'],
                    'status': {'state': 'completed'},
                    'final': True
                })
            
            logger.info(f"✅ Task {task['id']} completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Error in task {task['id']}: {e}")
            logger.error(f"📍 Exception type: {type(e).__name__}")
            logger.error(f"📍 Traceback: {traceback.format_exc()}")
            
            # Send error status update
            await self.broker.send_stream_event(params['id'], {
                'kind': 'status-update',
                'task_id': params['id'],
                'context_id': task['context_id'],
                'status': {'state': 'failed'},
                'final': True,
                'metadata': {
                    'message': f'Evaluation failed: {str(e)}',
                    'error': str(e)
                }
            })
            
            # Handle errors
            error_message = Message(
                role='agent',
                parts=[TextPart(text=f"Evaluation failed: {str(e)}", kind='text')],
                kind='message',
                message_id=str(uuid.uuid4())
            )
            
            try:
                await self.storage.update_context(task['context_id'], context + [error_message])
                await self.storage.update_task(
                    task['id'],
                    state='failed',
                    new_messages=[error_message]
                )
                
                logger.info(f"💾 Task {task['id']} marked as failed")
            except Exception as storage_error:
                logger.error(f"💥 Failed to save error state for task {task['id']}: {storage_error}")
    
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
        """Extract evaluation request from message - supports both legacy and AgentBeats formats."""
        for part in message['parts']:
            if part['kind'] == 'data':
                data = part.get('data', {})
                # Legacy format check
                if 'purple_agent_url' in data and 'tasks' in data:
                    return data
                # New AgentBeats format check
                elif 'participants' in data and 'config' in data:
                    return self._convert_agentbeats_request(data)
            elif part['kind'] == 'text':
                try:
                    data = json.loads(part['text'])
                    # Legacy format check
                    if 'purple_agent_url' in data and 'tasks' in data:
                        return data
                    # New AgentBeats format check
                    elif 'participants' in data and 'config' in data:
                        return self._convert_agentbeats_request(data)
                except (json.JSONDecodeError, ValueError):
                    continue
        return None

    def _convert_agentbeats_request(self, agentbeats_data: dict) -> dict:
        """Convert AgentBeats format to internal evaluation format."""
        participants = agentbeats_data['participants']
        config = agentbeats_data['config']
        
        # Extract purple agent URL (assumes single participant for DABench)
        purple_agent_url = None
        for role, url in participants.items():
            purple_agent_url = url
            break  # Take the first participant as the agent to evaluate
        
        if not purple_agent_url:
            raise ValueError("No participants found in AgentBeats request")
        
        # Load tasks from dataset based on config
        tasks = self._load_tasks_from_dataset(config)
        
        return {
            'purple_agent_url': purple_agent_url,
            'tasks': tasks,
            'config': config  # Pass through config for potential future use
        }

    def _load_tasks_from_dataset(self, config: dict) -> List[dict]:
        """Load DABench tasks from dataset based on configuration."""
        dataset_path = config.get('dataset_path', 'data-dabench/')
        num_tasks = config.get('num_tasks', 3)
        quick_sample = config.get('quick_sample', True)
        
        logger.info(f"📂 Loading tasks from {dataset_path}")
        logger.info(f"🔢 Requesting {num_tasks} tasks (quick_sample: {quick_sample})")
        
        # Load tasks from the DABench dataset
        try:
            from ..data_loader import load_dabench_tasks
        except ImportError:
            sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
            from data_loader import load_dabench_tasks
        
        try:
            # Load the dataset using the data directory
            dataset = load_dabench_tasks(dataset_path)
            
            # Sample tasks based on configuration
            if quick_sample:
                # Take first num_tasks for predictable testing
                selected_tasks = dataset[:num_tasks]
                logger.info(f"🎯 Selected first {len(selected_tasks)} tasks for quick sample")
            else:
                # Take all requested tasks or all available tasks
                selected_tasks = dataset[:num_tasks] if num_tasks < len(dataset) else dataset
                logger.info(f"🎯 Selected {len(selected_tasks)} tasks from dataset")
            
            # Convert to the expected format
            converted_tasks = []
            for task in selected_tasks:
                converted_task = {
                    'task_id': f"task_{task.get('task_id', len(converted_tasks))}",
                    'question': task['question'],
                    'constraints': task.get('constraints', ''),
                    'format': task.get('format', ''),
                    'file_name': task.get('file_name', ''),
                    'correct_answer': task.get('correct_answer', ''),
                    'concepts': task.get('concepts', []),
                    'level': task.get('level', 'medium')
                }
                converted_tasks.append(converted_task)
            
            logger.info(f"✅ Successfully loaded {len(converted_tasks)} tasks")
            return converted_tasks
            
        except Exception as e:
            logger.error(f"❌ Failed to load tasks from dataset: {e}")
    
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
            response_text = "Hello! I'm the Green Agent, an evaluator that can assess other AI agents using benchmark tasks with Pydantic Eval. How can I help you today?"
        elif 'test' in user_text or 'check' in user_text:
            response_text = "I'm working properly! I can evaluate A2A agents using benchmark tasks with Pydantic Eval and LLM as judge. Send me an evaluation request with a purple agent URL and tasks to get started."
        elif any(word in user_text for word in ['help', 'what', 'how', 'explain']):
            response_text = """I'm the Green Agent! Here's what I can do:

🎯 **Primary Function**: Evaluate AI agents using benchmark tasks
📋 **Evaluation Process**: Send tasks to target agents and score their responses using Pydantic Eval
📊 **Scoring**: Use LLM as judge with structured evaluation criteria
🔍 **Framework**: Built on pydantic-evals for robust evaluation pipelines

To run an evaluation, send me a message with:
- `purple_agent_url`: URL of the agent to test
- `tasks`: Array of tasks with questions and correct answers

Example: "Please evaluate the agent at http://localhost:9019 using these tasks: [...]"
"""
        else:
            response_text = f"I received your message: '{user_text}'. I'm specialized in evaluating AI agents using benchmark tasks with Pydantic Eval. Would you like me to run an evaluation?"
        
        return Message(
            role='agent',
            parts=[TextPart(text=response_text, kind='text')],
            kind='message',
            message_id=str(uuid.uuid4())
        )

    async def _evaluate_agent_pydantic(self, eval_request: dict, task_id: str) -> dict:
        """Evaluate an agent using Pydantic Eval framework."""
        purple_agent_url = eval_request['purple_agent_url']
        tasks = eval_request['tasks']
        
        # Clear completed tasks from previous evaluations
        self.completed_tasks.clear()
        
        logger.info(f"🤖 Starting Pydantic Eval assessment of agent at {purple_agent_url}")
        logger.info(f"📝 Evaluating {len(tasks)} tasks")
        
        # Send progress update for evaluation setup
        await self.broker.send_stream_event(task_id, {
            'kind': 'status-update',
            'task_id': task_id,
            'context_id': task_id,  # Using task_id as fallback for context_id
            'status': {'state': 'working'},
            'final': False,
            'metadata': {
                'message': f'Setting up evaluation for {len(tasks)} tasks...',
                'progress': {'current': 0, 'total': len(tasks), 'phase': 'setup'}
            }
        })
        
        # Create timeout configuration for purple agent communication
        timeout = httpx.Timeout(
            connect=30.0,   # Connection timeout: 30 seconds
            read=1800.0,     # Read timeout: 30 minutes (data analysis can take time)
            write=30.0,     # Write timeout: 30 seconds
            pool=None       # No pool timeout - create fresh connections
        )
        
        # Create evaluation function that calls the purple agent
        async def evaluate_purple_agent(task_input: dict) -> str:
            """Function that sends tasks to the purple agent and returns the response."""
            http_client = None  # Initialize to None
            try:
                # The task_input is already a dictionary, no need to load it from a string.
                task_data = task_input
                question = task_data['question']
                constraints = task_data['constraints']
                format_info = task_data['format']
                file_name = task_data['file_name']
                case_name = task_data['case_name']  # Get the case name
                
                # Create task prompt for purple agent using DABench format
                task_prompt = f"""You are an expert data analyst and you will answer the question using the tools at your disposal.

Here is the question you need to answer:
{question}"""

                task_prompt += f"\n\nUse the data file: {file_name}"
                task_prompt += f"\n\nConstraints: {constraints}"
                task_prompt += f"\n\nExpected output format: {format_info}"
    
                # Create ONE HTTP client for this task (completely isolated)
                limits = httpx.Limits(max_keepalive_connections=0, max_connections=1)
                http_client = httpx.AsyncClient(timeout=timeout, limits=limits)
                client = A2AClient(base_url=purple_agent_url, http_client=http_client)
                
                # Create task message for purple agent
                message_id = str(uuid.uuid4())
                logger.info(f"   📤 Sending task to purple agent: {question[:50]}{'...' if len(question) > 50 else ''}")
                
                # Use streaming message endpoint instead of send_message
                try:
                    # Prepare streaming request
                    stream_payload = {
                        'jsonrpc': '2.0',
                        'id': str(uuid.uuid4()),
                        'method': 'message/stream',
                        'params': {
                            'message': {
                                'role': 'user',
                                'parts': [{'kind': 'text', 'text': task_prompt}],
                                'kind': 'message',
                                'messageId': message_id
                            }
                        }
                    }
                    
                    logger.info(f"   📡 Using streaming endpoint for real-time updates...")
                    
                    # Stream the response
                    final_task = None
                    agent_answer = None
                    token_usage_info = None
                    
                    async with http_client.stream('POST', purple_agent_url, json=stream_payload) as response:
                        if response.status_code != 200:
                            logger.error(f"   ❌ Streaming request failed: {response.status_code}")
                            return f"[Streaming request failed: {response.status_code}]"
                        
                        async for line in response.aiter_lines():
                            if line.startswith('data: '):
                                try:
                                    data = json.loads(line[6:])
                                    event = data.get('result', {})
                                    
                                    if event.get('kind') == 'status-update':
                                        status = event.get('status', {})
                                        state = status.get('state', 'unknown')
                                        metadata = event.get('metadata', {})
                                        message = metadata.get('message', '')
                                        
                                        # Extract token usage from metadata if available
                                        if 'token_usage' in metadata:
                                            token_usage_info = metadata['token_usage']
                                            logger.info(f"   🪙 Token usage: {token_usage_info}")
                                        
                                        logger.info(f"   📊 Purple agent: {state} - {message}")
                                        
                                        if event.get('final', False) and state == 'completed':
                                            logger.info(f"   ✅ Purple agent completed task")
                                            # We'll get the final result from the completed task event
                                            break
                                    
                                    elif event.get('kind') == 'task':
                                        final_task = event
                                        purple_task_id = event.get('id')
                                        logger.info(f"   🆔 Got task ID: {purple_task_id}")
                                    
                                    elif event.get('kind') == 'message' and event.get('role') == 'agent':
                                        # Extract the agent's answer from the message
                                        parts = event.get('parts', [])
                                        for part in parts:
                                            if part.get('kind') == 'text':
                                                agent_answer = part.get('text', '')
                                                logger.info(f"   💬 Got agent response from message: {agent_answer[:100]}...")
                                                break
                                    
                                    elif event.get('kind') == 'artifact-update':
                                        # Extract the agent's answer from the artifact
                                        artifact = event.get('artifact', {})
                                        logger.info(f"   📎 Received artifact: {artifact.get('name', 'unknown')}")
                                        
                                        # Look for answer in artifact parts
                                        for part in artifact.get('parts', []):
                                            if part.get('kind') == 'text':
                                                text_answer = part.get('text', '').strip()
                                                if text_answer and not agent_answer:
                                                    agent_answer = text_answer
                                                    logger.info(f"   💬 Got agent response from artifact text: {agent_answer[:100]}...")
                                            elif part.get('kind') == 'data':
                                                data = part.get('data', {})
                                                if isinstance(data, dict):
                                                    # Extract answer
                                                    if 'answer' in data and not agent_answer:
                                                        data_answer = str(data['answer']).strip()
                                                        if data_answer:
                                                            agent_answer = data_answer
                                                            logger.info(f"   💬 Got agent response from artifact data: {agent_answer[:100]}...")
                                                    
                                                    # Extract token usage if available
                                                    if 'token_usage' in data:
                                                        token_usage_info = data['token_usage']
                                                        logger.info(f"   🪙 Token usage from artifact: {token_usage_info}")
                                    
                                except json.JSONDecodeError as e:
                                    logger.warning(f"   ⚠️ Failed to parse streaming data: {e}")
                                    continue
                    
                    if agent_answer:
                        # Store completed task info for token usage aggregation
                        if final_task:
                            final_task_with_id = dict(final_task)
                            final_task_with_id['case_name'] = case_name
                            
                            # Add token usage information if we collected it
                            if token_usage_info:
                                final_task_with_id['token_usage'] = token_usage_info
                                logger.info(f"   📊 Storing token usage for task: {token_usage_info}")
                            
                            self.completed_tasks.append(final_task_with_id)
                        
                        return agent_answer
                    else:
                        logger.warning(f"   ⚠️ No answer received from streaming response")
                        return "[No answer received from streaming]"
                        
                except httpx.ReadTimeout as e:
                    logger.warning(f"   ⏰ Streaming timeout after {timeout.read}s")
                    return "[Streaming timeout]"
                    
                except Exception as e:
                    logger.error(f"   ❌ Error in streaming communication: {e}")
                    return f"[Streaming communication error: {e}]"
                else:
                    error = response.get('error', 'Unknown error')
                    logger.error(f"   ❌ Failed to send message to purple agent: {error}")
                    logger.error(f"   🔍 Full response: {response}")
                    return f"[Message send failed: {error}]"
                        
            except Exception as e:
                logger.error(f"   ❌ Exception during purple agent evaluation: {e}")
                logger.error(f"   🔍 Exception type: {type(e).__name__}")
                logger.error(f"   📍 Full traceback: {traceback.format_exc()}")
                return f"[Error: {str(e)}]"
            finally:
                # Ensure the client is always closed
                if http_client:
                    await http_client.aclose()
                    logger.info("   🔒 HTTP client closed.")
        
        # Create pydantic eval cases from DABench tasks
        cases = []
        for i, task in enumerate(tasks):
            
            case = Case(
                name=task['task_id'],
                inputs={'question': task['question'], 'constraints': task['constraints'], 'format': task['format'], 'file_name': task['file_name'], 'case_name': task['task_id']},  # Add case_name to inputs
                expected_output=task['correct_answer'],
                metadata={
                    'level': task['level'],
                    'task_id': task['task_id'],
                    'concepts': task['concepts'],
                }
            )
            cases.append(case)
        
        # Create evaluators
        evaluators = []
        
        # Add LLM Judge evaluator

        
        evaluators.append(
            LLMJudge(
                rubric="""
                A response is considered correct if and only if:
                1. The value(s) in the response match those in "expected_output".
                2. The response strictly follows the "format" specification.
                Both value correctness and format compliance are required for a pass.
                """
                include_input=True,
                include_expected_output=True,
                model=self.pydantic_ai_model,
                model_settings=ModelSettings(temperature=0.0)  # Deterministic evaluation
            )
        )
        
        # Create dataset
        dataset = Dataset(
            cases=cases,
            evaluators=evaluators
        )
        
        logger.info(f"🔍 Running Pydantic Eval with {len(cases)} cases and {len(evaluators)} evaluators...")
        
        # Send evaluation start progress update  
        await self.broker.send_stream_event(task_id, {
            'kind': 'status-update',
            'task_id': task_id,
            'context_id': task_id,
            'status': {'state': 'working'},
            'final': False,
            'metadata': {
                'message': f'Starting evaluation of {len(cases)} tasks...',
                'progress': {'current': 0, 'total': len(cases), 'phase': 'evaluation'}
            }
        })
        
        # Create a wrapper function to track progress during evaluation
        evaluation_progress = {'completed': 0}
        
        async def evaluate_purple_agent_with_progress(task_input: dict) -> str:
            """Wrapper around the evaluation function that sends progress updates."""
            try:
                # Call the original evaluation function
                result = await evaluate_purple_agent(task_input)
                
                # Update progress
                evaluation_progress['completed'] += 1
                
                # Send progress update
                await self.broker.send_stream_event(task_id, {
                    'kind': 'status-update',
                    'task_id': task_id,
                    'context_id': task_id,
                    'status': {'state': 'working'},
                    'final': False,
                    'metadata': {
                        'message': f'Completed {evaluation_progress["completed"]}/{len(cases)} tasks',
                        'progress': {
                            'current': evaluation_progress['completed'], 
                            'total': len(cases), 
                            'phase': 'evaluation'
                        }
                    }
                })
                
                return result
            except Exception as e:
                evaluation_progress['completed'] += 1
                logger.error(f"❌ Task evaluation failed: {e}")
                return f"[Error: {str(e)}]"
        
        # Run evaluation with limited concurrency to avoid overwhelming the purple agent
        start_time = time.time()
        try:
            report = await dataset.evaluate(evaluate_purple_agent_with_progress, max_concurrency=1)
            duration = time.time() - start_time
            evaluation_time = start_time  # Store when evaluation started, not duration
            
            # Store duration immediately for use in artifacts
            self.duration = duration
            
            # Send final evaluation completion update
            await self.broker.send_stream_event(task_id, {
                'kind': 'status-update',
                'task_id': task_id,
                'context_id': task_id,
                'status': {'state': 'working'},
                'final': False,
                'metadata': {
                    'message': f'Evaluation completed! Processed {len(cases)} tasks in {duration:.2f}s',
                    'progress': {'current': len(cases), 'total': len(cases), 'phase': 'completed'},
                    'duration': duration
                }
            })
            
            logger.info(f"✅ Pydantic Eval completed in {duration:.2f} seconds")
            report.print(include_reasons=True, include_output=True, include_expected_output=True)
            
            # Save report to results folder
            await self._save_evaluation_report(report, evaluation_time, len(cases), self.completed_tasks, duration)
            
            # Return the report object directly
            return report
            
        except Exception as e:
            logger.error(f"❌ Pydantic Eval failed: {e}")
            logger.error(f"📍 Traceback: {traceback.format_exc()}")
            
            # Set duration to 0 for failed evaluations
            self.duration = 0
            
            return {
                'success': False,
                'error': str(e),
                'evaluation_time': start_time,  # Store start time even for failures
                'total_cases': len(cases)
            }
    
    async def _save_evaluation_report(self, report, evaluation_time: float, total_cases: int, completed_tasks: List[Dict[str, Any]] = None, duration: float = None) -> None:
        """Save the pydantic evaluation report to a results folder."""
        try:
            
            # Create results directory if it doesn't exist
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            results_dir = os.path.join(project_root, 'results')
            os.makedirs(results_dir, exist_ok=True)
            
            # Generate timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Get model names for both agents
            green_model = os.getenv("GREEN_AGENT_MODEL")
            purple_model = os.getenv("PURPLE_AGENT_MODEL")
            
            # Clean model names for filename (replace special characters)
            green_model_clean = green_model.replace(":", "_").replace("-", "_")
            purple_model_clean = purple_model.replace(":", "_").replace("-", "_")
            
            # Create filename with timestamp, model names, and case count
            filename = f"pydantic_eval_report_{timestamp}_{green_model_clean}_vs_{purple_model_clean}_{total_cases}cases.json"
            filepath = os.path.join(results_dir, filename)
            
            
            # Extract structured data from EvaluationReport using centralized function
            if hasattr(report, 'cases'):
                report_data = self._extract_evaluation_summary(report, completed_tasks)
            else:
                # Fallback for other report types
                if isinstance(report, dict):
                    report_data = report
                    report_data.setdefault('success_rate', 0.0)
                else:
                    report_data = {
                        'report_text': str(report),
                        'type': str(type(report).__name__),
                        'success_rate': 0.0
                    }
            
            # Simplify token usage aggregation
            token_usage_summary = {}
            if completed_tasks:
                token_usage_summary = self._aggregate_token_usage(completed_tasks)
            
            # Create final report structure
            evaluation_metadata = {
                "evaluation_start_time": evaluation_time,  # When the evaluation started
                "evaluation_duration_seconds": self.duration,  # How long it took
                "total_cases": total_cases,
                "timestamp": timestamp,
                "success_rate": report_data.get('success_rate', 0.0),
                "generated_by": "Green Agent - Pydantic Eval",
                "token_usage": token_usage_summary
            }
            
            full_report = {
                "metadata": evaluation_metadata,
                "report": report_data
            }
            
            # Save to JSON file with proper serialization
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(full_report, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"📄 Evaluation report saved to: {filepath}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to save evaluation report: {e}")
            # Don't raise the exception - just log it, so the evaluation can continue
    
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

    def _get_evaluation_metadata(self) -> Dict[str, Any]:
        """Get evaluation metadata including purple agent model, duration, and token usage."""
        metadata = {}
        
        purple_agent_model = os.getenv("PURPLE_AGENT_MODEL")
        if purple_agent_model:
            metadata['purple_agent_model'] = purple_agent_model
        
        evaluation_duration = getattr(self, 'duration', None)
        if evaluation_duration is not None:
            metadata['evaluation_duration_seconds'] = evaluation_duration
        
        # Add aggregated token usage
        if hasattr(self, 'completed_tasks'):
            token_usage_summary = self._aggregate_token_usage(self.completed_tasks)
            metadata['token_usage'] = token_usage_summary
            logger.info(f"🪙 Adding token usage to metadata: {token_usage_summary}")
            
        return metadata

    def _process_report_cases(self, report, completed_tasks: List[Dict[str, Any]]) -> tuple:
        """Process report cases and return successful/failed cases lists."""
        successful_cases = []
        failed_cases = []
        
        # Process evaluation cases
        for case in report.cases:
            case_passed = self._check_case_success(case)
            case_data = {
                'name': case.name,
                'inputs': case.inputs,
                'expected_output': case.expected_output,
                'output': case.output,
                'metadata': case.metadata,
                'passed': case_passed
            }
            
            # Add token usage if available
            if completed_tasks:
                for task in completed_tasks:
                    if case.name == task.get('case_name'):
                        task_token_usage = self._extract_token_usage(task)
                        if task_token_usage:
                            case_data['token_usage'] = task_token_usage
                        break
            
            if case_passed:
                successful_cases.append(case_data)
            else:
                failed_cases.append(case_data)
        
        # Process execution failures
        for failure in getattr(report, 'failures', []):
            failure_data = {
                'name': failure.name,
                'inputs': failure.inputs,
                'expected_output': failure.expected_output,
                'error_message': failure.error_message,
                'error_stacktrace': failure.error_stacktrace,
                'metadata': failure.metadata,
                'passed': False
            }
            failed_cases.append(failure_data)
            
        return successful_cases, failed_cases

    def _generate_formatted_summary(self, report) -> List[str]:
        """Generate formatted summary from report."""
        if hasattr(report, 'render'):
            try:
                formatted_table = report.render(
                    include_output=True,
                    include_expected_output=True,
                    include_reasons=True
                )
                return formatted_table.split('\n')
            except Exception:
                return str(report).split('\n')
        else:
            return str(report).split('\n')

    def _extract_evaluation_summary(self, report, completed_tasks: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract standardized evaluation summary from report for both artifacts and JSON saving."""
        # Get evaluation metadata
        metadata = self._get_evaluation_metadata()
        
        # Base summary structure
        base_summary = {
            'name': getattr(report, 'name', 'Evaluation Report'),
            'total_cases': 0,
            'successful_cases': 0,
            'failed_cases': 0,
            'success_rate': 0.0,
            'cases': [],
            'failures': [],
            'formatted_summary': str(report).split('\n') if report else []
        }
        base_summary.update(metadata)
        
        # Handle reports without cases
        if not hasattr(report, 'cases'):
            return base_summary
        
        # Process cases
        successful_cases, failed_cases = self._process_report_cases(report, completed_tasks)
        
        # Calculate statistics
        total_cases = len(successful_cases) + len(failed_cases)
        success_count = len(successful_cases)
        success_rate = success_count / total_cases if total_cases > 0 else 0
        
        # Build final summary
        summary = {
            'name': getattr(report, 'name', 'evaluate_purple_agent'),
            'total_cases': total_cases,
            'successful_cases': success_count,
            'failed_cases': len(failed_cases),
            'success_rate': success_rate,
            'cases': successful_cases + failed_cases,
            'failures': [],  # Keep empty for backwards compatibility
            'experiment_metadata': getattr(report, 'experiment_metadata', None),
            'formatted_summary': self._generate_formatted_summary(report)
        }
        summary.update(metadata)
        
        return summary

    def _check_case_success(self, case) -> bool:
        """Check if a case passed based on evaluation results using proper Pydantic AI API."""
        # Use the correct Pydantic AI API structure
        if hasattr(case, 'assertions') and case.assertions:
            # case.assertions is a dict where values have .value and .reason attributes
            for assertion_name, result in case.assertions.items():
                if hasattr(result, 'value') and result.value is True:
                    return True
            # If we have assertions but none are True, it failed
            return False
        
        # Fallback: check if output matches expected (simple string comparison)
        if hasattr(case, 'output') and hasattr(case, 'expected_output'):
            return str(case.output).strip() == str(case.expected_output).strip()
        
        # If we can't determine success, assume failure to be conservative
        return False

    def _extract_token_usage(self, task_result: Dict[str, Any]) -> Optional[Dict[str, int]]:
        """Extract token usage information from task result."""
        # First check if token_usage is directly on the task (from streaming)
        if 'token_usage' in task_result:
            return task_result['token_usage']
        
        # Fallback: check in artifacts for compatibility
        if 'artifacts' in task_result:
            for artifact in task_result['artifacts']:
                for part in artifact.get('parts', []):
                    if part.get('kind') == 'data':
                        data = part.get('data', {})
                        if isinstance(data, dict) and 'token_usage' in data:
                            return data['token_usage']
        return None

    def _aggregate_token_usage(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate token usage across all completed tasks."""
        total_usage = {
            'total_requests': 0,
            'total_request_tokens': 0,
            'total_response_tokens': 0,
            'total_tokens': 0,
            'tasks_with_usage': 0,
            'average_tokens_per_task': 0
        }
        
        usage_data = []
        for task in tasks:
            token_usage = self._extract_token_usage(task)
            if token_usage:
                usage_data.append(token_usage)
                total_usage['total_requests'] += token_usage.get('requests', 0)
                total_usage['total_request_tokens'] += token_usage.get('request_tokens', 0)
                total_usage['total_response_tokens'] += token_usage.get('response_tokens', 0)
                total_usage['total_tokens'] += token_usage.get('total_tokens', 0)
                total_usage['tasks_with_usage'] += 1
        
        # Calculate averages
        if total_usage['tasks_with_usage'] > 0:
            total_usage['average_tokens_per_task'] = total_usage['total_tokens'] / total_usage['tasks_with_usage']
            total_usage['average_request_tokens_per_task'] = total_usage['total_request_tokens'] / total_usage['tasks_with_usage']
            total_usage['average_response_tokens_per_task'] = total_usage['total_response_tokens'] / total_usage['tasks_with_usage']
        
        return total_usage
    
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
        # Use centralized evaluation summary extraction
        if hasattr(report, 'cases'):
            report_data = self._extract_evaluation_summary(report, getattr(self, 'completed_tasks', []))
        else:
            # Fallback for other report types
            report_data = {"report": str(report), "success_rate": 0.0}
        
        # Create single artifact with the processed report
        return [Artifact(
            artifact_id=str(uuid.uuid4()),
            name="pydantic_eval_report",
            description="Pydantic Eval assessment report with correct success rate",
            parts=[DataPart(data=report_data, kind='data')]
        )]
        
    def _extract_text_from_parts(self, parts: List[Any]) -> str:
        """Extract text from message parts."""
        text_parts = []
        for part in parts:
            if part['kind'] == 'text':
                text_parts.append(part['text'])
        return ' '.join(text_parts)


def create_green_agent(card_url: str = None) -> FastA2A:
    """Create the green agent with Pydantic Eval."""
    
    # Initialize storage and broker
    storage = InMemoryStorage[Context]()
    broker = InMemoryBroker()
    
    # Define skills
    evaluation_skill = Skill(
        id="pydantic-eval-assessment",
        name="Pydantic Eval Assessment",
        description="Evaluates A2A agents using benchmark tasks with Pydantic Eval framework and LLM as judge",
        tags=["evaluation", "pydantic-eval", "llm-judge", "benchmark", "scoring"],
        examples=[
            "Evaluate an A2A agent on benchmark tasks using Pydantic Eval",
            "Run benchmark assessment with LLM as judge and return structured report"
        ],
        input_modes=["application/json"],
        output_modes=["application/json"]
    )
    
    # Agent provider - configurable via environment
    provider = AgentProvider(
        organization=os.getenv("AGENT_ORGANIZATION", "Local"),
        url=os.getenv("AGENT_PROVIDER_URL", card_url or f"http://localhost:{DEFAULT_PORT}")
    )
    
    # Create worker
    worker = GreenWorker(broker=broker, storage=storage)
    
    # Add lifespan to start worker (same pattern as purple agent)
    @asynccontextmanager
    async def lifespan(app):
        logger.info("🚀 Starting Green Agent worker...")
        try:
            async with app.task_manager:
                async with worker.run():
                    logger.info("✅ Green Agent worker started successfully")
                    yield
        except Exception as e:
            logger.error(f"❌ Failed to start Green Agent worker: {e}")
            logger.error(f"📍 Traceback: {traceback.format_exc()}")
            raise
        finally:
            logger.info("🛑 Green Agent worker stopped")
    
    # Create app with lifespan - use the card_url parameter
    app = FastA2A(
        storage=storage,
        broker=broker,
        name="Green Agent (Pydantic Eval)",
        description="A2A-compatible green agent that evaluates other agents using benchmark tasks with Pydantic Eval framework",
        url=card_url or f"http://localhost:{DEFAULT_PORT}",  # Use the provided card_url
        version="1.0.0",
        provider=provider,
        skills=[evaluation_skill],
        streaming=True,
        lifespan=lifespan
    )
    
    return app


def main():
    """Main entry point for the green agent."""
    
    # Parse command line arguments for AgentBeats compatibility
    parser = argparse.ArgumentParser(description="Green Agent (Evaluator)")
    parser.add_argument("--host", default="0.0.0.0", help="Host address to bind to")
    parser.add_argument("--port", type=int, default=9009, help="Port to listen on")
    parser.add_argument("--card-url", help="URL to advertise in the agent card (optional)")
    args = parser.parse_args()
    
    host = args.host
    port = args.port
    card_url = args.card_url or f"http://{host}:{port}"
    
    # Set global card URL
    global CARD_URL
    CARD_URL = card_url

    logger.info("🟢 Starting Green Agent with Pydantic Eval...")
    logger.info(f"📋 Agent Card: {card_url}/.well-known/agent-card.json")
    logger.info(f"🔗 A2A Endpoint: {card_url}/")
    logger.info("🎯 Evaluation Framework: Pydantic Eval with LLM as Judge")
    logger.info(f"🌐 Binding to {host}:{port}")

    # Create the app directly instead of using a factory
    app = create_green_agent(card_url)

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()