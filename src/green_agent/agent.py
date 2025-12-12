"""
Green Agent - A2A-compatible evaluator agent using Pydantic Eval.

This agent:
1. Receives evaluation requests containing tasks and target agent URL
2. Sends tasks to the purple agent (agent under test)
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


# Load environment variables from .env file
from dotenv import load_dotenv
# Load from project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
dotenv_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path)
print(f"🔧 Green Agent loaded environment variables from {dotenv_path}")

DEFAULT_PORT = 9009

# Import shared utils
try:
    from ..shared_utils import get_pydantic_ai_model, setup_logger
except ImportError:
    # Fallback for when running as script
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from shared_utils import get_pydantic_ai_model, setup_logger

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
        
        try:
            # Load context
            context = await self.storage.load_context(task['context_id']) or []
            context.extend(task.get('history', []))
            
            # Extract message content
            message = params['message']
            eval_request = self._extract_evaluation_request(message)
            
            if eval_request:
                # Handle evaluation request using Pydantic Eval
                logger.info("🔍 Processing evaluation request with Pydantic Eval...")
                
                # Update status to working and send initial status message
                await self.storage.update_task(task['id'], state='working')
                initial_message = Message(
                    role='agent',
                    parts=[TextPart(text="Starting evaluation with Pydantic Eval framework...", kind='text')],
                    kind='message',
                    message_id=str(uuid.uuid4())
                )
                await self.storage.update_task(
                    task['id'], 
                    new_messages=[initial_message]
                )
                
                report = await self._evaluate_agent_pydantic(eval_request, task['id'])
                response_message = self._create_response_message(report)
                artifacts = self._create_response_artifacts(report)
            else:
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
            logger.info(f"✅ Task {task['id']} completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Error in task {task['id']}: {e}")
            logger.error(f"📍 Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"📍 Traceback: {traceback.format_exc()}")
            
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
            # Fallback for when relative imports fail
            import sys
            import os
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
        
        # Send progress update
        progress_message = Message(
            role='agent',
            parts=[TextPart(text=f"Starting evaluation of {len(tasks)} tasks using Pydantic Eval framework", kind='text')],
            kind='message',
            message_id=str(uuid.uuid4())
        )
        await self.storage.update_task(task_id, new_messages=[progress_message])
        
        # Create timeout configuration for purple agent communication
        import httpx
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
                
                # Send progress update for this specific task
                task_progress_message = Message(
                    role='agent',
                    parts=[TextPart(text=f"Running task: {case_name} - {question[:50]}{'...' if len(question) > 50 else ''}", kind='text')],
                    kind='message',
                    message_id=str(uuid.uuid4())
                )
                await self.storage.update_task(task_id, new_messages=[task_progress_message])
                
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
                task_message = Message(
                    role='user',
                    parts=[TextPart(
                        text=task_prompt,
                        kind='text'
                    )],
                    kind='message',
                    message_id=str(uuid.uuid4())
                )
                logger.info(f"   📤 Sending task to purple agent: {question[:50]}{'...' if len(question) > 50 else ''}")
                
                # Send task to purple agent
                try:
                    response = await client.send_message(task_message)
                except httpx.ReadTimeout as e:
                    logger.warning(f"   ⏰ Timeout sending message after {timeout.read}s")
                    return "[Timeout during message sending]"
                except Exception as e:
                    logger.error(f"   ❌ Error sending message: {e}")
                    return f"[Error sending message: {e}]"
                                
                if 'result' in response:
                    # Get task result
                    agent_task = response['result']
                    task_id = agent_task['id']
                                        
                    # Wait for completion
                    max_wait_time = 1800  # 30 minutes (total time for task completion)
                    poll_interval = 10   # Check every 10 seconds
                    elapsed_time = 0
                    
                    logger.info(f"   ⏳ Waiting for purple agent to complete task...")
                    
                    while elapsed_time < max_wait_time:
                        await asyncio.sleep(poll_interval)
                        elapsed_time += poll_interval
                        
                        try:
                            task_response = await client.get_task(task_id)
                            
                            if 'result' in task_response:
                                final_task = task_response['result']
                                task_status = final_task.get('status', {}).get('state', 'unknown')
                                
                                logger.info(f"   📊 Task status: {task_status} (elapsed: {elapsed_time}s)")
                                
                                if task_status == 'completed':
                                    logger.info(f"   ✅ Purple agent completed task after {elapsed_time}s")
                                    break
                                elif task_status == 'failed':
                                    logger.warning(f"   ❌ Purple agent task failed after {elapsed_time}s")
                                    return f"[Task failed: {task_status}]"
                                elif task_status in ['working', 'submitted']:
                                    continue  # Keep waiting
                                else:
                                    logger.warning(f"   ⚠️ Unknown task status: {task_status}")
                                    break
                            else:
                                logger.error(f"   ❌ Failed to get purple agent task status: {task_response}")
                                return "[Failed to get task status]"
                        except httpx.ReadTimeout as e:
                            logger.warning(f"   ⏰ Status check timeout after {timeout.read}s (elapsed: {elapsed_time}s)")
                            logger.info(f"   💭 Purple agent may be overloaded - continuing to wait...")
                            continue  # Keep trying
                        except Exception as e:
                            logger.warning(f"   ⚠️ Error checking purple agent status: {e}")
                            continue
                    
                    if elapsed_time >= max_wait_time:
                        logger.warning(f"   ⏰ Purple agent task timed out after {max_wait_time}s")
                        return "[Task timed out]"
                    
                    # Get final task result and extract answer
                    try:
                        task_response = await client.get_task(task_id)
                    except httpx.ReadTimeout as e:
                        logger.warning(f"   ⏰ Final result timeout after {timeout.read}s")
                        return "[Final result retrieval timeout]"
                                        
                    if 'result' in task_response:
                        final_task = task_response['result']
                        
                        # Store completed task with case name for token usage aggregation
                        final_task_with_id = dict(final_task)
                        final_task_with_id['case_name'] = case_name  # Use the case name from inputs
                        self.completed_tasks.append(final_task_with_id)
                        
                        # Use the outer self reference
                        agent_answer = self._extract_agent_answer(final_task)
                        
                        if agent_answer:
                            logger.info(f"   💬 Purple agent answered: '{agent_answer[:100]}{'...' if len(agent_answer) > 100 else ''}'")
                            return agent_answer
                        else:
                            logger.warning(f"   ⚠️ Could not extract answer from purple agent response")
                            logger.warning(f"   🔍 Final task structure: {list(final_task.keys()) if isinstance(final_task, dict) else type(final_task)}")
                            return "[No answer extracted]"
                    else:
                        error = task_response.get('error', 'Unknown error')
                        logger.error(f"   ❌ Purple agent task failed: {error}")
                        return f"[Task failed: {error}]"
                else:
                    error = response.get('error', 'Unknown error')
                    logger.error(f"   ❌ Failed to send message to purple agent: {error}")
                    logger.error(f"   🔍 Full response: {response}")
                    return f"[Message send failed: {error}]"
                        
            except Exception as e:
                logger.error(f"   ❌ Exception during purple agent evaluation: {e}")
                logger.error(f"   🔍 Exception type: {type(e).__name__}")
                import traceback
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
        
        logger.info(f"🔍 Running Pydantic Eval with {len(cases)} cases and {len(evaluators)} evaluators...")
        
        # Send progress update before starting evaluation
        eval_start_message = Message(
            role='agent',
            parts=[TextPart(text=f"Running Pydantic Eval with {len(cases)} cases and {len(evaluators)} evaluators...", kind='text')],
            kind='message',
            message_id=str(uuid.uuid4())
        )
        await self.storage.update_task(task_id, new_messages=[eval_start_message])
        
        # Run evaluation with limited concurrency to avoid overwhelming the purple agent
        start_time = time.time()
        try:
            report = await dataset.evaluate(evaluate_purple_agent, max_concurrency=1)
            evaluation_time = time.time() - start_time
            
            logger.info(f"✅ Pydantic Eval completed in {evaluation_time:.2f} seconds")
            report.print(include_reasons=True, include_output=True, include_expected_output=True)
            
            # Save report to results folder
            await self._save_evaluation_report(report, evaluation_time, len(cases), self.completed_tasks)
            
            # Return the report object directly
            return report
            
        except Exception as e:
            logger.error(f"❌ Pydantic Eval failed: {e}")
            import traceback
            logger.error(f"📍 Traceback: {traceback.format_exc()}")
            
            return {
                'success': False,
                'error': str(e),
                'evaluation_time': time.time() - start_time,
                'total_cases': len(cases)
            }
    
    async def _save_evaluation_report(self, report, evaluation_time: float, total_cases: int, completed_tasks: List[Dict[str, Any]] = None) -> None:
        """Save the pydantic evaluation report to a results folder."""
        try:
            import os
            from datetime import datetime
            
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
            
            # Extract structured data from EvaluationReport using the proper API
            if hasattr(report, 'cases') and hasattr(report, 'failures'):
                # This is a proper EvaluationReport object
                report_data = {
                    'name': getattr(report, 'name', 'Unnamed Evaluation'),
                    'cases': [],
                    'failures': [],
                    'experiment_metadata': getattr(report, 'experiment_metadata', None),
                }
                
                # Process successful cases
                for case in report.cases:
                    case_data = {
                        'name': case.name,
                        'inputs': case.inputs,
                        'expected_output': case.expected_output,
                        'output': case.output,
                        'metadata': case.metadata,
                    }
                    
                    # Find the corresponding completed task for this case and add token usage
                    task_token_usage = None
                    if completed_tasks:
                        # Match by case name (which is the task_id)
                        for task in completed_tasks:
                            if case.name == task.get('case_name'):
                                task_token_usage = self._extract_token_usage(task)
                                break
                    
                    if task_token_usage:
                        case_data['token_usage'] = task_token_usage
                        
                    report_data['cases'].append(case_data)
                
                # Process failures
                for failure in report.failures:
                    failure_data = {
                        'name': failure.name,
                        'inputs': failure.inputs,
                        'expected_output': failure.expected_output,
                        'error_message': failure.error_message,
                        'error_stacktrace': failure.error_stacktrace,
                        'metadata': failure.metadata,
                    }
                    report_data['failures'].append(failure_data)
                
                # Calculate summary statistics
                total_cases_processed = len(report.cases) + len(report.failures)
                success_count = len(report.cases)
                success_rate = success_count / total_cases_processed if total_cases_processed > 0 else 0
                
                report_data.update({
                    'total_cases': total_cases_processed,
                    'successful_cases': success_count,
                    'failed_cases': len(report.failures),
                    'success_rate': success_rate,
                })
                
                # Add a nicely formatted table representation
                if hasattr(report, 'render'):
                    try:
                        # Use the render method for a clean, readable text table
                        formatted_table = report.render(
                            include_output=True,
                            include_expected_output=True,
                            include_reasons=True
                        )
                        # Split the table into lines for better JSON readability
                        report_data['formatted_summary'] = formatted_table.split('\n')
                    except Exception:
                        # Fallback if render fails
                        report_data['formatted_summary'] = str(report).split('\n')
                else:
                    report_data['formatted_summary'] = str(report).split('\n')
                
            elif hasattr(report, 'model_dump'):
                # Pydantic v2 style serialization
                report_data = report.model_dump()
                success_rate = getattr(report, 'success_rate', None)
            elif hasattr(report, 'dict'):
                # Pydantic v1 style serialization
                report_data = report.dict()
                success_rate = getattr(report, 'success_rate', None)
            elif isinstance(report, dict):
                # Already a dictionary (error case)
                report_data = report
                success_rate = report.get('success_rate')
            else:
                # Fallback - convert to string
                report_data = {
                    'report_text': str(report),
                    'type': str(type(report).__name__)
                }
                success_rate = None
            
            # Aggregate token usage from completed tasks
            token_usage_summary = {}
            if completed_tasks:
                token_usage_summary = self._aggregate_token_usage(completed_tasks)
                logger.info(f"📊 Token usage summary - Total: {token_usage_summary.get('total_tokens', 0)}, "
                           f"Tasks with usage: {token_usage_summary.get('tasks_with_usage', 0)}")
            
            # Add our custom metadata
            evaluation_metadata = {
                "evaluation_time_seconds": evaluation_time,
                "total_cases": total_cases,
                "timestamp": timestamp,
                "success_rate": report_data.get('success_rate', success_rate),
                "generated_by": "Green Agent - Pydantic Eval",
                "token_usage": token_usage_summary
            }
            
            # Combine metadata with report
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

    def _extract_token_usage(self, task_result: Dict[str, Any]) -> Optional[Dict[str, int]]:
        """Extract token usage information from task result artifacts."""
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
        artifacts = []
        
        # Calculate simple metrics from report
        total_cases = len(getattr(report, 'cases', [])) + len(getattr(report, 'failures', []))
        successful_cases = len(getattr(report, 'cases', []))
        metrics = {
            'total_cases': total_cases,
            'successful_cases': successful_cases,
            'failed_cases': len(getattr(report, 'failures', [])),
            'success_rate': successful_cases / total_cases if total_cases > 0 else 0
        }
        
        # Convert EvaluationReport to serializable format using proper API
        if hasattr(report, 'cases') and hasattr(report, 'failures'):
            # This is a proper EvaluationReport object - extract using documented API
            cases_data = []
            for case in getattr(report, 'cases', []):
                case_data = {
                    'name': case.name,
                    'inputs': case.inputs,
                    'expected_output': case.expected_output,
                    'output': case.output,
                }
                
                # Find the corresponding completed task for this case and add token usage
                task_token_usage = None
                if hasattr(self, 'completed_tasks') and self.completed_tasks:
                    # Match by case name (which is the task_id)
                    for task in self.completed_tasks:
                        if case.name == task.get('case_name'):
                            task_token_usage = self._extract_token_usage(task)
                            break
                
                if task_token_usage:
                    case_data['token_usage'] = task_token_usage
                    
                cases_data.append(case_data)
            
            report_data = {
                'name': getattr(report, 'name', 'Evaluation Report'),
                'total_cases': len(getattr(report, 'cases', [])) + len(getattr(report, 'failures', [])),
                'successful_cases': len(getattr(report, 'cases', [])),
                'failed_cases': len(getattr(report, 'failures', [])),
                'cases': cases_data,
                'failures': [
                    {
                        'name': failure.name,
                        'error_message': failure.error_message,
                        'inputs': failure.inputs,
                    }
                    for failure in getattr(report, 'failures', [])
                ],
                'metrics': metrics
            }
        elif hasattr(report, 'model_dump'):
            # Pydantic v2 style serialization
            report_data = report.model_dump()
        elif hasattr(report, 'dict'):
            # Pydantic v1 style serialization
            report_data = report.dict()
        elif isinstance(report, dict):
            # Already a dictionary (error case)
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
    import os
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
            import traceback
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
        lifespan=lifespan
    )
    
    return app


def main():
    """Main entry point for the green agent."""
    import argparse
    import uvicorn
    
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

    # Create a factory function that uses the card_url
    def create_app():
        return create_green_agent(card_url)

    uvicorn.run(
        create_app,
        host=host,
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()