#!/usr/bin/env python3
"""
AgentBeats Launcher

This script launches both the green agent (evaluator) and purple agent (test subject)
and provides a simple interface to run evaluations.

Usage:
    python launcher.py                    # Start both agents
    python launcher.py --green-only       # Start only green agent
    python launcher.py --purple-only       # Start only purple agent
    python launcher.py --evaluate         # Start agents and run evaluation
"""

import os
import sys
import json
import asyncio
import argparse
import subprocess
import time
import signal
from typing import Optional, List, Dict, Any
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("🔧 Loaded environment variables from .env")
except ImportError:
    print("⚠️  python-dotenv not installed, skipping .env file loading")

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

try:
    from fasta2a.client import A2AClient
    from fasta2a.schema import Message, TextPart, DataPart
except ImportError:
    print("❌ fasta2a not installed. Please install with: pip install fasta2a")
    sys.exit(1)


class Launcher:
    """Manages launching and coordinating agents."""
    
    def __init__(self, log_dir: Path):
        self.green_process: Optional[subprocess.Popen] = None
        self.purple_process: Optional[subprocess.Popen] = None
        self.green_port = 9009
        self.purple_port = 9019
        self.jupyter_port = 8888  # For reference only - purple agent manages its own Jupyter
        self.jupyter_token: Optional[str] = None
        self.log_dir = log_dir
        self.green_log_handle: Optional[Any] = None
        self.purple_log_handle: Optional[Any] = None
    
    def start_green_agent(self) -> bool:
        """Start the green agent (evaluator)."""
        try:
            print("🟢 Starting Green Agent (Evaluator)...")
            
            green_dir = project_root / "src" / "green_agent"
            if not green_dir.exists():
                print(f"❌ Green agent directory not found: {green_dir}")
                return False
            
            # Log to a file inside the timestamped log directory
            green_log_path = self.log_dir / "green_agent.log"
            print(f"📝 Green agent logs will be written to: {green_log_path}")
            self.green_log_handle = open(green_log_path, "w")

            self.green_process = subprocess.Popen(
                [sys.executable, "agent.py"],
                cwd=green_dir,
                stdout=self.green_log_handle,
                stderr=self.green_log_handle
            )
            
            # Wait for startup
            await_time = 5
            print(f"\n⏳ Waiting {await_time}s for green agent startup...")
            time.sleep(await_time)
            
            # Check if process is still running
            if self.green_process.poll() is None:
                print(f"✅ Green agent started on port {self.green_port}")
                print("─" * 50)
                return True
            else:
                print(f"❌ Green agent failed to start (process exited)")
                return False
                
        except Exception as e:
            print(f"❌ Error starting green agent: {e}")
            return False
    
    def start_purple_agent(self) -> bool:
        """Start the purple agent (test subject)."""
        try:
            print("🟣 Starting Purple Agent (Test Subject)...")
            
            purple_dir = project_root / "src" / "purple_agent"
            if not purple_dir.exists():
                print(f"❌ Purple agent directory not found: {purple_dir}")
                return False
            
            # Log to a file inside the timestamped log directory
            purple_log_path = self.log_dir / "purple_agent.log"
            print(f"📝 Purple agent logs will be written to: {purple_log_path}")
            self.purple_log_handle = open(purple_log_path, "w")

            self.purple_process = subprocess.Popen(
                [sys.executable, "agent.py"],
                cwd=purple_dir,
                stdout=self.purple_log_handle,
                stderr=self.purple_log_handle
            )
            
            # Wait for startup
            await_time = 5
            print(f"\n⏳ Waiting {await_time}s for purple agent startup...")
            time.sleep(await_time)
            
            # Check if process is still running
            if self.purple_process.poll() is None:
                print(f"✅ Purple agent started on port {self.purple_port}")
                print("─" * 50)
                return True
            else:
                print(f"❌ Purple agent failed to start (process exited)")
                return False
                
        except Exception as e:
            print(f"❌ Error starting purple agent: {e}")
            return False

    def stop_agents(self):
        """Stop all agents and services."""
        if self.green_process and self.green_process.poll() is None:
            print("🟢 Stopping green agent...")
            self.green_process.terminate()
            try:
                self.green_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.green_process.kill()
        
        if self.purple_process and self.purple_process.poll() is None:
            print("🟣 Stopping purple agent (and embedded Jupyter MCP)...")
            self.purple_process.terminate()
            try:
                self.purple_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.purple_process.kill()

        # Close log file handles
        if self.green_log_handle:
            self.green_log_handle.close()
        if self.purple_log_handle:
            self.purple_log_handle.close()
    
    async def check_agent_health(self, url: str, name: str) -> bool:
        """Check if agent is healthy and responding."""
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{url}/.well-known/agent-card.json", timeout=5.0)
                if response.status_code == 200:
                    agent_card = response.json()
                    print(f"✅ {name} is healthy: {agent_card.get('name', 'Unknown')}")
                    return True
                else:
                    print(f"❌ {name} health check failed: {response.status_code}")
                    return False
        except ImportError:
            print(f"⚠️  httpx not available, using curl for {name}")
            return await self._check_health_with_curl(url, name)
        except Exception as e:
            print(f"❌ {name} health check error: {e}")
            return False
    
    async def _check_health_with_curl(self, url: str, name: str) -> bool:
        """Fallback health check using curl."""
        try:
            import subprocess
            result = subprocess.run(
                ['curl', '-s', f"{url}/.well-known/agent-card.json"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                print(f"✅ {name} is healthy (via curl)")
                return True
            else:
                print(f"❌ {name} health check failed (curl)")
                return False
        except Exception as e:
            print(f"❌ {name} curl health check error: {e}")
            return False
    
    async def run_sample_evaluation(self, sample_mode: str = "quick", quick_sample_size: int = 3, use_agentbeats_format: bool = False) -> bool:
        """Run an evaluation with real data."""
        try:
            print(f"\n🎯 Running Evaluation ({sample_mode} mode)...")
            
            # Check agent health
            green_url = f"http://localhost:{self.green_port}"
            purple_url = f"http://localhost:{self.purple_port}"
            
            green_healthy = await self.check_agent_health(green_url, "Green Agent")
            purple_healthy = await self.check_agent_health(purple_url, "Purple Agent")
            
            if not (green_healthy and purple_healthy):
                print("❌ Agents are not healthy, cannot run evaluation")
                return False
            
            # Load real DABench tasks
            try:
                # Add src to path for importing data_loader
                import sys
                src_path = project_root / "src"
                if str(src_path) not in sys.path:
                    sys.path.append(str(src_path))
                
                from data_loader import load_dabench_tasks, get_sample_tasks, get_task_statistics
                
                # Get tasks based on mode - use DABench only
                if sample_mode == "quick":
                    sample_tasks = get_sample_tasks(quick_sample_size)
                    print(f"📋 Using {len(sample_tasks)} quick sample DABench tasks")
                elif sample_mode == "dev":
                    sample_tasks = load_dabench_tasks()  # Load all DABench dev tasks
                    print(f"📋 Using {len(sample_tasks)} DABench development tasks")
                elif sample_mode == "full":
                    sample_tasks = load_dabench_tasks()  # DABench only has dev set currently
                    print(f"📋 Using {len(sample_tasks)} DABench tasks")
                else:
                    raise ValueError(f"Invalid sample mode: {sample_mode}. Use: quick, dev, full")
                
                # Show dataset statistics
                stats = get_task_statistics(sample_tasks)
                print(f"📊 Dataset stats: {stats['total']} tasks, levels: {stats['by_level']}")
                
            except (ImportError, FileNotFoundError) as e:
                print(f"⚠️  Could not load data: {e}")
                sample_tasks = []
            
            # Create evaluation request
            if use_agentbeats_format:
                print(f"🤖 Creating AgentBeats assessment request...")
                eval_request = {
                    "participants": {
                        "data_analyst": purple_url
                    },
                    "config": {
                        "num_tasks": len(sample_tasks) if sample_tasks else quick_sample_size,
                        "quick_sample": sample_mode == "quick",
                        "dataset_path": "data-dabench/"
                    }
                }
                message_text = "AgentBeats assessment request for DABench evaluation."
            else:
                print(f"📋 Creating legacy evaluation request...")
                eval_request = {
                    "purple_agent_url": purple_url,
                    "tasks": sample_tasks
                }
                message_text = "Please evaluate the purple agent using the provided tasks."
            
            # Create message for green agent
            message = Message(
                role='user',
                parts=[
                    TextPart(
                        text=message_text,
                        kind='text'
                    ),
                    DataPart(data=eval_request, kind='data')
                ],
                kind='message',
                message_id="sample_eval_001"
            )
            
            # Send evaluation request to green agent
            print(f"📤 Sending evaluation request to green agent...")
            print(f"📋 Request contains {len(sample_tasks)} tasks")
            
            # Create A2A client with reasonable timeout handling
            import httpx
            timeout = httpx.Timeout(
                connect=30.0,   # Connection timeout
                read=120.0,     # Read timeout (2 minutes should be plenty)
                write=30.0,     # Write timeout
                pool=30.0       # Pool timeout
            )
            
            async with httpx.AsyncClient(timeout=timeout) as http_client:
                client = A2AClient(base_url=green_url, http_client=http_client)
                
                try:
                    print("⏳ Sending evaluation request (timeout: 120s)...")
                    print(f"📋 Message details: type={type(message)}")
                    print(f"📋 Message content: {json.dumps(message.model_dump() if hasattr(message, 'model_dump') else str(message), indent=2)[:500]}...")
                    response = await client.send_message(message)
                    print(f"📬 Response received: {response}")
                except Exception as e:
                    print(f"❌ Send message error: {type(e).__name__}: {str(e)}")
                    import traceback
                    print(f"📍 Full traceback:\n{traceback.format_exc()}")
                    return False
                
                if 'result' in response:
                    task_id = str(response['result']['id'])
                    print(f"📋 Evaluation task created: {task_id}")
                    
                    # Wait for actual completion instead of arbitrary timeout
                    print("⏳ Waiting for green agent to complete evaluation...")
                    
                    task_response = None
                    max_wait_time = 43200  # 12 hours
                    start_time = time.time()
                    
                    while True:
                        # Safety check for timeout
                        elapsed = time.time() - start_time
                        if elapsed > max_wait_time:
                            print(f"⚠️  Safety timeout reached ({max_wait_time}s), stopping wait")
                            break
                            
                        try:
                            # Check task status
                            task_response = await client.get_task(task_id=task_id)
                            
                            if 'result' in task_response:
                                final_task = task_response['result']
                                task_status = final_task.get('status', {})
                                task_state = task_status.get('state', 'unknown')
                                
                                if task_state == 'completed':
                                    print("✅ Green agent completed evaluation!")
                                    break
                                elif task_state in ['failed', 'cancelled', 'error']:
                                    print(f"❌ Task failed with state: {task_state}")
                                    break
                                else:
                                    # Task still in progress
                                    print(f"   🔄 Green agent working... (status: {task_state}, {elapsed:.0f}s elapsed)")
                                    await asyncio.sleep(15)  # Brief wait before next check
                            else:
                                print(f"⚠️  Could not get task status: {task_response}")
                                await asyncio.sleep(15)
                                
                        except Exception as e:
                            print(f"❌ Error checking task status: {e}")
                            # Still wait a bit and try again
                            await asyncio.sleep(15)
                    
                    # We already have the final task_response from the loop above
                    print(f"📋 Final task response: {task_response}")
                    
                    if 'result' in task_response:
                        final_task = task_response['result']
                        task_status = final_task.get('status', {})
                        task_state = task_status.get('state', 'unknown')
                        
                        print(f"📊 Task state: {task_state}")
                        
                        if task_state == 'completed':
                            print("✅ Evaluation completed successfully!")
                            
                            # Display results
                            if 'artifacts' in final_task:
                                for artifact in final_task['artifacts']:
                                    print(f"📄 Artifact: {artifact.get('name', 'unknown')}")
                                    if artifact.get('name') == 'evaluation_summary':
                                        for part in artifact.get('parts', []):
                                            if part.get('type') == 'data':
                                                summary = part.get('data', {})
                                                print(f"\n📊 Evaluation Results:")
                                                print(f"   Accuracy: {summary.get('accuracy', 0) * 100:.1f}%")
                                                print(f"   Correct: {summary.get('correct_tasks', 0)}/{summary.get('total_tasks', 0)}")
                                                print(f"   Time: {summary.get('time_used', 0):.2f}s")
                            
                            return True
                        elif task_state in ['working', 'submitted']:
                            print(f"⏳ Task still in progress: {task_state}")
                            print("💡 Try increasing the wait time or check task status later")
                            return False
                        else:
                            print(f"❌ Evaluation failed: {task_state}")
                            return False
                    else:
                        print(f"❌ Failed to get task results: {task_response.get('error', 'Unknown error')}")
                        return False
                else:
                    print(f"❌ Failed to send evaluation request: {response.get('error', 'Unknown error')}")
                    return False
                
        except Exception as e:
            print(f"❌ Sample evaluation failed: {e}")
            return False
    
    def run_interactive_mode(self):
        """Run interactive mode for custom evaluations."""
        print("\n🎮 Interactive Mode")
        print("Commands:")
        print("  eval [quick|dev|full] - Run evaluation (default: quick)")
        print("  agentbeats - Toggle AgentBeats format (default: legacy)")
        print("  status - Check agent status")
        print("  quit - Exit")
        
        use_agentbeats_format = False  # Default to legacy format
        
        while True:
            try:
                command_input = input("\n> ").strip()
                parts = command_input.split()
                command = parts[0].lower() if parts else ""
                
                if command == 'quit':
                    break
                elif command == 'eval':
                    # Parse eval mode
                    mode = parts[1] if len(parts) > 1 else "quick"
                    if mode not in ["quick", "dev", "full"]:
                        print("❌ Invalid mode. Use: quick, dev, or full")
                        continue
                    
                    sample_size = 3 if mode == "quick" else 10
                    format_info = "AgentBeats" if use_agentbeats_format else "legacy"
                    print(f"🚀 Running evaluation in {mode} mode with {format_info} format...")
                    asyncio.run(self.run_sample_evaluation(mode, sample_size, use_agentbeats_format))
                elif command == 'agentbeats':
                    use_agentbeats_format = not use_agentbeats_format
                    format_status = "AgentBeats" if use_agentbeats_format else "legacy"
                    print(f"🔄 Message format switched to: {format_status}")
                elif command == 'status':
                    asyncio.run(self._check_status())
                elif command == 'help':
                    print("Commands:")
                    print("  eval [quick|dev|full] - Run evaluation")
                    print("    quick: 3 sample tasks (fast)")
                    print("    dev: 10 development tasks")
                    print("    full: 450 full dataset tasks")
                    print(f"  agentbeats - Toggle format (current: {'AgentBeats' if use_agentbeats_format else 'legacy'})")
                    print("  status - Check agent health")
                    print("  quit - Exit")
                else:
                    print("Unknown command. Try 'help' for available commands.")
                    
            except KeyboardInterrupt:
                break
        
        print("\n👋 Goodbye!")
    
    async def _check_status(self):
        """Check status of both agents."""
        green_url = f"http://localhost:{self.green_port}"
        purple_url = f"http://localhost:{self.purple_port}"
        
        await self.check_agent_health(green_url, "Green Agent")
        await self.check_agent_health(purple_url, "Purple Agent")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AgentBeats Launcher - A2A compatible benchmark evaluation",
        epilog="""
Examples:
  python launcher.py --evaluate                    # Quick evaluation (3 tasks)
  python launcher.py --evaluate --sample-mode dev  # Dev evaluation (10 tasks)  
  python launcher.py --evaluate --full-dataset     # Full evaluation (450 tasks)
  python launcher.py --interactive                 # Interactive mode
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--green-only", action="store_true", help="Start only green agent")
    parser.add_argument("--purple-only", action="store_true", help="Start only purple agent")
    parser.add_argument("--evaluate", action="store_true", help="Start agents and run evaluation")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    # Dataset options  
    parser.add_argument("--full-dataset", action="store_true", 
                       help="Use full dataset")
    parser.add_argument("--quick-sample", type=int, default=3, metavar="N",
                       help="Number of tasks for quick testing (default: 3)")
    parser.add_argument("--sample-mode", choices=["quick", "dev", "full"], default="quick",
                       help="Dataset mode: quick (3 DABench), dev (DABench), full (DABench)")
    parser.add_argument("--agentbeats", action="store_true", 
                       help="Use AgentBeats format for evaluation requests")
    
    args = parser.parse_args()
    
    # --- Setup Logging Directory ---
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = project_root / "logs" / f"run_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    os.environ["AGENT_LOG_DIR"] = str(log_dir)
    print(f"🪵  Logging for this run will be stored in: {log_dir}")
    
    launcher = Launcher(log_dir=log_dir)
    
    def signal_handler(signum, frame):
        print("\n🛑 Shutting down...")
        launcher.stop_agents()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        print("🚀 AgentBeats Launcher")
        print("=" * 40)
        
        success = True
        
        if args.purple_only:
            # Purple agent now has embedded Jupyter MCP server
            success = launcher.start_purple_agent()
        elif args.green_only:
            success = launcher.start_green_agent()
        else:
            # Purple agent has embedded Jupyter MCP server, so just start both agents
            success = (launcher.start_purple_agent() and 
                      launcher.start_green_agent())
        
        if not success:
            print("❌ Failed to start agents")
            launcher.stop_agents()
            sys.exit(1)
        
        if args.evaluate:
            # Determine sample mode from arguments
            if args.full_dataset:
                sample_mode = "full"
            elif args.sample_mode == "dev":
                sample_mode = "dev"
            elif args.sample_mode == "quick":
                sample_mode = "quick"
            else:
                sample_mode = args.sample_mode
            
            print(f"\n⏳ Running evaluation in 5 seconds (mode: {sample_mode})...")
            time.sleep(5)
            eval_success = asyncio.run(launcher.run_sample_evaluation(sample_mode, args.quick_sample, args.agentbeats))
            if eval_success:
                print("\n✅ Evaluation completed successfully!")
            else:
                print("\n❌ Evaluation failed")
        elif args.interactive:
            launcher.run_interactive_mode()
        else:
            print("\n✅ All services started successfully!")
            print(f" Green Agent: http://localhost:{launcher.green_port}")
            print(f"🟣 Purple Agent: http://localhost:{launcher.purple_port} (with embedded Jupyter MCP)")
            print("\nPress Ctrl+C to stop all services")
            
            # Keep running until interrupted
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
    
    finally:
        launcher.stop_agents()


if __name__ == "__main__":
    main()