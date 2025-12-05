#!/usr/bin/env python3
"""
Send evaluation request directly to green agent.
"""
import sys
import os
import json
import asyncio
import httpx

# Add src to path for imports
sys.path.append('src')
sys.path.append('.')

from fasta2a.client import A2AClient
from fasta2a.schema import Message, TextPart, DataPart
from src.data_loader import get_sample_tasks, load_dabench_tasks

async def send_evaluation_request(green_url="http://localhost:8000", purple_url="http://localhost:8001", num_tasks=3, monitor=False):
    """Send evaluation request to green agent."""
    try:
        if num_tasks == 0:
            print("🔍 Loading full DABench dataset...")
            from src.data_loader import load_dabench_tasks
            tasks = load_dabench_tasks()
            print(f"📊 Loaded {len(tasks)} tasks from full DABench dataset")
        else:
            print(f"🔍 Loading {num_tasks} sample tasks...")
            tasks = get_sample_tasks(num_tasks)
            print(f"📊 Loaded {len(tasks)} sample tasks")
        
        if not tasks:
            print("❌ No tasks loaded! Check data files.")
            return False
        
        print(f"📦 Creating evaluation request...")
        eval_request = {
            'purple_agent_url': purple_url,
            'tasks': tasks
        }
        
        message = Message(
            role='user',
            parts=[
                TextPart(
                    text='Please evaluate the purple agent using the provided tasks.',
                    kind='text'
                ),
                DataPart(
                    data=eval_request,
                    kind='data'
                )
            ],
            kind='message',
            message_id='eval_001'
        )
        
        print(f"📤 Sending evaluation request to green agent at {green_url}...")
        
        # Create timeout configuration like launcher.py
        timeout = httpx.Timeout(
            connect=30.0,   
            read=120.0,     
            write=30.0,     
            pool=30.0       
        )
        
        async with httpx.AsyncClient(timeout=timeout) as http_client:
            client = A2AClient(base_url=green_url, http_client=http_client)
            response = await client.send_message(message)
            print("✅ Evaluation request sent successfully!")
                        
            if monitor:
                print("🔍 Monitoring evaluation progress... (Press Ctrl+C to stop)")
                
                if 'result' in response:
                    task_id = response['result']['id']
                    print(f"📋 Task ID: {task_id}")
                    print("📝 Real-time status checking every 30 seconds")
                    
                    # Monitor with proper task status checking
                    max_monitor_time = 5400  # 90 minutes max monitoring 
                    check_interval = 30      # Check every 30 seconds
                    elapsed_time = 0
                    
                    try:
                        while elapsed_time < max_monitor_time:
                            await asyncio.sleep(check_interval)
                            elapsed_time += check_interval
                            
                            # Check task status
                            try:
                                task_response = await client.get_task(task_id)
                                
                                if 'result' in task_response:
                                    task_data = task_response['result']
                                    task_status = task_data.get('status', {}).get('state', 'unknown')
                                    
                                    print(f"📊 Task status: {task_status} (elapsed: {elapsed_time}s)")
                                    
                                    if task_status == 'completed':
                                        print("🎉 Evaluation completed successfully!")
                                        print("📊 Check results/ directory for evaluation report")
                                        print(f"📋 Task response: {task_data}")
                                        return True
                                    elif task_status == 'failed':
                                        print(f"❌ Evaluation failed")
                                        return False
                                    elif task_status in ['working', 'submitted']:
                                        # Show progress info if available
                                        if 'new_messages' in task_data and task_data['new_messages']:
                                            latest_msg = task_data['new_messages'][-1]
                                            if 'parts' in latest_msg:
                                                for part in latest_msg['parts']:
                                                    if part.get('kind') == 'text':
                                                        msg_text = part.get('text', '')[:100]
                                                        print(f"💭 Latest: {msg_text}...")
                                        continue  # Keep monitoring
                                    else:
                                        print(f"⚠️ Unknown task status: {task_status}")
                                        continue
                                else:
                                    print(f"❌ Failed to get task status: {task_response}")
                                    continue
                                    
                            except httpx.ReadTimeout:
                                print(f"⏰ Status check timeout (elapsed: {elapsed_time}s) - continuing...")
                                continue
                            except Exception as e:
                                print(f"⚠️ Error checking status: {e}")
                                continue
                        
                        print("⏰ Monitoring timeout reached. Evaluation may still be running.")
                        print("📊 Check green agent logs or results/ directory for status.")
                        
                    except KeyboardInterrupt:
                        print("\n🛑 Monitoring stopped. Evaluation continues in background.")
                        if 'result' in response:
                            print(f"📋 Task ID: {response['result']['id']} (use this to check status manually)")
                        return True
                else:
                    print(f"⚠️ No task ID in response - cannot monitor: {response}")
                    return True
            else:
                if 'result' in response:
                    task_id = response['result']['id']
                    print(f"📍 Evaluation started! Task ID: {task_id}")
                    print("📊 Check green agent logs for progress or use --monitor flag next time.")
                else:
                    print(f"⚠️ Unexpected response format: {response}")
                    return False
            
            return True
            
    except Exception as e:
        print(f"❌ Error sending evaluation request: {e}")
        return False

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Send evaluation request to green agent')
    parser.add_argument('--green-url', default='http://localhost:8000', help='Green agent URL')
    parser.add_argument('--purple-url', default='http://localhost:8001', help='Purple agent URL')
    parser.add_argument('--tasks', type=int, default=3, help='Number of tasks to evaluate (0 = full dataset)')
    parser.add_argument('--monitor', action='store_true', help='Keep monitoring evaluation progress (keeps terminal busy)')
    
    args = parser.parse_args()
    
    success = asyncio.run(send_evaluation_request(args.green_url, args.purple_url, args.tasks, args.monitor))
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()