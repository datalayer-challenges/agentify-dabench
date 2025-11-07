"""
DABSTEP Data Loader

Loads DABSTEP benchmark tasks from the data directory.
"""

import json
from typing import List, Dict, Any
from pathlib import Path


def load_dabstep_tasks(sample_mode: bool = True, data_dir: str = None) -> List[Dict[str, Any]]:
    """
    Load DABSTEP tasks from the data directory.
    
    Args:
        sample_mode: If True, load dev.jsonl (10 tasks), if False load all.jsonl (450 tasks)
        data_dir: Path to data directory (defaults to project root/data)
    
    Returns:
        List of task dictionaries
    """
    if data_dir is None:
        # Default to data directory relative to this file
        current_dir = Path(__file__).parent.parent
        data_dir = current_dir / "data" / "tasks"
    else:
        data_dir = Path(data_dir)
    
    # Choose dataset based on sample mode
    filename = "dev.jsonl" if sample_mode else "all.jsonl"
    file_path = data_dir / filename
    
    if not file_path.exists():
        raise FileNotFoundError(f"DABSTEP task file not found: {file_path}")
    
    tasks = []
    print(f"📂 Loading DABSTEP tasks from: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                task = json.loads(line)
                
                # Standardize task format
                standardized_task = {
                    'task_id': str(task['task_id']),
                    'question': task['question'],
                    'correct_answer': task['answer'],
                    'level': task.get('level', 'unknown'),
                    'guidelines': task.get('guidelines', '')
                }
                
                tasks.append(standardized_task)
                
            except json.JSONDecodeError as e:
                print(f"⚠️  Error parsing line {line_num}: {e}")
                continue
            except KeyError as e:
                print(f"⚠️  Missing required field {e} in line {line_num}")
                continue
    
    dataset_type = "sample (dev)" if sample_mode else "full (all)"
    print(f"✅ Loaded {len(tasks)} DABSTEP tasks from {dataset_type} dataset")
    
    return tasks


def get_sample_tasks(num_tasks: int = 3) -> List[Dict[str, Any]]:
    """
    Get a small sample of tasks for quick testing.
    
    Args:
        num_tasks: Number of tasks to return
    
    Returns:
        List of sample task dictionaries
    """
    try:
        tasks = load_dabstep_tasks(sample_mode=True)
        return tasks[:num_tasks]
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"⚠️  Could not load DABSTEP tasks: {e}")
        # Fallback to hardcoded sample
        return [
            {
                "task_id": "math_basic_1",
                "question": "What is 15 + 27?",
                "correct_answer": "42",
                "level": "basic",
                "guidelines": "Provide the numerical answer."
            },
            {
                "task_id": "math_basic_2", 
                "question": "Calculate 8 * 7",
                "correct_answer": "56",
                "level": "basic",
                "guidelines": "Provide the numerical answer."
            },
            {
                "task_id": "comparison_1",
                "question": "Is 25 greater than 30?",
                "correct_answer": "no",
                "level": "basic",
                "guidelines": "Answer with yes or no."
            }
        ]


def validate_task_format(task: Dict[str, Any]) -> bool:
    """
    Validate that a task has the required fields.
    
    Args:
        task: Task dictionary to validate
    
    Returns:
        True if valid, False otherwise
    """
    required_fields = ['task_id', 'question', 'correct_answer']
    return all(field in task for field in required_fields)


def get_tasks_by_level(tasks: List[Dict[str, Any]], level: str) -> List[Dict[str, Any]]:
    """
    Filter tasks by difficulty level.
    
    Args:
        tasks: List of tasks
        level: Difficulty level ('easy', 'medium', 'hard')
    
    Returns:
        Filtered list of tasks
    """
    return [task for task in tasks if task.get('level', '').lower() == level.lower()]


def get_task_statistics(tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Get statistics about the task dataset.
    
    Args:
        tasks: List of tasks
    
    Returns:
        Dictionary with statistics
    """
    if not tasks:
        return {'total': 0, 'by_level': {}}
    
    levels = {}
    for task in tasks:
        level = task.get('level', 'unknown')
        levels[level] = levels.get(level, 0) + 1
    
    return {
        'total': len(tasks),
        'by_level': levels,
        'task_ids': [task['task_id'] for task in tasks[:5]]  # First 5 task IDs
    }


if __name__ == "__main__":
    # Test the data loader
    print("🧪 Testing DABSTEP data loader...")
    
    # Test sample mode
    sample_tasks = load_dabstep_tasks(sample_mode=True)
    print(f"📊 Sample dataset: {get_task_statistics(sample_tasks)}")
    
    # Test full mode
    try:
        full_tasks = load_dabstep_tasks(sample_mode=False)
        print(f"📊 Full dataset: {get_task_statistics(full_tasks)}")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"⚠️  Could not load full dataset: {e}")
    
    # Test quick sample
    quick_sample = get_sample_tasks(2)
    print(f"🎯 Quick sample (2 tasks): {[t['task_id'] for t in quick_sample]}")