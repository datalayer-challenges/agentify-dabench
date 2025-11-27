"""
DABench Data Loader

Loads DABench benchmark tasks from the data directory.
"""

import json
from typing import List, Dict, Any
from pathlib import Path


def load_dabench_tasks(data_dir: str = None) -> List[Dict[str, Any]]:
    """
    Load DABench tasks from the data-dabench directory.
    
    Args:
        data_dir: Path to data-dabench directory (defaults to project root/data-dabench)
    
    Returns:
        List of task dictionaries in standardized format
    """
    if data_dir is None:
        # Default to data-dabench directory relative to this file
        current_dir = Path(__file__).parent.parent
        data_dir = current_dir / "data-dabench"
    else:
        data_dir = Path(data_dir)
    
    questions_file = data_dir / "da-dev-questions.jsonl"
    labels_file = data_dir / "da-dev-labels.jsonl"
    
    if not questions_file.exists():
        raise FileNotFoundError(f"DABench questions file not found: {questions_file}")
    
    if not labels_file.exists():
        raise FileNotFoundError(f"DABench labels file not found: {labels_file}")
    
    print(f"📂 Loading DABench tasks from: {data_dir}")
    
    # Load questions
    questions = {}
    with open(questions_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                question = json.loads(line)
                questions[question['id']] = question
            except json.JSONDecodeError as e:
                print(f"⚠️  Error parsing questions line {line_num}: {e}")
                continue
    
    # Load labels
    labels = {}
    with open(labels_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                label = json.loads(line)
                labels[label['id']] = label
            except json.JSONDecodeError as e:
                print(f"⚠️  Error parsing labels line {line_num}: {e}")
                continue
    
    # Combine questions and labels into standardized format
    tasks = []
    for task_id, question_data in questions.items():
        if task_id not in labels:
            print(f"⚠️  No label found for task ID {task_id}, skipping")
            continue
        
        label_data = labels[task_id]
        
        # Extract expected answers from common_answers
        common_answers = label_data.get('common_answers', [])
        
        # Convert to standardized format
        standardized_task = {
            'task_id': str(task_id),
            'question': question_data['question'],
            'correct_answer': _format_dabench_answers(common_answers),
            'level': question_data.get('level', 'medium'),
            'file_name': question_data.get('file_name', ''),
            'concepts': question_data.get('concepts', []),
            'constraints': question_data.get('constraints', ''),
            'format': question_data.get('format', '')
        }
        
        tasks.append(standardized_task)
    
    print(f"✅ Loaded {len(tasks)} DABench tasks")
    
    return tasks


def _format_dabench_answers(common_answers: List[List[str]]) -> str:
    """
    Format DABench answers into a readable string for evaluation.
    
    Args:
        common_answers: List of [key, value] pairs
    
    Returns:
        Formatted answer string
    """
    if not common_answers:
        return "No answer provided"
    
    # Convert to dictionary for easier access
    answers_dict = {pair[0]: pair[1] for pair in common_answers}
    
    # If there's only one answer, return it directly
    if len(answers_dict) == 1:
        key, value = list(answers_dict.items())[0]
        return f"{key}: {value}"
    
    # For multiple answers, format them nicely
    formatted_parts = []
    for key, value in answers_dict.items():
        formatted_parts.append(f"{key}: {value}")
    
    return ", ".join(formatted_parts)


def get_sample_tasks(num_tasks: int = 3) -> List[Dict[str, Any]]:
    """
    Get a small sample of tasks for quick testing.
    
    Args:
        num_tasks: Number of tasks to return
    
    Returns:
        List of sample task dictionaries
    """
    try:
        tasks = load_dabench_tasks()
        return tasks[:num_tasks]
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"⚠️  Could not load tasks: {e}")
        return []


def validate_task_format(task: Dict[str, Any]) -> bool:
    """
    Validate that a task has the required fields.
    
    Args:
        task: Task dictionary to validate
    
    Returns:
        True if valid, False otherwise
    """
    required_fields = ['task_id', 'question', 'correct_answer', 'level', 'guidelines']
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
    print("🧪 Testing data loaders...")
    
    # Test DABench dataset
    print("\n📊 Testing DABench dataset...")
    try:
        dabench_tasks = load_dabench_tasks()
        print(f"📊 DABench dataset: {get_task_statistics(dabench_tasks)}")
        if dabench_tasks:
            print(f"📝 Sample DABench task: {dabench_tasks[0]['task_id']} - {dabench_tasks[0]['question'][:100]}...")
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        print(f"⚠️  Could not load DABench dataset: {e}")
    
    # Test quick sample
    print("\n🎯 Testing quick sample...")
    quick_sample = get_sample_tasks(2)
    if quick_sample:
        print(f"🎯 Quick sample (2 tasks): {[t['task_id'] for t in quick_sample]}")