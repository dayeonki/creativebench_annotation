import json
from azure_openai.aoai_gpt import AoaiGptInterface
import time
import argparse
import os
from typing import List, Dict, Any

# Configuration
MAX_RETRIES = 3  # Default max retry attempts
DELAY_BETWEEN_RETRIES = 5  # seconds
CHECKPOINT_FILE = 'gpt_processing_checkpoint.json'

def load_checkpoint() -> Dict[int, Any]:
    """Load checkpoint of completed tasks"""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_checkpoint(completed_tasks: Dict[int, Any]):
    """Save checkpoint of completed tasks"""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(completed_tasks, f)

def save_task_result(task_id: int, result: Dict, output_file: str):
    """Save individual task result to output file"""
    with open(output_file, 'a') as f:
        json.dump(result, f)
        f.write('\n')

def generate_gpt_answer(gpt_interface, task, max_retries: int = MAX_RETRIES) -> Dict:
    """Generate GPT answer for a single task with retry logic"""
    for attempt in range(max_retries):
        try:
            # Modified system prompt to be more neutral
            messages = [
                {"role": "system", "content": """You are a design consultant helping to analyze design requirements and visual elements. Your task is to:
1. Review the design brief and reference images
2. Suggest appropriate design elements
3. Provide confidence levels for suggestions
4. Note areas needing additional review
5. Structure response in JSON format"""},
            ]

            # Simplified user prompt to avoid content policy issues
            user_content = [
                {
                    "type": "text",
                    "text": f"""Please review this design brief and the reference images:

Brief: {task['user_query']}

Provide your analysis in this JSON format:
{{
    "background_color": {{"suggestion": "str", "confidence": "low/medium/high"}},
    "text_elements": {{"suggestions": [], "confidence": "low/medium/high"}},
    "visual_elements": {{"suggestions": [], "confidence": "low/medium/high"}},
    "review_points": ["areas needing additional review"],
    "overall_confidence": "low/medium/high"
}}"""
                }
            ]

            # Add images
            for image_set in task['images']:
                for url in image_set['urls']:
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": url}
                    })

            messages.append({"role": "user", "content": user_content})

            # Removed temperature parameter as it's not supported
            response = gpt_interface.client.chat.completions.create(
                model=gpt_interface.deployment_name,
                messages=messages,
                seed=42
            )
            return json.loads(response.choices[0].message.content)
        
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Error on task {task['ID']} (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"Retrying in {DELAY_BETWEEN_RETRIES} seconds...")
                time.sleep(DELAY_BETWEEN_RETRIES)
            else:
                print(f"Failed to process task {task['ID']} after {max_retries} attempts: {e}")
                return None

def process_data(trial_mode: bool = False, max_retries: int = MAX_RETRIES):
    """Process data with trial mode and checkpointing support"""
    # Initialize GPT interface
    gpt_interface = AoaiGptInterface()
    gpt_interface.select_config()

    # Load original data
    with open('data.jsonl', 'r') as f:
        tasks = [json.loads(line) for line in f]

    # Load checkpoint
    completed_tasks = load_checkpoint()
    
    # In trial mode, process enough tasks to make complete batches
    if trial_mode:
        batch_size = 5  # smaller batch size for trial
        num_tasks = batch_size * 2  # process 2 complete batches
        tasks = tasks[:num_tasks]
        print(f"Trial mode: Processing first {len(tasks)} tasks")

    output_file = 'data_with_gpt_trial.jsonl' if trial_mode else 'data_with_gpt.jsonl'
    
    # Process each task
    for task in tasks:
        task_id = task['ID']
        
        # Skip if already completed
        if str(task_id) in completed_tasks:
            print(f"Skipping task {task_id} (already completed)")
            continue

        print(f"Processing task {task_id}...")
        gpt_answer = generate_gpt_answer(gpt_interface, task, max_retries)
        
        if gpt_answer:
            task['gpt_answer'] = gpt_answer
            # Save individual result
            save_task_result(task_id, task, output_file)
            # Update checkpoint
            completed_tasks[str(task_id)] = True
            save_checkpoint(completed_tasks)
            print(f"Successfully processed task {task_id}")
        else:
            print(f"Skipping task {task_id} due to failure")
        
        time.sleep(1)  # Rate limiting

    print(f"Processing complete. Results saved to {output_file}")
    print(f"Processed {len(completed_tasks)} tasks in total")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process data with GPT answers')
    parser.add_argument('--trial', action='store_true', help='Run in trial mode (process only 10 tasks)')
    parser.add_argument('--max-retries', type=int, default=MAX_RETRIES, help='Maximum number of retry attempts per task')
    args = parser.parse_args()

    process_data(trial_mode=args.trial, max_retries=args.max_retries) 