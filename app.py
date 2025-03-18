from flask import Flask, render_template, request, redirect, url_for
import json
import random
import uuid
import os
import time
import argparse

random.seed(42)
app = Flask(__name__)
app.jinja_env.globals.update(enumerate=enumerate)  # Make enumerate available in templates

# Global variables
SESSIONS = dict()
INPUT_JSONL = 'data'
BATCH_COUNT_FILE = 'batch_count.json'
TRIAL_MODE = False
batch_size = 30
data = None
batches = None
batch_counts = None

def init_app(trial_mode=False):
    """Initialize app settings based on mode"""
    global INPUT_JSONL, BATCH_COUNT_FILE, TRIAL_MODE, batch_size, data, batches, batch_counts
    
    TRIAL_MODE = trial_mode
    if TRIAL_MODE:
        INPUT_JSONL = 'data_with_gpt_trial'
        BATCH_COUNT_FILE = 'batch_count_trial.json'
        batch_size = 5
        print("Running in trial mode")
    else:
        INPUT_JSONL = 'data_with_gpt'
        BATCH_COUNT_FILE = 'batch_count.json'
        batch_size = 30
        print("Running in full mode")

    # Initialize all data-dependent variables
    data = load_data()
    print("Total length of data: ", len(data))
    batches = shuffle_and_batch_data(data, batch_size)
    print("Total length of batches: ", len(batches))
    batch_counts = initialize_batch_counts(batches)

def load_data():
    with open(f'{INPUT_JSONL}.jsonl', 'r') as file:
        return [json.loads(line) for line in file]

def shuffle_and_batch_data(data, batch_size):
    random.shuffle(data)
    batches_with_ids = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        batch_id = i // batch_size + 1
        batches_with_ids.append({'batch_id': batch_id, 'batch_data': batch})
    return batches_with_ids

def initialize_batch_counts(batches):
    if not os.path.exists(BATCH_COUNT_FILE):
        batch_counts = {batch['batch_id']: 0 for batch in batches}
        save_batch_counts(batch_counts)
    else:
        with open(BATCH_COUNT_FILE, 'r') as file:
            batch_counts = json.load(file)
    
    print(f"Batch counts initialized: {batch_counts}")    
    return batch_counts

def save_batch_counts(batch_counts):
    """Save the batch assignment counts to a file."""
    with open(BATCH_COUNT_FILE, 'w') as file:
        json.dump(batch_counts, file)

def get_least_assigned_batch(batch_counts):
    """Get the batch with the lowest assignment count."""
    return min(batch_counts, key=batch_counts.get)

def generate_session_id():
    return str(uuid.uuid4())

session_id = generate_session_id()
print(f"Assigned session ID: {session_id}")
SESSIONS[session_id] = {
    'prolific_id': None,
    'user_batch_ids': None,
    'user_batch_unique_id': None,
    'index': None,
    'design_usage': None,
    'adobe_app': None,
    'qid': None,
    'user_query': None,
    'background_color': None,
    'text_elemnets': None,
    'explanation': None,
    'image_ranks': None,
    'user_batch_ids': None,
}
session = SESSIONS[session_id]

RESPONSE_DIR = 'responses'  # Directory to store responses

def save_annotation_to_file(session_data):
    if not os.path.exists(RESPONSE_DIR):
        os.makedirs(RESPONSE_DIR, exist_ok=True)
    annotator_id = session_data.get('prolific_id', 'unknown')
    save_name = time.strftime("%Y%m%d-%H%M%S")  # Timestamp
    response_file = os.path.join(RESPONSE_DIR, f'{annotator_id}_{save_name}.jsonl')

    with open(response_file, 'a', encoding='utf-8') as f:
        json.dump(session_data, f)
        f.write('\n')

    print(f"Saved annotation to {response_file}")

@app.route('/')
def intro():
    return render_template('introduction.html')

@app.route('/success')
def success():
    return render_template('success.html')

@app.route('/failure')
def failure():
    return render_template('failure.html')

@app.route('/prolific')
def prolific():
    return render_template('prolific.html')

@app.route('/save_prolific', methods=['POST'])
def save_prolific():
    prolific_id = request.form.get('prolific_id')
    session['prolific_id'] = prolific_id

    with open(BATCH_COUNT_FILE, 'r') as file:
        batch_counts = json.load(file)
    
    least_assigned_batch_id = get_least_assigned_batch(batch_counts)
    
    print(f"Available batch IDs: {[batch['batch_id'] for batch in batches]}")

    try:
        user_batch_info = next(batch for batch in batches if batch['batch_id'] == int(least_assigned_batch_id))
    except StopIteration:
        raise ValueError(f"No batch found with batch_id {least_assigned_batch_id}")
    
    user_batch = user_batch_info['batch_data']
    
    session['user_batch_ids'] = [item['ID'] for item in user_batch]
    session['user_batch_unique_id'] = least_assigned_batch_id
    print(f"Assigned user_batch_unique_id: {least_assigned_batch_id}")

    batch_counts[least_assigned_batch_id] += 1
    save_batch_counts(batch_counts)

    session['index'] = 0
    return redirect(url_for('design_tool'))

@app.route('/design_tool')
def design_tool():
    return render_template('design_tool_usage.html')

@app.route('/save_design_tool', methods=['POST'])
def save_design_tool():
    design_usage = request.form.get('design_usage')
    session['design_usage'] = design_usage
    return redirect(url_for('adobe_app'))

@app.route('/adobe_app')
def adobe_app():
    return render_template('adobe_app.html')

@app.route('/save_adobe_app', methods=['POST'])
def save_adobe_app():
    adobe_app = request.form.get('adobe_app')
    session['adobe_app'] = adobe_app
    return redirect(url_for('start'))

@app.route('/start')
def start():
    return redirect(url_for('annotate'))

@app.route('/end')
def end():
    print(f"Session {session_id} completed.")
    return render_template('end.html')

@app.route('/annotate', methods=['GET', 'POST'])
def annotate():
    index = session.get('index', 0)
    user_batch_ids = session.get('user_batch_ids')
    user_batch_unique_id = session.get('user_batch_unique_id')
    
    if index >= len(user_batch_ids):
        return redirect(url_for('end'))

    current_data = next(item for item in data if item['ID'] == user_batch_ids[index])
    qid = current_data['ID']
    user_query = current_data['user_query']
    design_choices = current_data['design_choices']
    images = current_data['images']
    # Get GPT answer if it exists in the data
    gpt_answer = current_data.get('gpt_answer', {
        'overall_confidence': 'N/A',
        'background_color': {'suggestion': 'N/A', 'confidence': 'N/A'},
        'text_elements': {'suggestions': [], 'confidence': 'N/A'},
        'visual_elements': {'suggestions': [], 'confidence': 'N/A'},
        'review_notes': []
    })

    if request.method == 'POST':
        background_color = request.form.get('background_color')
        explanation = request.form.get('explanation')

        text_alignments = {}
        if 'text' in design_choices:
            for text_item in design_choices['text'].keys():
                alignment_response = request.form.get(text_item)
                text_item_r = text_item.replace("_", " ")
                text_alignments[text_item_r] = alignment_response

        image_ranks = {}
        for i, image_set in enumerate(images):
            image_ranks[f'rank_image_{i+1}'] = {}
            for j in range(len(image_set['urls'])):
                rank_key = f'rank_image_{i+1}_{j+1}'
                rank_value = request.form.get(rank_key)
                image_ranks[f'rank_image_{i+1}'][f'image_{j+1}'] = rank_value

        session_data = {
            'session_id': session_id,
            'prolific_id': session.get('prolific_id'),
            'user_batch_unique_id': user_batch_unique_id,
            'design_usage': session.get('design_usage'),
            'adobe_app': session.get('adobe_app'),
            'qid': qid,
            'user_query': user_query,
            'background_color': background_color,
            'text_elements': text_alignments,
            'explanation': explanation,
            'image_ranks': image_ranks,
            'user_batch_ids': user_batch_ids,
        }
        
        save_annotation_to_file(session_data)  # Save each row separately

        session['index'] = index + 1
        return redirect(url_for('annotate'))

    # Replace _ to " " in images
    for image in images:
        image["content"] = image["content"].replace("_", " ")

    return render_template('annotation_r.html', 
                         index=index, 
                         user_query=user_query, 
                         design_choices=design_choices, 
                         images=images,
                         gpt_answer=gpt_answer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run annotation web app')
    parser.add_argument('--trial', action='store_true', help='Run in trial mode')
    args = parser.parse_args()
    
    init_app(trial_mode=args.trial)
    app.run(debug=True, host='0.0.0.0', port=5001)
