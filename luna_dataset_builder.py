import gradio as gr
import json
import os
import requests
from werkzeug.utils import secure_filename
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI API Key and Assistant ID
OPENAI_API_KEY = os.getenv('LUNAS_OPENAI_API_KEY')
ASSISTANT_ID = "asst_Fd7cOhXiCamVGAC5h7Gd5qVw"

# Specify the folder for storing datasets
DATASET_FOLDER = r"G:\My Drive\Luna_dataset\datasets\datasets jsonl"

def send_to_openai(user_text, input_type):
    headers = {
        'Authorization': f'Bearer {OPENAI_API_KEY}',
        'Content-Type': 'application/json',
        'OpenAI-Beta': 'assistants=v2'
    }

    thread_response = requests.post('https://api.openai.com/v1/threads', headers=headers)
    if thread_response.status_code != 200:
        return f"Error creating thread: {thread_response.text}"
    thread_id = thread_response.json().get('id')

    message_data = {'role': 'user', 'content': user_text}
    message_response = requests.post(f'https://api.openai.com/v1/threads/{thread_id}/messages', headers=headers, json=message_data)
    if message_response.status_code != 200:
        return f"Error adding message to thread: {message_response.text}"

    run_data = {
        'assistant_id': ASSISTANT_ID,
        'instructions': f"""Generate 3-5 diverse JSONL entries for fine-tuning a language model on {input_type} processing tasks. Each entry should be a complete JSON object on a single line, containing a 'messages' array with 'system', 'user', and 'assistant' messages. Vary the system messages, rephrase user inputs, and provide diverse assistant responses. Ensure entries are substantive and showcase different aspects of {input_type} understanding and generation.""",
        'stream': True
    }
    run_response = requests.post(f'https://api.openai.com/v1/threads/{thread_id}/runs', headers=headers, json=run_data, stream=True)
    if run_response.status_code != 200:
        return f"Error running assistant on thread: {run_response.text}"

    full_response = ""
    for line in run_response.iter_lines():
        if line:
            try:
                line_text = line.decode('utf-8')
                if line_text.startswith('data: '):
                    data = json.loads(line_text[6:])
                    if data['object'] == 'thread.message.delta':
                        if 'content' in data and len(data['content']) > 0 and 'text' in data['content'][0]:
                            content = data['content'][0]['text']['value']
                            full_response += content
                            print(content, end='', flush=True)
                    elif data['object'] == 'thread.run.completed':
                        break
            except json.JSONDecodeError:
                print(f"Error decoding JSON: {line_text}")
            except Exception as e:
                print(f"Error processing line: {e}")

    if not full_response:
        print("No response generated. Retrieving messages from thread.")
        messages_response = requests.get(f'https://api.openai.com/v1/threads/{thread_id}/messages', headers=headers)
        if messages_response.status_code == 200:
            messages = messages_response.json().get('data', [])
            assistant_messages = [msg for msg in messages if msg['role'] == 'assistant']
            if assistant_messages:
                full_response = assistant_messages[0]['content'][0]['text']['value']

    return full_response

def extract_youtube_transcript(url):
    try:
        parsed_url = urlparse(url)
        if parsed_url.netloc == 'youtu.be':
            video_id = parsed_url.path[1:]
        elif parsed_url.netloc in ('www.youtube.com', 'youtube.com'):
            if parsed_url.path == '/watch':
                video_id = parse_qs(parsed_url.query)['v'][0]
            elif parsed_url.path.startswith(('/embed/', '/v/')):
                video_id = parsed_url.path.split('/')[2]
        else:
            return "Invalid YouTube URL"

        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        full_transcript = " ".join([entry['text'] for entry in transcript])
        return full_transcript
    except Exception as e:
        return f"Error extracting transcript: {str(e)}"

def analyze_image(image_path):
    return f"Analysis of image at {image_path}"

def scrape_url(url):
    return f"Content scraped from {url}"

def process_document(file_path):
    return f"Content extracted from document: {file_path}"

def get_datasets():
    os.makedirs(DATASET_FOLDER, exist_ok=True)
    return [f for f in os.listdir(DATASET_FOLDER) if f.endswith('.jsonl')]

def create_dataset(name):
    filename = secure_filename(f"{name}.jsonl")
    full_path = os.path.join(DATASET_FOLDER, filename)
    with open(full_path, 'w', encoding='utf-8') as f:
        pass
    return full_path

def process_input(dataset_option, new_dataset_name, input_type, content, existing_dataset, document_file):
    if "Create New Dataset" in dataset_option:
        if not new_dataset_name:
            return "Please provide a name for the new dataset."
        dataset = create_dataset(new_dataset_name)
    elif "Select Existing Dataset" in dataset_option:
        if not existing_dataset:
            return "Please select an existing dataset."
        dataset = os.path.join(DATASET_FOLDER, existing_dataset)
    else:
        return "Please select a dataset option and provide necessary information."

    if not input_type:
        return "Please select an input type."

    if input_type == "YouTube Link":
        if not content:
            return "Please provide a YouTube URL."
        processed_content = extract_youtube_transcript(content)
    elif input_type == "Image/Diagram":
        if not content:
            return "Please provide an image file path."
        processed_content = analyze_image(content)
    elif input_type == "Text":
        if not content:
            return "Please provide some text content."
        processed_content = content
    elif input_type == "Documents":
        if not document_file:
            return "Please upload a document file."
        processed_content = process_document(document_file.name)
    elif input_type == "URL Scraping":
        if not content:
            return "Please provide a URL to scrape."
        processed_content = scrape_url(content)
    else:
        return f"Invalid input type: {input_type}"

    assistant_response = send_to_openai(processed_content, input_type)

    # Parse the assistant's response
    jsonl_entries = []
    for line in assistant_response.strip().split('\n'):
        try:
            entry = json.loads(line)
            jsonl_entries.append(entry)
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON entry: {line}")

    if not jsonl_entries:
        return "No valid entries were generated. Please try again."

    # Write valid entries to the JSONL file
    try:
        with open(dataset, 'a', encoding='utf-8') as f:
            for entry in jsonl_entries:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')
    except IOError as e:
        return f"Error writing to dataset: {str(e)}"

    # Format the output for display
    formatted_output = "Content processed and added to dataset {}:\n\n".format(dataset)
    for entry in jsonl_entries:
        formatted_output += json.dumps(entry, indent=2, ensure_ascii=False) + "\n\n"

    return formatted_output

# Gradio interface
with gr.Blocks() as app:
    gr.Markdown("# Dataset Builder with OpenAI Assistant Integration")

    with gr.Row():
        dataset_option = gr.CheckboxGroup(["Select Existing Dataset", "Create New Dataset"], label="Dataset Option")
        existing_dataset = gr.Dropdown(choices=get_datasets(), label="Existing Datasets")
        new_dataset_name = gr.Textbox(label="New Dataset Name")

    with gr.Row():
        input_type = gr.Radio(["YouTube Link", "Image/Diagram", "Text", "Documents", "URL Scraping"], label="Input Type")
        content_input = gr.Textbox(label="Content (URL, text, or file path)")
        document_file = gr.File(label="Upload Document", visible=False)

    submit_btn = gr.Button("Process and Add to Dataset")
    output = gr.Textbox(label="Output")

    def update_visibility_and_datasets(dataset_choice):
        datasets = get_datasets()
        return {
            existing_dataset: gr.update(visible="Select Existing Dataset" in dataset_choice, choices=datasets),
            new_dataset_name: gr.update(visible="Create New Dataset" in dataset_choice)
        }

    def update_input_visibility(input_type):
        return {
            content_input: gr.update(visible=input_type != "Documents"),
            document_file: gr.update(visible=input_type == "Documents")
        }

    dataset_option.change(update_visibility_and_datasets, dataset_option, [existing_dataset, new_dataset_name])
    input_type.change(update_input_visibility, input_type, [content_input, document_file])

    def process_and_update(dataset_option, new_dataset_name, input_type, content, existing_dataset, document_file):
        result = process_input(dataset_option, new_dataset_name, input_type, content, existing_dataset, document_file)
        updated_datasets = get_datasets()
        return result, gr.update(choices=updated_datasets)

    submit_btn.click(
        process_and_update,
        inputs=[dataset_option, new_dataset_name, input_type, content_input, existing_dataset, document_file],
        outputs=[output, existing_dataset]
    )

if __name__ == "__main__":
    app.launch(share=True)