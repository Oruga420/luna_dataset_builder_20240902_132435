import gradio as gr
import json
import os
import requests
from werkzeug.utils import secure_filename
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv
import PyPDF2  # Add this import
import io
import csv
from docx import Document

# Load environment variables
load_dotenv()

# OpenAI API Key and Assistant ID
OPENAI_API_KEY = os.getenv('LUNAS_OPENAI_API_KEY')
ASSISTANT_ID = "asst_Fd7cOhXiCamVGAC5h7Gd5qVw"

# Specify the folder for storing datasets
DATASET_FOLDER = r"G:\My Drive\Luna_dataset\datasets\datasets jsonl"

def send_to_openai(user_text, input_type):
    if not user_text.strip():
        return "Error: Empty content. Please provide some text to process."

    headers = {
        'Authorization': f'Bearer {OPENAI_API_KEY}',
        'Content-Type': 'application/json',
        'OpenAI-Beta': 'assistants=v2'
    }

    # Create a new thread
    thread_response = requests.post('https://api.openai.com/v1/threads', headers=headers)
    if thread_response.status_code != 200:
        return f"Error creating thread: {thread_response.text}"
    thread_id = thread_response.json().get('id')

    # Add the user's message to the thread
    message_data = {'role': 'user', 'content': user_text}
    message_response = requests.post(f'https://api.openai.com/v1/threads/{thread_id}/messages', headers=headers, json=message_data)
    if message_response.status_code != 200:
        return f"Error adding message to thread: {message_response.text}"

    # Prepare instructions based on input type
    instructions = {
        "YouTube Link": """You are processing a YouTube video transcript. Summarize key points and generate 3-5 diverse JSONL entries for fine-tuning a language model. Each entry should be a complete JSON object on a single line, containing a 'messages' array with 'system', 'user', and 'assistant' messages. 
        For the system message, create a context related to the video content.
        For the user message, formulate a relevant question or request based on the transcript.
        For the assistant message, provide an informative response drawing from the video content.
        Ensure entries are substantive and showcase different aspects of understanding and generating content related to the video transcript.""",
        "URL Scraping": """You are processing scraped web content. Summarize key points and generate 3-5 diverse JSONL entries for fine-tuning a language model. Each entry should be a complete JSON object on a single line, containing a 'messages' array with 'system', 'user', and 'assistant' messages. 
        For the system message, create a context related to the web content.
        For the user message, formulate a relevant question or request based on the scraped text.
        For the assistant message, provide an informative response drawing from the web content.
        Ensure entries are substantive and showcase different aspects of understanding and generating content related to the scraped web page.""",
        "Image/Diagram": """You are processing an image or diagram description. Generate 3-5 diverse JSONL entries for fine-tuning a language model. Each entry should be a complete JSON object on a single line, containing a 'messages' array with 'system', 'user', and 'assistant' messages. 
        For the system message, create a context related to image analysis or the diagram's topic.
        For the user message, formulate a relevant question or request based on the image description.
        For the assistant message, provide an informative response interpreting or explaining the image content.
        Ensure entries are substantive and showcase different aspects of understanding and generating content related to visual information.""",
        "Text": """You are processing a text input. Generate 3-5 diverse JSONL entries for fine-tuning a language model. Each entry should be a complete JSON object on a single line, containing a 'messages' array with 'system', 'user', and 'assistant' messages. 
        For the system message, create a context related to the text content.
        For the user message, formulate a relevant question or request based on the text.
        For the assistant message, provide an informative response drawing from the text content.
        Ensure entries are substantive and showcase different aspects of understanding and generating content related to the given text.""",
        "Documents": """You are processing document content. Summarize key points and generate 3-5 diverse JSONL entries for fine-tuning a language model. Each entry should be a complete JSON object on a single line, containing a 'messages' array with 'system', 'user', and 'assistant' messages. 
        For the system message, create a context related to the document's topic or type.
        For the user message, formulate a relevant question or request based on the document content.
        For the assistant message, provide an informative response drawing from the document.
        Ensure entries are substantive and showcase different aspects of understanding and generating content related to the document."""
    }

    instruction = instructions.get(input_type, f"""Generate 3-5 diverse JSONL entries for fine-tuning a language model on {input_type} processing tasks. Each entry should be a complete JSON object on a single line, containing a 'messages' array with 'system', 'user', and 'assistant' messages. 
        Vary the system messages to provide different contexts.
        Create relevant user queries based on the content.
        Provide informative and diverse assistant responses.
        Ensure entries are substantive and showcase different aspects of {input_type} understanding and generation.""")

    # Run the assistant on the thread with streaming enabled
    run_data = {
        'assistant_id': ASSISTANT_ID,
        'instructions': instruction,
        'stream': True
    }
    run_response = requests.post(f'https://api.openai.com/v1/threads/{thread_id}/runs', headers=headers, json=run_data, stream=True)
    if run_response.status_code != 200:
        return f"Error running assistant on thread: {run_response.text}"

    # Process the streaming response
    full_response = ""
    for line in run_response.iter_lines():
        if line:
            try:
                line_text = line.decode('utf-8')
                if line_text.startswith('data: '):
                    data = json.loads(line_text[6:])  # Remove 'data: ' prefix
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
    # Handle dataset selection or creation
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

    # Process different input types
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
        processed_content = process_file(document_file)
    elif input_type == "URL Scraping":
        if not content:
            return "Please provide a URL to scrape."
        processed_content = scrape_url(content)
    else:
        return f"Invalid input type: {input_type}"

    if processed_content.startswith("Error"):
        return processed_content

    if not processed_content.strip():
        return "Error: No content was extracted from the input. Please check your file or input and try again."
    # Send processed content to OpenAI assistant
    assistant_response = send_to_openai(processed_content, input_type)
    
    if assistant_response.startswith("Error"):
        return assistant_response

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


def process_file(file):
    if not file:
        return "Error: No file provided."

    try:
        print(f"File object: {file}")
        print(f"File attributes: {dir(file)}")
        print(f"File name: {file.name}")
        print(f"Original name: {getattr(file, 'orig_name', 'Not available')}")

        # Try to get file size
        try:
            file_size = os.path.getsize(file.name)
            print(f"File size (os.path.getsize): {file_size} bytes")
        except Exception as e:
            print(f"Error getting file size: {str(e)}")

        # Try to read file content directly
        try:
            with open(file.name, 'rb') as f:
                content = f.read()
            print(f"Successfully read {len(content)} bytes directly from file")
        except Exception as e:
            print(f"Error reading file directly: {str(e)}")
            content = b''

        file_extension = file.name.split('.')[-1].lower()
        print(f"Processing file: {file.name}, Type: {file_extension}")

        if len(content) == 0:
            return "Error: The file is empty or could not be read."

        if file_extension == 'pdf':
            return process_pdf(file)
        elif file_extension == 'txt':
            return process_txt(file)
        elif file_extension in ['doc', 'docx']:
            return process_word(file)
        elif file_extension == 'csv':
            return process_csv(file)
        else:
            return f"Error: Unsupported file type '{file_extension}'. Please upload a PDF, TXT, Word document, or CSV file."
    except Exception as e:
        return f"Error processing file: {str(e)}"

def process_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        if not text.strip():
            return "Error: Unable to extract text from PDF. The file might be empty or protected."
        print(f"Extracted {len(text)} characters from PDF")
        return text
    except Exception as e:
        return f"Error processing PDF: {str(e)}"

def process_txt(file):
    try:
        print(f"Starting to process TXT file: {file.name}")

        with open(file.name, 'rb') as f:
            content = f.read()
        print(f"Read {len(content)} bytes from file")

        if len(content) == 0:
            return "Error: The TXT file is empty (0 bytes)."

        try:
            text = content.decode('utf-8')
            print("Successfully decoded with UTF-8")
        except UnicodeDecodeError:
            print("UTF-8 decoding failed, trying Latin-1")
            text = content.decode('latin-1')
            print("Successfully decoded with Latin-1")

        if not text.strip():
            return f"Error: The TXT file content is empty or contains only whitespace. Raw content length: {len(text)}"

        print(f"Extracted {len(text)} characters from TXT file")
        return text
    except Exception as e:
        return f"Error processing TXT file: {str(e)}"

def process_word(file):
    try:
        doc = Document(file)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        if not text.strip():
            return "Error: Unable to extract text from Word document. The file might be empty."
        print(f"Extracted {len(text)} characters from Word document")
        return text
    except Exception as e:
        return f"Error processing Word document: {str(e)}"

def process_csv(file):
    try:
        content = file.read()
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        csv_reader = csv.reader(io.StringIO(content))
        text = "\n".join([", ".join(row) for row in csv_reader])
        if not text.strip():
            return "Error: The CSV file is empty or contains no valid data."
        print(f"Extracted {len(text)} characters from CSV file")
        return text
    except Exception as e:
        return f"Error processing CSV file: {str(e)}"

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