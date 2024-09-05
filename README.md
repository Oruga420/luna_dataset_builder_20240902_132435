# Luna Dataset Builder

## Project Overview

Luna Dataset Builder is a powerful tool designed to create and manage datasets for fine-tuning language models, with a specific focus on integrating OpenAI's API. This application provides a user-friendly interface for processing various types of input and generating high-quality, diverse datasets.

## Key Features

1. **Multiple Input Types**: Supports processing of YouTube links, images/diagrams, text, documents (PDF, TXT, DOCX, CSV), and URL scraping.

2. **OpenAI Integration**: Utilizes OpenAI's API to generate diverse and relevant dataset entries.

3. **Dataset Management**: Allows users to create new datasets or add to existing ones.

4. **File Processing**: Handles various file types including PDFs, text files, Word documents, and CSV files.

5. **Image Analysis**: Processes images and generates detailed descriptions using OpenAI's vision model.

6. **User-Friendly Interface**: Built with Gradio for an intuitive and accessible user experience.

## How It Works

1. Users select an input type and provide content (URL, text, or file upload).
2. The application processes the input, extracting relevant information.
3. Processed content is sent to OpenAI's API for analysis and dataset entry generation.
4. Generated entries are added to the selected dataset in JSONL format.

## Use Cases

- Creating training datasets for fine-tuning language models
- Generating diverse Q&A pairs from various sources
- Automating the process of dataset creation for AI research and development

## Requirements

- Python 3.x
- OpenAI API key
- Various Python libraries (gradio, requests, PyPDF2, python-docx, etc.)

## Setup and Usage

1. Clone the repository
2. Install required dependencies
3. Set up your OpenAI API key in the environment variables
4. Run the script to launch the Gradio interface
5. Use the interface to process inputs and build your datasets

## Note

This tool is designed to streamline the process of creating high-quality datasets for AI training and research purposes. It's particularly useful for those working on fine-tuning language models or developing AI applications that require diverse and well-structured training data.
