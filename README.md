# AI Text Summarizer

A minor project that uses a fine-tuned **T5 transformer model** to generate concise summaries from longer text or dialogues.

## Project Overview

This project demonstrates how a transformer-based NLP model can be trained and integrated into a simple web application.

The trained T5 model takes a dialogue or text as input and generates a shorter summary while preserving the main information.

## Features

* Text preprocessing
* T5-based abstractive text summarization
* Fine-tuned transformer model
* FastAPI backend
* Simple web interface
* Runs locally through a browser

## Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* T5
* FastAPI
* HTML/CSS/JavaScript
* Jupyter Notebook

## Project Structure

```text
Text-Summarizer/
│
├── APP/
│   ├── app.py
│   └── index.html
│
├── DATASETS/
│   └── SAMSum dataset files
│
├── MODEL/
│   ├── results/
│   ├── save_summary_model/
│   └── Model.ipynb
│
└── README.md
```

## How It Works

1. Text is entered through the web interface.
2. FastAPI receives the input.
3. The input is cleaned and tokenized using the T5 tokenizer.
4. The trained T5 model generates a summary.
5. The generated summary is returned and displayed in the browser.

## How to Run

Clone the repository and install the required Python packages.

Then navigate to the application folder:

```bash
cd APP
```

Start the FastAPI server:

```bash
uvicorn app:app --reload
```

Open the application in your browser:

```text
http://127.0.0.1:8000
```

## Project Type

This was developed as a **minor/course project** to practice NLP, transformer models, model deployment, and building a simple ML application.

## Author

Krishna
