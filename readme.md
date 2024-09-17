# Tungo 文件搜尋引擎 (Document Search Engine)

## Project Overview

Tungo is a powerful and versatile full-text retrieval tool designed for efficient searching and analysis of textual documents. Developed as part of the Biomedical Information Retrieval course at National Cheng Kung University, this project addresses the need for a robust document search system capable of handling multiple languages and document formats.

## Features

- **Multi-format Support**: Handles both XML (PubMed) and JSON (Twitter) formats
- **Bilingual Capability**: Supports English and Chinese documents
- **Efficient Indexing**: Uses inverted index for fast keyword-based searches
- **Comprehensive Document Analysis**: Provides character count, word count, and sentence count
- **Smart Sentence Detection**: Implements language-aware sentence tokenization
- **Keyword Frequency Analysis**: Identifies top 10 most frequent keywords per document
- **User-friendly Web Interface**: Intuitive, responsive design for easy interaction

## Tech Stack

- **Backend**: Python, Flask
- **Frontend**: Vue.js, Tailwind CSS
- **NLP Libraries**: NLTK (English), Jieba (Chinese)
- **Data Storage**: Local JSON-based storage

## Usage

1. **Upload Documents**: 
   - Click the "Upload Document" button
   - Select a file (XML or JSON format)
   - The document will be processed and indexed automatically

2. **Search Documents**:
   - Enter keywords in the search bar
   - Click "Search" or press Enter
   - View results, including document statistics and previews

3. **View Document Details**:
   - Click on a document title in the search results to view the full document

## Project Structure

```
tungo-search-engine/
│
├── app.py                 # Main Flask application
├── templates/
│   └── index.html         # Main HTML template
├── public/
│   └── documents/         # Uploaded documents
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

