# Tungo 文件搜尋引擎 (Document Search Engine)

## Project Overview
Tungo is a powerful and versatile full-text retrieval tool designed for efficient searching and analysis of textual documents. Developed as part of the Artificial Intelligence Information Extraction course at National Cheng Kung University, this project addresses the need for a robust document search system capable of handling multiple languages and document formats.

## Features
- **Multi-format Support**: Handles plain text, JSON, and XML document formats
- **Multilingual Capability**: Supports English, Simplified Chinese, and Traditional Chinese documents
- **Efficient Indexing**: Uses an inverted index for fast keyword-based searches
- **Advanced Query Syntax**: Supports AND, OR, and NOT operators for complex queries
- **Comprehensive Document Analysis**: Provides character count (with/without spaces), word count, sentence count, non-ASCII character count, and non-ASCII word count
- **Smart Language Detection**: Automatically detects the language of uploaded documents
- **Keyword Frequency Analysis**: Identifies the top 10 most frequent keywords per document
- **User-friendly Web Interface**: Intuitive, responsive design built with Vue.js and Tailwind CSS for easy interaction
- **Document Preview**: Shows a preview of each document in the search results
- **Document Viewing**: Allows viewing the full content of an uploaded document

## Tech Stack
- **Backend**: Python, Flask
- **Frontend**: Vue.js, Tailwind CSS
- **NLP Libraries**: NLTK (English), Jieba (Chinese)
- **Data Storage**: Local JSON-based storage

## Usage
1. **Upload Documents**: 
   - Click the "Upload Document" button
   - Select a file (plain text, JSON, or XML format)
   - The document will be processed, analyzed, and indexed automatically

2. **Search Documents**:
   - Enter keywords in the search bar
     - Enter AND,  OR, and NOT options if need
   - Click "Search" or press Enter
   - View search results, including document statistics, keyword frequencies, and previews

3. **View Document Details**:
   - Click on a document title in the search results to view the full document content

## Project Structure
```
tungo-search-engine/
│
├── app.py                 # Main Flask application
├── templates/
│   └── index.html         # Main HTML template
├── public/
│   └── documents/         # Uploaded documents
├── local_database.json    # Local JSON-based storage for index and document data 
├── chinese_stop_words.txt # Chinese stop words list
├── requirements.txt       # Python dependencies
└── README.md              # Project overview and documentation
```
