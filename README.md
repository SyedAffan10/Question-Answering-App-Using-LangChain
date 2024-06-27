# Question Answering App Using LangChain

This Streamlit application allows users to upload multiple PDF files and ask questions based on the content of these PDFs. The app uses the Langchain library to process the PDFs and OpenAI's API to generate answers.

## Features

- **Upload Multiple PDFs**: Users can upload multiple PDF files.
- **Text Extraction and Splitting**: Extracts text from PDFs and splits them into manageable chunks for processing.
- **Vector Store Creation**: Uses OpenAI's embeddings to create a vector store from the text chunks.
- **Conversational Retrieval**: Allows users to query the PDF contents and get relevant answers.

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key, which should be stored in a `.env` file.

## Key Components

### Streamlit

- **`file_uploader`**: Allows users to upload PDF files.
- **`text_input`**: Provides an interface for users to input their questions.

### Langchain

- **`PyPDFLoader`**: Loads PDF files and extracts their text content.
- **`CharacterTextSplitter`**: Splits the text content into chunks.
- **`OpenAIEmbeddings`**: Generates embeddings for the text chunks using OpenAI's API.
- **`Chroma`**: Creates a vector store from the embeddings for efficient retrieval.
- **`ConversationalRetrievalChain`**: Handles the conversational retrieval process to answer user queries.

## Usage

1. **Upload PDF Files**: Click the file uploader and select one or more PDF files.
2. **Ask Questions**: Enter your question in the text input field.
3. **Get Answers**: The app processes the PDFs and returns answers based on their content.

## How to Run

1. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Create a `.env` file** in the root directory of your project and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ```

3. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```
