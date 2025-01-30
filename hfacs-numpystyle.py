
import pandas as pd
import numpy as np
from llama_index.readers.file.docs import PDFReader
from llama_index.core import VectorStoreIndex
from llama_index.llms.openai import OpenAI
import os

# Set OpenAI API key
os.environ['OPENAI_API_KEY'] = "API_KEY_PATH"
def load_documents(file_paths):
    """
    Load PDF documents using PDFReader.
    
    Parameters
    ----------
    file_paths : list of str
        List of file paths to the PDF documents.

    Returns
    -------
    list
        List of loaded documents.
    """
    pdf_reader = PDFReader()
    documents = []
    
    for file_path in file_paths:
        docs = pdf_reader.load_data(file=file_path)
        print(f"Loaded {len(docs)} documents from {file_path}")
        documents.extend(docs)
    
    return documents

def create_index(documents):
    """
    Create a VectorStoreIndex from documents.
    
    Parameters
    ----------
    documents : list
        List of documents to be indexed.

    Returns
    -------
    VectorStoreIndex
        The created index for vector search.
    """
    return VectorStoreIndex.from_documents(documents)

def create_chat_engine(index):
    """
    Create a chat engine using OpenAI's GPT model.
    
    Parameters
    ----------
    index : VectorStoreIndex
        The indexed documents.

    Returns
    -------
    ChatEngine
        The initialized chat engine.
    """
    llm = OpenAI(model="gpt-4o-mini", api_key=os.environ['OPENAI_API_KEY'])
    return index.as_chat_engine(chat_mode="condense_question", llm=llm, verbose=True)

def get_responses(documents, questions, chat_engine):
    """
    Process each document with predefined questions and collect responses.

    Parameters
    ----------
    documents : list
        List of loaded documents.
    questions : list of str
        List of questions to ask for each document.
    chat_engine : ChatEngine
        The chat engine to interact with.

    Returns
    -------
    np.ndarray
        Numpy structured array containing responses.
    """
    responses = []
    
    for doc in documents:
        file_name = doc.metadata['file_name']
        for question in questions:
            prompt = f"Please answer the following question with 'Yes' or 'No': {question}"
            response = chat_engine.chat(prompt)
            responses.append((file_name, question, response))

    dtype = np.dtype([
        ('Document', 'U200'),
        ('Question', 'U300'),
        ('Answer', 'U10')
    ])

    return np.array(responses, dtype=dtype)

def save_responses_to_csv(responses, file_name="responses.csv"):
    """
    Save responses to a CSV file.

    Parameters
    ----------
    responses : np.ndarray
        Structured NumPy array with responses.
    file_name : str, optional
        Name of the output CSV file (default is "responses.csv").
    """
    df = pd.DataFrame(responses)
    df.to_csv(file_name, index=False)
    print(f"Responses saved to {file_name}")

# Define input file paths
file_paths = [
    r"path",
    r"path"
]

# Define questions
questions = [
    "Is the lack of organizational culture one of the contributing factors to this accident?",
    "Is the operational process issue one of the contributing factors to this accident?",
    "Is the resource management problem one of the contributing factors to this accident?",
    "Is the inadequate supervision one of the contributing factors to this accident?",
    "Is the planned inappropriate operations one of the contributing factors to this accident?",
    "Is the failure to correct known problems one of the contributing factors to this accident?",
    "Is the supervisory violation one of the contributing factors to this accident?",
    "Are physical environment factors like wind, rain, or snow contributing factors to this accident?",
    "Are tool or technology issues contributing factors to this accident?",
    "Is the operational process one of the contributing factors to this accident?",
    "Are communication, coordination, or planning problems contributing factors to this accident?",
    "Is not being fit for duty or responsibility one of the contributing factors to this accident?",
    "Are mental state or mental problems one of the contributing factors to this accident?",
    "Is the physiological state one of the contributing factors to this accident?",
    "Are physical or mental limitations one of the contributing factors to this accident?",
    "Is decision error or making a wrong decision one of the contributing factors to this accident?",
    "Are skill-based errors contributing factors to this accident?",
    "Is perceptual error one of the contributing factors to this accident?",
    "Is routine violation one of the contributing factors to this accident?",
    "Is exceptional error one of the contributing factors to this accident?",
]

# Load documents
documents = load_documents(file_paths)

# Create index
index = create_index(documents)

# Create chat engine
chat_engine = create_chat_engine(index)

# Get responses
responses = get_responses(documents, questions, chat_engine)

# Save responses to CSV
save_responses_to_csv(responses)
