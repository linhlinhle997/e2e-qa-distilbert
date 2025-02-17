# End-to-End Question Answering System

This project aims to build an end-to-end question answering (QA) system that utilizes a retrieval-based approach, similar to Retrieval Augmented Generation (RAG). The system retrieves relevant documents and extracts the most accurate answer.

## Overview

The system consists of two primary modules:

*   **Reader**: Extracts the most accurate answer from the retrieved documents and the input question.
*   **Retriever**:  Queries a database to retrieve relevant documents related to the input question.

## Key Components

*   **Dataset**: SQuAD2.0 (Reading comprehension dataset)
*   **Vector Database**: FAISS (Facebook AI Similarity Search) for efficient similarity search
*   **Model**: DistilBERT (pre-trained for the Reader and used for creating vector embeddings)

## Implementation Details

1. **Reader (DistilBERT)**

*   The Reader model is built using `DistilBERT` for answer extraction.
*   The SQuAD2.0 dataset is tokenized for training and validation.
*   The model is fine-tuned on the SQuAD2.0 dataset to answer questions accurately.

2. **Retriever (FAISS)**

*   A vector database is constructed from the SQuAD2.0 dataset's question embeddings.
*   `DistilBERT` is used to create vector embeddings of the questions.
*   `FAISS` is employed to perform similarity search and retrieve the most relevant context based on the input question.

## Setup and Installation

1.  **Clone the GitHub Repository**:

```bash
  git clone https://github.com/linhlinhle997/qa-with-faiss-and-bert.git
  cd qa-with-faiss-and-bert
```

4. **Run the Notebook**:

Open the notebook file `qa_system.ipynb`
The notebook will handle the following tasks automatically:
- Install Required Libraries (`transformers`, `datasets`, `evaluate`, `faiss-gpu`).
- Download the SQuAD2.0 dataset using Hugging Face.
- Prompt you to log in to Hugging Face to access necessary datasets and models.
Once the notebook completes the setup, you can proceed with model training, evaluation, and inference.

## Usage

1.   **Reader**: Fine-tuned `DistilBERT` model to extract answers from the retrieved contexts.
2.   **Retriever**: Implemented using `FAISS` to create a vector database of question embeddings for efficient context retrieval.

## Project Structure

The project includes code for:

1. **Data Preprocessing**: Tokenizing the dataset and preparing it for training and evaluation.
2. **Model Training**: Fine-tuning the `DistilBERT` model on the SQuAD2.0 dataset for answering questions.
3. **Retriever (FAISS)**: Uses DistilBERT to embed questions and FAISS to retrieve relevant contexts.
4. **Reader (DistilBERT + FAISS)**: Extracts answers from retrieved contexts using the fine-tuned DistilBERT model.
6. **Evaluation**: Measures performance with the SQuAD metric using the `evaluate` library.
7. **Inference**: Uses the trained model for question answering with the retrieval and reading pipeline.
