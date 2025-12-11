# Email Order Extraction

AI-powered extraction of order information from emails and PDF attachments using Azure OpenAI and Langfuse for evaluation and monitoring.

## Overview

This project automates the extraction of structured order data from German business correspondence (emails with PDF attachments). It uses Azure OpenAI's GPT-4o vision model to parse both email content and PDF documents, extracting buyer information, order details, delivery addresses, and product line items.

The system includes comprehensive evaluation capabilities through Langfuse, enabling dataset-based testing and performance monitoring.

## Features

- **Multi-source Data Extraction**: Synthesizes information from email text and PDF images
- **Structured Output**: Returns JSON with strict schema validation using Pydantic
- **DIN 5008 Layout Support**: Specialized logic for German business letter formats
- **Langfuse Integration**: Built-in experiment tracking and evaluation
- **Multiple Evaluators**: Exact match, buyer info, order info, address info, and product scores.

## Project Structure

```
.
├── src/
│   ├── task.py                    # Main extraction and Langfuse experiment runner
│   ├── data_processing.py         # Utility functions for data preparation
│   └── langfuse_integration.py    # Langfuse dataset creation and testing utilities
├── data/
│   ├── pdfs/                      # PDF order documents
│   ├── emails.txt                 # Email content (optional, for data preparation)
│   ├── expected_output.txt        # Expected outputs (optional, for data preparation)
│   └── matched_emails_output.csv  # Prepared dataset for Langfuse
├── pyproject.toml                 # Project dependencies and configuration
├── .env                          # Environment variables (not in repo)
└── README.md                     # This file
```

## Installation

### Requirements

- Python 3.8+
- Azure OpenAI API access
- Langfuse account (for experiment tracking)

### Setup

1. **Install dependencies**:
```bash
pip install -e .
```

Or install from pyproject.toml:
```bash
pip install openai python-dotenv PyMuPDF pydantic langfuse pandas
```

2. **Configure environment variables**:

Create a `.env` file in the project root:

```env
AZURE_OPENAI_KEY=your_azure_openai_api_key
AZURE_OPENAI_RESOURCE_URL=https://your-resource.openai.azure.com/

LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

3. **Prepare your data**:

Ensure you have:
- PDF files in `data/pdfs/`
- A Langfuse dataset named `email_order_extraction` with items containing:
  - `input`: `{"filename": "order.pdf", "email": "email text"}`
  - `expected_output`: Expected JSON structure

## Usage

### Running Experiments

The main script runs a Langfuse experiment on your dataset:

```bash
python -m src.task
```

This will:
1. Load the dataset from Langfuse
2. Process each email and PDF pair
3. Extract order information using Azure OpenAI
4. Evaluate results against expected outputs
5. Log all results to Langfuse with detailed metrics

### Data Preparation

If you need to prepare data from raw files:

```bash
python -m src.data_processing
```

This processes raw email and expected output text files into the required CSV format.

### Langfuse Dataset Management

Create or update the Langfuse dataset:

```bash
python -m src.langfuse_integration
```

Skip dataset creation if it already exists:

```bash
python -m src.langfuse_integration --skip-dataset
```

## Output Schema

The extraction produces JSON with the following structure:

```json
{
  "buyer_company_name": "Company Name GmbH",
  "buyer_person_name": "Name Lantname",
  "buyer_email_address": "name.lastname@company.de",
  "order_number": "1234567890",
  "order_date": "15.03.2024",
  "delivery_address_street": "Somestraße 123",
  "delivery_address_city": "Berlin",
  "delivery_address_postal_code": "11111",
  "products": [
    {
      "position": 1,
      "article_code": "X1234567",
      "quantity": 10
    }
  ]
}
```

## Evaluation Metrics

The system uses multiple evaluators:

1. **exact_match**: Binary score (1 if all fields match exactly, 0 otherwise)
2. **buyer_info**: Proportion of buyer fields that match (0.0-1.0)
3. **order_info**: Proportion of order fields that match (0.0-1.0)
4. **address_info**: Proportion of address fields that match (0.0-1.0)
5. **products**: Binary score for product list match (1.0 if exact match, 0.0 otherwise)
6. **avg_score**: Average across all field-level evaluators excluding exact_match field.

## Architecture

### Core Components

**task.py**: Main extraction engine
- `convert_pdf_to_images()`: Converts PDF pages to base64-encoded PNG images
- `create_extraction_prompt()`: Returns the specialized system prompt for DIN 5008 documents
- `call_azure_openai_with_vision()`: Executes the API call with vision and structured output
- `run_langfuse_experiment()`: Orchestrates the full experiment workflow with evaluators

**data_processing.py**: Data preparation utilities
- Parses raw text files into structured JSON
- Matches emails with expected outputs
- Creates CSV datasets for Langfuse

**langfuse_integration.py**: Langfuse operations
- Dataset creation and management
