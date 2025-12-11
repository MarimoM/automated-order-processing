import json
import base64
import os
import re
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from openai import AzureOpenAI
import fitz
from pydantic import BaseModel, Field
from langfuse import Langfuse
from jinja2 import Environment, FileSystemLoader

try:
    from langfuse.client import Evaluation
except ImportError:
    from langfuse import Evaluation


MODEL_NAME = "gpt-4o"


class Product(BaseModel):
    model_config = {"extra": "forbid"}

    position: int = Field(description="Position number of the product in the order")
    article_code: str = Field(description="Article/supplier code for the product")
    quantity: int = Field(description="Quantity ordered")


class OrderExtraction(BaseModel):
    model_config = {"extra": "forbid"}

    buyer_company_name: str = Field(description="Name of the buying company")
    buyer_person_name: str = Field(description="Full name of the buyer person")
    buyer_email_address: str = Field(description="Email address of the buyer")
    order_number: str = Field(description="Order number from the PDF")
    order_date: str = Field(description="Order date in DD.MM.YYYY format")
    delivery_address_street: str = Field(description="Delivery street address including house number")
    delivery_address_city: str = Field(description="Delivery city")
    delivery_address_postal_code: str = Field(description="Delivery postal code")
    products: List[Product] = Field(description="List of products in the order")


def convert_pdf_to_images(pdf_path: str) -> List[str]:
    pdf_document = fitz.open(pdf_path)
    base64_images = []

    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_bytes = pix.tobytes("png")
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        base64_images.append(img_base64)

    pdf_document.close()
    return base64_images


def create_extraction_prompt() -> str:
    template_dir = Path(__file__).parent
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template('extraction_prompt.jinja')
    return template.render()


def call_azure_openai_with_vision(email_text: str, pdf_path: str, expected_output: dict = None) -> dict:
    load_dotenv()

    api_key = os.getenv('AZURE_OPENAI_KEY')
    endpoint = os.getenv('AZURE_OPENAI_RESOURCE_URL')

    if not api_key or not endpoint:
        raise ValueError("Azure OpenAI credentials not found in .env file")

    client = AzureOpenAI(
        api_key=api_key,
        api_version="2024-08-01-preview",
        azure_endpoint=endpoint
    )

    pdf_images = convert_pdf_to_images(pdf_path)
    system_prompt = create_extraction_prompt()

    user_content = [
        {
            "type": "text",
            "text": f"**EMAIL:**\n\n{email_text}\n\n**Extract the order information from the email above and the PDF images below:**"
        }
    ]

    for i, img_base64 in enumerate(pdf_images):
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img_base64}",
                "detail": "high"
            }
        })

    schema = OrderExtraction.model_json_schema()

    def add_additional_properties_false(obj):
        if isinstance(obj, dict):
            if obj.get("type") == "object":
                obj["additionalProperties"] = False
            for value in obj.values():
                add_additional_properties_false(value)
        elif isinstance(obj, list):
            for item in obj:
                add_additional_properties_false(item)

    add_additional_properties_false(schema)

    api_params = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_content
            }
        ],
        "max_completion_tokens": 3000,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "order_extraction_schema",
                "schema": schema,
                "strict": True
            }
        }
    }

    response = client.chat.completions.create(**api_params)
    extracted_text = response.choices[0].message.content

    try:
        if not extracted_text or extracted_text.strip() == "":
            raise ValueError("Empty response from API")

        json_match = re.search(r'```json\s*(.*?)\s*```', extracted_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = extracted_text

        extracted_data = json.loads(json_str)

    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parsing failed: {extracted_text[:500] if extracted_text else 'EMPTY'}")
    except Exception as e:
        raise ValueError(f"Schema validation failed: {str(e)}")

    result = {
        "filename": os.path.basename(pdf_path),
        "pdf_path": pdf_path,
        "extracted_data": extracted_data,
        "expected_output": expected_output,
        "raw_response": extracted_text
    }

    return result


def run_langfuse_experiment(dataset_name: str = "email_order_extraction", pdfs_dir: str = None):
    load_dotenv()

    client = Langfuse(
        secret_key=os.getenv('LANGFUSE_SECRET_KEY').strip('"'),
        public_key=os.getenv('LANGFUSE_PUBLIC_KEY').strip('"'),
        host=os.getenv('LANGFUSE_BASE_URL').strip('"')
    )

    dataset = client.get_dataset(name=dataset_name)

    def extraction_task(item):
        filename = item.input['filename']
        email_text = item.input['email']
        expected_output = item.expected_output

        pdf_path = os.path.join(pdfs_dir, filename)

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        result = call_azure_openai_with_vision(
            email_text=email_text,
            pdf_path=pdf_path,
            expected_output=expected_output
        )

        return result['extracted_data']

    def exact_match_evaluator(output, expected_output, **kwargs):
        fields_to_compare = [
            'buyer_company_name', 'buyer_person_name', 'buyer_email_address',
            'order_number', 'order_date',
            'delivery_address_street', 'delivery_address_city', 'delivery_address_postal_code',
            'products'
        ]

        if expected_output is None or output is None:
            score = 0
        else:
            score = 1 if all(expected_output.get(f) == output.get(f) for f in fields_to_compare) else 0

        return Evaluation(
            name="exact_match",
            value=score,
            comment="1 if exact match, 0 if any differences"
        )

    def buyer_info_evaluator(output, expected_output, **kwargs):
        if not output or not expected_output:
            return Evaluation(name="buyer_info", value=0.0, comment="Missing output or expected output")

        buyer_fields = ['buyer_company_name', 'buyer_person_name', 'buyer_email_address']
        matches = sum(1 for f in buyer_fields if output.get(f) == expected_output.get(f))
        score = matches / len(buyer_fields)

        return Evaluation(
            name="buyer_info",
            value=score,
            comment=f"{matches}/{len(buyer_fields)} buyer fields match"
        )

    def order_info_evaluator(output, expected_output, **kwargs):
        if not output or not expected_output:
            return Evaluation(name="order_info", value=0.0, comment="Missing output or expected output")

        order_fields = ['order_number', 'order_date']
        matches = sum(1 for f in order_fields if output.get(f) == expected_output.get(f))
        score = matches / len(order_fields)

        return Evaluation(
            name="order_info",
            value=score,
            comment=f"{matches}/{len(order_fields)} order fields match"
        )

    def address_info_evaluator(output, expected_output, **kwargs):
        if not output or not expected_output:
            return Evaluation(name="address_info", value=0.0, comment="Missing output or expected output")

        address_fields = ['delivery_address_street', 'delivery_address_city', 'delivery_address_postal_code']
        matches = sum(1 for f in address_fields if output.get(f) == expected_output.get(f))
        score = matches / len(address_fields)

        return Evaluation(
            name="address_info",
            value=score,
            comment=f"{matches}/{len(address_fields)} address fields match"
        )

    def products_evaluator(output, expected_output, **kwargs):
        if not output or not expected_output:
            return Evaluation(name="products", value=0.0, comment="Missing output or expected output")

        output_products = output.get('products', [])
        expected_products = expected_output.get('products', [])

        score = 1.0 if output_products == expected_products else 0.0

        return Evaluation(
            name="products",
            value=score,
            comment=f"{len(output_products)} products extracted, {len(expected_products)} expected, match={score==1.0}"
        )

    def average_score(*, item_results, **kwargs):
        scores = [
            evaluation.value for result in item_results
            for evaluation in result.evaluations
            if evaluation.name in [
                "buyer_info",
                "order_info",
                "address_info",
                "products"
            ]
        ]

        if not scores:
            return Evaluation(name="avg_score", value=None, comment="No scores available")

        avg = sum(scores) / len(scores)

        return Evaluation(
            name="avg_score",
            value=avg,
            comment=f"Average score across all evaluators: {avg:.2%} ({len(scores)} total evaluations)"
        )

    dataset.run_experiment(
        name=f"{MODEL_NAME} Order Extraction",
        description="Extract order information from emails and PDF attachments",
        task=extraction_task,
        evaluators=[
            exact_match_evaluator,
            buyer_info_evaluator,
            order_info_evaluator,
            address_info_evaluator,
            products_evaluator
        ],
        run_evaluators=[
            average_score
        ],
        metadata={
            'model': MODEL_NAME,
            'approach': 'Extraction with strict JSON schema and Pydantic validation'
        }
    )

    client.flush()


if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent
    pdfs_directory = base_dir / "data" / "pdfs"

    run_langfuse_experiment(
        dataset_name="email_order_extraction",
        pdfs_dir=str(pdfs_directory)
    )
