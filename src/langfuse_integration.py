import os
import json
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from langfuse import Langfuse


def init_langfuse() -> Langfuse:
    load_dotenv()

    secret_key = os.getenv('LANGFUSE_SECRET_KEY')
    public_key = os.getenv('LANGFUSE_PUBLIC_KEY')
    base_url = os.getenv('LANGFUSE_BASE_URL')

    if not all([secret_key, public_key, base_url]):
        raise ValueError("Langfuse credentials not found in .env file")

    secret_key = secret_key.strip('"')
    public_key = public_key.strip('"')
    base_url = base_url.strip('"')

    client = Langfuse(
        secret_key=secret_key,
        public_key=public_key,
        host=base_url
    )

    return client


def create_langfuse_dataset(csv_path: str, dataset_name: str = "email_order_extraction") -> None:
    client = init_langfuse()

    df = pd.read_csv(csv_path)

    client.create_dataset(name=dataset_name)

    for idx, row in df.iterrows():
        expected_output = json.loads(row['expected_output']) if pd.notna(row['expected_output']) else None

        input_data = {
            "filename": row['filename'],
            "email": row['email']
        }

        client.create_dataset_item(
            dataset_name=dataset_name,
            input=input_data,
            expected_output=expected_output
        )


if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent
    csv_file = base_dir / "data" / "matched_emails_output.csv"

    create_langfuse_dataset(
        csv_path=str(csv_file),
        dataset_name="email_order_extraction"
    )
