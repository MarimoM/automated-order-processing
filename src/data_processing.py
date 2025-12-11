import json
import re
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd


def parse_expected_output_to_json(input_file: str, output_file: str) -> None:
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    records = []
    current_record = {}
    current_products = []

    lines = content.strip().split('\n')

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line == "Buyer:":
            if current_record:
                if current_products:
                    current_record['products'] = current_products
                records.append(current_record)
                current_record = {}
                current_products = []
            i += 1
            continue

        if line == "Order:":
            i += 1
            continue

        if line in ["Product:", "Products:"]:
            i += 1
            continue

        if line.startswith('•'):
            field_line = line[1:].strip()
            if ':' in field_line:
                key, value = field_line.split(':', 1)
                key = key.strip()
                value = value.strip()

                if key == 'position':
                    current_products.append({'position': int(value)})
                elif key in ['article_code', 'quantity']:
                    if current_products:
                        if key == 'quantity':
                            current_products[-1][key] = int(value)
                        else:
                            current_products[-1][key] = value
                else:
                    current_record[key] = value

        i += 1

    if current_record:
        if current_products:
            current_record['products'] = current_products
        records.append(current_record)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)


def parse_emails_and_match(emails_file: str, expected_output_file: str) -> List[Dict[str, Any]]:
    with open(expected_output_file, 'r', encoding='utf-8') as f:
        expected_outputs = json.load(f)

    order_lookup = {record['order_number']: record for record in expected_outputs}

    with open(emails_file, 'r', encoding='utf-8') as f:
        emails_content = f.read()

    email_blocks = []
    current_email = []

    for line in emails_content.split('\n'):
        if line.startswith('attachment:'):
            if current_email:
                email_blocks.append('\n'.join(current_email))
                current_email = []
            current_email.append(line)
        else:
            current_email.append(line)

    if current_email:
        email_blocks.append('\n'.join(current_email))

    matched_data = []

    for email_block in email_blocks:
        attachment_match = re.search(r'attachment:\s*(.+?)(?:\n|$)', email_block)
        attachment_filename = attachment_match.group(1).strip() if attachment_match else None

        if attachment_filename:
            attachment_filename = re.sub(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}:', '', attachment_filename)

        email_content_clean = re.sub(r'^attachment:.*?\n', '', email_block, flags=re.MULTILINE).strip()

        email_data = {
            'email_content': email_content_clean,
            'attachment': attachment_filename,
            'sender_email': None,
            'sender_name': None,
            'order_number': None,
            'expected_output': None
        }

        sender_match = re.search(r'Von:\s*(.+?)\s+([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', email_block)
        if sender_match:
            email_data['sender_name'] = sender_match.group(1).strip()
            email_data['sender_email'] = sender_match.group(2).strip()

        order_patterns = [
            r'Bestellung\s+BT\s+(\d+)',
            r'Hofbauer[_-](\d+)',
            r'Bestellung\s+(?:Nr\.\s+)?(\S+)',
            r'order[_\s]+number[:\s]+(\S+)',
        ]

        for pattern in order_patterns:
            order_match = re.search(pattern, email_block, re.IGNORECASE)
            if order_match:
                email_data['order_number'] = order_match.group(1).strip()
                break

        if email_data['order_number'] and email_data['order_number'] in order_lookup:
            email_data['expected_output'] = order_lookup[email_data['order_number']]

        matched_data.append(email_data)

    return matched_data


def create_dataframe_and_save(matched_data: List[Dict[str, Any]], output_file: str) -> None:
    rows = []
    for item in matched_data:
        row = {
            'filename': item['attachment'],
            'email': item['email_content'],
            'expected_output': json.dumps(item['expected_output'], ensure_ascii=False) if item['expected_output'] else None
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False, encoding='utf-8')


if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent
    input_file = base_dir / "data" / "expected_output.txt"
    output_json = base_dir / "data" / "expected_output.json"
    emails_file = base_dir / "data" / "emails.txt"
    matched_csv = base_dir / "data" / "matched_emails_output.csv"

    parse_expected_output_to_json(str(input_file), str(output_json))
    matched_data = parse_emails_and_match(str(emails_file), str(output_json))
    create_dataframe_and_save(matched_data, str(matched_csv))
