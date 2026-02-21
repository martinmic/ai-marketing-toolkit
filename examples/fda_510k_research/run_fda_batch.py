#!/usr/bin/env python3
import sys
import os
import pandas as pd
import argparse
import requests
from dotenv import load_dotenv

# This allows the script to find 'product_first_lead.py' in the root folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../research_agent')))

try:
    from product_first_lead import research_contact
except ImportError:
    print("Error: Could not find 'product_first_lead.py' in the root directory.")
    sys.exit(1)

load_dotenv()


def format_timing_line(timing_data):
    mode_names = ["name_company", "product_company", "company_fallback"]
    parts = [f"total={timing_data.get('total_s', 0.0):.2f}s"]
    modes = timing_data.get("modes", {})

    for mode in mode_names:
        mode_data = modes.get(mode)
        if not mode_data:
            parts.append(f"{mode}(skipped)")
            continue
        if mode_data.get("skipped"):
            parts.append(f"{mode}(skipped)")
            continue
        gemini_s = mode_data.get("gemini_s", 0.0)
        gpt_s = mode_data.get("gpt_s", 0.0)
        parts.append(f"{mode}(gemini={gemini_s:.2f}s,gpt={gpt_s:.2f}s)")

    return "Timing: " + " | ".join(parts)


def normalize_fda_id(value):
    if pd.isna(value):
        return None
    normalized = str(value).strip()
    return normalized or None


def fetch_existing_fda_ids(table_name, supabase_url=None, service_role_key=None):
    supabase_url = supabase_url or os.getenv("SUPABASE_URL")
    service_role_key = service_role_key or os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not supabase_url or not service_role_key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set for Supabase output."
        )

    endpoint = f"{supabase_url.rstrip('/')}/rest/v1/{table_name}"
    headers = {
        "apikey": service_role_key,
        "Authorization": f"Bearer {service_role_key}",
        "Range-Unit": "items",
    }
    params = {
        "select": "fda_id",
        "fda_id": "not.is.null",
    }

    existing_ids = set()
    page_size = 1000
    offset = 0

    while True:
        page_headers = dict(headers)
        page_headers["Range"] = f"{offset}-{offset + page_size - 1}"

        response = requests.get(
            endpoint,
            params=params,
            headers=page_headers,
            timeout=30
        )
        response.raise_for_status()

        rows = response.json()
        if not rows:
            break

        for row in rows:
            fda_id = normalize_fda_id(row.get("fda_id"))
            if fda_id:
                existing_ids.add(fda_id)

        if len(rows) < page_size:
            break
        offset += page_size

    return existing_ids


def save_results_to_csv(results, output_path):
    out_df = pd.DataFrame(results)
    out_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


def build_supabase_insert_context(table_name, supabase_url=None, service_role_key=None):
    supabase_url = supabase_url or os.getenv("SUPABASE_URL")
    service_role_key = service_role_key or os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not supabase_url or not service_role_key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set for Supabase output."
        )

    endpoint = f"{supabase_url.rstrip('/')}/rest/v1/{table_name}"
    headers = {
        "apikey": service_role_key,
        "Authorization": f"Bearer {service_role_key}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }
    return endpoint, headers


def insert_result_to_supabase(row, endpoint, headers):
    payload = {
        "company": row["company"],
        "person": row["person"],
        "product": row["product"],
        "fda_id": row.get("fda_id") or row.get("knumber"),
        "email": row["email"],
        "source": row["source"],
    }

    try:
        response = requests.post(endpoint, json=payload, headers=headers, timeout=30)
    except requests.RequestException as exc:
        print(
            f"   WARNING: Request failed for {row['person']} at {row['company']}: {exc}"
        )
        return "failed"

    if response.status_code in (200, 201, 204):
        return "inserted"

    if response.status_code == 409:
        return "duplicate"

    print(
        f"   WARNING: Failed to insert row for {row['person']} at {row['company']}. "
        f"Status {response.status_code}: {response.text}"
    )
    return "failed"


def process_fda_list(
    file_path,
    limit=None,
    output_mode="csv",
    output_path="fda_research_results.csv",
    table_name="fda510k_leads",
    supabase_url=None,
    service_role_key=None,
    test_mode=False,
):
    try:
        df = pd.read_csv(file_path, sep='\t', encoding='latin1')
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    initial_count = len(df)
    df = df[df['COUNTRY_CODE'] == 'US']
    filtered_count = initial_count - len(df)
    
    print(f"Loaded {initial_count} records. Skipped {filtered_count} non-US entries.")
    if test_mode:
        print("Running in test mode: skipping research_contact and using placeholder@email.com")

    if limit is not None:
        df = df.head(limit)
        print(f"Limiting processing to first {limit} entries.")

    existing_fda_ids = set()
    supabase_endpoint = None
    supabase_headers = None
    if output_mode == "supabase":
        try:
            existing_fda_ids = fetch_existing_fda_ids(
                table_name=table_name,
                supabase_url=supabase_url,
                service_role_key=service_role_key,
            )
            supabase_endpoint, supabase_headers = build_supabase_insert_context(
                table_name=table_name,
                supabase_url=supabase_url,
                service_role_key=service_role_key,
            )
            print(f"Loaded {len(existing_fda_ids)} existing fda_id values from Supabase.")
        except Exception as e:
            print(f"Error loading existing fda_id values: {e}")
            return

    results = []
    found_count = 0
    skipped_existing = 0
    inserted = 0
    duplicates = 0
    failed = 0
    timing_rows = []

    for _, row in df.iterrows():
        company = row['APPLICANT']
        person = row['CONTACT']
        product = row['DEVICENAME']
        knumber = normalize_fda_id(row['KNUMBER'])

        if output_mode == "supabase" and knumber and knumber in existing_fda_ids:
            skipped_existing += 1
            print(f"\nSkipping: {person} ({company}) - already processed (FDA ID: {knumber})")
            continue

        print(f"\nResearching: {person} ({company})")
        print(f"   Product: {product} (FDA ID: {knumber})")

        try:
            if test_mode:
                class _TestResult:
                    email_found = True
                    email_address = "placeholder@email.com"
                    source_url = "test_mode"

                result = _TestResult()
                timing_data = {"total_s": 0.0, "modes": {}}
            else:
                result, timing_data = research_contact(
                    name=person, 
                    company=company, 
                    product=product, 
                    verbose=False,
                    return_timing=True,
                )
            print(f"   {format_timing_line(timing_data)}")
            timing_rows.append({"fda_id": knumber, "timing": timing_data})

            if result and result.email_found:
                found_count += 1
                print(f"   SUCCESS: {result.email_address}")
                result_row = {
                    "knumber": knumber,
                    "fda_id": knumber,
                    "company": company,
                    "person": person,
                    "product": product,
                    "email": result.email_address,
                    "source": result.source_url or "unknown"
                }
                if output_mode == "csv":
                    results.append(result_row)
                else:
                    write_status = insert_result_to_supabase(
                        row=result_row,
                        endpoint=supabase_endpoint,
                        headers=supabase_headers,
                    )
                    if write_status == "inserted":
                        inserted += 1
                        if knumber:
                            existing_fda_ids.add(knumber)
                    elif write_status == "duplicate":
                        duplicates += 1
                        if knumber:
                            existing_fda_ids.add(knumber)
                    else:
                        failed += 1
            else:
                print(f"   NOT FOUND")

        except Exception as e:
            print(f"   WARNING: ERROR processing this entry: {e}")

    print("\n" + "="*40)
    print(f"BATCH COMPLETE: Found {found_count} of {len(df)} attempts.")
    if output_mode == "supabase":
        print(f"Skipped existing in DB: {skipped_existing}")
        print(
            f"Supabase write complete. Inserted: {inserted}, duplicates skipped: {duplicates}, failed: {failed}"
        )
    if timing_rows:
        total_values = [item["timing"].get("total_s", 0.0) for item in timing_rows]
        gemini_values = []
        gpt_values = []
        for item in timing_rows:
            for mode_data in item["timing"].get("modes", {}).values():
                if mode_data.get("skipped"):
                    continue
                gemini_values.append(mode_data.get("gemini_s", 0.0))
                gpt_values.append(mode_data.get("gpt_s", 0.0))

        slowest_row = max(timing_rows, key=lambda item: item["timing"].get("total_s", 0.0))
        avg_total = sum(total_values) / len(total_values) if total_values else 0.0
        avg_gemini = sum(gemini_values) / len(gemini_values) if gemini_values else 0.0
        avg_gpt = sum(gpt_values) / len(gpt_values) if gpt_values else 0.0
        slowest_id = slowest_row["fda_id"] or "unknown"
        slowest_s = slowest_row["timing"].get("total_s", 0.0)
        print(
            f"Timing Summary: contacts={len(timing_rows)} | avg_total={avg_total:.2f}s | "
            f"avg_gemini={avg_gemini:.2f}s | avg_gpt={avg_gpt:.2f}s | "
            f"slowest_fda_id={slowest_id} ({slowest_s:.2f}s)"
        )
    print("="*40)

    if output_mode == "csv" and results:
        save_results_to_csv(results, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FDA 510(K) Contact Research Batch Tool")
    parser.add_argument("--input", default="sample_input.txt", help="Path to tab-delimited FDA data")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of records to process (default: no limit)",
    )
    parser.add_argument(
        "--output-mode",
        choices=["csv", "supabase"],
        default="csv",
        help="Where to save output (default: csv)",
    )
    parser.add_argument(
        "--output-path",
        default="fda_research_results.csv",
        help="CSV output path when --output-mode=csv",
    )
    parser.add_argument(
        "--supabase-table",
        default="fda510k_leads",
        help="Supabase table name when --output-mode=supabase",
    )
    parser.add_argument(
        "--supabase-url",
        default=None,
        help="Supabase project URL (overrides SUPABASE_URL env var)",
    )
    parser.add_argument(
        "--supabase-service-role-key",
        default=None,
        help="Supabase service role key (overrides SUPABASE_SERVICE_ROLE_KEY env var)",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Skip AI research and use placeholder@email.com for fast pipeline testing.",
    )

    args = parser.parse_args()

    process_fda_list(
        file_path=args.input,
        limit=args.limit,
        output_mode=args.output_mode,
        output_path=args.output_path,
        table_name=args.supabase_table,
        supabase_url=args.supabase_url,
        service_role_key=args.supabase_service_role_key,
        test_mode=args.test_mode,
    )
