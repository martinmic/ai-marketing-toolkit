#!/usr/bin/env python3
import sys
import os
import pandas as pd
import argparse

# This allows the script to find 'product_first_lead.py' in the root folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../research_agent')))

try:
    from product_first_lead import research_contact
except ImportError:
    print("Error: Could not find 'product_first_lead.py' in the root directory.")
    sys.exit(1)

def process_fda_list(file_path, limit=None):
    try:
        df = pd.read_csv(file_path, sep='\t', encoding='latin1')
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    initial_count = len(df)
    df = df[df['COUNTRY_CODE'] == 'US']
    filtered_count = initial_count - len(df)
    
    print(f"Loaded {initial_count} records. Skipped {filtered_count} non-US entries.")

    if limit:
        df = df.head(limit)
        print(f"Limiting processing to first {limit} entries.")

    results = []

    for _, row in df.iterrows():
        company = row['APPLICANT']
        person = row['CONTACT']
        product = row['DEVICENAME']
        knumber = row['KNUMBER']

        print(f"\nResearching: {person} ({company})")
        print(f"   Product: {product} (FDA ID: {knumber})")

        try:
            result = research_contact(
                name=person, 
                company=company, 
                product=product, 
                verbose=False
            )

            if result and result.email_found:
                print(f"   SUCCESS: {result.email_address}")
                results.append({
                    "knumber": knumber,
                    "applicant": company,
                    "contact": person,
                    "email": result.email_address,
                    "source": result.source_url
                })
            else:
                print(f"   NOT FOUND")

        except Exception as e:
            print(f"   ⚠️ ERROR processing this entry: {e}")

    print("\n" + "="*40)
    print(f"BATCH COMPLETE: Found {len(results)} of {len(df)} attempts.")
    print("="*40)

    # Optional: Save results to a CSV
    if results:
        out_df = pd.DataFrame(results)
        out_df.to_csv("fda_research_results.csv", index=False)
        print("Results saved to fda_research_results.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FDA 510(K) Contact Research Batch Tool")
    parser.add_argument("--input", default="sample_input.txt", help="Path to tab-delimited FDA data")
    parser.add_argument("--limit", type=int, default=10, help="Number of records to process (default: 10)")

    args = parser.parse_args()

    process_fda_list(file_path=args.input, limit=args.limit)

