#!/usr/bin/env python3
"""
product_first_lead.py

Uses a dual-model AI pipeline (Gemini + OpenAI) to find verified contact
info for stakeholders based on Name, Product, and Company inputs.
"""

# TODO: Add Supabase integration to store verified stakeholders
# TODO: Implement retry logic if Gemini times out using MAX_GEMINI_RETRIES

import os
from openai import OpenAI
from typing import Optional
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# ensure the user actually set up their .env
if not os.getenv("GEMINI_API_KEY") or not os.getenv("OPENAI_API_KEY"):
    print("Error: Missing API keys in .env file.")
    print("Please ensure GEMINI_API_KEY and OPENAI_API_KEY are set.")
    exit(1)

client_google = genai.Client()
client_openai = OpenAI()

# -----------------------------
# Configuration placeholders
# -----------------------------

# Gemini 3 Flash Thinking Level (adjust as needed)
# - available options: minimal/low/medium/high
# - for more info visit: https://ai.google.dev/gemini-api/docs/thinking
GENAI_THINKING_LEVEL = "medium" 

# -----------------------------
# Logging setup
# -----------------------------

import logging
from datetime import datetime

logging.basicConfig(
    filename='research.log',
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# -----------------------------
# Gemini system instructions
# -----------------------------

COMMON_RULES = """
You are a verification-focused web research assistant.

Rules (strict):
- ONLY include email addresses that appear verbatim in a public source.
- Each email MUST have a source URL where it appears.
- DO NOT guess, infer, construct, or suggest email formats.
- DO NOT include speculative, likely, or pattern-based emails.
- If an email does not appear verbatim in a source, it MUST NOT be included.
"""

COMMON_TASK = """
Search beyond the company website, including:
- academic papers
- NIH / PMC articles
- FDA documents
- patents
- conference papers
Use queries that may include the string "Email:", "gmail" and the provided company name.
"""

GEMINI_SYSTEM_INSTRUCTIONS = {
    "name_company": f"""

{COMMON_RULES}

Task:
Perform a broad web search for public, explicitly published email addresses related to the provided name and company.

{COMMON_TASK}

Input format:
first_name last_name, company_name

    """,
    "product_company": f"""

{COMMON_RULES}

Task:
Perform a broad web search to identify appropriate, explicitly published email addresses of persons or departments responsible for product marketing, communications, or business development of the specified product.

{COMMON_TASK}

Input format:
product_name, company_name

    """,
    "company_fallback": f"""

{COMMON_RULES}

Task:
Perform a broad web search to identify appropriate, explicitly published email addresses of persons or departments responsible for product marketing, communications, or business development at the specified company.

{COMMON_TASK}

Input format:
company_name

    """
}

# -----------------------------
# GPT system instructions
# -----------------------------

GPT_SYSTEM_INSTRUCTIONS = """
You are an evaluation and decision-making assistant.

You are given raw text produced by a Gemini model that performed a web search
to find publicly published email addresses based on:
- a person’s name and company, and/or
- a product name and company.

The Gemini model was explicitly instructed to:
- include ONLY email addresses that appear verbatim in public sources
- include a source URL for each email address it reports

Your task:
- Analyze the provided text
- Determine whether a valid, usable email address is present
- Prefer emails that are clearly associated with the intended person, role, product, or company
- Reject emails that are ambiguous, unsupported, or lack a clear source URL

Your output MUST conform to the given structure.

Rules:
- If no valid email with a clear source URL is present, return false.
- Do NOT invent, guess, or modify email addresses.
- Do NOT include commentary, explanations, or extra fields.
"""


# -----------------------------
# Gemini interface (web research)
# -----------------------------

def call_gemini(query: str, mode: str) -> str:
    """
    Executes a web-discovery search using Google Gemini to surface contact data.

    Args:
        query: The search intent string.
        mode: The strategy key used to select specific system instructions.
    """
    grounding_tool = types.Tool(
        google_search=types.GoogleSearch()
    )

    config = types.GenerateContentConfig(
        tools=[grounding_tool],
        temperature=0.1, # forcing it to be factual, accurate, and tied to external, verifiable information
        system_instruction=GEMINI_SYSTEM_INSTRUCTIONS[mode],
        thinking_config=types.ThinkingConfig(thinking_level=GENAI_THINKING_LEVEL)
    )

    response = client_google.models.generate_content(
        model="gemini-3-flash-preview",
        contents=query,
        config=config
    )

    return response.text


# -----------------------------
# GPT interface (Gemini result evaluator)
# -----------------------------

class EmailExtraction(BaseModel):
    """
    Data schema for contact information extracted by the LLM.

    Attributes:
        email_found (bool): Indicates if a valid email was found or not.
        email_address (str): The discovered email address (null if not found).
        source_url (str): The specific URL where the contact info was discovered (null if not found).
    """
    email_found: bool
    email_address: Optional[str] = None
    source_url: Optional[str] = None

def call_gpt_for_evaluation(query: str, raw_text: str) -> EmailExtraction:
    """
    Performs a cognitive evaluation of raw web-search data to extract
    and validate structured contact information.

    Uses OpenAI's Structured Outputs to ensure the
    response adheres to the EmailExtraction schema.

    Args:
        query: The original research intent used as the contextual anchor.
        raw_text: The unformatted search results from the discovery phase.
    """
    evaluation_input = f"""
        Target Context:
        ---
        {query}
        ---

        Gemini discovery results:
        ---
        {raw_text}
        ---
    """

    response = client_openai.responses.parse(
        model="gpt-4.1",
        instructions=GPT_SYSTEM_INSTRUCTIONS,
        input=evaluation_input,
        text_format=EmailExtraction
    )
    return response.output_parsed


# -----------------------------
# Controller (orchestration)
# -----------------------------

def research_contact(name, company, product, verbose=False):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"logs/{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    logger.info(f"NEW RESEARCH: Target='{name}', Company='{company}', Product='{product}'")

    modes = ["name_company", "product_company", "company_fallback"]
    os.makedirs("logs", exist_ok=True)

    for mode in modes:
        if verbose: print(f"Researching via mode: {mode} ({run_dir})")
        if mode == "name_company":
            query = f"Name: {name}, Company: {company}"
        elif mode == "product_company":
            if not product: continue # skip product mode if no product was provided
            query = f"Product: {product}, Company: {company}"
        else:  # company_fallback
            query = f"Company: {company}"

        raw_text = call_gemini(query, mode)

        # write Gemini raw output to a mode-specific file
        with open(f"{run_dir}/gemini_{mode}.txt", "w", encoding="utf-8") as f:
            f.write(raw_text or "")

        response = call_gpt_for_evaluation(query, raw_text)

        log_msg = f"EVAL: Mode={mode} | Found={response.email_found} | Email={response.email_address}"
        logger.info(log_msg)

        if response.email_found:
            with open(f"{run_dir}/final_result.json", "w", encoding="utf-8") as f:
                f.write(response.model_dump_json(indent=2))

            logger.info(f"SUCCESS: Lead found in mode '{mode}' ({run_dir})")
            return response

    logger.warning(f"FAILURE: No contact found for '{name}' ({run_dir})")
    return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Product-First Lead Discovery Agent")

    parser.add_argument("--name", required=True, help="Target's full name")
    parser.add_argument("--company", required=True, help="Target's company")
    parser.add_argument("--product", required=False, help="Associated product (optional)")

    args = parser.parse_args()
    
    result = research_contact(
        name=args.name,
        company=args.company,
        product=args.product,
        verbose=True
    )

    if result and result.email_found:
        print(f"Discovery Successful!")
        print(f"Email:  {result.email_address}")
        print(f"Source: {result.source_url}")
    else:
        print("No verified contact info found.")
        exit(1)


