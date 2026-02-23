#!/usr/bin/env python3
"""
product_first_lead.py

Uses a dual-model AI pipeline (Gemini + OpenAI) to find verified contact
info for stakeholders based on Name, Product, and Company inputs.
"""

import os
import time
import threading
from openai import OpenAI
from queue import Queue
from typing import Any, Callable, Optional
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
GENAI_THINKING_LEVEL = "minimal" 
MAX_GEMINI_RETRIES = 3
DEFAULT_LLM_TIMEOUT_SECONDS = 60
DEFAULT_LLM_MAX_RETRIES = MAX_GEMINI_RETRIES

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
- regulatory filings and public datasets
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

class LLMCallTimeoutError(TimeoutError):
    pass


def is_timeout_error(exc: Exception) -> bool:
    error_text = f"{type(exc).__name__}: {exc}".lower()
    return isinstance(exc, TimeoutError) or any(
        token in error_text for token in ("timeout", "timed out", "deadline exceeded", "504")
    )


def run_with_timeout(fn: Callable[[], Any], timeout_seconds: float):
    result_queue = Queue(maxsize=1)

    def target():
        try:
            result_queue.put(("ok", fn()))
        except Exception as exc:
            result_queue.put(("err", exc))

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout_seconds)

    if thread.is_alive():
        raise LLMCallTimeoutError(f"LLM call exceeded timeout of {timeout_seconds}s")

    status, payload = result_queue.get()
    if status == "err":
        raise payload
    return payload


def call_with_timeout_retries(
    call_name: str,
    fn: Callable[[], Any],
    timeout_seconds: float,
    max_retries: int,
    mode: Optional[str] = None,
):
    max_attempts = max(0, int(max_retries)) + 1

    for attempt in range(1, max_attempts + 1):
        try:
            return run_with_timeout(fn, timeout_seconds)
        except Exception as exc:
            if not is_timeout_error(exc) or attempt >= max_attempts:
                raise
            if mode:
                logger.warning(
                    "%s timeout on attempt %s/%s for mode '%s'; retrying.",
                    call_name,
                    attempt,
                    max_attempts,
                    mode,
                )
            else:
                logger.warning(
                    "%s timeout on attempt %s/%s; retrying.",
                    call_name,
                    attempt,
                    max_attempts,
                )

    raise RuntimeError(f"{call_name} retry loop exited unexpectedly")


def call_gemini(
    query: str,
    mode: str,
    llm_timeout_seconds: float = DEFAULT_LLM_TIMEOUT_SECONDS,
    llm_max_retries: int = DEFAULT_LLM_MAX_RETRIES,
) -> str:
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

    def do_call():
        response = client_google.models.generate_content(
            model="gemini-3-flash-preview",
            contents=query,
            config=config
        )
        return response.text

    return call_with_timeout_retries(
        call_name="Gemini",
        fn=do_call,
        timeout_seconds=llm_timeout_seconds,
        max_retries=llm_max_retries,
        mode=mode,
    )


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

def call_gpt_for_evaluation(
    query: str,
    raw_text: str,
    llm_timeout_seconds: float = DEFAULT_LLM_TIMEOUT_SECONDS,
    llm_max_retries: int = DEFAULT_LLM_MAX_RETRIES,
) -> EmailExtraction:
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

    def do_call():
        response = client_openai.responses.parse(
            model="gpt-4.1",
            instructions=GPT_SYSTEM_INSTRUCTIONS,
            input=evaluation_input,
            text_format=EmailExtraction
        )
        return response.output_parsed

    return call_with_timeout_retries(
        call_name="GPT",
        fn=do_call,
        timeout_seconds=llm_timeout_seconds,
        max_retries=llm_max_retries,
    )


# -----------------------------
# Controller (orchestration)
# -----------------------------

def research_contact(
    name,
    company,
    product,
    verbose=False,
    return_timing=False,
    llm_timeout_seconds=DEFAULT_LLM_TIMEOUT_SECONDS,
    llm_max_retries=DEFAULT_LLM_MAX_RETRIES,
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"logs/{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    logger.info(f"NEW RESEARCH: Target='{name}', Company='{company}', Product='{product}'")
    total_start = time.perf_counter()
    timing_data = {
        "total_s": 0.0,
        "modes": {}
    }

    def finalize(result):
        timing_data["total_s"] = time.perf_counter() - total_start
        if return_timing:
            return result, timing_data
        return result

    modes = ["name_company", "product_company", "company_fallback"]
    os.makedirs("logs", exist_ok=True)

    for mode in modes:
        if verbose: print(f"Researching via mode: {mode} ({run_dir})")
        if mode == "name_company":
            query = f"Name: {name}, Company: {company}"
        elif mode == "product_company":
            if not product:
                timing_data["modes"][mode] = {"skipped": True}
                continue  # skip product mode if no product was provided
            query = f"Product: {product}, Company: {company}"
        else:  # company_fallback
            query = f"Company: {company}"

        gemini_start = time.perf_counter()
        raw_text = call_gemini(
            query=query,
            mode=mode,
            llm_timeout_seconds=llm_timeout_seconds,
            llm_max_retries=llm_max_retries,
        )
        gemini_duration_s = time.perf_counter() - gemini_start

        # write Gemini raw output to a mode-specific file
        with open(f"{run_dir}/gemini_{mode}.txt", "w", encoding="utf-8") as f:
            f.write(raw_text or "")

        gpt_start = time.perf_counter()
        response = call_gpt_for_evaluation(
            query=query,
            raw_text=raw_text,
            llm_timeout_seconds=llm_timeout_seconds,
            llm_max_retries=llm_max_retries,
        )
        gpt_duration_s = time.perf_counter() - gpt_start
        timing_data["modes"][mode] = {
            "gemini_s": gemini_duration_s,
            "gpt_s": gpt_duration_s
        }

        log_msg = f"EVAL: Mode={mode} | Found={response.email_found} | Email={response.email_address}"
        logger.info(log_msg)

        if response.email_found:
            with open(f"{run_dir}/final_result.json", "w", encoding="utf-8") as f:
                f.write(response.model_dump_json(indent=2))

            logger.info(f"SUCCESS: Lead found in mode '{mode}' ({run_dir})")
            return finalize(response)

    logger.warning(f"FAILURE: No contact found for '{name}' ({run_dir})")
    return finalize(None)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Product-First Lead Discovery Agent")

    parser.add_argument("--name", required=True, help="Target's full name")
    parser.add_argument("--company", required=True, help="Target's company")
    parser.add_argument("--product", required=False, help="Associated product (optional)")
    parser.add_argument(
        "--llm-timeout-seconds",
        type=float,
        default=DEFAULT_LLM_TIMEOUT_SECONDS,
        help="Max seconds per Gemini/GPT call before timeout retry is triggered.",
    )
    parser.add_argument(
        "--llm-max-retries",
        type=int,
        default=DEFAULT_LLM_MAX_RETRIES,
        help="Number of retries per Gemini/GPT call after timeout.",
    )

    args = parser.parse_args()
    
    result = research_contact(
        name=args.name,
        company=args.company,
        product=args.product,
        verbose=True,
        llm_timeout_seconds=args.llm_timeout_seconds,
        llm_max_retries=args.llm_max_retries,
    )

    if result and result.email_found:
        print(f"Discovery Successful!")
        print(f"Email:  {result.email_address}")
        print(f"Source: {result.source_url}")
    else:
        print("No verified contact info found.")
        exit(1)
