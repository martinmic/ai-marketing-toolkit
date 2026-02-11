<h1 align="center">AI Marketing Toolkit</h1>

<p align="center">
 The fastest way to research leads.
</p>

<p align="center">
  <a href="#overview"><strong>Overview</strong></a> ·
  <a href="#quick-start"><strong>Quick Start</strong></a> ·
  <a href="#tools-description"><strong>Tools Description</strong></a> ·
  <a href="#prerequisites"><strong>Prerequisites</strong></a>
</p>
<br/>

## Overview

A collection of high-accuracy research and automation utilities designed for modern Marketing Operations.

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/martinmic/ai-marketing-toolkit.git
   cd ai-marketing-toolkit
   ```

2. Install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Rename `.env.example` to `.env` and update the following:
   ```env
   OPENAI_API_KEY=[INSERT OPENAI API KEY]
   GEMINI_API_KEY=[INSERT GOOGLE AI STUDIO API KEY]
   ```

4. Run a simple test from the CLI:
   ```bash
   ./product_first_lead.py --name "first last" --company "name"
   ```


## Tools Description

### product_first_lead.py
This script separates discovery from cognition to solve the common problem of LLM hallucinations in market research. Its purpose is to identify the specific human responsible for a product's success.

### run_fda_batch.py
Automates batch execution of `product_first_lead.py`. It ingests an input file containing person names, company names, and the associated FDA‑submitted product, then runs the research workflow for each record.


See [research_agent/README.md](research_agent/README.md) for more details.

## Prerequisites

To run these utilities, you will need **Python 3.9+** and active API keys from the following providers. 

1. OpenAI API Key
   - Setup: Create an account at [OpenAI Platform](https://platform.openai.com/).
   - Billing: OpenAI requires pre-paid credits to use their API. You must navigate to Settings (little gear icon in the top right corner). Then, click on Billing in the left menu and "Add to credit balance".
   
2. Google Gemini API Key
   - Setup: Create an account at [Google AI Studio](https://aistudio.google.com/).
   - Billing: Google offers a "Pay-as-you-go" model. You must set up an active billing account for real-time usage.
