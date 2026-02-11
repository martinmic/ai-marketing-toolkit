## Overview

This script separates discovery from cognition to solve the common problem of LLM hallucinations in market research. Its purpose is to identify the specific human responsible for a product's success.

- **Phase 1 (Discovery):** Utilizes Google Gemini's web-search capabilities to surface raw data and source URLs.
- **Phase 2 (Evaluation):** Passes discovery data to an OpenAI LLM for context-aware ranking, credibility scoring, and usability decision-making.

## Logic Flow
The agent employs a recursive fallback model:
1. **Direct Match:** Targeted search for a specific [Name] at the [Company].
2. **Product-Led Fallback:** Identifies the persona/stakeholder responsible for a specific [Product] at that organization.
3. **Domain-Wide Fallback:** Surfaces key technical or business decision-makers at the [Company].

## Configuration

You can tune the performance of the agent by modifying the variables at the top of `product_first_lead.py`:

- **`THINKING_LEVEL`**: Adjusts the reasoning depth of the Gemini model. 
  - `low`: Quick discovery for well-known entities.
  - `medium`: Balanced (default).
  - `high`: Deep research for obscure products or complex company structures.

## Testing with CLI

The Product-First Lead agent is designed for high-precision, automated discovery via the command line. By providing a target Name, Company, and Product, the utility executes a multi-stage research cycle. If a direct person-match isn't found, the engine automatically pivots to identifying the persona responsible for the specific product or business unit.

Run a single research task:

```bash
cd research_agent
./product_first_lead.py --name "James Tamplin" --company "Firebase"
```

Sample output:

```bash
Researching via mode: name_company (logs/20260211_132214)
Discovery Successful!
Email:  james@firebase.com
Source: https://www.thedomains.com/2015/10/13/willing-com-evolve-com-zeros-com-and-more-domain-movers/
```

Key Flags:

```bash
--name / --company: (Required) The core identity of your research target.
--product: (Optional) Enables the "Product-Led" fallback mode if the individual is not found.
--verbose: Enables the Discovery Log, providing real-time visibility into the Gemini search results and GPT reasoning process.
```

## Running a batch script

The FDA 510(K) Batch Tool demonstrates the toolkit’s capability to handle high-volume, industry-specific datasets. Designed to process the FDA’s premarket submission data, this script automates the ingestion of tab-delimited records, applies geographic filtering (US-only), and executes the recursive discovery engine at scale. It transforms a "messy" regulatory spreadsheet into a structured lead list, outputting results directly to a CSV for seamless integration into CRM or database workflows.

The data input files for this test are available for download from the FDA website. The examples/fda_510k_research folder contains a downloaded sample, renamed to sample_input.txt (this file is automatically picked up when the --input flag is omitted). Keep in mind these files are tab-delimited so when opening in Excel you need to define the tab as a delimiter.

Process the first 10 entries of the FDA dataset:

```bash
cd ai-marketing-toolkit/examples/fda_510k_research/
./run_fda_batch.py --limit 10 --verbose
```

Customization Options:

```bash
--limit [N]: Controls API usage by limiting the number of processed records (defaults to 10).
--input [510k_file]: Path to your tab-delimited dataset (defaults to sample_input.txt).
--verbose: Provides a live progress bar and detailed research status for each record in the batch.
```

## Log files

After running a single CLI command or a batch script two types of logs will be generated.

1. Main research log
```bash
$ tail research.log
2026-02-11 13:22:14 - INFO - NEW RESEARCH: Target='James Tamplin', Company='Firebase', Product='None'
2026-02-11 13:23:49 - INFO - EVAL: Mode=name_company | Found=True | Email=james@firebase.com
2026-02-11 13:23:49 - INFO - SUCCESS: Lead found in mode 'name_company' (logs/20260211_132214)
```

2. More detailed running logs
```bash
$ cd logs/20260211_132214
$ ls
final_result.json
gemini_name_company.txt
```

Note that the location of the running log will be printed inside the `research.log` as well as in the STDOUT of the `product_first_lead.py`.