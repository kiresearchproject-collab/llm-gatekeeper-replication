# LLMs as the Gatekeeper: Replication Materials

Replication materials for "LLMs as the Gatekeeper: How Product Type Determines Opposite Persuasion Effects Across Language Models"

## Overview

This repository contains the experimental data, analysis code, and complete prompt templates used in the study examining how large language models respond to persuasive marketing techniques across different product types.

**Key Finding:** Identical persuasive marketing content produces opposite effects depending on product type, amplifying recommendation certainty for hedonic products while reducing it for utilitarian products.

## Repository Structure

```
llm-gatekeeper-replication/
├── README.md
├── requirements.txt
├── data/
│   └── llm_gatekeeper_dataset.csv      # Full dataset (13,500 observations)
├── code/
│   ├── experiment_runner.py             # Experiment execution code
│   └── statistical_analysis.py          # Statistical analysis script
└── prompts/
    └── experimental_conditions.md       # All 30 prompt templates
```

## Dataset

The dataset contains 13,500 observations from a 5×6×3 factorial experiment:

- **5 Conditions:** Control, Authority, Social Proof, Scarcity, Reciprocity
- **6 Products:** 3 Utilitarian (laptop, mobile plan, software) + 3 Hedonic (concert tickets, spa retreat, wine tasting)
- **3 Models:** GPT-4.1 Mini, GPT-5 Mini, Kimi K2
- **150 trials** per unique condition

### Variables

| Variable | Description |
|----------|-------------|
| `trial_id` | Unique trial identifier |
| `product_id` | Product identifier |
| `category` | Product category (hedonic/utilitarian) |
| `influence_condition` | Experimental condition |
| `model_name` | LLM used |
| `recommendation` | Binary recommendation (True/False) |
| `certainty` | Certainty rating (1-10 scale) |
| `response_text` | LLM reasoning text |
| `reasoning_length` | Word count of reasoning |
| `timestamp` | Trial timestamp |

## Replication

### Requirements

```bash
pip install pandas numpy scipy statsmodels aiohttp
```

### Running the Analysis

To replicate the statistical analysis reported in the paper:

```bash
python code/statistical_analysis.py
```

This will output all statistics reported in Section 4 of the paper.

### Running the Experiment

To re-run the experiment (requires OpenRouter API key):

```bash
python code/experiment_runner.py
```

You will be prompted to enter your API key. The experiment generates approximately 13,500 API calls.

## Key Results Summary

| Effect | Hedonic | Utilitarian |
|--------|---------|-------------|
| Authority | +0.191*** | -0.106*** |
| Social Proof | +0.345*** | -0.035 ns |
| Scarcity | +0.000 ns | -0.084** |
| Reciprocity | +0.109*** | +0.054 ns |

\*\*\* p < 0.001, \*\* p < 0.01, ns = not significant

## Citation

```bibtex
@article{pescher2025llmgatekeeper,
  title={LLMs as the Gatekeeper: How Product Type Determines Opposite Persuasion Effects Across Language Models},
  author={Pescher, Andrea},
  journal={Marketing Science},
  year={2025}
}
```

## Contact

Andrea Pescher

## Acknowledgments

Data collection was conducted using the OpenRouter API platform.
