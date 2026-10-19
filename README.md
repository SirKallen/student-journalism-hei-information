# Student Journalism and the Informational Role of Higher Education Institutions

This repository contains code, prompts, and supporting materials for the study:

**Student Journalism and the Informational Role of Higher Education Institutions: A Rural–Urban Comparative Analysis**

## Overview

This study examines whether differences across rural and urban higher education institutions are reflected in the informational characteristics of HEI-affiliated student journalism.

The analysis compares three dimensions of student-produced news:

- Textual distinctiveness
- Critical Information Needs (CIN) coverage
- Localness

The study analyzes approximately 13,000 articles from more than 800 student journalism outlets.

## Methods

The study uses:

- Soft cosine similarity to evaluate textual distinctiveness
- LLM-assisted classification for CIN coverage and localness
- Human validation of LLM classifications
- Logistic regression with outlet-clustered standard errors
- Bootstrap analysis for pairwise textual similarity

## Repository Contents

- `code/` — scripts used for data processing, classification, statistical analysis, and figure/table generation
- `prompts/` — LLM prompts, codebooks, and classification instructions
- `results/` — selected tables, figures, and non-sensitive aggregate outputs
- `manuscript/` — manuscript or preprint materials
- `requirements.txt` — software dependencies

## Data Availability

The underlying article-level dataset is not included in this repository.

The dataset was externally constructed and contains full-text student journalism articles and related metadata. Because these materials cannot be redistributed through this repository, users will need to obtain access to the original data source independently.

The code in this repository is provided to document the analytical workflow and support methodological transparency.

## Reproducibility

Because the underlying article-level data are not publicly distributed here, the full analysis cannot be reproduced from this repository alone.

However, the repository provides:

- data-processing procedures
- classification code
- LLM prompt structures
- coding frameworks
- statistical analysis scripts
- figure and table generation procedures

These materials are intended to make the computational workflow as transparent as possible while respecting data-sharing restrictions.

## Citation

Citation information will be added upon publication.
