# student-journalism-hei-information
This repository contains the code and supporting materials for the study:  Student Journalism and the Informational Role of Higher Education Institutions: A Rural–Urban Comparative Analysis

## Overview

This study examines whether differences across rural and urban higher education institutions are reflected in the informational characteristics of HEI-affiliated student journalism.

We compare three dimensions of student-produced news:

- Textual distinctiveness
- Critical Information Needs (CIN) coverage
- Localness

The analysis uses approximately 13,000 articles from more than 800 student journalism outlets.

## Methods

The study uses:

- Soft cosine similarity for textual distinctiveness
- LLM-assisted classification for CIN coverage and localness
- Human validation of LLM classifications
- Logistic regression with outlet-clustered standard errors
- Bootstrap analysis for pairwise textual similarity

## Repository Structure

- `code/` — data processing and statistical analysis scripts
- `prompts/` — LLM prompts and coding frameworks
- `data/` — processed data and data documentation
- `results/` — generated tables and figures
- `manuscript/` — manuscript or preprint

## Data

The underlying student journalism dataset was externally constructed and is not necessarily redistributed in this repository.

See `data/README.md` for information on data access and preparation.

## Citation

Citation information will be added upon publication.
