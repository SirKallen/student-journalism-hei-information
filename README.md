# Student Journalism and the Informational Role of Higher Education Institutions

This repository contains code, prompts, and manuscript materials for the study:

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
- Human validation and review of LLM classifications
- Logistic regression with outlet-clustered standard errors
- Equal-sized bootstrap analysis for pairwise textual similarity

## Repository Contents

- `code/` — scripts used for institutional matching, textual similarity analysis, LLM-assisted classification, and logistic regression
- `prompts/` — CIN and locality codebooks, classification prompts, and structured output documentation
- `manuscript/` — manuscript and related publication materials

## Data Availability

The underlying article-level dataset and article text files are not included in this repository.

The study uses externally constructed student journalism data containing article text and related metadata. These materials are not redistributed through this repository.

Accordingly, this repository is intended to document the analytical and coding workflow rather than provide a fully reproducible data package.

## Reproducibility

Because the underlying article-level data are not included, the full analysis cannot be reproduced from this repository alone.

The repository provides:

- institutional matching and rurality classification procedures
- textual similarity analysis code
- CIN classification code
- locality classification code
- LLM classification prompts
- CIN and locality codebooks
- logistic regression scripts
- bootstrap procedures
- manuscript documentation

These materials are intended to make the computational and coding procedures as transparent as possible while respecting data-sharing restrictions.

## Repository Structure

```text
student-journalism-hei-information/
├── README.md
├── code/
├── prompts/
└── manuscript/
