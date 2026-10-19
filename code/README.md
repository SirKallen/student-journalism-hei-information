# Code

This folder contains the analysis scripts used for the study:

**Student Journalism and the Informational Role of Higher Education Institutions: A Rural–Urban Comparative Analysis**

The scripts cover institutional rurality matching, textual similarity analysis, LLM-assisted article classification, and logistic regression.

## Files

### `match_hei_rurality.py`

Merges higher education institution records with IPEDS locale information to assign a rurality indicator to each HEI.

### `soft_cosine_analysis.py`

Computes outlet-level soft cosine similarity across publication corpora.

The script:

- preprocesses outlet-level text corpora;
- constructs TF-IDF representations;
- uses pretrained GloVe embeddings to build a term-similarity matrix;
- calculates pairwise soft cosine similarity across outlets;
- categorizes outlet pairs as rural-rural, rural-urban, or urban-urban; and
- performs equal-sized bootstrap comparisons across pair types.

### `cin_binary_classification.py`

Classifies articles for binary Critical Information Needs (CIN) coverage using an LLM and a predefined CIN codebook.

The script:

- reads article text from a ZIP archive;
- applies the CIN classification prompt and codebook;
- returns a binary CIN indicator;
- records supporting excerpts;
- flags ambiguous or incomplete cases for manual review; and
- exports article-level classification results.

The script expects the OpenAI API key to be stored in the `OPENAI_API_KEY` environment variable.

### `cin_logistic_regression.py`

Estimates differences in CIN coverage between rural and urban outlets.

The script produces:

- descriptive CIN rates by rurality;
- unclustered logistic regression estimates;
- logistic regression estimates with standard errors clustered by outlet;
- odds ratios and confidence intervals; and
- predicted probabilities by rurality group.

### `locality_binary_classification.py`

Classifies articles for binary geographic locality using an LLM and a predefined locality codebook.

The script:

- reads article text from a ZIP archive;
- applies the locality classification prompt and codebook;
- returns a binary locality indicator;
- records supporting excerpts;
- flags ambiguous or incomplete cases for manual review; and
- exports article-level classification results.

The script expects the OpenAI API key to be stored in the `OPENAI_API_KEY` environment variable.

### `locality_logistic_regression.py`

Estimates differences in article localness between rural and urban outlets.

The script produces:

- descriptive locality rates by rurality;
- unclustered logistic regression estimates;
- logistic regression estimates with standard errors clustered by outlet;
- odds ratios and confidence intervals; and
- predicted probabilities by rurality group.

## Dependencies

The main Python dependencies are:

```text
pandas
numpy
gensim
openai
statsmodels
