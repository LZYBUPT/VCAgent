"""
Configuration file - adjust these parameters as needed.
"""

# LLM configuration
LLM_MODEL = "gpt-4o-mini"  # Options: gpt-4o, gpt-3.5-turbo, claude-3-sonnet-20240229
LLM_TEMPERATURE = 0.7

# Data sampling configuration
NUM_SAMPLES_PER_API = 5  # Number of data samples per API
RANDOM_SEED = 42         # Random seed for reproducibility

# Similarity threshold configuration
SIMILARITY_THRESHOLD = 0.3  # Minimum similarity to display in the relationship graph
# 0.5 - Strict mode (only show strong associations)
# 0.3 - Default mode (show moderate and above associations)
# 0.1 - Relaxed mode (show almost all associations)

# Sentence embedding model configuration
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"
# Alternative models:
# - "all-MiniLM-L6-v2" (lightweight, optimized for English, fast)
# - "paraphrase-multilingual-MiniLM-L12-v2" (multilingual support)
# - "all-mpnet-base-v2" (higher accuracy, slightly slower)

# Path configuration
API_DATA_DIR = "../api_data"
OUTPUT_DIR = "output"

# Graph configuration
FIGURE_SIZE = (16, 12)
NODE_SIZE = 3000
NODE_COLOR = "lightblue"
EDGE_WIDTH_MULTIPLIER = 5
DPI = 300

# API file mapping
API_FILES = {
    "CCLE": "ccle_cell_line_data.json",
    "Cellosaurus": "cellosaurus_cell_line_data.json",
    "DepMap": "depmap_cell_line_data.json",
    "Ensembl": "ensembl_gene_data.json",
    "KEGG": "kegg_gene_data.json",
    "NCBI": "ncbi_gene_data.json",
    "Reactome": "reactome_gene_data.json",
    "UniProt": "uniprot_gene_data.json",
    "PubChem": "pubchem_drug_data.json",
    "DrugBank": "drugbank_drug_data.json"
}

# LLM Prompt template
DESCRIPTION_PROMPT_TEMPLATE = """Please convert the following structured data from the {api_name} database into a concise natural language description.
The description should include the core information and features of the data to facilitate understanding of its content and purpose.

Data ID: {data_key}
Data Content:
{data_content}

Please provide the description directly without any prefixes or explanations."""
