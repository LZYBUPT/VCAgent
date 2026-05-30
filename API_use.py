"""
API call module - reads data from local JSON files and returns formatted results.
Each API corresponds to a function that returns a formatted information string.
"""

import json
import os
from typing import Dict, Optional, Any


class APIClient:
    """API client class for reading data from local JSON files."""

    def __init__(self, api_data_dir: str = "api_data"):
        """
        Initialize the API client.

        Args:
            api_data_dir: Path to the API data directory, defaults to "api_data".
        """
        self.api_data_dir = api_data_dir
        self._data_cache = {}

        self._load_all_data()

    def _load_all_data(self):
        """Preload all API data files into memory."""
        data_files = {
            'ccle': 'ccle_cell_line_data.json',
            'cellosaurus': 'cellosaurus_cell_line_data.json',
            'depmap': 'depmap_cell_line_data.json',
            'pubchem': 'pubchem_drug_data.json',
            'drugbank': 'drugbank_drug_data.json',
            'ensembl': 'ensembl_gene_data.json',
            'kegg': 'kegg_gene_data.json',
            'ncbi': 'ncbi_gene_data.json',
            'reactome': 'reactome_gene_data.json',
            'uniprot': 'uniprot_gene_data.json'
        }
        
        for key, filename in data_files.items():
            filepath = os.path.join(self.api_data_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    self._data_cache[key] = json.load(f)
                print(f"Successfully loaded {filename}")
            except Exception as e:
                print(f"Failed to load {filename}: {e}")
                self._data_cache[key] = {}
    
    def _format_value(self, value: Any) -> str:
        """
        Format a value, handling None, N/A, empty lists, etc.

        Args:
            value: The value to format.

        Returns:
            Formatted string; returns "No relevant information available" if empty.
        """
        if value is None:
            return "No relevant information available"

        if isinstance(value, str):
            if value.strip() in ["", "N/A", "-", "None"]:
                return "No relevant information available"
            return value.strip()

        if isinstance(value, list):
            if not value or value == [] or value == ["N/A"]:
                return "No relevant information available"
            filtered = [str(item) for item in value if item and str(item).strip() not in ["N/A", "-", "", "None"]]
            if not filtered:
                return "No relevant information available"
            return ", ".join(filtered)

        if isinstance(value, bool):
            return "Expression profile data available" if value else "No expression profile data available"

        return str(value)
    
    def _format_list_summary(self, items: list, max_items: int = 3) -> str:
        """
        Format a list into a summary string.

        Args:
            items: List items.
            max_items: Maximum number of items to display.

        Returns:
            Formatted summary string.
        """
        if not items or items == [] or items == ["N/A"]:
            return "No relevant information available"

        filtered = [str(item) for item in items if item and str(item).strip() not in ["N/A", "-", "", "None"]]
        if not filtered:
            return "No relevant information available"

        if len(filtered) <= max_items:
            return ", ".join(filtered)
        else:
            return ", ".join(filtered[:max_items]) + f" etc. (total {len(filtered)} items)"
    
    def get_ccle_info(self, cell_line_name: str) -> Optional[Dict]:
        """
        Get cell line information from CCLE.

        Args:
            cell_line_name: Cell line name.

        Returns:
            CCLE information dictionary (with formatted fields).
        """
        data = self._data_cache.get('ccle', {}).get(cell_line_name)
        if not data:
            return None

        return {
            'cell_line_name': self._format_value(data.get('cell_line_name')),
            'ccle_name': self._format_value(data.get('ccle_name')),
            'source': self._format_value(data.get('source')),
            'primary_disease': self._format_value(data.get('primary_disease')),
            'lineage': self._format_value(data.get('lineage')),
            'subtype': self._format_value(data.get('subtype')),
            'expression_profile_available': self._format_value(data.get('expression_profile_available'))
        }
    
    def get_cellosaurus_info(self, cell_line_name: str) -> Optional[Dict]:
        """
        Get cell line information from Cellosaurus.

        Args:
            cell_line_name: Cell line name.

        Returns:
            Cellosaurus information dictionary (with formatted fields).
        """
        data = self._data_cache.get('cellosaurus', {}).get(cell_line_name)
        if not data:
            return None

        return {
            'cell_line_name': self._format_value(data.get('cell_line_name')),
            'accession_id': self._format_value(data.get('accession_id')),
            'category': self._format_value(data.get('category')),
            'species': self._format_value(data.get('species')),
            'tissue_origin': self._format_value(data.get('tissue_origin')),
            'disease': self._format_value(data.get('disease')),
            'cell_type': self._format_value(data.get('cell_type')),
            'sex': self._format_value(data.get('sex')),
            'age': self._format_value(data.get('age')),
            'population': self._format_value(data.get('population')),
            'doubling_time': self._format_value(data.get('doubling_time')),
            'key_mutations': self._format_value(data.get('key_mutations'))
        }

    def get_depmap_info(self, cell_line_name: str) -> Optional[Dict]:
        """
        Get cell line information from DepMap.

        Args:
            cell_line_name: Cell line name.

        Returns:
            DepMap information dictionary (with formatted fields).
        """
        data = self._data_cache.get('depmap', {}).get(cell_line_name)
        if not data:
            return None

        return {
            'cell_line_name': self._format_value(data.get('cell_line_name')),
            'depmap_id': self._format_value(data.get('depmap_id')),
            'ccle_name': self._format_value(data.get('ccle_name')),
            'source': self._format_value(data.get('source')),
            'tissue': self._format_value(data.get('tissue')),
            'lineage': self._format_value(data.get('lineage')),
            'disease': self._format_value(data.get('disease')),
            'growth_pattern': self._format_value(data.get('growth_pattern'))
        }

    def get_pubchem_info(self, drug_name: str) -> Optional[Dict]:
        """
        Get drug information from PubChem.

        Args:
            drug_name: Drug name.

        Returns:
            PubChem information dictionary (with formatted fields).
        """
        data = self._data_cache.get('pubchem', {}).get(drug_name)
        if not data:
            return None

        return {
            'drug_name': self._format_value(data.get('drug_name')),
            'pubchem_cid': self._format_value(data.get('pubchem_cid')),
            'molecular_formula': self._format_value(data.get('molecular_formula')),
            'molecular_weight': self._format_value(data.get('molecular_weight')),
            'iupac_name': self._format_value(data.get('iupac_name')),
            'common_names': self._format_value(data.get('common_names')),
            'logp': self._format_value(data.get('logp')),
            'h_bond_donor': self._format_value(data.get('h_bond_donor')),
            'h_bond_acceptor': self._format_value(data.get('h_bond_acceptor')),
            'canonical_smiles': self._format_value(data.get('canonical_smiles'))
        }
    
    def get_ensembl_info(self, gene_symbol: str) -> Optional[Dict]:
        """
        Get gene information from Ensembl.

        Args:
            gene_symbol: Gene symbol.

        Returns:
            Ensembl information dictionary (with formatted fields).
        """
        data = self._data_cache.get('ensembl', {}).get(gene_symbol)
        if not data:
            return None

        basic_info = data.get('basic_info', {})
        transcripts = data.get('transcripts', [])
        identifiers = data.get('gene_identifiers', {})

        transcripts_list = []
        for t in transcripts[:3]:
            if t.get('transcript_name'):
                transcripts_list.append(
                    f"{t.get('transcript_name')} (length {t.get('length_bp', 'N/A')} bp, "
                    f"protein {t.get('protein_length_aa', 'N/A')} amino acids)"
                )
        transcripts_summary = self._format_list_summary(transcripts_list)

        return {
            'gene_symbol': self._format_value(basic_info.get('gene_symbol')),
            'ensembl_id': self._format_value(basic_info.get('ensembl_id')),
            'gene_full_name': self._format_value(basic_info.get('gene_full_name')),
            'gene_type': self._format_value(basic_info.get('gene_type')),
            'location': self._format_value(basic_info.get('location')),
            'strand': self._format_value(basic_info.get('strand')),
            'transcripts_summary': transcripts_summary,
            'NCBI_Gene_ID': self._format_value(identifiers.get('NCBI_Gene_ID')),
            'HGNC_ID': self._format_value(identifiers.get('HGNC_ID'))
        }

    def get_kegg_info(self, gene_symbol: str) -> Optional[Dict]:
        """
        Get gene information from KEGG.

        Args:
            gene_symbol: Gene symbol.

        Returns:
            KEGG information dictionary (with formatted fields).
        """
        data = self._data_cache.get('kegg', {}).get(gene_symbol)
        if not data:
            return None

        pathways = data.get('pathways', [])
        diseases = data.get('diseases', [])
        drug_targets = data.get('drug_targets', [])
        total_pathways = data.get('total_pathways', 0)

        pathways_summary = self._format_list_summary(pathways)
        diseases_summary = self._format_list_summary(diseases)
        drug_targets_summary = self._format_list_summary(drug_targets)

        return {
            'gene_name': self._format_value(data.get('gene_name')),
            'kegg_gene_id': self._format_value(data.get('kegg_gene_id')),
            'pathways_summary': pathways_summary,
            'total_pathways': str(total_pathways if pathways_summary != "No relevant information available" else 0),
            'diseases_summary': diseases_summary,
            'drug_targets_summary': drug_targets_summary
        }

    def get_ncbi_gene_info(self, gene_symbol: str) -> Optional[Dict]:
        """
        Get gene information from NCBI.

        Args:
            gene_symbol: Gene symbol.

        Returns:
            NCBI information dictionary (with formatted fields).
        """
        data = self._data_cache.get('ncbi', {}).get(gene_symbol)
        if not data:
            return None

        aliases = data.get('aliases', [])
        aliases_summary = self._format_list_summary(aliases)

        return {
            'symbol': self._format_value(data.get('symbol')),
            'gene_id': self._format_value(data.get('gene_id')),
            'description': self._format_value(data.get('description')),
            'source': self._format_value(data.get('source')),
            'chromosome': self._format_value(data.get('chromosome')),
            'map_location': self._format_value(data.get('map_location')),
            'summary': self._format_value(data.get('summary')),
            'aliases_summary': aliases_summary
        }

    def get_reactome_pathways(self, gene_symbol: str) -> Optional[Dict]:
        """
        Get gene pathway information from Reactome.

        Args:
            gene_symbol: Gene symbol.

        Returns:
            Reactome information dictionary (with formatted fields).
        """
        data = self._data_cache.get('reactome', {}).get(gene_symbol)
        if not data:
            return None

        pathways = data.get('pathways', [])
        total_pathways = data.get('total_pathways', len(pathways))

        pathway_ids = [p.get('pathway_id') for p in pathways if p.get('pathway_id')]
        reactome_pathways_summary = self._format_list_summary(pathway_ids)

        return {
            'gene_symbol': self._format_value(data.get('gene_symbol')),
            'source': self._format_value(data.get('source', 'Reactome')),
            'reactome_pathways_summary': reactome_pathways_summary,
            'total_pathways': str(total_pathways if reactome_pathways_summary != "No relevant information available" else 0)
        }

    def get_uniprot_info(self, gene_symbol: str) -> Optional[Dict]:
        """
        Get protein information from UniProt.

        Args:
            gene_symbol: Gene symbol.

        Returns:
            UniProt information dictionary (with formatted fields).
        """
        data = self._data_cache.get('uniprot', {}).get(gene_symbol)
        if not data:
            return None
        
        subcellular_locations = data.get('subcellular_locations', [])
        subcellular_summary = self._format_list_summary(subcellular_locations)
        
        keywords = data.get('keywords', [])
        keywords_summary = self._format_list_summary(keywords)

        return {
            'gene_symbol': self._format_value(data.get('gene_symbol')),
            'protein_id': self._format_value(data.get('protein_id')),
            'protein_name': self._format_value(data.get('protein_name')),
            'function': self._format_value(data.get('function')),
            'subcellular_locations_summary': subcellular_summary,
            'keywords_summary': keywords_summary
        }

    def get_drugbank_info(self, drug_name: str) -> Optional[Dict]:
        """
        Get drug information from DrugBank.

        Args:
            drug_name: Drug name.

        Returns:
            DrugBank information dictionary (with formatted fields).
        """
        data = self._data_cache.get('drugbank', {}).get(drug_name)
        if not data:
            return None

        return {
            'name': self._format_value(data.get('name')),
            'drugbank_id': self._format_value(data.get('drugbank_id')),
            'description_drug': self._format_value(data.get('description')),  # Renamed to avoid conflict with gene description
            'groups': self._format_value(data.get('groups')),
            'mechanism_of_action': self._format_value(data.get('mechanism_of_action')),
            'pharmacodynamics': self._format_value(data.get('pharmacodynamics')),
            'cas_number': self._format_value(data.get('cas_number'))
        }


api_client = APIClient()


def get_ncbi_gene_data(gene_symbol: str) -> Optional[str]:
    """Get NCBI gene data (formatted string)."""
    return api_client.get_ncbi_gene_info(gene_symbol)


def get_uniprot_data(gene_symbol: str) -> Optional[str]:
    """Get UniProt protein data (formatted string)."""
    return api_client.get_uniprot_info(gene_symbol)


def get_reactome_data(gene_symbol: str) -> Optional[str]:
    """Get Reactome pathway data (formatted string)."""
    return api_client.get_reactome_pathways(gene_symbol)


def get_kegg_data(gene_symbol: str) -> Optional[str]:
    """Get KEGG pathway data (formatted string)."""
    return api_client.get_kegg_info(gene_symbol)


def get_ensembl_data(gene_symbol: str) -> Optional[str]:
    """Get Ensembl gene data (formatted string)."""
    return api_client.get_ensembl_info(gene_symbol)


def get_cellosaurus_data(cell_line: str) -> Optional[str]:
    """Get Cellosaurus cell line data (formatted string)."""
    return api_client.get_cellosaurus_info(cell_line)


def get_ccle_data(cell_line: str) -> Optional[str]:
    """Get CCLE cell line data (formatted string)."""
    return api_client.get_ccle_info(cell_line)


def get_depmap_data(cell_line: str) -> Optional[str]:
    """Get DepMap cell line data (formatted string)."""
    return api_client.get_depmap_info(cell_line)


def get_pubchem_data(drug_name: str) -> Optional[str]:
    """Get PubChem drug data (formatted string)."""
    return api_client.get_pubchem_info(drug_name)


def get_drugbank_data(drug_name: str) -> Optional[str]:
    """Get DrugBank drug data (formatted string)."""
    return api_client.get_drugbank_info(drug_name)
