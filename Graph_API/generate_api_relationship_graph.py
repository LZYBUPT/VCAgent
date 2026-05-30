"""
API relationship graph generator.

Features:
1. Randomly sample data from each API's JSON file
2. Convert data into natural language descriptions using an LLM
3. Compute semantic similarity between descriptions
4. Generate an API relationship graph based on similarity
"""

import json
import random
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
# os.environ.pop("OPENAI_API_BASE", None)
# os.environ["OPENAI_BASE_URL"] = 
# os.environ["OPENAI_API_KEY"] = 
# os.environ["HF_TOKEN"] = 
# Ensure api key exists (you must set it externally)
assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY is missing!"
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import litellm
except ImportError:
    print("Please install litellm: pip install litellm")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Please install sentence-transformers: pip install sentence-transformers")
    sys.exit(1)

try:
    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
except ImportError:
    print("Please install networkx and matplotlib: pip install networkx matplotlib")
    sys.exit(1)

from sklearn.metrics.pairwise import cosine_similarity


class APIRelationshipAnalyzer:
    """API relationship analyzer."""

    def __init__(self, api_data_dir: str = "api_data", output_dir: str = "Graph_API/output"):
        self.api_data_dir = Path(api_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.api_files = {
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
        
        print("Loading sentence embedding model...")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        print("Sentence embedding model loaded!")

        # Store data
        self.sampled_data: Dict[str, List[Dict]] = {}
        self.descriptions: Dict[str, List[str]] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
    
    def load_and_sample_data(self, num_samples: int = 5):
        """Load and randomly sample data from each API file."""
        print("\n=== Step 1: Load and sample data ===")

        for api_name, filename in self.api_files.items():
            filepath = self.api_data_dir / filename

            if not filepath.exists():
                print(f"Warning: {filepath} does not exist, skipping")
                continue

            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Random sampling
            all_keys = list(data.keys())
            sample_size = min(num_samples, len(all_keys))
            sampled_keys = random.sample(all_keys, sample_size)

            self.sampled_data[api_name] = [
                {"key": key, "data": data[key]} for key in sampled_keys
            ]

            print(f"  {api_name}: sampled {sample_size}/{len(all_keys)} entries")
    
    def generate_descriptions_with_llm(self, model: str = "gpt-4o-mini"):
        """Use an LLM to convert data into natural language descriptions."""
        print(f"\n=== Step 2: Generate natural language descriptions using {model} ===")

        for api_name, samples in self.sampled_data.items():
            self.descriptions[api_name] = []

            print(f"\nProcessing {api_name} database:")

            for i, sample in enumerate(samples, 1):
                prompt = f"""Please convert the following structured data from the {api_name} database into a concise natural language description.
The description should include the core information and features of the data to facilitate understanding of its content and purpose.

Data ID: {sample['key']}
Data Content:
{json.dumps(sample['data'], ensure_ascii=False, indent=2)}

Please provide the description directly without any prefixes or explanations."""

                try:
                    response = litellm.completion(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.7
                    )

                    description = response.choices[0].message.content.strip()
                    self.descriptions[api_name].append(description)

                    print(f"  [{i}/{len(samples)}] {sample['key'][:30]}: OK")

                except Exception as e:
                    print(f"  [{i}/{len(samples)}] {sample['key'][:30]}: ERROR: {e}")
                    self.descriptions[api_name].append(
                        f"Information about {sample['key']} from the {api_name} database"
                    )

            print(f"  {api_name}: generated {len(self.descriptions[api_name])} descriptions")
    
    def compute_embeddings(self):
        """Compute sentence embeddings for all descriptions."""
        print("\n=== Step 3: Compute sentence embeddings ===")
        
        for api_name, descriptions in self.descriptions.items():
            if descriptions:
                embeddings = self.encoder.encode(descriptions)
                self.embeddings[api_name] = embeddings
                print(f"✓ {api_name}: {embeddings.shape}")
    
    def calculate_api_similarities(self) -> Dict[Tuple[str, str], float]:
        """Calculate average semantic similarity between APIs."""
        print("\n=== Step 4: Calculate semantic similarities between APIs ===")
        
        similarities = {}
        api_names = list(self.embeddings.keys())
        
        for i, api1 in enumerate(api_names):
            for api2 in api_names[i+1:]:
                emb1 = self.embeddings[api1]
                emb2 = self.embeddings[api2]

                sim_matrix = cosine_similarity(emb1, emb2)

                avg_similarity = np.mean(sim_matrix)

                similarities[(api1, api2)] = avg_similarity
                print(f"  {api1} <-> {api2}: {avg_similarity:.4f}")

        return similarities

    def generate_relationship_graph(self, similarities: Dict[Tuple[str, str], float],
                                   threshold: float = 0.3):
        """Generate and visualize the API relationship graph."""
        print(f"\n=== Step 5: Generate relationship graph (similarity threshold >= {threshold}) ===")

        G = nx.Graph()

        for api_name in self.embeddings.keys():
            G.add_node(api_name)

        edges_data = []
        for (api1, api2), similarity in similarities.items():
            if similarity >= threshold:
                G.add_edge(api1, api2, weight=similarity)
                edges_data.append((api1, api2, similarity))
                print(f"  {api1} -- {api2}: {similarity:.4f}")

        plt.figure(figsize=(16, 12))

        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                              node_size=3000, alpha=0.9)

        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, width=[w*5 for w in weights],
                              alpha=0.6, edge_color='gray')

        nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold')

        edge_labels = {(u, v): f"{G[u][v]['weight']:.3f}"
                      for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=10)

        plt.title("API Database Semantic Relationship Graph\n(edge thickness indicates similarity strength)",
                 fontsize=18, fontweight='bold', pad=20)
        plt.axis('off')
        plt.tight_layout()

        output_file = self.output_dir / "api_relationship_graph.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n  Relationship graph saved to: {output_file}")

        plt.show()

        return G

    def save_results(self, similarities: Dict[Tuple[str, str], float]):
        """Save analysis results to JSON files."""
        print("\n=== Save results ===")

        descriptions_file = self.output_dir / "api_descriptions.json"
        with open(descriptions_file, 'w', encoding='utf-8') as f:
            json.dump(self.descriptions, f, ensure_ascii=False, indent=2)
        print(f"  Descriptions saved to: {descriptions_file}")

        similarities_data = {
            f"{api1} <-> {api2}": float(sim)
            for (api1, api2), sim in sorted(similarities.items(),
                                           key=lambda x: x[1], reverse=True)
        }
        similarities_file = self.output_dir / "api_similarities.json"
        with open(similarities_file, 'w', encoding='utf-8') as f:
            json.dump(similarities_data, f, ensure_ascii=False, indent=2)
        print(f"  Similarities saved to: {similarities_file}")

        report_file = self.output_dir / "analysis_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("API Database Relationship Analysis Report\n")
            f.write("=" * 80 + "\n\n")

            f.write("1. Data Sampling\n")
            f.write("-" * 80 + "\n")
            for api_name, samples in self.sampled_data.items():
                f.write(f"  {api_name}: {len(samples)} samples\n")

            f.write("\n2. Similarity Ranking (Top 10)\n")
            f.write("-" * 80 + "\n")
            sorted_sims = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            for i, ((api1, api2), sim) in enumerate(sorted_sims[:10], 1):
                f.write(f"  {i}. {api1} <-> {api2}: {sim:.4f}\n")

            f.write("\n3. Relationship Analysis\n")
            f.write("-" * 80 + "\n")
            f.write("High similarity (>0.5): Strong association between databases, may contain similar or complementary information\n")
            f.write("Medium similarity (0.3-0.5): Some association between databases, partial information overlap\n")
            f.write("Low similarity (<0.3): Weak association between databases, different information types\n\n")

            high_sim = [(k, v) for k, v in sorted_sims if v > 0.5]
            mid_sim = [(k, v) for k, v in sorted_sims if 0.3 <= v <= 0.5]

            f.write(f"  High similarity pairs: {len(high_sim)}\n")
            f.write(f"  Medium similarity pairs: {len(mid_sim)}\n")
            f.write(f"  Low similarity pairs: {len(sorted_sims) - len(high_sim) - len(mid_sim)}\n")

        print(f"  Analysis report saved to: {report_file}")

    def run_analysis(self, num_samples: int = 5, model: str = "gpt-4o-mini",
                    threshold: float = 0.3):
        """Run the complete analysis pipeline."""
        print("=" * 80)
        print("API Database Relationship Analysis")
        print("=" * 80)

        self.load_and_sample_data(num_samples)

        self.generate_descriptions_with_llm(model)

        self.compute_embeddings()

        similarities = self.calculate_api_similarities()

        graph = self.generate_relationship_graph(similarities, threshold)

        self.save_results(similarities)

        print("\n" + "=" * 80)
        print("Analysis complete!")
        print("=" * 80)

        return graph, similarities


def main():
    """Main entry point."""
    analyzer = APIRelationshipAnalyzer(
        api_data_dir=r".\api_data",
        output_dir="Graph_API/output"
    )

    # Parameters:
    # - num_samples: number of data samples per API (default 5)
    # - model: LLM model to use (default gpt-4o-mini)
    # - threshold: minimum similarity threshold for showing edges in the graph (default 0.3)
    analyzer.run_analysis(
        num_samples=5,
        model="gpt-4o-mini",
        threshold=0.3
    )


if __name__ == "__main__":
    main()
