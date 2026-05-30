# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""
MCTS visualization tools.

Features:
1. Pareto front score vs. metric calls curve
2. MCTS tree structure visualization
"""

import json
import os
from typing import Dict, List, Optional, Any

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import networkx as nx
    from networkx.drawing.nx_pydot import graphviz_layout
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


class MCTSVisualizer:
    """MCTS training process visualizer"""

    def __init__(self, output_dir: str = "output/mcts_monitoring"):
        """
        Initialize the visualizer.

        Args:
            output_dir: Output directory path
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # History data file path
        self.history_file = os.path.join(output_dir, "visualization_history.json")

        # Training history records (stores only essential data)
        self.metric_calls_history: List[int] = []
        self.pareto_score_history: List[float] = []
        self.best_valset_score_history: List[float] = []  # New: record Best score on valset

        # Attempt to load existing history data
        self._load_history()
    
    def _load_history(self):
        """Load history data from file."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    history_data = json.load(f)
                    self.metric_calls_history = history_data.get('metric_calls_history', [])
                    self.pareto_score_history = history_data.get('pareto_score_history', [])
                    self.best_valset_score_history = history_data.get('best_valset_score_history', [])
                print(f"[MCTSVisualizer] Successfully loaded history data, {len(self.metric_calls_history)} records")
            except Exception as e:
                print(f"[MCTSVisualizer] Failed to load history data: {e}. Starting fresh.")

    def _save_history(self):
        """Save history data to file."""
        try:
            history_data = {
                'metric_calls_history': self.metric_calls_history,
                'pareto_score_history': self.pareto_score_history,
                'best_valset_score_history': self.best_valset_score_history
            }
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2)
        except Exception as e:
            print(f"[MCTSVisualizer] Failed to save history data: {e}")
    
    def log_iteration(
        self,
        iteration: int,
        total_metric_calls: int,
        best_valset_score: float,
        best_program_idx: int,
        pareto_front_aggregate_score: float,
        num_candidates: int,
        new_program_idx: Optional[int] = None,
        new_program_valset_score: Optional[float] = None,
    ):
        """
        Record information for a single iteration.

        Args:
            total_metric_calls: Cumulative evaluation count
            best_valset_score: Current best validation set score
            pareto_front_aggregate_score: Pareto front aggregate score
        """
        self.metric_calls_history.append(total_metric_calls)
        self.pareto_score_history.append(pareto_front_aggregate_score)
        self.best_valset_score_history.append(best_valset_score)  # New: save best validation set score

        # Save to file immediately after each record
        self._save_history()
    
    def plot_pareto_score_curve(self):
        """Generate Pareto front score vs. metric calls curve."""
        if not HAS_MATPLOTLIB:
            print("[MCTSVisualizer] matplotlib not available, skipping plot")
            return

        if len(self.metric_calls_history) == 0:
            return

        plt.figure(figsize=(12, 6))

        plt.plot(
            self.metric_calls_history,
            self.pareto_score_history,
            'r-o',
            linewidth=2,
            markersize=5,
            label='Pareto Front Aggregate Score'
        )

        plt.xlabel('Total Metric Calls', fontsize=14, fontweight='bold')
        plt.ylabel('Pareto Front Score', fontsize=14, fontweight='bold')
        plt.title('Pareto Front Score vs Metric Calls', fontsize=16, fontweight='bold')
        plt.legend(loc='best', fontsize=12)
        plt.grid(True, alpha=0.3)

        plot_path = os.path.join(self.output_dir, "pareto_score_curve.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"[MCTS] Pareto score curve saved to: {plot_path}")
    
    def plot_best_valset_score_curve(self):
        """Generate best validation set score vs. metric calls curve."""
        if not HAS_MATPLOTLIB:
            print("[MCTSVisualizer] matplotlib not available, skipping plot")
            return

        if len(self.metric_calls_history) == 0 or len(self.best_valset_score_history) == 0:
            return

        plt.figure(figsize=(12, 6))

        plt.plot(
            self.metric_calls_history,
            self.best_valset_score_history,
            'b-o',
            linewidth=2,
            markersize=5,
            label='Best Score on Valset'
        )

        # Annotate the maximum value
        if self.best_valset_score_history:
            max_score = max(self.best_valset_score_history)
            max_idx = self.best_valset_score_history.index(max_score)
            max_calls = self.metric_calls_history[max_idx]
            plt.axhline(y=max_score, color='g', linestyle='--', alpha=0.5, label=f'Max Score: {max_score:.4f}')
            plt.plot(max_calls, max_score, 'g*', markersize=15, label=f'Peak at {max_calls} calls')

        plt.xlabel('Total Metric Calls', fontsize=14, fontweight='bold')
        plt.ylabel('Best Valset Score', fontsize=14, fontweight='bold')
        plt.title('Best Validation Score vs Metric Calls', fontsize=16, fontweight='bold')
        plt.legend(loc='best', fontsize=12)
        plt.grid(True, alpha=0.3)

        plot_path = os.path.join(self.output_dir, "best_valset_score_curve.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"[MCTS] Best valset score curve saved to: {plot_path}")
    
    def visualize_mcts_tree(self, mcts_tree, state=None):
        """
        Visualize the MCTS tree structure.

        Args:
            mcts_tree: MCTSTree object
            state: GEPAState object (used to retrieve score information)
        """
        if not HAS_NETWORKX:
            print("[MCTSVisualizer] networkx not available, skipping tree visualization")
            return

        G = nx.DiGraph()
        self._add_nodes_to_graph(G, mcts_tree.root, state)

        try:
            pos = graphviz_layout(G, prog='dot')
        except Exception:
            # Fall back to hierarchical layout if graphviz is unavailable
            pos = self._hierarchical_layout(G, mcts_tree.root)

        plt.figure(figsize=(16, 12))

        node_colors = []
        node_sizes = []
        for node_id in G.nodes():
            node_data = G.nodes[node_id]
            # Color by success rate
            if node_data['visit_count'] > 0:
                success_rate = node_data['success_count'] / node_data['visit_count']
                node_colors.append(success_rate)
            else:
                node_colors.append(0)
            # Size by visit count
            node_sizes.append(300 + node_data['visit_count'] * 50)

        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=node_sizes,
            cmap=plt.cm.RdYlGn,
            vmin=0, vmax=1,
            alpha=0.9
        )

        nx.draw_networkx_edges(
            G, pos,
            edge_color='gray',
            arrows=True,
            arrowsize=20,
            width=2,
            alpha=0.6
        )

        labels = {}
        for node_id in G.nodes():
            node_data = G.nodes[node_id]
            label = f"gen{node_data['generation']}\n"
            if node_data['module_type'] != 'root':
                label += f"{node_data['module_type']}\n"
            label += f"V:{node_data['visit_count']} S:{node_data['success_count']}"
            if node_data['visit_count'] > 0:
                q = node_data['success_count'] / node_data['visit_count']
                label += f"\nQ:{q:.2f}"
            labels[node_id] = label

        nx.draw_networkx_labels(
            G, pos,
            labels,
            font_size=8,
            font_weight='bold'
        )

        plt.title('MCTS Tree Structure', fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()

        tree_plot_path = os.path.join(self.output_dir, "mcts_tree.png")
        plt.savefig(tree_plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"[MCTS] Tree visualization saved to: {tree_plot_path}")
    
    def _add_nodes_to_graph(self, G: nx.DiGraph, node, state, parent_id=None):
        """Recursively add nodes to the graph."""
        node_id = id(node)

        G.add_node(
            node_id,
            generation=node.generation,
            module_type=node.module_type,
            visit_count=node.visit_count,
            success_count=node.success_count,
        )

        if parent_id is not None:
            G.add_edge(parent_id, node_id)

        for child in node.children:
            self._add_nodes_to_graph(G, child, state, node_id)
    
    def _hierarchical_layout(self, G: nx.DiGraph, root) -> Dict:
        """Create a hierarchical layout (used when graphviz is unavailable)."""
        pos = {}

        def assign_positions(node, x=0, y=0, layer_width=2.0, visited=None):
            if visited is None:
                visited = set()

            node_id = id(node)
            if node_id in visited:
                return
            visited.add(node_id)

            pos[node_id] = (x, -y)

            # Compute child node positions
            num_children = len(node.children)
            if num_children > 0:
                child_width = layer_width / num_children
                start_x = x - layer_width / 2
                for i, child in enumerate(node.children):
                    child_x = start_x + (i + 0.5) * child_width
                    assign_positions(child, child_x, y + 1, child_width, visited)

        assign_positions(root)
        return pos
    
    def plot_candidate_evolution(self, state):
        """
        Draw the candidate program evolution tree (for non-MCTS mode).
        Shows the actual parent-to-child evolution tree structure,
        supporting multiple children from the same parent.

        Args:
            state: GEPAState object
        """
        if not HAS_MATPLOTLIB or not HAS_NETWORKX:
            print("[Visualizer] matplotlib or networkx not available, skipping candidate evolution plot")
            return
        
        try:
            G = nx.DiGraph()

            num_candidates = len(state.program_candidates)

            if num_candidates == 0:
                print("[Visualizer] No candidates to visualize")
                return

            # Retrieve scores for each candidate
            candidate_scores = []
            for idx in range(num_candidates):
                if idx < len(state.program_full_scores_val_set):
                    score = state.program_full_scores_val_set[idx]
                else:
                    score = 0.0
                candidate_scores.append(score)

            # Calculate generation depth for each node based on parent-child relationships
            generations = self._calculate_generations(state.parent_program_for_candidate, num_candidates)

            # Add nodes (one per candidate)
            for idx in range(num_candidates):
                G.add_node(
                    idx,
                    score=candidate_scores[idx],
                    generation=generations[idx],
                    label=f"#{idx}\nGen:{generations[idx]}\nScore:{candidate_scores[idx]:.3f}"
                )

            # Add edges using actual parent-child relationships
            for child_idx in range(num_candidates):
                parents = state.parent_program_for_candidate[child_idx]
                if parents:
                    for parent_idx in parents:
                        if parent_idx is not None and 0 <= parent_idx < num_candidates:
                            G.add_edge(parent_idx, child_idx)

            # Collect tree structure statistics
            roots = [i for i in range(num_candidates) if not state.parent_program_for_candidate[i] or
                    all(p is None for p in state.parent_program_for_candidate[i])]
            num_edges = G.number_of_edges()
            max_generation = max(generations) if generations else 0

            print(f"[Visualizer] Tree structure: {num_candidates} nodes, {num_edges} edges, "
                  f"{len(roots)} roots, max generation: {max_generation}")

            try:
                pos = graphviz_layout(G, prog='dot')
            except Exception:
                print("[Visualizer] Graphviz not available, using custom hierarchical layout")
                pos = self._hierarchical_layout_tree(G, generations, num_candidates)

            plt.figure(figsize=(20, 12))

            # Node color by score
            if candidate_scores and max(candidate_scores) > min(candidate_scores):
                max_score = max(candidate_scores)
                min_score = min(candidate_scores)
                score_range = max_score - min_score
                node_colors = [(score - min_score) / score_range for score in candidate_scores]
            else:
                node_colors = [0.5] * num_candidates

            # Node size: larger for higher scores
            base_size = 500
            if candidate_scores and max(candidate_scores) > 0:
                max_score = max(candidate_scores)
                node_sizes = [base_size + (score / max_score * 1000) for score in candidate_scores]
            else:
                node_sizes = [base_size] * num_candidates
            
            nx.draw_networkx_nodes(
                G, pos,
                node_color=node_colors,
                node_size=node_sizes,
                cmap=plt.cm.RdYlGn,
                vmin=0, vmax=1,
                alpha=0.9,
                edgecolors='black',
                linewidths=2
            )

            nx.draw_networkx_edges(
                G, pos,
                edge_color='gray',
                arrows=True,
                arrowsize=15,
                width=1.5,
                alpha=0.6,
                connectionstyle='arc3,rad=0.1',
                node_size=node_sizes
            )

            labels = {idx: G.nodes[idx]['label'] for idx in G.nodes()}
            nx.draw_networkx_labels(
                G, pos,
                labels,
                font_size=8,
                font_weight='bold',
                font_color='black'
            )

            if candidate_scores:
                sm = plt.cm.ScalarMappable(
                    cmap=plt.cm.RdYlGn,
                    norm=plt.Normalize(vmin=min(candidate_scores), vmax=max(candidate_scores))
                )
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=plt.gca(), fraction=0.03, pad=0.02)
                cbar.set_label('Validation Score', fontsize=12, fontweight='bold')

            plt.title(f'Candidate Program Evolution Tree\n{num_candidates} Candidates, {num_edges} Edges, Max Generation: {max_generation}',
                     fontsize=16, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()

            plot_path = os.path.join(self.output_dir, "candidate_evolution_tree.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"[Visualization] Candidate evolution tree saved to: {plot_path}")

            # Save evolution statistics (including parent-child relationships)
            parent_child_relationships = {}
            for child_idx in range(num_candidates):
                parents = state.parent_program_for_candidate[child_idx]
                if parents:
                    parent_child_relationships[child_idx] = [p for p in parents if p is not None]
            
            evolution_stats = {
                "total_candidates": num_candidates,
                "total_edges": num_edges,
                "max_generation": max_generation,
                "num_roots": len(roots),
                "best_score": max(candidate_scores) if candidate_scores else 0,
                "worst_score": min(candidate_scores) if candidate_scores else 0,
                "average_score": sum(candidate_scores) / len(candidate_scores) if candidate_scores else 0,
                "score_improvement": candidate_scores[-1] - candidate_scores[0] if len(candidate_scores) > 1 else 0,
                "all_scores": candidate_scores,
                "generations": generations,
                "parent_child_relationships": parent_child_relationships
            }
            
            stats_path = os.path.join(self.output_dir, "candidate_evolution_stats.json")
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(evolution_stats, f, indent=2)
            
            print(f"[Visualization] Evolution statistics saved to: {stats_path}")
            
        except Exception as e:
            import traceback
            print(f"[Visualizer] Error creating candidate evolution plot: {e}")
            print(traceback.format_exc())
    
    def _calculate_generations(self, parent_program_for_candidate: list, num_candidates: int) -> list:
        """
        Calculate the generation of each candidate program based on parent-child relationships.

        Args:
            parent_program_for_candidate: List of parent indices
            num_candidates: Total number of candidate programs

        Returns:
            List of generation numbers for each candidate program
        """
        generations = [0] * num_candidates

        # Calculate generations using ancestor chain traversal
        for idx in range(num_candidates):
            parents = parent_program_for_candidate[idx]
            if not parents or all(p is None for p in parents):
                generations[idx] = 0  # Root node
            else:
                # Generation = max parent generation + 1
                valid_parents = [p for p in parents if p is not None and p < idx]
                if valid_parents:
                    generations[idx] = max(generations[p] for p in valid_parents) + 1
                else:
                    generations[idx] = 0

        return generations
    
    def _hierarchical_layout_tree(self, G: nx.DiGraph, generations: list, num_nodes: int) -> Dict:
        """
        Create a generation-based hierarchical layout (used for candidate program evolution tree).
        Nodes of the same generation are placed on the same horizontal level.

        Args:
            G: NetworkX graph
            generations: Generation number for each node
            num_nodes: Number of nodes

        Returns:
            Position dictionary {node_id: (x, y)}
        """
        pos = {}

        # Group nodes by generation
        nodes_by_generation = {}
        for idx in range(num_nodes):
            gen = generations[idx]
            if gen not in nodes_by_generation:
                nodes_by_generation[gen] = []
            nodes_by_generation[gen].append(idx)

        # Assign positions for each generation
        y_spacing = 3.0

        for gen, nodes in sorted(nodes_by_generation.items()):
            num_nodes_in_gen = len(nodes)
            x_spacing = 4.0
            total_width = (num_nodes_in_gen - 1) * x_spacing
            start_x = -total_width / 2

            for i, node_idx in enumerate(nodes):
                x = start_x + i * x_spacing
                y = -gen * y_spacing
                pos[node_idx] = (x, y)

        return pos
    
    def plot_api_component_evolution(self, state):
        """
        Draw the API component evolution heatmap (for non-MCTS mode).
        Shows the usage of each API information slot across iterations.

        Args:
            state: GEPAState object
        """
        if not HAS_MATPLOTLIB:
            print("[Visualizer] matplotlib not available, skipping API component evolution plot")
            return

        try:
            # Collect API usage info from all candidate programs
            api_usage = {}
            for i in range(1, 11):  # API_information1 through API_information10
                api_key = f"API_information{i}"
                api_usage[api_key] = []

            # Analyze which APIs each candidate program uses
            for idx, candidate in enumerate(state.program_candidates):
                for i in range(1, 11):
                    api_key = f"API_information{i}"
                    if api_key in candidate:
                        api_text = candidate[api_key].strip()
                        # Check if API is actually used (not the default unfavorable message)
                        is_used = api_text and "unfavorable" not in api_text.lower()
                        api_usage[api_key].append(1 if is_used else 0)
                    else:
                        api_usage[api_key].append(0)

            fig, ax = plt.subplots(figsize=(14, 8))

            # Prepare data matrix
            data_matrix = []
            labels = []
            for api_key in sorted(api_usage.keys()):
                data_matrix.append(api_usage[api_key])
                labels.append(api_key)

            im = ax.imshow(data_matrix, cmap='YlGn', aspect='auto', interpolation='nearest')

            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=10)
            ax.set_xlabel('Candidate Program Index', fontsize=12, fontweight='bold')
            ax.set_ylabel('API Information Slot', fontsize=12, fontweight='bold')
            ax.set_title('API Component Usage Evolution', fontsize=14, fontweight='bold')

            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Usage (0=Unused, 1=Used)', fontsize=10)

            plt.tight_layout()

            plot_path = os.path.join(self.output_dir, "api_component_evolution.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"[Visualization] API component evolution plot saved to: {plot_path}")

        except Exception as e:
            print(f"[Visualizer] Error creating API component evolution plot: {e}")
    
    def plot_pareto_programs(self, state):
        """
        Draw the Pareto front programs information (for non-MCTS mode).
        Shows the programs in the Pareto front and their score distribution.

        Args:
            state: GEPAState object
        """
        if not HAS_MATPLOTLIB:
            print("[Visualizer] matplotlib not available, skipping Pareto programs plot")
            return

        try:
            # Retrieve Pareto front program indices and scores
            pareto_programs = []
            pareto_scores = []

            if state.frontier_type == "instance":
                # Instance-level Pareto front
                for data_id, score in state.pareto_front_valset.items():
                    if data_id in state.program_at_pareto_front_valset:
                        program_indices = state.program_at_pareto_front_valset[data_id]
                        for prog_idx in program_indices:
                            pareto_programs.append(prog_idx)
                            pareto_scores.append(score)
            elif state.frontier_type == "objective":
                # Objective-level Pareto front
                for obj_id, score in state.objective_pareto_front.items():
                    if obj_id in state.program_at_pareto_front_objectives:
                        program_indices = state.program_at_pareto_front_objectives[obj_id]
                        for prog_idx in program_indices:
                            pareto_programs.append(prog_idx)
                            pareto_scores.append(score)

            if not pareto_programs:
                print("[Visualizer] No Pareto programs to plot")
                return

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            # Left: scatter plot of program index vs score
            ax1.scatter(pareto_programs, pareto_scores, s=100, alpha=0.6, c='blue', edgecolors='black')
            ax1.set_xlabel('Program Index', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
            ax1.set_title('Pareto Front Programs: Index vs Score', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)

            # Right: frequency distribution of program indices (bar chart)
            from collections import Counter
            program_counts = Counter(pareto_programs)
            programs = sorted(program_counts.keys())
            counts = [program_counts[p] for p in programs]

            ax2.bar(programs, counts, color='green', alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Program Index', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Frequency in Pareto Front', fontsize=12, fontweight='bold')
            ax2.set_title('Program Representation in Pareto Front', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()

            plot_path = os.path.join(self.output_dir, "pareto_programs.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"[Visualization] Pareto programs plot saved to: {plot_path}")

            # Additionally save Pareto program statistics to JSON
            stats = {
                "total_pareto_programs": len(pareto_programs),
                "unique_programs": len(program_counts),
                "program_frequency": dict(program_counts),
                "average_score": sum(pareto_scores) / len(pareto_scores) if pareto_scores else 0,
                "max_score": max(pareto_scores) if pareto_scores else 0,
                "min_score": min(pareto_scores) if pareto_scores else 0,
            }

            stats_path = os.path.join(self.output_dir, "pareto_programs_stats.json")
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2)

            print(f"[Visualization] Pareto programs statistics saved to: {stats_path}")

        except Exception as e:
            import traceback
            print(f"[Visualizer] Error creating Pareto programs plot: {e}")
            print(traceback.format_exc())