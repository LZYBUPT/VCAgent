# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""
Demonstration of the MCTS tree selection mechanism.

This file shows how to use the new MCTS tree PUCT selection algorithm to replace the
original candidate_selector and module_selector mechanisms in GEPA.
"""

from gepa.proposer.reflective_mutation.mcts_tree import MCTSTree, MCTSNode, should_force_system_prompt_update


def demo_mcts_basic_usage():
    """Demonstrate basic usage of the MCTS tree."""

    # Create an initial candidate solution
    initial_candidate = {
        "system_prompt": "You are an expert in biology.",
        "API_information1": "NCBI Gene information...",
        "API_information2": "Uniprot protein information...",
        "API_information3": "Reactome pathway information...",
    }

    # Create the MCTS tree
    mcts_tree = MCTSTree(root_candidate=initial_candidate)

    print("MCTS tree created")
    print(f"Root node generation: {mcts_tree.root.generation}")
    print(f"Root node candidate ID: {id(initial_candidate)}")


def demo_generation_logic():
    """Demonstrate generation logic."""

    root_candidate = {"system_prompt": "test", "API_information1": "test"}

    # Create root node
    root = MCTSNode(root_candidate, parent=None, module_type="root")
    print(f"Root node generation: {root.generation}")

    # Create child node
    child1 = MCTSNode(root_candidate.copy(), parent=root, module_type="system_prompt")
    print(f"First generation node: {child1.generation}")

    child2 = MCTSNode(root_candidate.copy(), parent=child1, module_type="API_information1")
    print(f"Second generation node: {child2.generation}")

    child3 = MCTSNode(root_candidate.copy(), parent=child2, module_type="API_information2")
    print(f"Third generation node: {child3.generation}")

    child4 = MCTSNode(root_candidate.copy(), parent=child3, module_type="system_prompt")
    print(f"Fourth generation node: {child4.generation}")


def demo_system_prompt_force_update():
    """Demonstrate system_prompt forced update rules."""

    root_candidate = {"system_prompt": "test"}

    # Create node chain
    nodes = []
    current = MCTSNode(root_candidate, parent=None, module_type="root")
    nodes.append(current)

    # Create consecutive nodes
    for i in range(10):
        next_node = MCTSNode(root_candidate.copy(), parent=current, module_type="API_information1")
        nodes.append(next_node)
        current = next_node

    print("Node generation and forced system_prompt update rule:")
    for i, node in enumerate(nodes):
        force_update = should_force_system_prompt_update(node)
        print(f"Node {i} (generation {node.generation}): force update system_prompt = {force_update}")

    print("\nForced updates occur at: generation 1 (0->1), 5th gen, 9th gen, 13th gen...")


def demo_system_prompt_failure_handling():
    """Demonstrate handling of system_prompt update failures."""

    root_candidate = {"system_prompt": "test"}

    # Create a 3rd-generation node (next gen should be 4th, forced update)
    gen3_node = MCTSNode(root_candidate, parent=None, module_type="root")
    gen3_node.generation = 3

    print(f"Generation 3 node, next gen (4th) should force system_prompt update: {should_force_system_prompt_update(gen3_node)}")

    # Simulate system_prompt update failure
    gen3_node.system_prompt_update_failed = True
    print(f"After marking system_prompt update as failed, still force update next gen? {should_force_system_prompt_update(gen3_node)}")

    # Generation 6 node (next gen should be 7th, forced update)
    gen6_node = MCTSNode(root_candidate, parent=None, module_type="root")
    gen6_node.generation = 6
    print(f"Generation 6 node, next gen (7th) should force system_prompt update: {should_force_system_prompt_update(gen6_node)}")


if __name__ == "__main__":
    print("=== MCTS Tree Selection Mechanism Demo ===\n")

    print("1. Basic usage:")
    demo_mcts_basic_usage()
    print()

    print("2. Generation logic:")
    demo_generation_logic()
    print()

    print("3. Forced system_prompt update rule:")
    demo_system_prompt_force_update()
    print()

    print("4. system_prompt failure handling:")
    demo_system_prompt_failure_handling()
    print()

    print("=== Demo complete ===")
