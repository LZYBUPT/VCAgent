# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa


import random
from typing import Any, Mapping


def json_default(x):
    """Default JSON encoder for objects that are not serializable by default."""
    try:
        return {**x}
    except Exception:
        return repr(x)


def idxmax(lst: list[float]) -> int:
    """Return the index of the maximum value in a list."""
    max_val = max(lst)
    return lst.index(max_val)

"""Determine whether program y is dominated: i.e., whether there exists some other program
that covers all validation tasks that y covers, making y redundant."""
def is_dominated(y, programs, program_at_pareto_front_valset):
    y_fronts = [front for front in program_at_pareto_front_valset.values() if y in front]
    for front in y_fronts:
        found_dominator_in_front = False
        for other_prog in front:
            if other_prog in programs:
                found_dominator_in_front = True
                break
        if not found_dominator_in_front:
            return False

    return True
"""In a Pareto front across multiple validation sets, remove globally dominated programs,
keeping only non-dominated programs. For example, if the Pareto fronts are (0,1), (0,1,2), (0,1),
then program 0 will be replaced by program 1 since 1 covers all of 0's tasks.

A program represents a candidate solution. A dominated program is one that is fully covered
by another program across all tasks."""
def remove_dominated_programs(program_at_pareto_front_valset, scores=None):
    freq = {}
    for front in program_at_pareto_front_valset.values():
        for p in front:
            freq[p] = freq.get(p, 0) + 1

    dominated = set()
    programs = list(freq.keys())

    if scores is None:
        scores = dict.fromkeys(programs, 1)

    # Programs with lower scores are more likely to be removed first
    programs = sorted(programs, key=lambda x: scores[x], reverse=False)
    found_to_remove = True
    while found_to_remove:
        found_to_remove = False
        for y in programs:
            if y in dominated:
                continue
            if is_dominated(y, set(programs).difference({y}).difference(dominated), program_at_pareto_front_valset):
                dominated.add(y)
                found_to_remove = True
                break
    # 'dominators' are programs that survive (not dominated by anything else)
    dominators = [p for p in programs if p not in dominated]
    # Verify that every front still has at least one dominator
    for front in program_at_pareto_front_valset.values():
        if not front:
            continue
        assert any(p in front for p in dominators)
    # Reconstruct the Pareto front with only non-dominated programs
    new_program_at_pareto_front_valset = {
        val_id: {prog_idx for prog_idx in front if prog_idx in dominators}
        for val_id, front in program_at_pareto_front_valset.items()
    }
    for val_id, front_new in new_program_at_pareto_front_valset.items():
        assert front_new.issubset(program_at_pareto_front_valset[val_id])

    return new_program_at_pareto_front_valset


def find_dominator_programs(pareto_front_programs, train_val_weighted_agg_scores_for_all_programs):
    train_val_pareto_front_programs = pareto_front_programs
    new_program_at_pareto_front_valset = remove_dominated_programs(
        train_val_pareto_front_programs, scores=train_val_weighted_agg_scores_for_all_programs
    )
    uniq_progs = []
    for front in new_program_at_pareto_front_valset.values():
        uniq_progs.extend(front)
    uniq_progs = set(uniq_progs)
    return list(uniq_progs)

"""Select a program candidate from the Pareto front. Removes dominated programs
and samples from the remaining non-dominated programs weighted by their frequency
of appearance across validation tasks."""
def select_program_candidate_from_pareto_front(
    pareto_front_programs: Mapping[Any, set[int]],
    train_val_weighted_agg_scores_for_all_programs: list[float],
    rng: random.Random,
) -> int:
    train_val_pareto_front_programs = pareto_front_programs
    new_program_at_pareto_front_valset = remove_dominated_programs(
        train_val_pareto_front_programs, scores=train_val_weighted_agg_scores_for_all_programs
    )
    program_frequency_in_validation_pareto_front = {}
    for testcase_pareto_front in new_program_at_pareto_front_valset.values():
        for prog_idx in testcase_pareto_front:
            if prog_idx not in program_frequency_in_validation_pareto_front:
                program_frequency_in_validation_pareto_front[prog_idx] = 0
            program_frequency_in_validation_pareto_front[prog_idx] += 1
    # Frequency-weighted sampling list, e.g. freq {0:4, 1:5} yields [0,0,0,0,1,1,1,1,1]
    sampling_list = [
        prog_idx for prog_idx, freq in program_frequency_in_validation_pareto_front.items() for _ in range(freq)
    ]

    # TODO: Determine if we need this fallback
    # if not sampling_list:
    #     # No Pareto programs survived; fall back to the globally highest-scoring program.
    #     return idxmax(train_val_weighted_agg_scores_for_all_programs)
    assert len(sampling_list) > 0
    # The selected program may not be the absolute best, but it is non-replaceable in the
    # current Pareto front, which makes it valuable in a multi-objective context
    curr_prog_id = rng.choice(sampling_list)
    return curr_prog_id
