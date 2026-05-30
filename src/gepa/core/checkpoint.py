"""
Checkpoint Manager for GEPA Engine

This module provides functionality to save and load checkpoints during training,
allowing the system to resume from the last saved state in case of interruption.
"""

import os
import pickle
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime


class CheckpointManager:
    """
    Manages checkpoint saving and loading for the GEPA engine.

    Features:
    - Periodically save training state
    - Resume from checkpoint after interruption
    - Retain multiple historical checkpoints
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        save_every: int = 5,
        keep_last_n: int = 3,
        auto_resume: bool = True
    ):
        """
        Initialize the checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            save_every: Save checkpoint every N iterations (default 5)
            keep_last_n: Keep the last N checkpoints (default 3)
            auto_resume: Whether to auto-resume from the latest checkpoint (default True)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_every = save_every
        self.keep_last_n = keep_last_n
        self.auto_resume = auto_resume
        
        print(f"[Checkpoint] Initialized checkpoint manager at: {self.checkpoint_dir}")
        print(f"[Checkpoint] Save every {save_every} iterations, keep last {keep_last_n} checkpoints")
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """
        Get the path of the latest checkpoint file.

        Returns:
            Path to the latest checkpoint, or None if no checkpoints exist.
        """
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_iter_*.pkl"))
        if not checkpoints:
            return None

        checkpoints.sort(key=lambda p: int(p.stem.split("_")[-1]))
        latest = checkpoints[-1]
        
        print(f"[Checkpoint] Found latest checkpoint: {latest.name}")
        return latest
    
    def should_save(self, iteration: int) -> bool:
        """
        Determine whether a checkpoint should be saved at the given iteration.

        Args:
            iteration: Current iteration number

        Returns:
            True if a checkpoint should be saved.
        """
        return iteration > 0 and iteration % self.save_every == 0
    
    def save_checkpoint(
        self,
        iteration: int,
        state: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save a checkpoint.

        Args:
            iteration: Current iteration number
            state: State dictionary to save
            metadata: Optional extra metadata

        Returns:
            Path to the saved checkpoint file.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = self.checkpoint_dir / f"checkpoint_iter_{iteration}.pkl"
        temp_path = self.checkpoint_dir / f"checkpoint_iter_{iteration}.tmp"

        checkpoint_data = {
            "iteration": iteration,
            "timestamp": timestamp,
            "state": state,
            "metadata": metadata or {}
        }

        try:
            with open(temp_path, 'wb') as f:
                pickle.dump(checkpoint_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Atomic rename to avoid partial writes
            shutil.move(str(temp_path), str(checkpoint_path))

            print(f"[Checkpoint] Saved checkpoint at iteration {iteration}")
            print(f"[Checkpoint]    File: {checkpoint_path.name}")

            # Also save a human-readable metadata JSON
            metadata_path = self.checkpoint_dir / f"checkpoint_iter_{iteration}_meta.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "iteration": iteration,
                    "timestamp": timestamp,
                    "metadata": metadata or {}
                }, f, indent=2)

            self._cleanup_old_checkpoints()

            return checkpoint_path

        except Exception as e:
            print(f"[Checkpoint] Failed to save checkpoint: {e}")
            if temp_path.exists():
                temp_path.unlink()
            raise
    
    def load_checkpoint(self, checkpoint_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
        """
        Load a checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file. If None, load the latest.

        Returns:
            Checkpoint data dictionary, or None if loading fails.
        """
        if checkpoint_path is None:
            checkpoint_path = self.get_latest_checkpoint()

        if checkpoint_path is None:
            print("[Checkpoint] No checkpoint found, starting from scratch")
            return None

        try:
            with open(checkpoint_path, 'rb') as f:
                checkpoint_data = pickle.load(f)

            iteration = checkpoint_data["iteration"]
            timestamp = checkpoint_data["timestamp"]

            print(f"[Checkpoint] Loaded checkpoint from iteration {iteration}")
            print(f"[Checkpoint]    Saved at: {timestamp}")
            print(f"[Checkpoint]    File: {checkpoint_path.name}")

            return checkpoint_data

        except Exception as e:
            print(f"[Checkpoint] Failed to load checkpoint: {e}")
            return None
    
    def _cleanup_old_checkpoints(self):
        """
        Clean up old checkpoint files, keeping only the most recent N.
        """
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_iter_*.pkl"))
        if len(checkpoints) <= self.keep_last_n:
            return

        checkpoints.sort(key=lambda p: int(p.stem.split("_")[-1]))

        for old_checkpoint in checkpoints[:-self.keep_last_n]:
            try:
                old_checkpoint.unlink()
                # Also remove the corresponding metadata file
                meta_file = old_checkpoint.with_suffix('').with_name(
                    old_checkpoint.stem + "_meta.json"
                )
                if meta_file.exists():
                    meta_file.unlink()

                print(f"[Checkpoint] Cleaned up old checkpoint: {old_checkpoint.name}")
            except Exception as e:
                print(f"[Checkpoint] Warning: Failed to delete {old_checkpoint.name}: {e}")
    
    def delete_all_checkpoints(self):
        """
        Delete all checkpoint files (use with caution).
        """
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_iter_*.*"))
        for checkpoint in checkpoints:
            try:
                checkpoint.unlink()
                print(f"[Checkpoint] Deleted: {checkpoint.name}")
            except Exception as e:
                print(f"[Checkpoint] Failed to delete {checkpoint.name}: {e}")
        
        print("[Checkpoint] All checkpoints deleted")
    
    def list_checkpoints(self) -> list[Dict[str, Any]]:
        """
        List all available checkpoints.

        Returns:
            List of checkpoint information dictionaries.
        """
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_iter_*.pkl"))
        checkpoints.sort(key=lambda p: int(p.stem.split("_")[-1]))

        checkpoint_info = []
        for cp in checkpoints:
            iteration = int(cp.stem.split("_")[-1])

            meta_file = cp.with_suffix('').with_name(cp.stem + "_meta.json")
            if meta_file.exists():
                try:
                    with open(meta_file, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                except:
                    meta = {}
            else:
                meta = {}
            
            checkpoint_info.append({
                "iteration": iteration,
                "path": str(cp),
                "size_mb": cp.stat().st_size / (1024 * 1024),
                "timestamp": meta.get("timestamp", "unknown"),
                "metadata": meta.get("metadata", {})
            })
        
        return checkpoint_info
    
    def print_checkpoint_status(self):
        """
        Print checkpoint status information.
        """
        checkpoints = self.list_checkpoints()
        
        print("\n" + "=" * 60)
        print("CHECKPOINT STATUS")
        print("=" * 60)
        print(f"Checkpoint directory: {self.checkpoint_dir}")
        print(f"Total checkpoints: {len(checkpoints)}")
        print(f"Save frequency: every {self.save_every} iterations")
        print(f"Keep last: {self.keep_last_n} checkpoints")
        
        if checkpoints:
            print("\nAvailable checkpoints:")
            for cp in checkpoints:
                print(f"  - Iteration {cp['iteration']:4d} | "
                      f"Size: {cp['size_mb']:.2f} MB | "
                      f"Time: {cp['timestamp']}")
        else:
            print("\nNo checkpoints available")
        
        print("=" * 60 + "\n")


def create_checkpoint_state(
    iteration: int,
    candidates: list,
    mcts_tree: Any,
    best_candidates: list,
    best_scores: Dict[str, Any],
    total_metric_calls: int,
    **kwargs
) -> Dict[str, Any]:
    """
    Create a checkpoint state dictionary.

    Args:
        iteration: Current iteration number
        candidates: Current list of candidate solutions
        mcts_tree: MCTS tree object
        best_candidates: List of best candidate solutions
        best_scores: Dictionary of best scores
        total_metric_calls: Total number of metric calls
        **kwargs: Additional state information

    Returns:
        State dictionary.
    """
    state = {
        "iteration": iteration,
        "candidates": candidates,
        "mcts_tree": mcts_tree,
        "best_candidates": best_candidates,
        "best_scores": best_scores,
        "total_metric_calls": total_metric_calls,
    }

    state.update(kwargs)
    
    return state
