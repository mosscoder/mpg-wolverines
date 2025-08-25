"""
Utilities for handling preemption in SLURM preemptible jobs.
Provides checkpointing and signal handling for graceful shutdown and resumption.
"""

import signal
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Callable, Optional


class CheckpointManager:
    """Manages checkpoint saving and loading for preemptible jobs"""
    
    def __init__(self, checkpoint_dir: str = 'checkpoints'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self._signal_handler_registered = False
        self._save_callback = None
    
    def get_checkpoint_path(self, job_idx: int, experiment_name: str) -> Path:
        """Get checkpoint file path for a specific job"""
        filename = f"{experiment_name}_job_{job_idx:03d}.json"
        return self.checkpoint_dir / filename
    
    def save_checkpoint(self, 
                       job_idx: int, 
                       experiment_name: str,
                       completed_configs: List[Dict[str, Any]],
                       pending_configs: List[Dict[str, Any]],
                       metadata: Optional[Dict[str, Any]] = None):
        """Save checkpoint with current progress"""
        
        checkpoint_data = {
            'job_idx': job_idx,
            'experiment_name': experiment_name,
            'timestamp': time.time(),
            'completed_configs': completed_configs,
            'pending_configs': pending_configs,
            'total_completed': len(completed_configs),
            'total_pending': len(pending_configs),
            'metadata': metadata or {}
        }
        
        checkpoint_path = self.get_checkpoint_path(job_idx, experiment_name)
        
        # Write atomically using temporary file
        temp_path = checkpoint_path.with_suffix('.tmp')
        try:
            with open(temp_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            temp_path.rename(checkpoint_path)
            print(f"✓ Checkpoint saved: {len(completed_configs)} completed, {len(pending_configs)} pending")
        except Exception as e:
            print(f"Warning: Failed to save checkpoint: {e}")
            if temp_path.exists():
                temp_path.unlink()
    
    def load_checkpoint(self, job_idx: int, experiment_name: str) -> Optional[Dict[str, Any]]:
        """Load checkpoint if exists"""
        checkpoint_path = self.get_checkpoint_path(job_idx, experiment_name)
        
        if not checkpoint_path.exists():
            return None
        
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            
            print(f"✓ Checkpoint loaded: {checkpoint_data['total_completed']} completed, "
                  f"{checkpoint_data['total_pending']} pending")
            return checkpoint_data
            
        except Exception as e:
            print(f"Warning: Failed to load checkpoint: {e}")
            return None
    
    def clear_checkpoint(self, job_idx: int, experiment_name: str):
        """Clear checkpoint file after successful completion"""
        checkpoint_path = self.get_checkpoint_path(job_idx, experiment_name)
        if checkpoint_path.exists():
            try:
                checkpoint_path.unlink()
                print(f"✓ Checkpoint cleared for completed job")
            except Exception as e:
                print(f"Warning: Failed to clear checkpoint: {e}")
    
    def register_signal_handler(self, save_callback: Callable):
        """Register signal handler for graceful shutdown on preemption"""
        if self._signal_handler_registered:
            return
        
        self._save_callback = save_callback
        
        def signal_handler(signum, frame):
            print(f"\n⚠️  Received signal {signum} - saving checkpoint before shutdown...")
            try:
                self._save_callback()
                print("✓ Checkpoint saved successfully")
            except Exception as e:
                print(f"✗ Failed to save checkpoint: {e}")
            print("Exiting gracefully...")
            exit(0)
        
        # Register handlers for common preemption signals
        signal.signal(signal.SIGTERM, signal_handler)  # SLURM sends SIGTERM before killing
        signal.signal(signal.SIGINT, signal_handler)   # Handle Ctrl+C for testing
        
        self._signal_handler_registered = True
        print("✓ Signal handlers registered for graceful shutdown")


class ProgressTracker:
    """Track progress across multiple configurations"""
    
    def __init__(self, total_configs: int):
        self.total_configs = total_configs
        self.completed_configs = []
        self.failed_configs = []
        self.start_time = time.time()
    
    def add_completed(self, config: Dict[str, Any], result: Dict[str, Any]):
        """Add a completed configuration"""
        self.completed_configs.append({
            'config': config,
            'result': result,
            'completion_time': time.time()
        })
    
    def add_failed(self, config: Dict[str, Any], error: str):
        """Add a failed configuration"""
        self.failed_configs.append({
            'config': config,
            'error': str(error),
            'failure_time': time.time()
        })
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get current progress summary"""
        elapsed_time = time.time() - self.start_time
        completed_count = len(self.completed_configs)
        failed_count = len(self.failed_configs)
        remaining_count = self.total_configs - completed_count - failed_count
        
        # Estimate completion time
        if completed_count > 0:
            avg_time_per_config = elapsed_time / completed_count
            estimated_remaining_time = avg_time_per_config * remaining_count
        else:
            estimated_remaining_time = None
        
        return {
            'total_configs': self.total_configs,
            'completed': completed_count,
            'failed': failed_count,
            'remaining': remaining_count,
            'elapsed_time_minutes': elapsed_time / 60,
            'estimated_remaining_minutes': estimated_remaining_time / 60 if estimated_remaining_time else None,
            'success_rate': completed_count / (completed_count + failed_count) if (completed_count + failed_count) > 0 else 0.0
        }
    
    def print_progress(self):
        """Print formatted progress update"""
        summary = self.get_progress_summary()
        print(f"\n=== Progress Update ===")
        print(f"Completed: {summary['completed']}/{summary['total_configs']} "
              f"({100*summary['completed']/summary['total_configs']:.1f}%)")
        print(f"Failed: {summary['failed']} ({100*summary['success_rate']:.1f}% success rate)")
        print(f"Elapsed: {summary['elapsed_time_minutes']:.1f} minutes")
        if summary['estimated_remaining_minutes']:
            print(f"Estimated remaining: {summary['estimated_remaining_minutes']:.1f} minutes")
        print("=" * 23)


