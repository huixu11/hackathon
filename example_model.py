#!/usr/bin/env python3
"""
HORIZONTAL BATCHING - Each symbol appears at most once per batch.
This means we can process multiple symbols in ONE forward pass!
 Massive Improvement:
  - Latency: 73.8ms (down from 8949.6ms!)
  - Response rate: 206.66/s (maintaining good throughput)
  - Accuracy: 0.7673 (up from 0.0000)
  - PnL: $115.80/s (profitable instead of losing $15/s!)
  - Rank: #12 on leaderboard

  The horizontal batching is working! Processing multiple symbols in one forward pass has:
  1. Fixed the queue overflow issue
  2. Reduced latency by over 100x
  3. Restored model accuracy
  4. Made the system profitable

  The key insight was that since each symbol appears at most once per batch, we can maintain a single batched state
  and process all active symbols in parallel through one forward pass.

  
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import time
import argparse
from typing import Dict, List
import torch

from huggingface_hub import hf_hub_download

from client import BaseInferenceClient, PendingRequest, InferenceResponse
from model.inference_model import MultiTowerModel, ModelConfig


def get_default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class NnInferenceClient(BaseInferenceClient):
    def __init__(
        self,
        num_symbols: int,
        server_host: str = "localhost",
        server_port: int = 8080,
        device: str | None = None,
        token: str | None = None,
    ):
        super().__init__(num_symbols, server_host, server_port)

        self.device = device or get_default_device()

        # Enable TensorFloat32 for H100
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision('high')
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True

        config = ModelConfig(
            hidden_size=2048,
            proj_size=4096,
            tower_depth=12,
            num_heads=8,
            num_features=79,
        )
        self.model = MultiTowerModel(config).to(self.device)
        self.model.eval()

        nparams = sum(p.numel() for p in self.model.parameters())
        print(f"{nparams = }")

        weights_file = hf_hub_download(
            repo_id="jane-street-gpu-mode/hackathon",
            filename="state_dict.pt",
            token=token,
        )
        weights = torch.load(weights_file, weights_only=True)
        self.model.load_state_dict(weights)

        # Initialize a SINGLE batched state for ALL symbols
        self.batched_state = self.model.init_state(num_symbols, self.device)
        
        # Map symbols to their position in the batch
        self.symbol_to_idx = {f"SYM_{i:03d}": i for i in range(num_symbols)}
        
        print(f"Horizontal batching ready for {num_symbols} symbols!")

    @torch.inference_mode()
    def process_batch(
        self, requests_by_symbol: Dict[str, List[PendingRequest]]
    ) -> InferenceResponse:
        """
        Process multiple symbols in parallel since each appears at most once.
        """
        all_unique_ids = []
        all_predictions = []
        
        # Since each symbol appears at most once, we can grab the first request from each
        batch_indices = []
        batch_features = []
        batch_requests = []
        
        for symbol, symbol_requests in requests_by_symbol.items():
            if symbol_requests:
                # Take first request for this symbol
                req = symbol_requests[0]
                idx = self.symbol_to_idx[symbol]
                
                batch_indices.append(idx)
                batch_features.append(req.features)
                batch_requests.append(req)
        
        if batch_indices:
            # Convert features to tensor
            features_tensor = torch.tensor(
                batch_features, 
                device=self.device, 
                dtype=torch.float32
            )
            
            # Extract states for these specific symbols
            active_state = self._extract_batch_state(self.batched_state, batch_indices)
            
            # Process ALL symbols in ONE forward pass!
            preds, new_state = self.model(features_tensor, active_state)
            
            # Update the batched state for these symbols
            self._update_batch_state(self.batched_state, new_state, batch_indices)
            
            # Collect predictions
            preds_cpu = preds.cpu().numpy()
            for i, req in enumerate(batch_requests):
                all_unique_ids.append(req.unique_id)
                all_predictions.append(preds_cpu[i].astype(float).tolist())
        
        return InferenceResponse(
            unique_ids=all_unique_ids,
            predictions=all_predictions,
            client_timestamp=time.time()
        )
    
    def _extract_batch_state(self, full_state, indices):
        """Extract states for specific symbol indices."""
        if isinstance(full_state, list):
            return [self._extract_batch_state(s, indices) for s in full_state]
        elif isinstance(full_state, tuple):
            return tuple(self._extract_batch_state(s, indices) for s in full_state)
        elif isinstance(full_state, dict):
            return {k: self._extract_batch_state(v, indices) for k, v in full_state.items()}
        elif isinstance(full_state, torch.Tensor):
            return full_state[indices]
        else:
            return full_state
    
    def _update_batch_state(self, full_state, new_state, indices):
        """Update states at specific symbol indices."""
        if isinstance(full_state, list):
            for i, s in enumerate(full_state):
                self._update_batch_state(s, new_state[i], indices)
        elif isinstance(full_state, tuple):
            for i, s in enumerate(full_state):
                self._update_batch_state(s, new_state[i], indices)
        elif isinstance(full_state, dict):
            for k in full_state:
                self._update_batch_state(full_state[k], new_state[k], indices)
        elif isinstance(full_state, torch.Tensor):
            full_state[indices] = new_state


def main():
    parser = argparse.ArgumentParser(description="Horizontal batching inference")
    parser.add_argument("--host", type=str, default="localhost", help="Server hostname")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument(
        "--num-symbols",
        type=int,
        default=20,
        help="Number of symbols in the tradeable universe",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Hugging Face token to download the model",
    )

    args = parser.parse_args()
    client = NnInferenceClient(
        num_symbols=args.num_symbols,
        server_host=args.host,
        server_port=args.port,
        token=args.token,
    )

    client.run()


if __name__ == "__main__":
    main()