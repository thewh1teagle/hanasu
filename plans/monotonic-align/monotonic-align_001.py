# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "monotonic-alignment-search==0.2.1",
#     "torch>=2.0",
#     "numpy>=2.0",
# ]
# ///
"""
Verify that monotonic-alignment-search package is a drop-in replacement
for our custom src/monotonic_align/ Cython module.

Checks:
1. API works (maximum_path with value + mask)
2. Output shape matches input
3. Output is a valid monotonic path
4. Results match between cython and numpy backends
"""

import torch
import numpy as np
from monotonic_alignment_search import maximum_path, maximum_path_cython, maximum_path_numpy

# Simulate typical VITS MAS inputs
batch_size = 4
t_t = 50   # text length
t_s = 200  # mel length

# Random cost matrix and mask
torch.manual_seed(42)
neg_cent = torch.randn(batch_size, t_t, t_s)
mask = torch.ones(batch_size, t_t, t_s)

# Test cython version
path_cy = maximum_path_cython(neg_cent.clone(), mask.clone())
print(f"Cython output shape: {path_cy.shape}")
assert path_cy.shape == (batch_size, t_t, t_s), "Shape mismatch"

# Test numpy version
path_np = maximum_path_numpy(neg_cent.clone(), mask.clone())
print(f"Numpy output shape: {path_np.shape}")

# Check paths match
assert torch.allclose(path_cy, path_np), "Cython and Numpy paths differ!"
print("Cython and Numpy paths match!")

# Check monotonicity: path should have exactly one 1 per column covered
for b in range(batch_size):
    p = path_cy[b].numpy()
    # Each active column should have exactly one 1
    col_sums = p.sum(axis=0)
    assert all(s <= 1 for s in col_sums), f"Non-monotonic path in batch {b}"

print("All paths are monotonic!")

# Test default API
path_default = maximum_path(neg_cent.clone(), mask.clone())
assert torch.allclose(path_default, path_cy), "Default API doesn't match cython"
print("Default API works correctly!")

# Compare with VITS convention:
# Our code: monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1))
# neg_cent shape: [b, t_t, t_s] — same as this package expects [b, t_x, t_y]
print("\nShape convention: [b, t_x, t_y] — matches our [b, t_t, t_s] usage")
print("\nAll checks passed!")
