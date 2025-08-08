"""Quick test script to verify multi-SN data loading and likelihood."""
import jax
import jax.numpy as jnp
import numpy as np
from jax_supernovae.multi_sn_utils import load_and_process_multiple_sne

# Enable float64
jax.config.update("jax_enable_x64", True)

# Test with 2 SNe
SN_NAMES = ['19agl', '19ahi']

print("Testing multi-SN data loading...")
try:
    sne_data, global_band_names, global_bridges = load_and_process_multiple_sne(
        SN_NAMES, data_dir='data', fix_z=True
    )
    print(f"✓ Successfully loaded data for {len(SN_NAMES)} supernovae")
    print(f"  Global bands: {global_band_names}")
    print(f"  Data shape (times): {sne_data['times'].shape}")
    print(f"  Valid mask shape: {sne_data['valid_mask'].shape}")
    print(f"  Fixed redshifts: {sne_data['fixed_z']}")
    
    # Check data integrity
    for i, sn_name in enumerate(SN_NAMES):
        n_points = sne_data['n_points'][i]
        n_bands = sne_data['n_bands'][i]
        print(f"\n  SN {sn_name}:")
        print(f"    Data points: {n_points}")
        print(f"    Bands used: {n_bands}")
        print(f"    Valid mask sum: {jnp.sum(sne_data['valid_mask'][i])}")
        
except Exception as e:
    print(f"✗ Error loading data: {e}")
    import traceback
    traceback.print_exc()

print("\nTest complete!")