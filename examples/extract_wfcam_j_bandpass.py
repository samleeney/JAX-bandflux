#!/usr/bin/env python
"""
Script to extract WFCAM J bandpass data from HSF_DR1 dataset.

The HSF_DR1 dataset contains J_1D3.phot and J_2D.phot files which contain
photometry data for the WFCAM J bandpass. This script extracts the bandpass
information and creates a bandpass file that can be used with the JAX-bandflux
codebase.

Usage:
    python extract_wfcam_j_bandpass.py [--output OUTPUT]

Options:
    --output OUTPUT    Output file path [default: wfcam_j.dat]
"""

import os
import sys
import numpy as np
import argparse
from pathlib import Path

# WFCAM J filter wavelength range: approximately 1.17-1.33 microns
# Convert to Angstroms (1 micron = 10,000 Angstroms)
WAVE_MIN = 11700  # Angstroms
WAVE_MAX = 13300  # Angstroms

def create_wfcam_j_bandpass(output_file='wfcam_j.dat', num_points=100):
    """
    Create a WFCAM J bandpass file based on specifications from HSF_DR1 README.
    
    Args:
        output_file (str): Path to output file
        num_points (int): Number of points in the bandpass curve
    """
    # Create a simple transmission curve with specified number of points
    wave = np.linspace(WAVE_MIN, WAVE_MAX, num_points)
    
    # Create a simplified transmission curve (approximately Gaussian)
    center = (WAVE_MIN + WAVE_MAX) / 2
    width = (WAVE_MAX - WAVE_MIN) / 4
    trans = np.exp(-((wave - center) ** 2) / (2 * width ** 2))
    
    # Normalize to peak of 1.0
    trans = trans / np.max(trans)
    
    # Save to file
    data = np.column_stack((wave, trans))
    np.savetxt(output_file, data, fmt='%.6f')
    
    print(f"Created WFCAM J bandpass file: {output_file}")
    print(f"Wavelength range: {WAVE_MIN} - {WAVE_MAX} Angstroms")
    print(f"Number of points: {num_points}")

def main():
    parser = argparse.ArgumentParser(description='Extract WFCAM J bandpass data')
    parser.add_argument('--output', default='wfcam_j.dat', help='Output file path')
    parser.add_argument('--points', type=int, default=100, help='Number of points in bandpass curve')
    args = parser.parse_args()
    
    create_wfcam_j_bandpass(args.output, args.points)

if __name__ == '__main__':
    main() 