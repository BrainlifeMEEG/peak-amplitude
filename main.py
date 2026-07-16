"""
Compute peak (mean) amplitude within a frequency band from a PSD table.

This app loads a per-channel PSD table (TSV), averages the amplitude
within a given frequency band per channel, and saves the result plus a
histogram plot.

Inputs:
    - psd: Path to per-channel PSD TSV file
    - fmin, fmax: Frequency band bounds

Outputs:
    - out_dir/psd.tsv: Per-channel peak amplitude table
    - out_figs/hist_peak_amplitude.png: Histogram of peak amplitudes
    - product.json: Metadata about the computed peaks
"""

# Copyright (c) 2026 brainlife.io
#
# Author: Guiomar Niso

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'brainlife_utils'))

# Standard imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import shared utilities
from brainlife_utils import (
    load_config,
    setup_matplotlib_backend,
    ensure_output_dirs,
    create_product_json,
    add_info_to_product,
    add_image_to_product,
    require_config_keys
)

# Set up matplotlib for headless execution
setup_matplotlib_backend()

# Ensure output directories exist
ensure_output_dirs('out_dir', 'out_figs')

# Load configuration
config = load_config()
require_config_keys(config, ['psd', 'fmin', 'fmax'])

# == GET CONFIG VALUES ==
fname = config['psd']
fmin = config['fmin']
fmax = config['fmax']

# == LOAD DATA ==
df_psd = pd.read_csv(fname, sep='\t')
canales = df_psd['channels'].copy()

# Number of frequencies computed for the PSD
nfreqs = df_psd.shape[1]
df = df_psd.iloc[:, 1:nfreqs].copy()  # To avoid the case where changing df also changes df_psd
# List of frequencies
freqs = df.columns.to_numpy()
freqs = freqs.astype(float)
# PSD values
psd_welch = df.to_numpy()
# Number of channels
nchannels = psd_welch.shape[0]

# Extract the frequencies that fall inside the band
ifreqs = [i for i, f in zip(range(0, len(freqs)), freqs) if f > fmin and f < fmax]
band_freqs = np.take(freqs, ifreqs)

# ==== FIND PEAK MEAN AMPLITUDE VALUE ====
channel_peak = np.mean(psd_welch[:, ifreqs], axis=1)

# Average value across all channels
mean_peak = np.mean(channel_peak, axis=0)

# == SAVE FILE ==
df_alpha = pd.DataFrame(channel_peak, index=canales, columns=['peak'])
tsv_path = os.path.join('out_dir', 'psd.tsv')
df_alpha.to_csv(tsv_path, sep='\t')

# ==== PLOT FIGURE ====
plt.figure(1)
sns.set_theme(style="ticks")
sns.histplot(data=channel_peak, binwidth=0.25, kde=True, kde_kws={'cut': 10})
plt.xlabel('Peak amplitude')
sns.despine()
fig_path = os.path.join('out_figs', 'hist_peak_amplitude.png')
plt.savefig(fig_path)
plt.close()

# == CREATE PRODUCT.JSON ==
product_items = []
add_info_to_product(product_items, f'Mean peak amplitude across channels ({fmin}-{fmax} Hz): {mean_peak:.4g}', 'success')
add_image_to_product(product_items, 'Peak amplitude histogram', filepath=fig_path)
create_product_json(product_items)
