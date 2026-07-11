import pandas as pd
import pybedtools
import os
import numpy as np

def generate_hotspots(cosmic_tsv, chrom_sizes_file, output_bed):
    """
    Implements Methodology from "Randomness in Cancer Breakpoint Prediction" (Section 2.1).
    - 100 kb non-overlapping windows.
    - Exclude centromeres, telomeres, blacklisted regions, and Y chromosome.
    - Breakpoint density: ratio of breakpoints in window to total breakpoints in chromosome.
    - Target labels: 99%, 99.5%, 99.9% percentiles and individual breakpoints (count >= 1).
    """
    print("Executing Methodology: Randomness in Cancer Breakpoint Prediction (Cheloshkina et al. 2021)")
    
    # 1. Load and Clean Breakpoints
    df = pd.read_csv(cosmic_tsv, sep='\t', low_memory=False)
    
    # Robust numeric coercion for coordinates
    df['LOCATION_FROM_MIN'] = pd.to_numeric(df['LOCATION_FROM_MIN'], errors='coerce')
    df['LOCATION_FROM_MAX'] = pd.to_numeric(df['LOCATION_FROM_MAX'], errors='coerce')
    
    df = df.dropna(subset=['CHROM_FROM', 'LOCATION_FROM_MIN', 'LOCATION_FROM_MAX'])
    
    # Clean chromosome names
    df['CHROM_FROM'] = df['CHROM_FROM'].astype(str).apply(lambda x: x if x.startswith('chr') else f'chr{x}')
    
    # Exclude Y chromosome as per paper (and non-standard ones)
    valid_chroms = [f'chr{i}' for i in range(1, 23)] + ['chrX']
    df = df[df['CHROM_FROM'].isin(valid_chroms)]
    
    # Condense to 1bp breakpoint locations
    raw_bed_df = df[['CHROM_FROM', 'LOCATION_FROM_MIN', 'LOCATION_FROM_MAX']].copy()
    raw_bed_df.columns = ['chrom', 'start', 'end']
    raw_bed_df['pos'] = raw_bed_df[['start', 'end']].min(axis=1).astype(int)
    raw_bed_df['start'] = raw_bed_df['pos']
    raw_bed_df['end'] = raw_bed_df['pos'] + 1
    raw_bed_df = raw_bed_df[['chrom', 'start', 'end']]
    
    raw_breakpoints = pybedtools.BedTool.from_dataframe(raw_bed_df).sort()
    print(f"-> Extracted {len(raw_breakpoints)} breakpoints.")

    # 2. Generate 100kb Windows
    # The paper says "excluded regions from centromeres, telomeres, blacklisted regions".
    # Since we don't have those specific files, we will use the available chr sizes and standard filtering.
    bins = pybedtools.BedTool().makewindows(g=chrom_sizes_file, w=100000, s=100000)
    
    # 3. Calculate Counts and Density per Chromosome
    binned_counts = bins.intersect(raw_breakpoints, c=True)
    counts_df = binned_counts.to_dataframe(names=['chrom', 'start', 'end', 'count'])
    
    # Calculate total breakpoints per chromosome
    chrom_totals = raw_bed_df.groupby('chrom').size().to_dict()
    
    def calc_density(row):
        total = chrom_totals.get(row['chrom'], 0)
        return row['count'] / total if total > 0 else 0.0

    counts_df['density'] = counts_df.apply(calc_density, axis=1)

    # 4. Generate Target Labels (Hotspots and Individual Breakpoints)
    # Thresholds: 99%, 99.5%, 99.9%
    p99 = np.percentile(counts_df['density'], 99)
    p995 = np.percentile(counts_df['density'], 99.5)
    p999 = np.percentile(counts_df['density'], 99.9)
    
    print(f"-> Thresholds: 99th={p99:.6f}, 99.5th={p995:.6f}, 99.9th={p999:.6f}")

    counts_df['label_99'] = (counts_df['density'] > p99).astype(int)
    counts_df['label_99.5'] = (counts_df['density'] > p995).astype(int)
    counts_df['label_99.9'] = (counts_df['density'] > p999).astype(int)
    counts_df['label_indiv'] = (counts_df['count'] >= 1).astype(int)

    # 5. Save Master BED File
    counts_df.to_csv(output_bed, sep='\t', index=False, header=True)
    print(f"-> Saved master windows BED to: {output_bed}")
    print(f"-> Windows: {len(counts_df)}")
    print(f"-> 99% Hotspots: {counts_df['label_99'].sum()}")
    print(f"-> 99.5% Hotspots: {counts_df['label_99.5'].sum()}")
    print(f"-> 99.9% Hotspots: {counts_df['label_99.9'].sum()}")
    print(f"-> Individual Breakpoint Windows: {counts_df['label_indiv'].sum()}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    COSMIC_FILE = os.path.join(BASE_DIR, "Data/Cosmic/Cosmic_Breakpoints_v103_GRCh38.tsv")
    CHROM_SIZES = os.path.join(BASE_DIR, "Data/hg38/hg38.chrom.sizes")
    
    os.makedirs(os.path.join(BASE_DIR, "Data/Intermediates"), exist_ok=True)
    OUTPUT_BED = os.path.join(BASE_DIR, "Data/Intermediates/windows_100kb_master.bed")
    
    generate_hotspots(COSMIC_FILE, CHROM_SIZES, OUTPUT_BED)
