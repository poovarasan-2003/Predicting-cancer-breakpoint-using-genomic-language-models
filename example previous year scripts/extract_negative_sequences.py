from Bio import SeqIO
import pandas as pd
import numpy as np
import random
import argparse
from intervaltree import IntervalTree

def load_fragile_regions(bed_file):
    """
    Load fragile regions from BED file into interval trees for efficient overlap checking.
    
    Returns:
    --------
    dict: Dictionary with chromosome names as keys and IntervalTree objects as values
    """
    print(f"Loading fragile regions from: {bed_file}")
    fragile_trees = {}
    
    try:
        # Read BED file
        bed_df = pd.read_csv(bed_file, sep='\t', header=None)
        print(f"BED file shape: {bed_df.shape}")
        
        # Assign column names
        if bed_df.shape[1] >= 3:
            bed_df.columns = ['chrom', 'start', 'end'] + [f'col{i}' for i in range(3, bed_df.shape[1])]
        else:
            raise ValueError("BED file must have at least 3 columns")
        
        # Show first few entries
        print("First 5 entries in BED file:")
        print(bed_df.head())
        
        # Build interval trees for each chromosome
        for _, row in bed_df.iterrows():
            chrom = str(row['chrom'])  # Ensure string type
            if chrom not in fragile_trees:
                fragile_trees[chrom] = IntervalTree()
            fragile_trees[chrom].addi(int(row['start']), int(row['end']))
        
        print(f"Loaded {len(bed_df)} fragile regions across {len(fragile_trees)} chromosomes")
        print(f"Chromosomes with fragile regions: {sorted(fragile_trees.keys())[:10]}...")
    except Exception as e:
        print(f"Error loading BED file: {e}")
        raise
    
    return fragile_trees

def get_chromosome_sizes(fasta_file):
    """
    Get the size of each chromosome from the FASTA file.
    
    Returns:
    --------
    dict: Dictionary with chromosome names as keys and lengths as values
    """
    print(f"Getting chromosome sizes from: {fasta_file}")
    chrom_sizes = {}
    
    try:
        for record in SeqIO.parse(fasta_file, "fasta"):
            chrom_sizes[record.id] = len(record.seq)
            print(f"  {record.id}: {len(record.seq):,} bp")
        
        print(f"Found {len(chrom_sizes)} chromosomes")
    except Exception as e:
        print(f"Error reading FASTA file: {e}")
        raise
    
    return chrom_sizes

def check_overlap(chrom, start, end, fragile_trees, buffer_size=1000):
    """
    Check if a region overlaps with any fragile region (with optional buffer).
    
    Parameters:
    -----------
    buffer_size : int
        Buffer around fragile regions to avoid (default: 1000bp)
    """
    if chrom not in fragile_trees:
        return False
    
    # Check with buffer
    overlaps = fragile_trees[chrom].overlap(start - buffer_size, end + buffer_size)
    return len(overlaps) > 0

def get_positive_sequence_stats(positive_fasta):
    """
    Analyze positive sequences to match their characteristics.
    Enhanced to return actual length list for sampling.
    """
    lengths = []
    gc_contents = []
    sequences_info = []
    
    print(f"Reading positive sequences from: {positive_fasta}")
    try:
        for i, record in enumerate(SeqIO.parse(positive_fasta, "fasta")):
            seq = str(record.seq).upper()
            seq_len = len(seq)
            lengths.append(seq_len)
            
            # Calculate GC content
            gc_count = seq.count('G') + seq.count('C')
            gc_content = gc_count / len(seq) if len(seq) > 0 else 0
            gc_contents.append(gc_content)
            
            sequences_info.append({
                'id': record.id,
                'length': seq_len
            })
            
            if i < 5:  # Show first few sequences
                print(f"  Seq {i+1}: {record.id}, length={len(seq)}")
    except Exception as e:
        print(f"Error reading positive FASTA file: {e}")
        return None
    
    if lengths:
        print(f"\nPositive sequence statistics:")
        print(f"  Count: {len(lengths)}")
        print(f"  Length: min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.1f}, std={np.std(lengths):.1f}")
        print(f"  GC content: mean={np.mean(gc_contents):.3f}, std={np.std(gc_contents):.3f}")
        
        # Show length distribution
        print(f"\nLength distribution (showing first 20 unique lengths):")
        unique_lengths = sorted(list(set(lengths)))
        for length in unique_lengths[:20]:
            count = lengths.count(length)
            print(f"  Length {length}: {count} sequences ({count/len(lengths)*100:.1f}%)")
        
        return {
            'count': len(lengths),
            'lengths': lengths,  # Return actual lengths for sampling
            'min_length': min(lengths),
            'max_length': max(lengths),
            'mean_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'mean_gc': np.mean(gc_contents)
        }
    else:
        print("No sequences found in positive FASTA file!")
        return None

def generate_random_regions(chrom_sizes, fragile_trees, num_regions=1000, 
                          region_lengths=None, max_attempts=10000, buffer_size=1000):
    """
    Generate random regions that don't overlap with fragile sites.
    Modified to use actual length distribution from positive sequences.
    
    Parameters:
    -----------
    region_lengths : list
        List of lengths to sample from (from positive sequences)
    """
    random_regions = []
    
    # Filter out small chromosomes and non-standard ones
    valid_chroms = [chrom for chrom, size in chrom_sizes.items() 
                   if size > 10000 and  # Reasonable minimum size
                   (chrom.startswith('chr') and len(chrom) <= 6)]  # chr1-22, chrX, chrY
    
    print(f"\nGenerating {num_regions} random regions")
    print(f"Using buffer of {buffer_size}bp around fragile sites")
    print(f"Valid chromosomes for placement: {len(valid_chroms)}")
    
    # Sample lengths from positive sequences with replacement
    if region_lengths is not None:
        # Use random.choices for sampling with replacement to match exact count
        sampled_lengths = random.choices(region_lengths, k=num_regions)
        print(f"Sampled lengths from positive sequences (mean: {np.mean(sampled_lengths):.1f}, std: {np.std(sampled_lengths):.1f})")
    else:
        sampled_lengths = [1000] * num_regions  # default
    
    failed_regions = []
    attempts_per_region = []
    
    for i, current_length in enumerate(sampled_lengths):
        success = False
        attempts = 0
        
        # Filter chromosomes that can accommodate this length
        suitable_chroms = [chrom for chrom in valid_chroms 
                          if chrom_sizes[chrom] > current_length + 2*buffer_size]
        
        if not suitable_chroms:
            print(f"Warning: No chromosome can accommodate length {current_length}")
            failed_regions.append((i, current_length))
            continue
        
        while not success and attempts < max_attempts:
            # Try random chromosome and position
            chrom = random.choice(suitable_chroms)
            chrom_size = chrom_sizes[chrom]
            
            # Calculate valid start range
            max_start = chrom_size - current_length - 1
            if max_start <= 0:
                attempts += 1
                continue
                
            start = random.randint(0, max_start)
            end = start + current_length
            
            # Check overlap with fragile regions
            if not check_overlap(chrom, start, end, fragile_trees, buffer_size):
                random_regions.append({
                    'chrom': chrom,
                    'start': start,
                    'end': end,
                    'strand': random.choice(['+', '-']),
                    'name': f"random_region_{i+1}",
                    'length': current_length
                })
                success = True
                attempts_per_region.append(attempts + 1)
            
            attempts += 1
        
        if not success:
            print(f"Failed to place region {i+1} of length {current_length} after {attempts} attempts")
            failed_regions.append((i, current_length))
        
        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{num_regions} regions...")
    
    print(f"\nSuccessfully generated {len(random_regions)}/{num_regions} non-overlapping regions")
    if failed_regions:
        print(f"Failed to place {len(failed_regions)} regions")
    if attempts_per_region:
        print(f"Average attempts per successful region: {np.mean(attempts_per_region):.1f}")
    
    return random_regions, failed_regions

def generate_with_retry(chrom_sizes, fragile_trees, target_count, positive_lengths, 
                       initial_buffer_size=1000, min_buffer_size=100):
    """
    Generate exactly target_count regions with retry logic.
    Gradually reduces buffer size if struggling to find enough regions.
    """
    all_regions = []
    remaining = target_count
    current_buffer = initial_buffer_size
    iteration = 0
    
    print(f"\nAttempting to generate exactly {target_count} regions...")
    
    while remaining > 0 and current_buffer >= min_buffer_size:
        iteration += 1
        print(f"\nIteration {iteration}: Generating {remaining} regions with buffer={current_buffer}bp")
        
        batch_regions, failed = generate_random_regions(
            chrom_sizes, fragile_trees, 
            num_regions=remaining,
            region_lengths=positive_lengths,
            buffer_size=current_buffer,
            max_attempts=10000
        )
        
        all_regions.extend(batch_regions)
        remaining = target_count - len(all_regions)
        
        if remaining > 0:
            # Reduce buffer size for next iteration
            current_buffer = int(current_buffer * 0.8)
            print(f"Still need {remaining} regions. Reducing buffer to {current_buffer}bp...")
    
    if len(all_regions) < target_count:
        print(f"\nWarning: Could only generate {len(all_regions)}/{target_count} regions")
        print(f"Consider reducing initial buffer size or relaxing chromosome constraints")
    else:
        print(f"\nSuccess: Generated all {target_count} regions!")
    
    return all_regions[:target_count]  # Ensure we don't exceed target

def extract_sequences(fasta_file, regions, output_file):
    """
    Extract sequences for the generated regions.
    """
    print(f"\nLoading genome from: {fasta_file}")
    genome = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        genome[record.id] = str(record.seq)
    
    print(f"Extracting sequences for {len(regions)} regions")
    
    sequences_written = 0
    with open(output_file, 'w') as f:
        for region in regions:
            chrom = region['chrom']
            if chrom not in genome:
                print(f"Warning: Chromosome {chrom} not found in FASTA")
                continue
            
            # Extract sequence
            seq = genome[chrom][region['start']:region['end']]
            
            # Reverse complement if negative strand
            if region['strand'] == '-':
                seq = reverse_complement(seq)
            
            # Write to file
            header = f">{region['name']}|{chrom}:{region['start']}-{region['end']}:{region['strand']}|length={region['length']}"
            f.write(f"{header}\n")
            
            # Format sequence with 60 characters per line
            for i in range(0, len(seq), 60):
                f.write(seq[i:i+60] + "\n")
            
            sequences_written += 1
    
    print(f"Sequences written to: {output_file}")
    print(f"Total sequences written: {sequences_written}")
    
    # Also save metadata
    metadata_file = output_file.replace('.fa', '_metadata.csv')
    pd.DataFrame(regions).to_csv(metadata_file, index=False)
    print(f"Metadata saved to: {metadata_file}")
    
    # Save summary statistics
    summary_file = output_file.replace('.fa', '_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"Negative Training Data Generation Summary\n")
        f.write(f"========================================\n")
        f.write(f"Total sequences generated: {len(regions)}\n")
        f.write(f"Sequences written: {sequences_written}\n")
        f.write(f"Length distribution:\n")
        lengths = [r['length'] for r in regions]
        f.write(f"  Min: {min(lengths)}\n")
        f.write(f"  Max: {max(lengths)}\n")
        f.write(f"  Mean: {np.mean(lengths):.1f}\n")
        f.write(f"  Std: {np.std(lengths):.1f}\n")
    print(f"Summary saved to: {summary_file}")

def reverse_complement(seq):
    """Return the reverse complement of a DNA sequence."""
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A',
                  'a': 't', 'c': 'g', 'g': 'c', 't': 'a',
                  'N': 'N', 'n': 'n'}
    return ''.join(complement.get(base, base) for base in reversed(seq))

def main():
    parser = argparse.ArgumentParser(
        description="Generate random non-fragile regions for Caduceus negative training data"
    )
    parser.add_argument("--genome", default="hg38.fa", 
                       help="Path to reference genome FASTA (default: hg38.fa)")
    parser.add_argument("--fragile-bed", default="merged_fragiles.bed", 
                       help="Path to fragile sites BED file (default: merged_fragiles.bed)")
    parser.add_argument("--positive-fasta", default=None,
                       help="Path to positive sequences to match characteristics (optional)")
    parser.add_argument("--output", default="negative_training_sequences.fa", 
                       help="Output FASTA file (default: negative_training_sequences.fa)")
    parser.add_argument("--num-regions", type=int, default=1000, 
                       help="Number of regions to generate (default: 1000, overridden by positive count)")
    parser.add_argument("--buffer", type=int, default=1000, 
                       help="Initial buffer around fragile sites in bp (default: 1000)")
    parser.add_argument("--min-buffer", type=int, default=100,
                       help="Minimum buffer size when retrying (default: 100)")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--use-retry", action="store_true",
                       help="Use retry mechanism to ensure exact count match")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print(f"Caduceus Negative Training Data Generator")
    print(f"=========================================")
    
    # Variables to store positive sequence info
    positive_lengths = None
    target_count = args.num_regions
    
    # Analyze positive sequences if provided
    if args.positive_fasta:
        pos_stats = get_positive_sequence_stats(args.positive_fasta)
        if pos_stats:
            # Use actual count and length distribution from positive sequences
            target_count = pos_stats['count']
            positive_lengths = pos_stats['lengths']
            print(f"\nMatching positive sequences: {target_count} sequences")
            print(f"Will sample lengths from positive sequence distribution")
    
    # Load fragile regions
    fragile_trees = load_fragile_regions(args.fragile_bed)
    
    # Get chromosome sizes
    chrom_sizes = get_chromosome_sizes(args.genome)
    
    # Generate random non-overlapping regions
    if args.use_retry and positive_lengths is not None:
        # Use retry mechanism for exact count
        random_regions = generate_with_retry(
            chrom_sizes, 
            fragile_trees, 
            target_count=target_count,
            positive_lengths=positive_lengths,
            initial_buffer_size=args.buffer,
            min_buffer_size=args.min_buffer
        )
    else:
        # Single attempt generation
        random_regions, _ = generate_random_regions(
            chrom_sizes, 
            fragile_trees, 
            num_regions=target_count,
            region_lengths=positive_lengths,
            buffer_size=args.buffer
        )
    
    # Extract sequences
    if random_regions:
        extract_sequences(args.genome, random_regions, args.output)
        
        print(f"\n{'='*50}")
        print(f"FINAL SUMMARY:")
        print(f"Generated {len(random_regions)} negative training sequences")
        if args.positive_fasta:
            print(f"Target was {target_count} sequences (from positive data)")
            if len(random_regions) == target_count:
                print("✓ Successfully matched positive sequence count!")
            else:
                print(f"✗ Warning: Generated {len(random_regions)}/{target_count} sequences")
        print(f"These regions do not overlap with fragile sites (with {args.buffer}bp initial buffer)")
        print(f"{'='*50}")
    else:
        print("ERROR: No regions could be generated!")

if __name__ == "__main__":
    main()
