import pandas as pd
from pyfaidx import Fasta
import os
from tqdm import tqdm

def extract_sequences(bed_file, ref_genome_fasta, output_fasta, label_filter):
    df = pd.read_csv(bed_file, sep='\t', header=None, 
                     names=['chrom', 'start', 'end', 'count', 'density', 'label'])
    
    df = df[df['label'] == label_filter]
    genome = Fasta(ref_genome_fasta)
    
    extracted = 0
    skipped = 0
    os.makedirs(os.path.dirname(output_fasta), exist_ok=True)
    
    with open(output_fasta, 'w') as out_f:
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting label={label_filter}"):
            chrom = str(row['chrom'])
            start, end = int(row['start']), int(row['end'])
            
            if chrom not in genome or end > len(genome[chrom]):
                skipped += 1
                continue
                
            raw_seq = str(genome[chrom][start:end]).upper()
            if len(raw_seq) != (end - start) or raw_seq.count('N') > (len(raw_seq) * 0.1):
                skipped += 1
                continue
                
            header = f">window_{chrom}:{start}-{end}|label={int(row['label'])}|density={row['density']}"
            out_f.write(f"{header}\n{raw_seq}\n")
            extracted += 1

    print(f"Extracted {extracted} 100kb sequences for label {label_filter} (Skipped: {skipped}).")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    BED_FILE = os.path.join(BASE_DIR, "Data/Intermediates/windows_100kb.bed") 
    REFERENCE_GENOME = os.path.join(BASE_DIR, "Data/hg38/hg38.ml.fa")
    OUTPUT_FILE = os.path.join(BASE_DIR, "Data/Intermediates/positive_sequences.fasta")
    
    if not os.path.exists(BED_FILE):
        print(f"Error: {BED_FILE} not found. Run hotspot_definition.py first.")
    else:
        extract_sequences(BED_FILE, REFERENCE_GENOME, OUTPUT_FILE, label_filter=1)