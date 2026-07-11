import pandas as pd
from pyfaidx import Fasta
import os
from tqdm import tqdm

def extract_pu_sequences(bed_file, ref_genome_fasta, output_fasta, max_n_ratio=0.1):
    """
    Extracts sequences for all windows in the master BED file.
    Calculates density and various hotspot labels.
    """
    print(f"Loading master windows from {bed_file}...")
    df = pd.read_csv(bed_file, sep='\t')
    
    genome = Fasta(ref_genome_fasta)
    
    extracted = 0
    skipped = 0
    os.makedirs(os.path.dirname(output_fasta), exist_ok=True)
    
    with open(output_fasta, 'w') as out_f:
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting 100kb sequences"):
            chrom = str(row['chrom'])
            start, end = int(row['start']), int(row['end'])
            
            if chrom not in genome or end > len(genome[chrom]):
                skipped += 1
                continue
                
            raw_seq = str(genome[chrom][start:end]).upper()
            
            # Basic validation
            if len(raw_seq) != (end - start) or raw_seq.count('N') > (len(raw_seq) * max_n_ratio):
                skipped += 1
                continue
            
            # Construct header with all paper-relevant labels
            header = (f">win_{chrom}:{start}-{end}|"
                      f"density={row['density']:.6f}|"
                      f"l99={int(row['label_99'])}|"
                      f"l99.5={int(row['label_99.5'])}|"
                      f"l99.9={int(row['label_99.9'])}|"
                      f"lind={int(row['label_indiv'])}")
            
            out_f.write(f"{header}\n{raw_seq}\n")
            extracted += 1

    print(f"\nExtraction Summary:")
    print(f"-> Successfully extracted: {extracted}")
    print(f"-> Skipped (missing/N-rich): {skipped}")
    print(f"-> Saved to: {output_fasta}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    BED_FILE = os.path.join(BASE_DIR, "Data/Intermediates/windows_100kb_master.bed") 
    REFERENCE_GENOME = os.path.join(BASE_DIR, "Data/hg38/hg38.ml.fa")
    OUTPUT_FILE = os.path.join(BASE_DIR, "Data/Intermediates/all_windows_100kb.fasta")
    
    if not os.path.exists(BED_FILE):
        print(f"Error: {BED_FILE} not found. Run hotspot_definition.py first.")
    else:
        extract_pu_sequences(BED_FILE, REFERENCE_GENOME, OUTPUT_FILE)
