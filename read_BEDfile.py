import pybedtools

def read_BED(file_read):
    bed = pybedtools.BedTool(file_read)
    
    # Convert to DataFrame
    df = bed.to_dataframe(dtype=str)

    # Keep only the first three columns
    diff = df.iloc[:,:3]

    # Focusing on only chromosome 22
    #diff = df[df['chrom']=='22']

    return diff

print(read_BED("bed_data/ENCFF052RRA.bed.gz"))
#print(read_BED('chr22.bed'))

# Task 1: Only focusing on chromosome 22
"""
grep '^22' bed_data/ENCFF052RRA.bed.gz > chr22.bed

In one of the BED files we received, we use grep to find chromosome 22 and make
a new BED file with only chromosome 22
"""