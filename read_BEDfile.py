import pandas as pd
import pybedtools
import numpy as np

# Load the bins
bins_bed = pybedtools.BedTool('chr22_bins_200bp.bed')
num_bins = len(bins_bed)
print(f"Number of bins created: {num_bins}")
# Create an array to store the labels (num_bins x 5)
labels_matrix = np.zeros((num_bins, 5), dtype=int)
bed_files = [f'track_{i+1}.bed' for i in range(5)]

for i, bed_file in enumerate(bed_files):
    track_bed = pybedtools.BedTool(bed_file)
    # Perform intersection
    # -a: bins, -b: track, -wao: write all original A fields plus overlap amount
    intersect_result = bins_bed.intersect(track_bed, wao=True)
    
    # Process the intersection results
    # The output columns are: bin_chr, bin_start, bin_end, track_chr, track_start, track_end, overlap_bp
    
    # Read into a pandas DataFrame for easy processing
    df = intersect_result.to_dataframe(header=None, names=['bin_chr', 'bin_start', 'bin_end', 
                                                           'track_chr', 'track_start', 'track_end', 'overlap_bp'])
    
    # Group by bin and find the total overlap with the track (a bin can overlap multiple peaks)
    overlap_df = df.groupby(['bin_chr', 'bin_start', 'bin_end'])['overlap_bp'].sum().reset_index()
    
    # Merge back to the full list of bins to ensure all bins are present
    bins_df = bins_bed.to_dataframe(header=None, names=['bin_chr', 'bin_start', 'bin_end'])
    labeled_df = bins_df.merge(overlap_df, on=['bin_chr', 'bin_start', 'bin_end'], how='left')
    
    # Fill NaN (bins with no overlap) with 0
    labeled_df['overlap_bp'] = labeled_df['overlap_bp'].fillna(0)
    
    # Apply the labeling rule: label = 1 if overlap >= 100 bp
    label_col = (labeled_df['overlap_bp'] >= 100).astype(int)
    labels_matrix[:, i] = label_col.values

# Save the labels matrix
np.save('chr22_labels.npy', labels_matrix)
print("Label generation complete.")