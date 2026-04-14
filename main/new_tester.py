from datasets import load_dataset

# or load the separate splits if the dataset has train/validation/test splits
ds = load_dataset("songlab/genomes-brassicales-balanced-v1")
print(ds[0])