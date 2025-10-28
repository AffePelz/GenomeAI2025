**Assignment: Preprocessing Genomic Data for a seq2fun Task**

**Objective**
The goal of this assignment is to practice turning raw genomic data into input–output pairs for a deep learning model. The workflow is inspired by the DeepSEA dataset and will show you how to prepare data for sequence-to-function (seq2fun) models.

Relevant references: 

* https://www.nature.com/articles/nmeth.3547
* https://www.biorxiv.org/content/10.1101/2025.02.21.639224v1


**Background**
Deep learning in biology can be used to predict how DNA sequences relate to biological functions, such as protein binding or chromatin accessibility. To train such models, DNA and experimental data must be turned into a consistent format. This assignment focuses on building that preprocessing pipeline step by step.

**Dataset Provided**

* Human reference genome sequence HG38 (FASTA): http://ftp.ensembl.org/pub/release-106/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz
* Five BED files (hg38 coordinates), each with genomic regions from functional experiments such as ATAC-seq or ChIP-seq.

More about the formats:

* FASTA: https://en.wikipedia.org/wiki/FASTA_format
* BED: https://en.wikipedia.org/wiki/BED_%28file_format%29

**Task Description**
You will prepare training data similar to DeepSEA. The main idea is to link short DNA sequences with labels that show whether they overlap functional regions from the BED files.

To simplify, only **chromosome 22** is required.

* Split the genome into **200 bp bins**.
* For each bin, assign a label from the BED files: if ≥100 bp (half of the bin) overlaps a peak, label = 1; else 0. With 5 BED files, this gives a label vector of length 5.
* For each bin, create a **1,000 bp DNA sequence** window (200 bp bin + 400 bp flanking bases on both sides).


**Hint:** You may use any bioinformatics tools or packages you like, but the final dataset must be usable in Python for deep learning, please document your enviroment for running for reproduce purpose.


**Suggested Steps**

1. **Genome binning** – Divide the genome into consecutive, non-overlapping 200 bp bins.
2. **Input sequence extraction** – For each bin, extract a 1,000 bp DNA sequence centered on it. Pad with `N`s if needed.
3. **Label generation** – For each bin, check overlaps with each of the 5 BED tracks. Build a 5-element binary label vector.
4. **Data representation** Convert DNA to one-hot encoding (`A, C, G, T` → vectors of length 4). Consider doing one-hot encoding on-the-fly during data loading.
5. **Output dataset** – Each training example is `(X, y)` where:

   * `X`: one-hot DNA sequence (1000 × 4)
   * `y`: binary label vector (5,)
     Save in any efficient format that support random access during training.

**Deliverables**

1. A reproducible workflow (script, Jupyter/Colab notebook, or bash pipeline). Document any environment setup (e.g., conda, pip).
2. A small output dataset from chromosome 22 (may be a few thousand training examples)
3. A short report including:
   * Brief explain your idea, steps, 
   * Number of bins created.
   * Label distribution for each track (plot)
   * Challenges faced and what you learned.

**Evaluation Criteria**
Your solution will be graded on:

1. **Efficiency** – Speed and memory use for preprocessing and dataset loading.
2. **Storage Optimization** – Compact size and fast read/write access.
3. **Scalability (most important)** – How your approach could handle 1,000 or even 10,000 tracks (e.g., parallelization, chunking, sparse storage, indexing).
4. **Reproducibility and Clarity** – Clean, modular code and clear documentation so others can easily reproduce results.

**Expected Outcome**
By completing this assignment, you will learn how to transform genome sequences and functional annotations into training-ready datasets for deep learning, while considering efficiency, storage, and scalability.
