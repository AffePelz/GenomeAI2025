# Genome AI 2025

In this projects, I am collaborating with researchers in optimizing and training deep learning models on genomic sequencing, understanding their functional implications and analysing and predicting genetic variants. This project requires background in mathematics (mashine learning and deep learning), informatics (mainly programming with Python and the library PyTorch) and bioinformatics (about DNA sequencing). Throughout this project, I am gaining practical experience in high-performance computation, designing, implementing and training deep learning models.

## First Assignment: Preprocessing Genomic Data

The goal of this assignment is to practice turning raw genomic data into input-output pairs for a deep learning model. The workshow is inspired by the DeepSEA dataset and will show you how to prepare data for sequence-to-function models. You will prepare training data similar to DeepSEA. The main idea is to link short DNA sequences with labels that show whether they overlap functional regions from the BED files. To simplify, only **chromosome 22** is required. You may use any bioinformatics tools or packages you like, but the final dataset must be usable in Python for deep learning, please document your enviroment for running for reproduce purpose.

1.  **Genome Binning** - Split the genome into consecutive, non-overlapping **200 bp bins**.
2.  **Label Generation** - For each bin, create a **1,000 bp DNA sequence** window (200 bp bin + 400 bp flanking bases on both sides).
3.  **Input Sequence Extraction** - For each bin, assign a label from the BED files: if ≥100 bp (half of the bin) overlaps a peak, label = 1; else 0. With 5 BED files, this gives a label vector of length 5.
4. **Data Representation** - Convert DNA to one-hot encoding (`A, C, G, T` → vectors of length 4). Consider doing one-hot encoding on-the-fly during data loading.
5.  **Output Dataset** – Each training example is `(X, y)` where:

      * `X`: one-hot DNA sequence (1000 × 4)
      * `y`: binary label vector (5,)
     Save in any efficient format that support random access during training.

**Background**

Deep learning in biology can be used to predict how DNA sequence relate to biological functions, such as protein binding or chromating accessibility. To train such models, DNA and experimental data must be turned into a consistent format. This assignment focuses on building that preprocessing pipeline step by step. 

**More Details about the Genomic Data Formats**

* FASTA: https://en.wikipedia.org/wiki/FASTA_format
* BED: https://en.wikipedia.org/wiki/BED_%28file_format%29

**Dataset Provided**

* Human reference genome sequence HG38 (FASTA): http://ftp.ensembl.org/pub/release-106/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz
* Five BED files (hg38 coordinates), each with genomic regions from functional experiments such as ATAC-seq or ChIP-seq.

Relevant references:

* https://www.nature.com/articles/nmeth.3547
* https://www.biorxiv.org/content/10.1101/2025.02.21.639224v1