# GenomeAI2025
In this project, I am collaborating with some researchers to train deep learning models related to DNA sequences. I am involved in this project to build practical experience with preparing training data of DNA sequences for deep learning models and eventually implement deep learning models.

**First Assignment: Preprocessing Genomic Data for a seq2fun Task**

The goal of this assignment is to practice turning raw genomic data into input–output pairs for a deep learning model. The workflow is inspired by the DeepSEA dataset and will show you how to prepare data for sequence-to-function (seq2fun) models.

Relevant references: 

* https://www.nature.com/articles/nmeth.3547
* https://www.biorxiv.org/content/10.1101/2025.02.21.639224v1

**Background**

Deep learning in biology can be used to predict how DNA sequences relate to biological functions, such as protein binding or chromatin accessibility. To train such models, DNA and experimental data must be turned into a consistent format. This assignment focuses on building that preprocessing pipeline step by step.

**Dataset Provided**

* Human reference genome sequence HG38 (FASTA): http://ftp.ensembl.org/pub/release-106/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz
* Five BED files (hg38 coordinates), each with genomic regions from functional experiments such as ATAC-seq or ChIP-seq.