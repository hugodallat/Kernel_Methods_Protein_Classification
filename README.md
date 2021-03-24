# Kernel methods for DNA sequence classification

Kernel machine learning methods for predicting whether a DNA sequence region is binding site to a specific transcription factor.

Transcription factors (TFs) are regulatory proteins that bind specific sequence motifs in the genome to activate or repress transcription of target genes.
Genome-wide protein-DNA binding maps can be profiled using some experimental techniques and thus all genomics can be classified into two classes for a TF of interest: bound or unbound.
In this challenge, we will work with three datasets corresponding to three different TFs.

The Datahandler class contains two standard embedding functions for string data (vanilla k-spectrum embedding and (k,m)-mismatch spectrum embedding). The Kernel class allows to perform every function required in
order to implement a kernel method for machine learning.

Everything is implemented solely with numpy, pandas and cvxopt.