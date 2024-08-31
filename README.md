## MutStab-ProteinMPNN
### About

Understanding the impact of missense mutations on protein stability is crucial for deciphering the mechanisms underlying genetic diseases. This understanding also provides important guidance for developing new therapeutic strategies and enhancing the performance of existing protein-based drugs. Based on their effects on protein stability, missense mutations can be categorized into destabilizing and stabilizing mutations. Experimental measurement of the impact of missense mutations on protein stability is often costly. In recent years, researchers have developed various computational methods to predict the effects of mutations on protein stability. However, these existing methods generally exhibit an imbalance in predicting destabilizing versus stabilizing mutations, with particularly poor performance in predicting stabilizing mutations. To address this, we have developed a machine learning-based predictive method, MutStab-ProteinMPNN, which focuses on improving the prediction accuracy for stabilizing mutations using protein structure information.

We propose an innovative data augmentation strategy that generates all possible mutations on the protein and integrates results from other predictive methods, resulting in the construction of a high-quality dataset containing 482,222 mutations. Our method is trained on this dataset and employs a transfer learning approach by extracting embedding features from the large protein model, ProteinMPNN. The architecture of our method is based on a Transformer Encoder, which focuses on the local information surrounding the mutation site—specifically, the 11 amino acid residues before and after the mutation site in the protein sequence—to predict the impact of the mutation on protein stability. Comparative analysis with other methods across multiple independent test sets demonstrates that our approach offers more reliable and stable predictive performance. Our method provides a new perspective and approach for accurately predicting the impact of mutations on protein stability, especially in cases where experimental data are limited, and is expected to have a positive impact in the fields of protein engineering and disease research.


### Installation


#### I. PREREQUISITES

<font size=4>
 

1. FoldX

   This is available at the FoldX website.

   http://foldxsuite.crg.eu/

2、Python packages: Pytorch

```
pip install torch
```


### Running





