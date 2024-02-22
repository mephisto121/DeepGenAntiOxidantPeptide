# De Novo Antioxidant Peptide Design Via Machine Learning and DFT studies

![DFT](https://github.com/mephisto121/DeepGenAntiOxidantPeptide/assets/71381384/6817d0e7-ade7-407a-9c0e-e1700a410f
This repository contains the source code for our article: De Novo Antioxidant Peptide Design Via Machine Learning and DFT studies.
In this project, we developed a deep generative model based on GRU layers to create novel antioxidant peptides. 
At first, A pretrained generative model was created using Tensorflow (02_GRU_Base.ipynb).
In the second step, a fine-tuned version of that model was trained to be specific towards antioxidant peptides (03_GRU_TL.ipynb).
The next step was to generate new peptide sequences from the fine-tuned model (04_Generate_data_TL.ipynb), and also develop a classification model (05_Conv1d_Classification.ipynb) to determine their predicted antioxidant activity.
At the end, we filter those remaining sequences (06_filter_gen_data.ipynb, 07_analysis_filter_cluster.ipynb) based on several factors and finally synthesis the remaining peptides for their activity.

### Dataset Availability
Please refer to the 

