# De Novo Antioxidant Peptide Design Via Machine Learning and DFT studies

![DFT](https://github.com/mephisto121/DeepGenAntiOxidantPeptide/assets/71381384/318a42cd-ee02-412e-a046-5d96155811f6)

## About

Welcome to the repository for our article: "De Novo Antioxidant Peptide Design Via Machine Learning and DFT Studies". This project revolves around the development of a deep generative model, leveraging GRU layers, to create antioxidant peptides.

### Project Overview

1. **Pretrained Generative Model**:
   We kick-started the project by crafting a pretrained generative model using TensorFlow. Refer to `02_GRU_Base.ipynb` for detailed insights.

2. **Fine-Tuning for Antioxidant Peptides**:
   We then fine-tuned this model to tailor its focus specifically towards generating antioxidant peptides. The fine-tuning process is documented in `03_GRU_TL.ipynb`.

3. **Peptide Generation and Classification**:
   Utilizing the fine-tuned model, we generated new peptide sequences (refer to `04_Generate_data_TL.ipynb`). Moreover, to predict antioxidant activity, we developed a classification model outlined in `05_Conv1d_Classification.ipynb`.

4. **Filtering and Synthesis**:
   Following generation and classification, we fiterd the generated sequences (`06_filter_gen_data.ipynb`, `07_analysis_filter_cluster.ipynb`) based on various criteria. The remaining peptides were then synthesized for further activity assessment.

### Dataset Availability

- The base model training dataset can be found [here](link_to_dataset).
- For training the fine-tuned generative and classification models, we utilized the dataset available [here](link_to_dataset).


