
# Deep Learning Recommendation Model (DLRMv1)

## Overview
This project implements the Deep Learning Recommendation Model (DLRMv1), previously a state-of-the-art architecture for personalized recommendation systems. DLRMv1 is designed to capture both explicit and implicit feature interactions, overcoming the limitations of traditional two-tower models by introducing early-stage user-item interaction layers. The implementation is inspired by the original Facebook DLRM [paper](https://arxiv.org/pdf/1906.00091) and supports advanced embedding compression techniques for memory efficiency.

## File Structure

```
DLRM-Ranking/
│
├── dlrm.py                # Main DLRM model implementation (PyTorch)
├── dlrm_data_pytorch.py   # Data generation and loading utilities for DLRM benchmarks
├── data_utils.py          # Downloading and preprocessing public datasets (Criteo, etc.)
├── demo.ipynb             # Example notebook: model instantiation and forward pass demo
├── tricks/                # Embedding compression modules
│   ├── qr_embedding_bag.py   # Quotient-Remainder Embedding implementation
│   └── md_embedding_bag.py   # Mixed-Dimension Embedding implementation
└── pyproject.toml         # Project metadata and dependencies
```

## Dataset
This project uses the [Criteo Kaggle Display Advertising Challenge dataset](https://www.kaggle.com/competitions/criteo-display-ad-challenge/overview), which contains 13 integer and 26 categorical features. The first column is the binary target (ad click). The goal is to maximize click-through rate (CTR) prediction accuracy.

## Evaluation Metrics
- Normalized Entropy / Normalized Entropy Rank
- Calibration / Calibration Rank
- AUC / AUC Rank

## DLRMv1 Architecture Summary
1. **Bottom-MLP**: Maps dense features to the same dimension as sparse feature embeddings.
2. **Interaction Layer**: Concatenates bottom-MLP output with sparse embeddings and computes all-to-all feature interactions (matrix multiplication and triangular extraction).
3. **Top-MLP**: Processes the combined features to produce the final prediction.

## Insights
- DLRM models are memory intensive due to large embedding tables for sparse features. This project includes advanced embedding compression:
    - **Quotient-Remainder Embedding**: Splits a large embedding into two smaller ones, reducing memory while avoiding hash collisions ([paper](https://arxiv.org/pdf/1909.02107)).
    - **Mixed-Dimension Embedding**: Adjusts embedding dimension based on feature cardinality and popularity ([paper](https://arxiv.org/pdf/1909.11810)).
- Embedding bags simplify categorical input representation for efficient batch processing.

## References
1. [Deep Learning Recommendation Model for Personalization and Recommendation Systems](https://arxiv.org/pdf/1906.00091)
2. [The Architectural Implications of Facebook’s DNN-based Personalized Recommendation](https://arxiv.org/pdf/1906.03109)
3. [DLRM Github Repository](https://github.com/facebookresearch/dlrm/blob/main/dlrm_s_pytorch.py)
4. [Criteo 1TB Click Logs Dataset](https://huggingface.co/datasets/criteo/CriteoClickLogs/blob/main/README.md)
5. [Criteo Kaggle Dataset](https://www.kaggle.com/competitions/criteo-display-ad-challenge/overview)
6. [Compositional Embeddings Using Complementary Partitions for Memory-Efficient Recommendation Systems](https://arxiv.org/pdf/1909.02107)
7. [Mixed Dimension Embeddings with Application to Memory-Efficient Recommendation Systems](https://arxiv.org/pdf/1909.11810)