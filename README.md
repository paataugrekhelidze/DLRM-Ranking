
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
├── DDP.py                 # Defines Trainer for multi-node (GPU) DDP. The script uses a dummy model architecture and data to simply test DDP
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


## Distributed Data Parallelism using Torchrun
Torchrun simplifies DDP by automating tasks such as spawing multiple processes, defining env variables (RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, MASTER_PORT), automatic restart of the processes during failure (customize to restart from a snapshot), scaling from a single to multi-node cluster...
Make sure that you have gpu devices with proper nvidia drivers and run the following on each node!
```bash
torchrun \
# value could vary between nodes, this one hace 2 GPUs
--nproc-per-node=2 \
# total count of nodes, that way torchrun begins when all nodes are ready
--nnodes=1
# unique per node, next node should use node_rank=1
--node_rank=0 \
# unique for the process group
--rdzv_id=456 \
--rdzv_backend=c10d \
# pick one of the nodes and unused port
# my example is for a single node multi-worker job
--rdzv_endpoint=127.0.0.1:29603 \
# custom train script
DDP.py 50 10 --aws_access_key=<AWS-KEY> --aws_access_secret="<AWS-SECRET>"
```

## 2D parralelism using DMPCollection API
Amazing [blog](https://pytorch.org/blog/scaling-recommendation-2d-sparse-parallelism) worth mentioning. It explains the concept of how TorchRec's DMPCollection API achieves 2D parallelism efficiently:
- The 2 dimensions of parallelism are data parallelism, for dense architecture, and model parallelism, for really large embedding tables
- each rank has two groups, sharding and replica, associated to it
- equivalent shards for different model replicas are placed on the same node (which can contain multiple ranks) to maximize intra-node communication during all reduce synchronization, which is fasteer than inter-node communication.
    - e.g. 8 GPUs (T=8) with a sharding group size of 4 (L=4), will create 2 model replicas (G = T/L). 
    - Using a simple formula, [i, G+i, 2G+i, ...], sharding group 0 (G = 0) is has its model shards placed at [0, 2, 4, 8] and sharding group 1 (G=1) has its shards placed at [1, 3, 5, 7]. This increases chances that equal replica groups or equivalent shards of different replicas models are placed close to each other, so during parameter synchronization, all reduce can run average within the same node.

Unfortunately, there are not a lot of practical examples for the implementation, so the best way to understand the DMPCollection API is by reading the [source code](https://github.com/meta-pytorch/torchrec/blob/main/torchrec/distributed/model_parallel.py#L1014).

Here is how I understand 2D parallelism:
- Sparse tables are sharded across the ranks in a sharding group.
- Dense layers are replicated across all global ranks, this is fine if they use less memory compared to the embedding tables
- Ranks in the same sharding group cooperate to execute one logical sparse model.
- Different sharding groups are replica copies of that logical model.

Saving DMP models are made simple using torch.distributed.checkpoint (see [doc](https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html))

Torchrun can be used to execute 2D parallelized script as well:
```bash
# could use --standalone flag in DDP as well since I am using only a single node
# my example is running a single node with 4 GPUs, world_size = 4, sharding_groups = 2, replica_groups = 2
torchrun --standalone --nproc-per-node=4 2DP.py 70 10 --sharding_group_size 2 --aws_access_key=<AWS-KEY> --aws_access_secret="<AWS-SECRET>"
```

## References
1. [Deep Learning Recommendation Model for Personalization and Recommendation Systems](https://arxiv.org/pdf/1906.00091)
2. [The Architectural Implications of Facebook’s DNN-based Personalized Recommendation](https://arxiv.org/pdf/1906.03109)
3. [DLRM Github Repository](https://github.com/facebookresearch/dlrm/blob/main/dlrm_s_pytorch.py)
4. [Criteo 1TB Click Logs Dataset](https://huggingface.co/datasets/criteo/CriteoClickLogs/blob/main/README.md)
5. [Criteo Kaggle Dataset](https://www.kaggle.com/competitions/criteo-display-ad-challenge/overview)
6. [Compositional Embeddings Using Complementary Partitions for Memory-Efficient Recommendation Systems](https://arxiv.org/pdf/1909.02107)
7. [Mixed Dimension Embeddings with Application to Memory-Efficient Recommendation Systems](https://arxiv.org/pdf/1909.11810)
8. [Scaling Recommendation Systems Training to Thousands of GPUs with 2D Sparse Parallelism](https://pytorch.org/blog/scaling-recommendation-2d-sparse-parallelism)