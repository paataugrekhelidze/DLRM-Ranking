import argparse
import os
from functools import partial
from typing import Dict, Any

import torch
import torch.distributed as dist
import torch.nn as nn
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchrec import EmbeddingBagCollection
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.model_parallel import DMPCollection
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter

from dlrm import DLRMDist
from test_utils import MyTrainDataset
from s3torchconnector.dcp import S3StorageReader, S3StorageWriter

import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.stateful import Stateful


class AppState(Stateful):
    """This is a useful wrapper for checkpointing the Application State. Since this object is compliant
    with the Stateful protocol, DCP will automatically call state_dict/load_stat_dict as needed in the
    dcp.save/load APIs.

    Note: We take advantage of this wrapper to hande calling distributed state dict methods on the model
    and optimizer.
    """

    def __init__(self, model, optimizer=None, epoch: int = -1):
        self.model = model
        self.optimizer = optimizer
        self.epoch = epoch

    def state_dict(self):
        state = {
            "model": self.model.state_dict(),
            "epoch": torch.tensor(self.epoch, dtype=torch.int64),
        }
        if self.optimizer is not None:
            state["optim"] = self.optimizer.state_dict()
        return state

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict["model"])
        if self.optimizer is not None and "optim" in state_dict:
            self.optimizer.load_state_dict(state_dict["optim"])
        self.epoch = int(state_dict["epoch"].item())

def load_model() -> DLRMDist:
    feat_tables = {
        "product_table": {
            "vocab_size": 11,
            "features": ["product"]
        },
        "user_table": {
            "vocab_size": 11,
            "features": ["user"]
        }
    }

    emb_dim = 32
    dense_dim = 5
    bottom_dense_layers = [64, 32]
    top_dense_layers = [128, 32, 1]

    return DLRMDist(
        feat_tables = feat_tables,
        emb_dim = emb_dim,
        device = "meta",
        # bottom MLP dense params
        dense_dim = dense_dim,
        bottom_dense_layers = bottom_dense_layers,
        # top MLP dense params
        top_dense_layers = top_dense_layers
    )

def load_train_objs():
    train_set = MyTrainDataset(2048)
    model = load_model()
    return train_set, model


def build_sparse_sharder(sparse_lr: float) -> EmbeddingBagCollectionSharder:
    fused_params = {
        # Sparse IDs are touched unevenly, and Adagrad often works well for that because it automatically shrinks the effective step size for frequently updated rows while letting rare rows move more aggressively. (maintains state to adapt)
        # rowwise means the optimizer state is stored per row rather than per individual embedding element, less expensive for large embedding tables

        "optimizer": EmbOptimType.EXACT_ROWWISE_ADAGRAD,
        "learning_rate": sparse_lr,
    }
    return EmbeddingBagCollectionSharder(fused_params=fused_params)

@torch.no_grad()
def init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        module.weight.fill_(1.0)
        if module.bias is not None:
            module.bias.zero_()
    elif isinstance(module, EmbeddingBagCollection):
        for param in module.parameters():
            nn.init.kaiming_normal_(param)


def init_distributed() -> tuple[int, int, int, torch.device]:
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{local_rank}")

    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl")
    return local_rank, global_rank, world_size, device


# def build_2d_layout(
#     world_size: int,
#     sharding_group_size: int,
#     use_inter_host_allreduce: bool,
# ) -> tuple[list[list[int]], list[list[int]]]:
#     if world_size % sharding_group_size != 0:
#         raise ValueError(
#             f"world_size={world_size} must be divisible by sharding_group_size={sharding_group_size}"
#         )

#     if use_inter_host_allreduce:
#         sharding_groups = [
#             list(range(start, start + sharding_group_size))
#             for start in range(0, world_size, sharding_group_size)
#         ]
#     else:
#         replica_count = world_size // sharding_group_size
#         sharding_groups = [
#             [local_shard_rank * replica_count + replica_id for local_shard_rank in range(sharding_group_size)]
#             for replica_id in range(replica_count)
#         ]

#     replica_groups = [list(group) for group in zip(*sharding_groups)]
#     return sharding_groups, replica_groups


def get_model_replica_rank(
    global_rank: int,
    world_size: int,
    sharding_group_size: int,
    use_inter_host_allreduce: bool,
) -> int:
    replica_count = world_size // sharding_group_size
    if use_inter_host_allreduce:
        return global_rank // sharding_group_size
    return global_rank % replica_count


def prepare_dataloader(
    dataset: Dataset,
    batch_size: int,
    model_replica_rank: int,
    model_replica_count: int,
) -> DataLoader:
    collate_with_args = partial(MyTrainDataset._collate_batch, user_v=10, product_v=10)
    sampler = DistributedSampler(
        dataset,
        num_replicas=model_replica_count,
        # expects model replica rank, not global rank. ranks [0, 2] form model_replica 0, ranks [1, 3] form model_replica 1
        rank=model_replica_rank,
        shuffle=False,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=sampler,
        collate_fn=collate_with_args,
    )

class Trainer:
    def __init__(self,
            model: torch.nn.Module,
            DMPConfig: Dict[str, Any],
            train_data: DataLoader,
            save_every: int,
            snapshot_path: str,
            aws_access_key: str, 
            aws_access_secret: str, 
            aws_region: str
        ) -> None:
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.device = DMPConfig["device"]
        self.global_rank = int(os.environ["RANK"])
        self.train_data = train_data
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        self.aws_region = aws_region

        # define env variables for AWS
        os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = aws_access_secret
        os.environ["AWS_DEFAULT_REGION"] = aws_region

        self.model = DMPCollection(
            module=model,
            device=DMPConfig["device"],
            plan=DMPConfig["plan"],
            world_size=DMPConfig["world_size"],
            sharding_group_size=DMPConfig["sharding_group_size"],
            global_pg=DMPConfig["global_pg"],
            sharders=DMPConfig["sharders"],
            use_inter_host_allreduce=DMPConfig["use_inter_host_allreduce"],
        )
        self.model.apply(init_weights)

        dense_optimizer = KeyedOptimizerWrapper(
            dict(in_backward_optimizer_filter(self.model.named_parameters())),
            lambda params: torch.optim.SGD(params, lr=DMPConfig["dense_lr"]),
        )
        self.optimizer = CombinedOptimizer([self.model.fused_optimizer, dense_optimizer])
        self.app_state = AppState(self.model, self.optimizer)

        try:
            self._load_snapshot()
        except Exception as exc:
            print(f"Start training from epoch 0: {exc}")

    def _load_snapshot(self):
        state_dict = {"app": self.app_state}
        storage_reader = S3StorageReader(region=self.aws_region, path=self.snapshot_path)
        dcp.load(
            state_dict=state_dict,
            storage_reader=storage_reader,
        )
        self.epochs_run = self.app_state.epoch + 1
        print(f"Resuming training from epoch {self.epochs_run}")
    
    def _save_snapshot(self, epoch):
        self.app_state.epoch = epoch
        state_dict = {"app": self.app_state}
        storage_writer = S3StorageWriter(region=self.aws_region, path=self.snapshot_path)
        # creates per-rank payload: __0_0.distcp, __1_0.distcp, ..., __<world_size-1>_0.distcp and metadata, which maps logical keys, such as model/optimizer, to byte ranges in *.distcp
        # file pairs (__0_0.distcp, __1_0.distcp) and (__2_0.distcp, __3_0.distcp) are not necessarily similar, even if the runtime replica symmetry is [0,1] and [2,3], the save planner does not necessarily preserve that symmetry in file sizes.
        dcp.save(state_dict=state_dict, storage_writer=storage_writer)
        print(f"Epoch {epoch} distributed snapshot saved at {self.snapshot_path}")

    def _run_batch(self, dense_x, kjt, targets):
        self.optimizer.zero_grad()
        output = self.model(dense_x, kjt)
        loss = nn.functional.binary_cross_entropy_with_logits(
                input=output,
                target=targets,
                reduction="mean",
            )
        loss.backward()
        self.optimizer.step()
        self.model.sync()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))["target"])
        print(f"[GPU{self.global_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        # needed to ensure different data shuffling across epochs
        # Each epoch gets a different shuffle while still maintaining synchronized, non-overlapping partitions across all ranks
        self.train_data.sampler.set_epoch(epoch)
        for batch in self.train_data:
            dense_x = batch["dense"].to(self.device)
            kjt = batch["sparse"].to(self.device)
            targets = batch["target"].to(self.device)
            self._run_batch(dense_x, kjt, targets)

    def train(self, max_epochs: int):
        self.model.train()
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            dist.barrier()
            if epoch % self.save_every == 0:
                self._save_snapshot(epoch)
            dist.barrier()


def main(
        total_epochs: int,
        save_every: int,
        batch_size: int, 
        aws_access_key: str,
        aws_access_secret: str,
        aws_region: str,
        sharding_group_size: int,
        snapshot_path: str, 
        use_inter_host_allreduce: bool
    ) -> None:
    _, global_rank, world_size, device = init_distributed()

    if world_size % sharding_group_size != 0:
        raise ValueError(
            f"world_size={world_size} must be divisible by sharding_group_size={sharding_group_size}"
        )

    replica_count = world_size // sharding_group_size
    model_replica_rank = get_model_replica_rank(
        global_rank=global_rank,
        world_size=world_size,
        sharding_group_size=sharding_group_size,
        use_inter_host_allreduce=use_inter_host_allreduce,
    )

    # DMPCollection(init_parameters=True) will take care of loadding the models to the right devices
    dataset, model = load_train_objs()
    
    planner = EmbeddingShardingPlanner(
        topology=Topology(
            world_size=world_size,
            local_world_size=sharding_group_size,
            compute_device=device.type,
        )
    )
    sparse_sharder = build_sparse_sharder(sparse_lr=0.05)
    DMPConfig ={
        "sharders": [sparse_sharder],
        "global_pg": dist.group.WORLD,
        "use_inter_host_allreduce": use_inter_host_allreduce,
        "device": device,
        "world_size": world_size,
        "sharding_group_size": sharding_group_size,
        "dense_lr": 1e-3,
    }
    DMPConfig["plan"] = planner.collective_plan(model, DMPConfig["sharders"], DMPConfig["global_pg"])

    train_data = prepare_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        model_replica_rank=model_replica_rank,
        model_replica_count=replica_count,
    )
    trainer = Trainer(model, DMPConfig, train_data, save_every, snapshot_path, aws_access_key, aws_access_secret, aws_region)
    trainer.train(total_epochs)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TorchRec 2D parallelism smoke test")
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--aws_access_key', type=str, help='Your AWS access key')
    parser.add_argument('--aws_access_secret', type=str, help='Your AWS secret key')
    parser.add_argument('--aws_region', default="us-west-2", type=str, help='AWS region (default: us-west-2)')
    parser.add_argument('--snapshot_path', default="s3://<MYBUCKET>/dlrm-ranking/train/dist", type=str, help='Distributed checkpoint root path')
    parser.add_argument("--sharding_group_size", default=2, type=int)
    parser.add_argument(
        "--use_inter_host_allreduce",
        action="store_true",
        help="Use contiguous sharding groups like [0, 1] and [2, 3] instead of alternating groups.",
    )
    args = parser.parse_args()

    main(
        total_epochs=args.total_epochs,
        save_every=args.save_every,
        batch_size=args.batch_size,
        aws_access_key=args.aws_access_key,
        aws_access_secret=args.aws_access_secret,
        aws_region=args.aws_region,
        snapshot_path=args.snapshot_path,
        sharding_group_size=args.sharding_group_size,
        use_inter_host_allreduce=args.use_inter_host_allreduce,
    )
