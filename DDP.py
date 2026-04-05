import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from test_utils import MyTrainDataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

from dlrm import DLRMDist
import torch.nn as nn
from functools import partial
from s3torchconnector import S3Checkpoint


class Trainer:
    def __init__(self,
            model: torch.nn.Module,
            train_data: DataLoader,
            optimizer: torch.optim.Optimizer,
            save_every: int,
            snapshot_path: str,
            aws_access_key: str, 
            aws_access_secret: str, 
            aws_region: str
        ) -> None:
        
        # e.g. lets say there are 2 nodes with 4 GPUs, each device has associated local and global ranks
        # node 0: local_ranks: [0, 1, 2, 3], global_ranks: [0, 1, 2, 3]
        # node 1: local_ranks: [0, 1, 2, 3], global_ranks: [4, 5, 6, 7]

        # Torchrun automatically creates and sets the LOCAL_RANK and RANK(global) env variables for each process it launches
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.model = model.to(self.local_rank)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path

        # define env variables for AWS
        os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = aws_access_secret
        os.environ["AWS_DEFAULT_REGION"] = aws_region
        self.checkpoint = S3Checkpoint(region="us-west-2")

        # if there is a snapshot, load it and continue from the last saved epoch
        try:
            self._load_snapshot()
        except:
            print("Start training from epoch 0!")
        
        # wrap the model with DDP
        self.model = DDP(self.model, device_ids=[self.local_rank])
    
    def _load_snapshot(self):
        loc = f"cuda:{self.local_rank}"
        with self.checkpoint.reader(self.snapshot_path) as reader:
            snapshot = torch.load(reader,  map_location=loc)
        self.model.module.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")
    
    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(), # state_dict under model.module instead of model besause of DDP wrapper
            "EPOCHS_RUN": epoch,
        }
        with self.checkpoint.writer(self.snapshot_path) as writer:
            torch.save(snapshot, writer)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

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

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))["target"])
        print(f"[GPU{self.global_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        # needed to ensure different data shuffling across epochs
        # Each epoch gets a different shuffle while still maintaining synchronized, non-overlapping partitions across all ranks
        self.train_data.sampler.set_epoch(epoch)
        for batch in self.train_data:
            dense_x = batch["dense"].to(self.local_rank)
            kjt = batch["sparse"].to(self.local_rank)
            targets = batch["target"].to(self.local_rank)
            self._run_batch(dense_x, kjt, targets)

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.global_rank == 0 and epoch % self.save_every == 0:
                # only save from one of the ranks since params are all identical after optimizer.step()
                self._save_snapshot(epoch)

def ddp_setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend="nccl") # TorchRun takes care of specifying current rank, world size and master env variables.

def load_model(device: str = "meta"):
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
        device = device,
        # bottom MLP dense params
        dense_dim = dense_dim,
        bottom_dense_layers = bottom_dense_layers,
        # top MLP dense params
        top_dense_layers = top_dense_layers
    )

def load_train_objs():
    train_set = MyTrainDataset(2048)  # load your dataset
    model = load_model(device = f"cuda:{os.environ["LOCAL_RANK"]}")
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer

def prepare_dataloader(dataset: Dataset, batch_size: int):
    collate_with_args = partial(MyTrainDataset._collate_batch, user_v=10, product_v=10)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False, # not used if sampler provided
        sampler=DistributedSampler(dataset), # makes sure that data chunks do not overlap across ranks
        collate_fn=collate_with_args
    )

def main(save_every: int, total_epochs: int, aws_access_key: str, aws_access_secret: str, aws_region: str, batch_size: int, snapshot_path: str):
    # distributed process group is initialized for each rank. It allows processes to discover each other and communicate
    ddp_setup()
    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, save_every, snapshot_path, aws_access_key, aws_access_secret, aws_region)
    trainer.train(total_epochs)
    destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--aws_access_key', type=str, help='Your AWS access key')
    parser.add_argument('--aws_access_secret', type=str, help='Your AWS secret key')
    parser.add_argument('--aws_region', default="us-west-2", type=str, help='AWS region (default: us-west-2)')
    parser.add_argument('--snapshot_path', default="s3://ml-paugre/dlrm-ranking/train/snapshot.pt", type=str, help='Snapshot Path for model artifacts')
    args = parser.parse_args()

    main(args.save_every, args.total_epochs, args.aws_access_key, args.aws_access_secret, args.aws_region, args.batch_size, args.snapshot_path)
