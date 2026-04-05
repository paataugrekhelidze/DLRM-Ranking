from typing import Dict, List, Union

import torch.nn as nn
import torch
from tricks.qr_embedding_bag import QREmbeddingBag
import math
import os
import time
from  torchrec import (
    EmbeddingBagCollection, 
    EmbeddingBagConfig, 
    PoolingType, 
    KeyedJaggedTensor
)

class DLRM(nn.Module):
    def __init__(self, input_size, b_mlp_layers, t_mlp_layers, emb_layers, qr_flag = True):
        """
        Deep Learning Recommendation Model implementation.

        Args:
            input_size (Int)
            b_mlp_layers (List): List of hidden layers for the bottom mlp architecture.
            t_mlp_layers (List): List of hidden layers for the top mlp architecture.
            emb_layers (List): List of K tuples, each specifying the embedding table dimensions.
        """
        super().__init__()
        b_layers = nn.ModuleList()
        t_layers = nn.ModuleList()

        # initialize bottom mlp
        for i in range(len(b_mlp_layers)):
            if i == 0:
                b_layers.append(nn.Linear(in_features=input_size, out_features=b_mlp_layers[i]))
            else:
                b_layers.append(nn.Linear(in_features=b_mlp_layers[i-1], out_features=b_mlp_layers[i]))
            b_layers.append(nn.ReLU())
        
        # flattened upper triangular of the interaction matrix
        # 1+K + 2N = (1+K)^2
        # N = (1+K)*(K) // 2

        # t_mlp_layers = dense features from botton_mlp + interactions
        K = len(emb_layers)
        t_mlp_input_size = b_mlp_layers[-1] + (1+K)*(K) // 2

        # initialize top mlp
        for i in range(len(t_mlp_layers)):
            if i == 0:
                t_layers.append(nn.Linear(in_features=t_mlp_input_size, out_features=t_mlp_layers[i]))
            else:
                t_layers.append(nn.Linear(in_features=t_mlp_layers[i-1], out_features=t_mlp_layers[i]))
            if i != len(t_mlp_layers)-1:
                t_layers.append(nn.ReLU())

        self.b_mlp = nn.Sequential(*b_layers)
        self.t_mlp = nn.Sequential(*t_layers)

        # initialize embedding tables
        self.emb_l = nn.ModuleList()        
        for i, shapes in enumerate(emb_layers):
            num_emb, emb_d = shapes
            if qr_flag:
                self.emb_l.append(QREmbeddingBag(num_categories=num_emb, embedding_dim=emb_d, num_collisions=int(math.sqrt(num_emb))))
            else:
                self.emb_l.append(nn.EmbeddingBag(num_embeddings=num_emb, embedding_dim=emb_d))
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            torch.nn.init.xavier_normal_(module.weight)

        
    def _interact(self, dense, sparse):
        """
        The following interact operation assumes only dot interactions between sparse and dense features
        Args:
            dense (torch.Tensor): dense input features of shape (B, D).
            sparse (list): list of K embedding lookups each of shape (B, D).
        Returns:
            Tensor output of shape (B, N) where N is the flattened upper triangular of second-order interactions.
        """
        # similar to factorization machine, combine features into a single matrix and run bmm against its transpose
        # get either upper or lower triangular since we do not need dup values
        
        # (B, (1+K)*D) -> (B, 1+K, D)
        batch_size, D = dense.shape
        T = torch.cat([dense] + sparse, dim=1).view((batch_size, -1, D))
        # print(f"T: {T.shape}")

        # (B, 1+K, D) x (B, D, 1+K) -> (B, 1+K, 1+K)
        Z = torch.bmm(T, torch.transpose(T, 1, 2))

        # print(f"Z: {Z.shape}")

        # get upper triangular for unique interactions, exlude diagonal
        row, col = torch.triu_indices(Z.shape[1], Z.shape[2], offset=1)
        # (B, 1+K, 1+K) -> (B, N) where N = (1+K)*(K) // 2
        Z_flat = Z[:, row, col]
        # print(f"Z_flat: {Z_flat.shape}")

        # combine original dense featues and flattened upper triangular of interactions
        # (B, N+D)
        combined = torch.cat([dense] + [Z_flat], dim=1)

        return combined
        

    def forward(self, x, emb_indices, emb_offsets):
        """
        Args:
            x (torch.Tensor): dense input features.
            emb_indices (torch.Tensor): embedding indices for k tables and B batch size of shape (k, B).
            emb_offsets (torch.Tensor): embedding offsets for k tables and B batch size of shape (k, B).
        Returns:
            Tensor output of shape (B, 1).
        """
        
        # step 1: score bottom MLP for dense features
        # (B, input) -> (B, D)
        b_mlp_out = self.b_mlp(x)

        # step 2: embedding lookup across all sparse features
        emb_out = []
        K = emb_indices.size(0)
        for k in range(K):
            emb_out.append(self.emb_l[k](emb_indices[k], emb_offsets[k]))
        
        # print(b_mlp_out.shape, len(emb_out))

        # step 3: calulate interaction matrix
        z = self._interact(b_mlp_out, emb_out)
        # print(z.shape)

        # step 4: score top MLP using output from the interaction op
        t_mlp_out = self.t_mlp(z)

        return t_mlp_out

class DLRMSolver:
    def __init__(self, 
                 model, 
                 data,
                 optimizer,
                 device, 
                 epochs, 
                 checkpoint_dir = "./checkpoints", 
                 checkpoint_every = 1, 
                 verbose = True, 
                 print_every = 1, 
                 reset = False
                ):
        self.model = model
        self.data = data
        self.device = device
        self.epochs = epochs
        self.verbose = verbose
        self.reset = reset
        self.print_every = print_every
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_every = checkpoint_every  
        self.optimizer = optimizer
        
        self.loss_history = []
        self.loss_history_batch = []

        self.model.to(device)
    
    def _step(self):

        total_loss = 0
        nbatches = len(self.data)
        # counter = 0
        for x, emb_offsets, emb_indices, target in self.data:
            # print(f"[{counter}/{nbatches}]")
            # counter += 1
            
            x = x.to(self.device)
            emb_indices = emb_indices.to(self.device)
            emb_offsets = emb_offsets.to(self.device)
            target = target.to(self.device).float()
            if target.dim() == 1:
                target = target.unsqueeze(1)


            self.optimizer.zero_grad(set_to_none=True)

            logits = self.model(x, emb_indices, emb_offsets)
            loss = nn.functional.binary_cross_entropy_with_logits(
                input=logits,
                target=target,
                reduction="mean",
            )

            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())
            self.loss_history_batch.append(float(loss.item()))

        return total_loss / nbatches

    
    def _save(self, loss, epoch, filepath):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint = {
            "epoch" : epoch,
            "loss" : loss,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved at {filepath}")

    def _load(self, filepath):
        if not os.path.exists(filepath):
            return None, None
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]
        return epoch, loss
    
    def train(self):
        start_epoch = 0
        if not self.reset:
            saved_epoch, saved_loss = self._load(os.path.join(self.checkpoint_dir, "last_checkpoint.pth"))

            # if loading a saved epoch, then continue from the last epoch
            if saved_epoch is not None:
                self.loss_history.append(saved_loss)
                print(f"Load [{saved_epoch}/{self.epochs}] Loss {self.loss_history[-1]:.6f}")
                start_epoch = saved_epoch + 1

        # Set the model to training mode
        self.model.train()

        for epoch in range(start_epoch, self.epochs):
            epoch_start_time = time.time()
            epoch_loss = self._step()
            epoch_end_time = time.time()
            
            if self.verbose and epoch % self.print_every == 0:
                print(f"[{epoch}/{self.epochs}] Loss: {epoch_loss:.6f} time: {(epoch_end_time - epoch_start_time) / 60.0}m")
            if epoch % self.checkpoint_every == 0:
                self._save(loss = epoch_loss, epoch = epoch, filepath = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth"))
                self._save(loss = epoch_loss, epoch = epoch, filepath = os.path.join(self.checkpoint_dir, f"last_checkpoint.pth"))
            
            self.loss_history.append(epoch_loss)



# sparse architecture using torchrec primitives such as EmbeddingBagCollection
# This enables optimizations such as embedding table lookups simultaneously across multiple tables with varying (jagged) features
# fused optimizers allows backward pass and step in a single kernel without doubling memory utilization, traditionally gradients require the same memory as the parameters
# enable DMP for sparse vectors, huge embedding tables are spread across ranks, especially useful when working with limited HBM
class Sparse(nn.Module):
    def __init__(
            self, 
            feat_tables: Dict[str, Dict[str, Union[int, List]]]= {},
            emb_dim: int = 32,
            device: str = "meta"
        ) -> None:
        """
        Args:
            feat_tables (Dict[str, Dict[str, Union[int, List]]]): dictionary with embedding tables, each specifying vocab size and features that belong to it
            emb_dim (int): size of the embeddings, must be equal for all tables to perform dot interactions.
            device (str): device to load tensors on
        """
        super().__init__()
        self.feat_tables = feat_tables
        self.emb_dim = emb_dim
        self.device = device

        # Q: why should we explicitly define device if it can be specified from model.to(device)?
        # A: if large embedding tables need to be moved to GPU this approach avoids the CPU intermediate allocation
        # ugly? Yes! practical? Yes!
        self.ebc = EmbeddingBagCollection(
            device = self.device,
            tables = [
                EmbeddingBagConfig(
                    name=table_name,
                    embedding_dim=self.emb_dim,
                    num_embeddings=table["vocab_size"],
                    feature_names=table["features"],
                    pooling=PoolingType.SUM,
                )
                for table_name, table in self.feat_tables.items()
            ]
        )

    def forward(self, kjt: KeyedJaggedTensor) -> torch.Tensor:
        """
        Args:
            kjt (torchrec.KeyedJaggedTensor): KeyedJaggedTensor data structure contains multi-hot, varying values for all sparse features for a batch of records
        Returns:
            output (torch.tensor): embedding lookup output tensor of size [B, F, D]. B - Batch, F - number of features, D - dimension of the pooled embedding lookup
        """
        output = self.ebc(kjt)
        B = output.values().shape[0]
        return output.values().reshape(B, -1, self.emb_dim)

# pretty much identical to the original dense layers
# goal is to enable DP for dense layers to increase compute throughput
class Dense(nn.Module):
    def __init__(
            self,
            input_dim: int = 5,
            dense_layers: List[int] = [],
            device: str = "meta"
        ) -> None:
        super().__init__()
        self.device = device

        self.dense = nn.ModuleList()
        for i in range(len(dense_layers)):
            if i == 0:
                self.dense.append(nn.Linear(in_features=input_dim, out_features=dense_layers[i], device=self.device))
            else:
                self.dense.append(nn.Linear(in_features=dense_layers[i-1], out_features=dense_layers[i], device=self.device))
            if i < len(dense_layers)-1:
                self.dense.append(nn.ReLU())
        
        self.dense = nn.Sequential(*self.dense)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor of size [B, M]. B - batch size, M - number of dense features
        Returns:
            output (torch.tensor): output tensor of size [B, D]
        """
        return self.dense(x)

class Interaction(nn.Module):
    def __init__(self, F: int) -> None:
        """
        It is assumed that at least 1 dense feature is provided
        Args:
            F (int): number of sparse features
        """
        super().__init__()
        # DMPCollection automatically moves buffered values, torch.triu_indices(...), to the right device 
        self.register_buffer(
            "triu_indices",
            # precompute rows and cols of upper triangular
            # notice that it is independent of batch size
            # saves time from recomputing on every forward pass
            torch.triu_indices(F + 1, F + 1, offset=1),
            persistent=False, # do not save in state_dict, not necessary
        )

    def forward(self, sparse_x: torch.Tensor, dense_x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sparse_x (torch.Tensor): sparse input tensor of size [B, F, D]
            dense_x (torch.Tensor): dense input tensor of size [B, D]
        Returns:
            output (torch.tensor): output tensor of size [B, K], upper triangular of the matrix product [B, F+1, D] @ [B, F+1, D].Transpose
        """

        dense_x = dense_x.unsqueeze(-1).transpose(1, 2)

        # concat([B, F, D], [B, 1, D]) -> [B, F+1, D]
        combined = torch.concat([sparse_x, dense_x], dim=1)
        # BMM([B, F+1, D], [B, D, F+1]) -> [B, F+1, F+1]
        interact = torch.bmm(combined, combined.transpose(1, 2))
        
        # get upper triangular for unique interactions, exlude diagonal since 
        # we only care about pairwise feature interactions with other features, not to themselves
        
        # this line not needes since upper triangular values were buffered during init
        # row, col = torch.triu_indices(interact.shape[1], interact.shape[2], offset=1)

        # (B, F+1, F+1) -> (B, K) where K = (F+1)*(F) // 2
        return interact[:, self.triu_indices[0], self.triu_indices[1]]



class DLRMDist(nn.Module):
    def __init__(
            self,
            # sparse params
            feat_tables: Dict[str, Dict[str, Union[int, List]]]= {},
            emb_dim: int = 32,
            device: str = "meta", # DMP will automatically handle materialization, without DMP the device must be specified explicitly for sparse layer along with model.to(device) for dense layers
            # bottom MLP dense params
            dense_dim: int = 5,
            bottom_dense_layers: List[int] = [],
            # top MLP dense params
            top_dense_layers: List[int] = []
            
        ) -> None:
        super().__init__()

        assert len(bottom_dense_layers) > 0, "dense_layer cannot be empty!" 
        assert bottom_dense_layers[-1] == emb_dim, "Output dimension for dense and sparse features but be equal!"


        F = sum([len(feat_tables[table]["features"]) for table in feat_tables]) # number of sparse features

        # initialize layers
        self.sparse = Sparse(
            feat_tables = feat_tables,
            emb_dim = emb_dim,
            device = device
        )

        self.bottom_mlp = Dense(
            input_dim = dense_dim,
            dense_layers = bottom_dense_layers,
            device = device
        )

        self.interaction = Interaction(
            F = F
        )

        self.top_mlp = Dense(
            input_dim = bottom_dense_layers[-1] + ((F+1)*(F) // 2),
            dense_layers = top_dense_layers,
            device = device
        )
    
    def forward(self, dense_x: torch.Tensor, kjt: KeyedJaggedTensor) -> torch.Tensor:
        """
        Args:
            dense_x (torch.Tensor): dense input tensor of size [B, D]
            kjt (torchrec.KeyedJaggedTensor): KeyedJaggedTensor data structure contains multi-hot, varying values for all sparse features for a batch of records
        Returns:
            output (torch.tensor): embedding lookup output tensor of size [B, F, D]. B - Batch, F - number of features, D - dimension of the pooled embedding lookup
        """
        
        sparse_x = self.sparse(kjt) # KJT -> [B, F, D]
        dense_x = self.bottom_mlp(dense_x) # [B, M] -> [B, D], B - batch size, M - num dense features
        interact = self.interaction(sparse_x, dense_x) # -> [B, K], K = (F+1)*F // 2
        
        return self.top_mlp(torch.concat([dense_x, interact], dim = 1)) # [B, K+D] -> [B, T], T - num tasks

