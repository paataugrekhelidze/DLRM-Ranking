import torch
from torch.utils.data import Dataset
import random
from torchrec import JaggedTensor, KeyedJaggedTensor


class MyTrainDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = [{
            "product": torch.randint(0, 10, (random.randint(0, 5),)), # product size varies from 1 to 5, representing multi-hot
            "user": torch.randint(0, 10, (1,)),
            "dense": torch.randn(5),
            "target": torch.rand(1)
        } for _ in range(size)]

    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        return self.data[index]
    

    @staticmethod
    def _collate_batch(batch, user_v, product_v):
        batched = {}
        jt_dict = {}
        for key in batch[0].keys():
            if key in ["product", "user"]:
                action_list, action_length = [], []
                for sample in batch:
                    actions = sample[key]

                    if len(actions) == 0:
                        processed_actions = torch.tensor([(product_v if key=="product" else user_v)], dtype=torch.int64)
                    else:
                        # hash real campaign ids into [0, product_v - 1], keeping product_v free for the null sentinel
                        processed_actions = torch.as_tensor(actions, dtype=torch.int64) % (product_v if key=="product" else user_v)
                    
                    action_list.append(processed_actions)
                    action_length.append(len(processed_actions))

                jt_dict[key] = JaggedTensor(values=torch.cat(action_list), lengths=torch.tensor(action_length, dtype=torch.int32)) 
            else:
                # stack other labels as usual
                batched[key] = torch.stack([sample[key] for sample in batch])
        # all sparse embedding lookups are organized in a signle kjt data sturcture
        batched["sparse"] = KeyedJaggedTensor.from_jt_dict(jt_dict)
        return batched