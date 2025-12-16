from typing import Iterable

import torch

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.workers.comet import BaseCometModel

from comet import load_from_checkpoint

__all__ = ['DataParallelComet']


class DataParallelComet(BaseCometModel):

    def __init__(self, config, comet_module):
        super().__init__(config=config)
        self.comet_model = comet_module
        self.ulysses_sequence_parallel_size = self.config.get('ulysses_sequence_parallel_size', 1)

    def _forward_batch(self, batch, micro_batch_size_per_gpu):
        # response_length = micro_batch['responses'].size(-1)

        print(f"dp_comet.py forward batch: {len(batch)}, micro_batch_size_per_gpu: {micro_batch_size_per_gpu}")
        comet_output = self.comet_model.predict(batch, batch_size=micro_batch_size_per_gpu, gpus=1)
        scaled_scores = [max(round(float(score), 4), 0.0) for score in comet_output.scores]

        # return comet_output.scores #for example: [0.84, 0.77, ...]
        return scaled_scores

    def compute_comet_score(self, data: list, micro_batch_size_per_gpu: int=1) -> torch.Tensor:

        # values_lst = []
        with torch.no_grad():
            scores = self._forward_batch(data, micro_batch_size_per_gpu)
        # values_lst.extend(scores)

        # for i in range(len(data)):
        #     reward_tensor[i] = values_lst[i]
        reward_tensor = torch.tensor(scores, dtype=torch.float32).unsqueeze(-1)

        return reward_tensor

    # def compute_valid_comet(self, data: DataProto) -> torch.Tensor:

    #     reward_tensor = torch.zeros((len(data.batch['responses']), 1), dtype=torch.float32)


    #     micro_batch_size = data.meta_info['micro_batch_size']
    #     # micro_batches = [triplet_list[i:i + micro_batch_size] for i in range(0, len(triplet_list), micro_batch_size)]
    #     micro_batches = [triplet_list[i:i + self.config.val_batch_size] for i in range(0, len(triplet_list), self.config.val_batch_size)]

    #     values_lst = []
    #     for micro_batch in micro_batches:
    #         with torch.no_grad():
    #             scores = self._forward_micro_batch(micro_batch)
    #         values_lst.extend(scores)

    #     for i in range(len(data)):
    #         reward_tensor[i] = values_lst[i]

    #     return reward_tensor