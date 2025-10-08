import torch
from torch.nn import DataParallel

class BalancedDataParallel(DataParallel):
    def __init__(self, gpu0_bsz, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gpu0_bsz = gpu0_bsz

    def forward(self, *inputs, **kwargs):
        if not self.device_ids or len(self.device_ids) == 1:
            return self.module(*inputs, **kwargs)
        
        # Overwrite scatter to control GPU0 load
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        if len(inputs) != len(self.device_ids):
            raise RuntimeError("Mismatch between input splits and device_ids")
        
        # Do not use parallel_apply on GPU0
        replicas = self.replicate(self.module, self.device_ids)
        outputs = self.parallel_apply(replicas, inputs, kwargs)
        return self.gather(outputs, self.output_device)
