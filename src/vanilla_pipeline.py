import os
import threading
import time

import torch
import torch.nn as nn
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.optim as optim
from torch.distributed.optim import DistributedOptimizer
from torch.distributed.rpc import RRef
from torch.distributed.nn.api.remote_module import RemoteModule

from torch.utils.data.dataloader import DataLoader
import torchvision

model = nn.Sequential(
    nn.Linear(784, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 10),
    nn.ReLU(),
)

MNISTShard1 = model[:2]
MNISTShard2 = model[2:4]
MNISTShard3 = model[4:]


class RemoteSeq(nn.Module):
    def __init__(self, seq: nn.Sequential):
        super().__init__()
        self.seq = seq
        self._lock = threading.Lock()

    def forward(self, input_rref: RRef[torch.Tensor]):
        input_ = input_rref.to_here()  # block
        with self._lock:
            # ensure that only one microbatch is trained at a time
            return self.seq(input_)


class DistMNIST(nn.Module):
    """
    Assemble three parts as an nn.Module and define pipelining logic
    """

    def __init__(self, workers, microbatch_size):
        super().__init__()

        self.microbatch_size = microbatch_size

        # Put the first part of the MNIST on workers[0]
        self.shard0 = RemoteModule(workers[0], RemoteSeq, (MNISTShard1,))
        self.shard1 = RemoteModule(workers[1], RemoteSeq, (MNISTShard2,))
        self.shard2 = RemoteModule(workers[2], RemoteSeq, (MNISTShard3,))

    def forward(self, xs):
        # Split the input batch xs into micro-batches, and collect async RPC
        # futures of the final outputs into a list
        out_futures = []
        for x in iter(xs.split(self.microbatch_size, dim=0)):
            x0_rref = RRef(x)
            x1_rref = self.shard0.module_rref.remote().forward(x0_rref)
            x2_rref = self.shard1.module_rref.remote().forward(x1_rref)
            z_fut = self.shard2.forward_async(x2_rref)
            out_futures.append(z_fut)

        # collect and cat all output tensors into one tensor.
        return torch.cat(torch.futures.wait_all(out_futures))

    def parameter_rrefs(self):
        remote_params = []
        remote_params.extend(self.shard0.remote_parameters())
        remote_params.extend(self.shard1.remote_parameters())
        remote_params.extend(self.shard2.remote_parameters())
        return remote_params


def run_master(microbatch_size):
    # TODO: Add support for CUDA, some minor changes needed in RemoteSeq.forward()
    model = DistMNIST(["worker1/cpu", "worker2/cpu", "worker3/cpu"], microbatch_size)
    loss_fn = nn.CrossEntropyLoss()
    opt = DistributedOptimizer(
        optim.SGD,
        model.parameter_rrefs(),
        lr=0.05,
    )

    trans = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        lambda x: x.flatten()
    ])
    mnist = torchvision.datasets.MNIST("../data", train=True, download=True, transform=trans)
    loader = DataLoader(mnist, batch_size=64, shuffle=True)
    mnist_test = torchvision.datasets.MNIST("../data", train=False, download=True, transform=trans)
    testloader = DataLoader(mnist_test, batch_size=64, shuffle=False)

    def test(model, testloader):
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in testloader:
                outputs = model(inputs)
                _, pred = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
        print(round(correct * 100 / total, 4))

    for i, data in enumerate(loader):
        if i % 100 == 0:
            print(f"Processing batch {i}")
            test(model, testloader)

        inputs, labels = data
        # The distributed autograd context is the dedicated scope for the
        # distributed backward pass to store gradients, which can later be
        # retrieved using the context_id by the distributed optimizer.
        with dist_autograd.context() as context_id:
            outputs: torch.Tensor = model(inputs)  # block
            dist_autograd.backward(context_id, [loss_fn(outputs, labels)])
            opt.step(context_id)

    print("Final accuracy")
    test(model, testloader)


def run_worker(rank, world_size, microbatch_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=256)

    if rank == 0:
        rpc.init_rpc(
            "master",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        run_master(microbatch_size)
    else:
        rpc.init_rpc(
            f"worker{rank}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        pass

    # block until all rpcs finish
    rpc.shutdown()


if __name__ == "__main__":
    world_size = 4
    for microbatch_size in (1, 2, 4, 8, 16, 32, 64):
        tik = time.time()
        mp.spawn(run_worker, args=(world_size, microbatch_size), nprocs=world_size, join=True)
        tok = time.time()
        print(f"{microbatch_size=}, execution time = {tok - tik}")
