import os
import threading
import time
from itertools import chain
from typing import Iterable

import torch
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.distributed.nn.api.remote_module import RemoteModule
from torch.distributed.optim import DistributedOptimizer
from torch.distributed.rpc import RRef
from torch.utils.data.dataloader import DataLoader

model = nn.Sequential(
    nn.Linear(784, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 10),
    nn.ReLU(),
)

shards = [model[:2], model[2:4], model[4:]]
BATCH_SIZE = 64


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

    def __init__(self, workers: Iterable[str], shards, microbatch_size):
        super().__init__()

        self.microbatch_size = microbatch_size

        # Put the first part of the MNIST on workers[0]
        self.shards = [RemoteModule(worker, RemoteSeq, (shard,))
                       for worker, shard in zip(workers, shards)]

    def forward(self, xs):
        # Split the input batch xs into micro-batches, and collect async RPC
        # futures of the final outputs into a list
        out_futures = []
        for x in iter(xs.split(self.microbatch_size, dim=0)):
            x_rref = RRef(x)
            for shard in self.shards[:-1]:
                x_rref = shard.module_rref.remote().forward(x_rref)
            z_fut = self.shards[-1].forward_async(x_rref)
            out_futures.append(z_fut)

        # collect and cat all output tensors into one tensor.
        return torch.cat(torch.futures.wait_all(out_futures))

    def parameter_rrefs(self):
        return list(chain.from_iterable(shard.remote_parameters() for shard in self.shards))


def run_master(microbatch_size, num_workers):
    workers = [f"worker{i}/cpu" for i in range(1, num_workers + 1)]
    # TODO: Add support for CUDA, some minor changes needed in RemoteSeq.forward()
    model = DistMNIST(workers, shards, microbatch_size)
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
    DATA_DIR = "../data/"
    mnist = torchvision.datasets.MNIST(DATA_DIR, train=True, download=True, transform=trans)
    loader = DataLoader(mnist, batch_size=BATCH_SIZE, shuffle=True)
    mnist_test = torchvision.datasets.MNIST(DATA_DIR, train=False, download=True, transform=trans)
    testloader = DataLoader(mnist_test, batch_size=BATCH_SIZE, shuffle=False)

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
        run_master(microbatch_size, world_size - 1)
    else:
        rpc.init_rpc(
            f"worker{rank}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )

    # block until all rpcs finish
    rpc.shutdown()


if __name__ == "__main__":
    world_size = 4
    sizes = [BATCH_SIZE]
    while sizes[-1] > 2:
        sizes.append(sizes[-1] // 2)
    for microbatch_size in sizes:
        tik = time.time()
        mp.spawn(run_worker, args=(world_size, microbatch_size), nprocs=world_size, join=True)
        tok = time.time()
        print(f"{microbatch_size=}, execution time = {tok - tik}")
