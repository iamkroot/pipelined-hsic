import argparse
from ipaddress import IPv4Address
import logging
from math import ceil
import os
import time
from typing import Any, NamedTuple, Type
from fairscale.experimental.nn.distributed_pipeline import DistributedLoss, DistributedPipeline, PipelineModulesGraph
from torch import nn, optim
import torch
import torch.multiprocessing as mp
from torch.distributed import rpc, autograd as dist_autograd
from torch.distributed.nn.api.remote_module import RemoteModule
from torch.distributed.optim.optimizer import DistributedOptimizer
from torch.distributed.rpc.api import RRef

from vanilla_pipeline import get_data


class RemoteModuleParams(NamedTuple):
    module_cls: Type[nn.Module]
    args: tuple
    kwargs: dict[str, Any]


def create_sequence_pipeline(
    layers: list[RemoteModuleParams], balance: list[int], devices: list[str], **kwargs
) -> DistributedPipeline:
    """A simple helper function to create a pipeline from list of pipeline-modules that run sequentially.
       Args:
           layers: list of modules. They should not be already assigned a remote-device.
           balance: a list of integers how layers should be paritioned. Sum of numbers in 'balance'
               should be equal to the number of layers.
           devices: specification of remote device for each partition. Should be of the same length
               as 'balance'.
    """
    remote_modules: list[RemoteModule] = []
    index = 0
    for num_layers, remote_device in zip(balance, devices):
        next_index = index + num_layers
        for li in range(index, next_index):
            remote_modules.append(RemoteModule(remote_device, **layers[li]._asdict()))
        index = next_index
    graph = PipelineModulesGraph()
    graph.add_sequence(remote_modules, [0])
    return DistributedPipeline(graph, **kwargs)


def evaluate(model, testloader):
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            outputs = model(inputs).to_here()
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    logging.info(f"Accuracy {correct * 100 / total:.2f}%")


def rpc_worker(rank, world_size, batch_size, microbatch_size):
    options = rpc.TensorPipeRpcBackendOptions(_transports=["ibv", "uv"])
    logging.info(f"Initting worker{rank}")
    rpc.init_rpc(
        "worker" + str(rank),
        rank=rank,
        world_size=world_size,
        backend=rpc.BackendType.TENSORPIPE,
        rpc_backend_options=options,
    )
    if rank == 0:
        run_master(world_size, batch_size, microbatch_size)

    rpc.shutdown()


def run_master(world_size, batch_size, microbatch_size):
    model = [RemoteModuleParams(nn.Linear, (784, 100),{}),
        RemoteModuleParams(nn.ReLU, (),{}),
        RemoteModuleParams(nn.Linear, (100, 100),{}),
        RemoteModuleParams(nn.ReLU, (),{}),
        RemoteModuleParams(nn.Linear, (100, 10),{}),
        RemoteModuleParams(nn.ReLU, (),{})
    ]
    torch.random.manual_seed(3)
    loss_fn = DistributedLoss(nn.CrossEntropyLoss)
    workers = [f"worker{i}/cpu" for i in range(0, world_size)]
    chunks = ceil(batch_size / microbatch_size)
    pipe = create_sequence_pipeline(model, [2] * world_size, workers, chunks=chunks)
    opt = DistributedOptimizer(
        optim.SGD,
        pipe.parameter_rrefs(),
        lr=0.05,
    )

    trainloader, testloader = get_data(batch_size)
    for i, (inputs, labels) in enumerate(trainloader):
        if i % 100 == 0:
            logging.info(f"Processing batch {i}")
            # evaluate(pipe, testloader)

        with dist_autograd.context() as context_id:
            outputs: RRef = pipe(inputs)
            loss = loss_fn(outputs, RRef(labels))  # block
            loss.backward(context_id)  # will run on last worker
            opt.step(context_id)
    evaluate(pipe, testloader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-addr", type=IPv4Address, default=IPv4Address("127.0.0.1"))
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--rank", type=int, default=-1)
    parser.add_argument("--world-size", "-s", type=int, default=3)
    parser.add_argument("--batch-size", "-b", type=int, default=64)
    parser.add_argument("--microbatch-size", "-m", type=int, default=64)
    args = parser.parse_args()

    os.environ['MASTER_ADDR'] = str(args.master_addr)
    os.environ['MASTER_PORT'] = str(args.master_port)
    world_size = args.world_size
    batch_size = args.batch_size
    microbatch_size = args.microbatch_size

    logging.info(f"{world_size=} {batch_size=} {microbatch_size=}")
    tik = time.time()
    if args.rank == -1:  # local run, spawn processes on current cpu
       mp.spawn(rpc_worker, args=(world_size, batch_size, microbatch_size), nprocs=world_size, join=True)
    else:  # distributed run, run only one worker with given rank
       rpc_worker(args.rank, world_size, batch_size, microbatch_size)
    tok = time.time()
    logging.info(f"execution_time={tok - tik:.3f}s")



if __name__ == '__main__':
    main()
