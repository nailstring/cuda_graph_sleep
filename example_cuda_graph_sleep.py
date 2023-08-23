import torch
import torch.nn as nn

# 반드시 import torch 아래애 쓰기
import cuda_graph_utils.sleep_ops

import random


if __name__ == "__main__":
    sleep_length = 100
    sleep_times = [sleep_length * random.uniform(0.0, 1.0) for i in range(100)]

    # Dummy operation
    dummy_x = torch.randn(4096, 4096).cuda()
    dummy_linear = nn.Linear(4096, 4096).cuda()

    # Warm-up
    for _ in range(3):
        dummy_output = dummy_linear(dummy_x)

    sleep_time_placeholder = torch.tensor([0]).cuda()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        cuda_graph_utils.sleep_ops.sleep(sleep_time_placeholder)
        dummy_output = dummy_linear(dummy_x)

    for sleep_time in sleep_times:
        print(f"sleep time : {sleep_time} ms")

        sleep_time = int(sleep_time * 1000000 * 1.54)
        sleep_time_placeholder.copy_(torch.tensor([sleep_time]).cuda())
        g.replay()

