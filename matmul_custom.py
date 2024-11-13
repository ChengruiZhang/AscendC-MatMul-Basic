#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2022-2023 Huawei Technologies Co., Ltd
import numpy as np
import sys
import torch
import torch_npu

def gen_golden_data(M, N, K):
    x1_gm_type = torch.float16
    x2_gm_type = torch.float16

    debug1 = False
    debug2 = False
    if(debug1):
        # 行
        x1_gm_0 = torch.arange(M).view(M, 1).repeat(1, K).to("npu") / 10
        x2_gm_0 = torch.arange(K).view(K, 1).repeat(1, N).to("npu") / 10
    elif (debug2):
        # 列
        x1_gm_0 = torch.arange(K).view(K, 1).repeat(1, M).to("npu").transpose(0, 1) / 10
        x2_gm_0 = torch.arange(N).view(N, 1).repeat(1, K).to("npu").transpose(0, 1) / 10
    else:
        x1_gm_0 = torch.randint(1, 4, [M, K], dtype=torch.float16, device="npu") / 10
        x2_gm_0 = torch.randint(1, 4, [K, N], dtype=torch.float16, device="npu") / 10

    transpose = False
    if(transpose):
        x1_gm = x1_gm_0.to(x1_gm_type).transpose(0, 1)
        x2_gm = x2_gm_0.to(x2_gm_type).transpose(0, 1)
        golden = torch.matmul(x2_gm, x1_gm).to(torch.float16)
    else:
        x1_gm = x1_gm_0.to(x1_gm_type)
        x2_gm = x2_gm_0.to(x2_gm_type)
        golden = torch.matmul(x1_gm, x2_gm).to(torch.float16)

    # c_init = torch.randint(1, 2, [M, N], dtype=torch.float16, device="npu") / 10

    print("compute done")

    x1_gm.cpu().numpy().tofile("./input/x1_gm.bin")
    x2_gm.cpu().numpy().tofile("./input/x2_gm.bin")
    # c_init.cpu().numpy().tofile("./c_init.bin")
    # x1_gm_test.cpu().numpy().tofile("./input/x1_gm_test.bin")
    # x2_gm_test.cpu().numpy().tofile("./input/x2_gm_test.bin")
    golden.cpu().numpy().tofile("./output/golden.bin")

    print("save done")


if __name__ == "__main__":
    
    gen_golden_data(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
