#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2022-2023 Huawei Technologies Co., Ltd
import numpy as np
import sys

def gen_golden_data(M, N, K):
    x1_gm_type = np.float16
    x2_gm_type = np.float16

    # M = 64
    # N = 64
    # K = 64

    x1_gm_0 = np.random.randint(1, 3, [M, K]) / 10
    # x1_gm_0 = np.arange(M * K).astype(x1_gm_type).reshape([M, K]) / 1000
    x1_gm = x1_gm_0.astype(x1_gm_type)
    x1_gm_test = x1_gm_0.astype(np.float32)
    x2_gm_0 = np.random.randint(1, 3, [K, N]).astype(x2_gm_type) / 10
    # x2_gm_0 = np.arange(K * N).astype(x2_gm_type).reshape([K, N]) / 1
    x2_gm = x2_gm_0.astype(x2_gm_type)
    x2_gm_test = x2_gm_0.astype(np.float32)
    golden = np.matmul(x1_gm.astype(np.float16), x2_gm.astype(np.float16)).astype(np.float16)

    print("compute done")

    x1_gm.tofile("./input/x1_gm.bin")
    x2_gm.tofile("./input/x2_gm.bin")
    x1_gm_test.tofile("./input/x1_gm_test.bin")
    x2_gm_test.tofile("./input/x2_gm_test.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    
    gen_golden_data(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
