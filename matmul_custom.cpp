/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 *
 * Function : c = a * b (matrix multiplication)
 * This sample is a very basic sample that implements Matmul on Ascend plaform.
 * In this sample:
 * Shape of matrix a is [m, k]: [32, 32]
 * Shape of matrix b is [k, n]: [32, 32]
 * Shape of matrix c is [m, n]: [32, 32]
 */

#include "kernel_operator.h"
using namespace AscendC;

class KernelMatmul {
public:
    __aicore__ inline KernelMatmul(uint16_t M, uint16_t N, uint16_t K)
    {
        m = M; n = N; k = K;
        aSize = m * k;
        bSize = k * n;
        cSize = m * m;
        mBlocks = m / 16;
        nBlocks = n / 16;
        kBlocks = k / 16;
    }
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c)
    {
        aGM.SetGlobalBuffer((__gm__ half*)a);
        bGM.SetGlobalBuffer((__gm__ half*)b);
        cGM.SetGlobalBuffer((__gm__ float*)c);
        pipe.InitBuffer(inQueueA1, 1, aSize * sizeof(half));
        pipe.InitBuffer(inQueueA2, 1, aSize * sizeof(half));
        pipe.InitBuffer(inQueueB1, 1, bSize * sizeof(half));
        pipe.InitBuffer(inQueueB2, 2, bSize * sizeof(half) / 2);
        pipe.InitBuffer(outQueueCO1, 2, cSize * sizeof(float) / 2);
        pipe.InitBuffer(outQueueCO2, 1, cSize * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        SplitA();

        LocalTensor<half> b1Local = inQueueB1.DeQue<half>();
        LocalTensor<half> a2Local = inQueueA2.DeQue<half>();
        // split matrix b into 2 parts, [32, 16] and [32, 16]
        for (int i = 0; i < 2; ++i) {
            SplitB(b1Local, i);
            Compute(a2Local);
            CopyOut(i);
        }
        inQueueB1.FreeTensor(b1Local);
        inQueueA2.FreeTensor(a2Local);

    }

private:
    __aicore__ inline void CopyND2NZ(const LocalTensor<half>& dst, const GlobalTensor<half>& src, const uint16_t height,
        const uint16_t width)
    {
        for (int i = 0; i < width / 16; ++i) {
            int srcOffset = i * 16;
            int dstOffset = i * 16 * height;
            DataCopy(dst[dstOffset], src[srcOffset], { height, 1, uint16_t(width / 16 - 1), 0 });
        }
    }
    __aicore__ inline void CopyIn()
    {
        LocalTensor<half> a1Local = inQueueA1.AllocTensor<half>();
        LocalTensor<half> b1Local = inQueueB1.AllocTensor<half>();

        CopyND2NZ(a1Local, aGM, m, k);
        CopyND2NZ(b1Local, bGM, k, n);

        inQueueA1.EnQue(a1Local);
        inQueueB1.EnQue(b1Local);
    }
    __aicore__ inline void SplitA()
    {
        int srcOffset = 0;
        int dstOffset = 0;
        LocalTensor<half> a1Local = inQueueA1.DeQue<half>();
        LocalTensor<half> a2Local = inQueueA2.AllocTensor<half>();

        LoadData2dParams loadDataParams;
        loadDataParams.repeatTimes = kBlocks;
        loadDataParams.srcStride = mBlocks;
        loadDataParams.ifTranspose = false;
        // transform nz to zz
        for (int i = 0; i < mBlocks; ++i) {

            LoadData(a2Local[dstOffset], a1Local[srcOffset], loadDataParams);

            srcOffset += 16 * 16;
            dstOffset += kBlocks * 16 * 16;
        }
        
        inQueueA2.EnQue<half>(a2Local);
        inQueueA1.FreeTensor(a1Local);
    }
    __aicore__ inline void SplitB(const LocalTensor<half>& b1Local, const int bSplitIdx)
    {
        LocalTensor<half> b2Local = inQueueB2.AllocTensor<half>();

        // transform nz to Zn
        LoadData2dParams loadDataParams;
        loadDataParams.repeatTimes = nBlocks / 2;
        loadDataParams.srcStride = kBlocks;
        loadDataParams.ifTranspose = true;

        for (int i = 0; i < kBlocks; i++){
            LoadData(b2Local[i * nBlocks / 2 * 16 * 16], b1Local[bSplitIdx * bSize / 2 + i * 16 * 16], loadDataParams);
        }

        inQueueB2.EnQue<half>(b2Local);
    }
    __aicore__ inline void Compute(const LocalTensor<half>& a2Local)
    {
        LocalTensor<half> b2Local = inQueueB2.DeQue<half>();
        LocalTensor<float> c1Local = outQueueCO1.AllocTensor<float>();

        Mmad(c1Local, a2Local, b2Local, { m, uint16_t(n / 2), k, false, 0, false, false, false });

        outQueueCO1.EnQue<float>(c1Local);
        inQueueB2.FreeTensor(b2Local);
    }
    
    __aicore__ inline void CopyOut(const int bSplitIdx)
    {
        LocalTensor<float> c1Local = outQueueCO1.DeQue<float>();
        FixpipeParamsV220 fixpipeParams;
        fixpipeParams.nSize = n / 2;
        fixpipeParams.mSize = m;
        fixpipeParams.srcStride = m;
        fixpipeParams.dstStride = n;
        fixpipeParams.ndNum = 1;
        fixpipeParams.srcNdStride = 2;
        fixpipeParams.dstNdStride = m * n;
        Fixpipe(cGM[bSplitIdx * n / 2], c1Local, fixpipeParams);

        outQueueCO1.FreeTensor(c1Local);
        // outQueueCO2.FreeTensor(c2Local);
    }

private:
    TPipe pipe;

    TQue<QuePosition::A1, 1> inQueueA1;
    TQue<QuePosition::A2, 1> inQueueA2;
    TQue<QuePosition::B1, 1> inQueueB1;
    TQue<QuePosition::B2, 2> inQueueB2;
    // dst queue
    TQue<QuePosition::CO1, 2> outQueueCO1;
    TQue<QuePosition::CO2, 1> outQueueCO2;

    GlobalTensor<half> aGM, bGM;
    GlobalTensor<float> cGM;

    uint16_t m, n, k;

    uint16_t aSize, bSize, cSize, mBlocks, nBlocks, kBlocks;
};

extern "C" __global__ __aicore__ void matmul_custom(GM_ADDR a, GM_ADDR b, GM_ADDR c, uint16_t M, uint16_t N, uint16_t K)
{
    KernelMatmul op(M, N, K);
    op.Init(a, b, c);
    op.Process();
}

enum OpType {
    fp32 = 4,
    fp16 = 2,
    int8 = 1,
};

struct Offset {
    uint16_t height;
    uint16_t width;

    Offset(uint16_t height_, uint16_t width_): height(height_), width(width_) {}
};

#ifndef __CCE_KT_TEST__
// call of kernel function
void matmul_custom_do(uint32_t blockDim, void* l2ctrl, void* stream, uint8_t* a, uint8_t* b, uint8_t* c, uint16_t M, uint16_t N, uint16_t K)
{
    uint16_t block = 16;
    uint16_t height = 4;
    uint16_t width = 5;

    uint32_t L1BufferSize = 1 * 1024 * 1024;
    uint32_t L0BufferSize = 64 * 1024;

    uint16_t BlockNumM = M / block;
    uint16_t BlockNumN = N / block;
    uint16_t BlockNumK = K / block;

    assert(BlockNumM > 0);
    assert(BlockNumN > 0);
    assert(BlockNumK > 0);

    uint32_t TotalResBlock = uint32_t(BlockNumM) * uint32_t(BlockNumN);
    if (blockDim > TotalResBlock) { blockDim = TotalResBlock; }

    uint16_t CoreTilingM = (BlockNumM - 1) / height;
    uint16_t CoreTilingN = (BlockNumN - 1) / width;

    OpType optype = OpType::fp16; 

    // for(uint16_t M_num = 0; H_num < M; )

    Offset offset(height, width);

    matmul_custom<<<blockDim, l2ctrl, stream>>>(a, b, c, M, N, K);
}
#endif
