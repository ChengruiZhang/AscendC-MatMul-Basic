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
    __aicore__ inline KernelMatmul()
    {
        aSize = m * k; // 1024
        bSize = k * n;
        cSize = m * n;
        mBlocks = m / 16;
        nBlocks = n / 16;
        kBlocks = k / 16;
    }
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c)
    {
        aGM.SetGlobalBuffer((__gm__ half*)a);
        bGM.SetGlobalBuffer((__gm__ half*)b);
        cGM.SetGlobalBuffer((__gm__ float*)c);
        pipe.InitBuffer(inQueueA1, 1, aSize * sizeof(half)); // 1024 * 2
        pipe.InitBuffer(inQueueA2, 1, aSize * sizeof(half)); // 1024 * 2
        pipe.InitBuffer(inQueueB1, 1, bSize * sizeof(half)); // 1024 * 2
        pipe.InitBuffer(inQueueB2, 2, bSize * sizeof(half) / 2); // 1024 * 2 / 2; b matrix double buffer
        pipe.InitBuffer(outQueueCO1, 2, cSize * sizeof(float) / 2); // 1024 * 4 / 2
        pipe.InitBuffer(outQueueCO2, 1, cSize * sizeof(float)); // 1024 * 4
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        SplitA();

        LocalTensor<half> b1Local = inQueueB1.DeQue<half>();
        LocalTensor<half> a2Local = inQueueA2.DeQue<half>();
        LocalTensor<float> c2Local = outQueueCO2.AllocTensor<float>();
        // split matrix b into 2 parts, [32, 16] and [32, 16]
        for (int i = 0; i < 2; ++i) { 
            SplitB(b1Local, i);
            Compute(a2Local);
            Aggregate(c2Local, i);
        }
        inQueueB1.FreeTensor(b1Local);
        inQueueA2.FreeTensor(a2Local);
        outQueueCO2.EnQue<float>(c2Local);

        CopyOut();
    }

private:
    __aicore__ inline void CopyND2NZ(const LocalTensor<half>& dst, const GlobalTensor<half>& src, const uint16_t height,
        const uint16_t width)
    {
        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = height; // 32
        dataCopyParams.blockLen = 1;
        dataCopyParams.srcStride = uint16_t(width / 16 - 1);
        dataCopyParams.dstStride = 0;
        
        for (int i = 0; i < width / 16; ++i) {
            int srcOffset = i * 16;
            int dstOffset = i * 16 * height;
            DataCopy(dst[dstOffset], src[srcOffset], dataCopyParams);
        }
    }
    __aicore__ inline void CopyIn()
    {
        LocalTensor<half> a1Local = inQueueA1.AllocTensor<half>(); // allocata tensor memory
        LocalTensor<half> b1Local = inQueueB1.AllocTensor<half>();

        // printf("fmt string %d\n", a1Local);
        
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

        // transform NZ to ZZ
        for (int i = 0; i < mBlocks; ++i) {
            LoadData2dParams loadDataParams;
            loadDataParams.repeatTimes = kBlocks;
            loadDataParams.srcStride = mBlocks;
            loadDataParams.ifTranspose = false;

            LoadData(a2Local[dstOffset], a1Local[srcOffset], loadDataParams);

            srcOffset += 16 * 16; // 1 blk
            dstOffset += k * 16; // k/16 blk
        }

        inQueueA2.EnQue<half>(a2Local);
        inQueueA1.FreeTensor(a1Local);
    }
    __aicore__ inline void SplitB(const LocalTensor<half>& b1Local, const int bSplitIdx)
    {
        LocalTensor<half> b2Local = inQueueB2.AllocTensor<half>();
 
        // transform nz to zn
        LoadData2dParams loadDataParams;
        loadDataParams.repeatTimes = kBlocks;
        loadDataParams.srcStride = 1;
        loadDataParams.ifTranspose = true;

        LoadData(b2Local, b1Local[bSplitIdx * bSize / 2], loadDataParams);

        inQueueB2.EnQue<half>(b2Local);
    }
    __aicore__ inline void Compute(const LocalTensor<half>& a2Local)
    {
        LocalTensor<half> b2Local = inQueueB2.DeQue<half>();
        LocalTensor<float> c1Local = outQueueCO1.AllocTensor<float>();

        // MmadParams mmadParams;
        // mmadParams.m = m;
        // mmadParams.n = uint16_t(n / 2);
        // mmadParams.k = k;
        
        // Mmad(c1Local, a2Local, b2Local, mmadParams);
        // Mmad(c1Local, a2Local, b2Local, { m, uint16_t(n / 2), k, false, 0, false, false, false });
        Mmad(c1Local, a2Local, b2Local, { m, uint16_t(n / 2), k, 0, false, true });

        outQueueCO1.EnQue<float>(c1Local);
        inQueueB2.FreeTensor(b2Local);
    }
    __aicore__ inline void Aggregate(const LocalTensor<float>& c2Local, const int bSplitIdx)
    {
        LocalTensor<float> c1Local = outQueueCO1.DeQue<float>();

        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = 2;
        DataCopyEnhancedParams enhancedParams;
        enhancedParams.blockMode = BlockMode::BLOCK_MODE_MATRIX;
        DataCopy(c2Local[bSplitIdx * cSize / 2], c1Local, dataCopyParams, enhancedParams);

        outQueueCO1.FreeTensor(c1Local);
    }
    __aicore__ inline void CopyOut()
    {
        LocalTensor<float> c2Local = outQueueCO2.DeQue<float>();
        // transform nz to nd
        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = m;
        dataCopyParams.blockLen = 2;
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = uint16_t((nBlocks - 1) * 2);
        for (int i = 0; i < nBlocks; i++) {
            DataCopy(cGM[i * 16], c2Local[i * m * 16], dataCopyParams);
        }
        // DataCopy(cGM[0], c2Local[0], dataCopyParams); 
        // DataCopy(cGM[16], c2Local[512], dataCopyParams); 
        outQueueCO2.FreeTensor(c2Local);
    }

private:
    TPipe pipe;

    TQue<QuePosition::A1, 1> inQueueA1; // L1 Buffer
    TQue<QuePosition::A2, 1> inQueueA2; // L0A Buffer
    TQue<QuePosition::B1, 1> inQueueB1; // L1 Buffer
    TQue<QuePosition::B2, 2> inQueueB2; // L0B Buffer
    // dst queue
    TQue<QuePosition::CO1, 2> outQueueCO1; // L0C Buffer
    TQue<QuePosition::CO2, 1> outQueueCO2; // UB

    GlobalTensor<half> aGM, bGM;
    GlobalTensor<float> cGM;

    uint16_t m = 32;
    uint16_t n = 32;
    uint16_t k = 32;

    uint16_t aSize, bSize, cSize, mBlocks, nBlocks, kBlocks;
};

extern "C" __global__ __aicore__ void matmul_custom(GM_ADDR a, GM_ADDR b, GM_ADDR c)
{
    KernelMatmul op;
    op.Init(a, b, c);
    op.Process();
}

#ifndef __CCE_KT_TEST__
// call of kernel function
void matmul_custom_do(uint32_t blockDim, void* l2ctrl, void* stream, uint8_t* a, uint8_t* b, uint8_t* c)
{
    matmul_custom<<<blockDim, l2ctrl, stream>>>(a, b, c);
}
#endif
