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
        pipe.InitBuffer(inQueueB2, 1, bSize * sizeof(half));
        pipe.InitBuffer(outQueueCO1, 1, cSize * sizeof(float));
        pipe.InitBuffer(outQueueCO2, 1, cSize * sizeof(float));
        // DumpTensor(aGM, 0, 16);
        // DumpTensor(bGM, 0, 16);
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        SplitA();

        LocalTensor<half> b1Local = inQueueB1.DeQue<half>();
        LocalTensor<half> a2Local = inQueueA2.DeQue<half>();
        // LocalTensor<float> c2Local = outQueueCO2.AllocTensor<float>();
        // split matrix b into 2 parts, [32, 16] and [32, 16]
        for (int i = 0; i < 1; ++i) {
            SplitB(b1Local, i);
            Compute(a2Local);
            // Aggregate(c2Local, i);
            CopyOut(i);
        }
        inQueueB1.FreeTensor(b1Local);
        inQueueA2.FreeTensor(a2Local);
        // outQueueCO2.EnQue<float>(c2Local);

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

        // DumpTensor(a1Local, 0, 16);
        // DumpTensor(b1Local, 0, 16);

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
        
        // DumpTensor(a2Local, 0, 16);

        inQueueA2.EnQue<half>(a2Local);
        inQueueA1.FreeTensor(a1Local);
    }
    __aicore__ inline void SplitB(const LocalTensor<half>& b1Local, const int bSplitIdx)
    {
        LocalTensor<half> b2Local = inQueueB2.AllocTensor<half>();

        // // transform nz to Zn
        // LoadData2dParams loadDataParams;
        // loadDataParams.repeatTimes = nBlocks;
        // loadDataParams.srcStride = kBlocks;
        // loadDataParams.ifTranspose = true;

        // for (int i = 0; i < kBlocks; i++){
        //     LoadData(b2Local[i * nBlocks * 16 * 16], b1Local[bSplitIdx * bSize / 2 + i * nBlocks * 16 * 16], loadDataParams);
        // }

        // transform nz to Nn
        LoadData2dParams loadDataParams;
        loadDataParams.repeatTimes = kBlocks;
        loadDataParams.srcStride = 1;
        loadDataParams.ifTranspose = true;

        for (int i = 0; i < nBlocks; i++){
            LoadData(b2Local[i * kBlocks * 16 * 16], b1Local[bSplitIdx * bSize / 2 + i * kBlocks * 16 * 16], loadDataParams);
        }


        // LoadData2dTransposeParams loadData2dTransposeParams; 
        // loadData2dTransposeParams.repeatTimes = kBlocks;
        // loadData2dTransposeParams.srcStride = 1;
        // loadData2dTransposeParams.dstGap = 1;
        // // LoadDataWithTranspose(b2Local, b1Local[ bSplitIdx * bSize / 2], loadData2dTransposeParams);
        
        // for (int i = 0; i < (nBlocks / 2); ++i) {
        //     LoadDataWithTranspose(b2Local[i * kBlocks * 16 * 16], b1Local[bSplitIdx * bSize / 2 + i * kBlocks * 16 * 16], loadData2dTransposeParams);
        // }

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

        // GemmTiling tilling = GetGemmTiling<half>(m, k, n);
        // tilling.loopMode = LoopMode::MODE_NM;
        // bool initValue = false;
        // Gemm(c1Local, a2Local, b2Local, m, k, n, tilling, false, initValue);

        Mmad(c1Local, a2Local, b2Local, { m, n, k, false, 0, false, false, false });

        // DumpTensor(c1Local, 0, 16);

        outQueueCO1.EnQue<float>(c1Local);
        inQueueB2.FreeTensor(b2Local);
    }
    
    
    __aicore__ inline void CopyOut(const int bSplitIdx)
    {
        LocalTensor<float> c1Local = outQueueCO1.DeQue<float>();
        FixpipeParamsV220 fixpipeParams;
        fixpipeParams.nSize = n;
        fixpipeParams.mSize = m;
        fixpipeParams.srcStride = m;
        fixpipeParams.dstStride = n;
        fixpipeParams.ndNum = 1;
        fixpipeParams.srcNdStride = 2;
        fixpipeParams.dstNdStride = m * n;
        Fixpipe(cGM[bSplitIdx * n], c1Local, fixpipeParams);
        // // transform nz to nd
        // for (int i = 0; i < nBlocks; ++i) {
        //     DataCopy(cGM[i * 16], c2Local[i * m * 16], { m, 2, 0, uint16_t((nBlocks - 1) * 2) });
        // }

        outQueueCO1.FreeTensor(c1Local);
        // outQueueCO2.FreeTensor(c2Local);
    }

private:
    TPipe pipe;

    TQue<QuePosition::A1, 1> inQueueA1;
    TQue<QuePosition::A2, 1> inQueueA2;
    TQue<QuePosition::B1, 1> inQueueB1;
    TQue<QuePosition::B2, 1> inQueueB2;
    // dst queue
    TQue<QuePosition::CO1, 1> outQueueCO1;
    TQue<QuePosition::CO2, 1> outQueueCO2;

    GlobalTensor<half> aGM, bGM;
    GlobalTensor<float> cGM;

    uint16_t m, n, k;
    // uint16_t m = 64;
    // uint16_t n = 64;
    // uint16_t k = 64;

    uint16_t aSize, bSize, cSize, mBlocks, nBlocks, kBlocks;
};

extern "C" __global__ __aicore__ void matmul_custom(GM_ADDR a, GM_ADDR b, GM_ADDR c, uint16_t M, uint16_t N, uint16_t K)
{
    KernelMatmul op(M, N, K);
    op.Init(a, b, c);
    op.Process();
}

#ifndef __CCE_KT_TEST__
// call of kernel function
void matmul_custom_do(uint32_t blockDim, void* l2ctrl, void* stream, uint8_t* a, uint8_t* b, uint8_t* c, uint16_t M, uint16_t N, uint16_t K)
{
    matmul_custom<<<blockDim, l2ctrl, stream>>>(a, b, c, M, N, K);
}
#endif
