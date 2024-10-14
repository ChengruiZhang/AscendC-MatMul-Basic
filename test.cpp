#include "test.h"

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
        auto coreidx = GetBlockIdx();
        aGM.SetGlobalBuffer((__gm__ half*)a);
        bGM.SetGlobalBuffer((__gm__ half*)b);
        cGM.SetGlobalBuffer((__gm__ float*)c);
        pipe.InitBuffer(inQueueA1, 1, 1 * 512 * 1024);
        pipe.InitBuffer(inQueueA2, 1, 1 * 64 * 1024);
        pipe.InitBuffer(inQueueB2, 1, 1 * 64 * 1024);
        pipe.InitBuffer(outQueueCO1, 1, 1 * 128 * 1024);
    }

    __aicore__ inline void Process_GM2L1();
    __aicore__ inline void Process_L12GM();
    __aicore__ inline void Process_GM2L0AB();
    __aicore__ inline void Process_L12L0AB();

    private:
    TPipe pipe;

    TQue<QuePosition::A1, 1> inQueueA1;
    TQue<QuePosition::A2, 1> inQueueA2;
    TQue<QuePosition::B1, 1> inQueueB1;
    TQue<QuePosition::B2, 1> inQueueB2;
    // dst queue
    TQue<QuePosition::CO1, 1> outQueueCO1;
    // TQue<QuePosition::CO2, 1> outQueueCO2;

    GlobalTensor<half> aGM, bGM;
    GlobalTensor<float> cGM;

    uint16_t m, n, k;

    uint16_t aSize, bSize, cSize, mBlocks, nBlocks, kBlocks;
};

__aicore__ inline void KernelMatmul::Process_GM2L1()
{
    LocalTensor<half> a1Local = inQueueA1.AllocTensor<half>();
    for (int j = 0; j < 100; ++j){
        for (int i = 0; i < 1; ++i) {
            int srcOffset = 0;
            int dstOffset = 0;
            DataCopy(a1Local[dstOffset], aGM[srcOffset], { 32, 512, 0, 0 }); // 32 Byte per iter
        }
    }
    inQueueA1.EnQue(a1Local);
}

__aicore__ inline void KernelMatmul::Process_L12GM()
{
    LocalTensor<half> a1Local = inQueueA1.AllocTensor<half>();
    // init
    DataCopy(a1Local[0], aGM[0], { 32, 512, 0, 0 });
    for (int j = 0; j < 100; ++j){
        for (int i = 0; i < 1; ++i) {
            int srcOffset = 0;
            int dstOffset = 0;
            DataCopy(aGM[dstOffset], a1Local[srcOffset], { 32, 512, 0, 0 }); // 32 Byte per iter
        }
    }
    inQueueA1.EnQue(a1Local);
}

__aicore__ inline void KernelMatmul::Process_GM2L0AB()
{
    LocalTensor<half> a1Local = inQueueA1.AllocTensor<half>();
    // init
    DataCopy(a1Local[0], aGM[0], { 32, 512, 0, 0 });
    for (int j = 0; j < 100; ++j){
        for (int i = 0; i < 1; ++i) {
            int srcOffset = 0;
            int dstOffset = 0;
            DataCopy(aGM[dstOffset], a1Local[srcOffset], { 32, 512, 0, 0 }); // 32 Byte per iter
        }
    }
    inQueueA1.EnQue(a1Local);
}

__aicore__ inline void KernelMatmul::Process_L12L0AB()
{
    LocalTensor<half> a1Local = inQueueA1.AllocTensor<half>();
    // init
    DataCopy(a1Local[0], aGM[0], { 32, 512, 0, 0 });
    for (int j = 0; j < 100; ++j){
        for (int i = 0; i < 1; ++i) {
            int srcOffset = 0;
            int dstOffset = 0;
            DataCopy(aGM[dstOffset], a1Local[srcOffset], { 32, 512, 0, 0 }); // 32 Byte per iter
        }
    }
    inQueueA1.EnQue(a1Local);
}



extern "C" __global__ __aicore__ void GM2L1(GM_ADDR a, GM_ADDR b, GM_ADDR c, uint16_t M, uint16_t N, uint16_t K)
{
    KernelMatmul op(M, N, K);
    op.Init(a, b, c);
    op.Process_GM2L1();
}

extern "C" __global__ __aicore__ void L12GM(GM_ADDR a, GM_ADDR b, GM_ADDR c, uint16_t M, uint16_t N, uint16_t K)
{
    KernelMatmul op(M, N, K);
    op.Init(a, b, c);
    op.Process_L12GM();
}

extern "C" __global__ __aicore__ void GM2L0AB(GM_ADDR a, GM_ADDR b, GM_ADDR c, uint16_t M, uint16_t N, uint16_t K)
{
    KernelMatmul op(M, N, K);
    op.Init(a, b, c);
    op.Process_GM2L0AB();
}

extern "C" __global__ __aicore__ void L12L0AB(GM_ADDR a, GM_ADDR b, GM_ADDR c, uint16_t M, uint16_t N, uint16_t K)
{
    KernelMatmul op(M, N, K);
    op.Init(a, b, c);
    op.Process_L12L0AB();
}

#ifndef __CCE_KT_TEST__
// call of kernel function
void GM2L1_do(uint32_t blockDim, void* l2ctrl, void* stream, uint8_t* a, uint8_t* b, uint8_t* c, uint16_t M, uint16_t N, uint16_t K)
{
    GM2L1<<<blockDim, l2ctrl, stream>>>(a, b, c, M, N, K);
}

void L12GM_do(uint32_t blockDim, void* l2ctrl, void* stream, uint8_t* a, uint8_t* b, uint8_t* c, uint16_t M, uint16_t N, uint16_t K)
{
    L12GM<<<blockDim, l2ctrl, stream>>>(a, b, c, M, N, K);
}

void GM2L0AB_do(uint32_t blockDim, void* l2ctrl, void* stream, uint8_t* a, uint8_t* b, uint8_t* c, uint16_t M, uint16_t N, uint16_t K)
{
    RGM_WL0AB<<<blockDim, l2ctrl, stream>>>(a, b, c, M, N, K);
}

void L12L0AB_do(uint32_t blockDim, void* l2ctrl, void* stream, uint8_t* a, uint8_t* b, uint8_t* c, uint16_t M, uint16_t N, uint16_t K)
{
    RL1_WL0AB<<<blockDim, l2ctrl, stream>>>(a, b, c, M, N, K);
}

#endif