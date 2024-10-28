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
// #include "data_utils.h"
using namespace AscendC;


enum OpType {
    fp32 = 4,
    fp16 = 2,
    int8 = 1,
};

// struct Offset {
//     uint16_t height;
//     uint16_t width;

//     Offset(uint16_t height_, uint16_t width_): height(height_), width(width_) {}
// };

struct Attr{
    uint16_t BaseM, BaseN, BaseK;
    uint16_t BlockNumM, BlockNumN, BlockNumK;
    uint16_t TilingL1K;
    Attr(uint16_t BaseM_, uint16_t BaseN_, uint16_t BaseK_): BaseM(BaseM_), BaseN(BaseN_), BaseK(BaseK_){}
};

class KernelMatmul {
public:
    // __aicore__ inline KernelMatmul(Attr attr_){
    __aicore__ inline KernelMatmul(uint16_t m, uint16_t n, uint16_t k){
        M = m; N = n; K = k;
    }
    __aicore__ inline void InitAttr(uint16_t BaseM_, uint16_t BaseN_, uint16_t BaseK_,
                                    uint16_t BlockNumM_, uint16_t BlockNumN_, uint16_t BlockNumK_,
                                    uint16_t TilingL1K_,
                                    uint16_t TotalResBlock_){
        // attr = attr_;
        BaseM = BaseM_; BaseN = BaseN_; BaseK = BaseK_;
        BlockNumM = BlockNumM_; BlockNumN = BlockNumN_; BlockNumK = BlockNumK_;
        TilingL1K = TilingL1K_;
        aL1Size = BaseM * TilingL1K * BaseK;
        bL1Size = BaseN * TilingL1K * BaseK;
        aL2Size = BaseM * BaseK;
        bL2Size = BaseK * BaseN;
        cSize = BaseM * BaseN;
        TotalResBlock = TotalResBlock_;
        
        pipe.InitBuffer(inQueueA1, 1, aL1Size * sizeof(half));
        pipe.InitBuffer(inQueueA2, 1, aL2Size * sizeof(half));
        pipe.InitBuffer(inQueueB1, 1, bL1Size * sizeof(half));
        pipe.InitBuffer(inQueueB2, 1, bL2Size * sizeof(half));
        pipe.InitBuffer(outQueueCO1, 1, cSize * sizeof(float));
    }
    __aicore__ inline void GetOffset(int64_t coreidx_){
        coreidx = coreidx_;
        coreM = coreidx / BlockNumN;
        coreN = coreidx % BlockNumN;
        MOffset = coreM * BaseM * K;
        NOffset = coreN * BaseN;
        ResOffset = coreM * BaseM * N + coreN * BaseN;
        MLen = (coreM + 1) > BlockNumM ? M - coreM * BaseM : BaseM;
        NLen = (coreN + 1) > BlockNumN ? N - coreN * BaseN : BaseN;
        mBlocks = MLen / 16;
        nBlocks = NLen / 16;
    }
    __aicore__ inline void Init(GM_ADDR A, GM_ADDR B, GM_ADDR C)
    {
        // auto coreidx = GetBlockIdx();
        // GetOffset(coreidx);
        // GetOffset(CoreIdx);
        aGM.SetGlobalBuffer((__gm__ half*)A);
        bGM.SetGlobalBuffer((__gm__ half*)B);
        cGM.SetGlobalBuffer((__gm__ float*)C);
        
        // aGM.SetGlobalBuffer((__gm__ half*)A + MOffset);
        // bGM.SetGlobalBuffer((__gm__ half*)B + NOffset);
        // cGM.SetGlobalBuffer((__gm__ float*)C + ResOffset);
    }
    __aicore__ inline void Process(){

        MmadParams mmadParams;
        mmadParams.m = MLen;
        mmadParams.n = NLen;
        mmadParams.cmatrixInitVal = true;

        LocalTensor<float> c1Local = outQueueCO1.AllocTensor<float>();

        // split K by L1
        for (int i = 0; i < BlockNumK; i += TilingL1K) {

            KL1Len = (i + TilingL1K) > BlockNumK ? K - i * BaseK * TilingL1K : BaseK * TilingL1K;
            kL1FineBlocks = KL1Len / 16;
            // bsize = KL1Len * NLen * 2;
            
            // ND2NZ, GM to L1
            CopyIn(i); // a1Local, b1Local: Alloc & EnQueue
            LocalTensor<half> a1Local = inQueueA1.DeQue<half>();
            LocalTensor<half> b1Local = inQueueB1.DeQue<half>();

            // split K by L0, read 1 base K to L0B, up to TilingL1K itrations
            uint16_t TilingL2K = KL1Len / BaseK;
            for (int j = 0; j < TilingL2K; j++){
                
                KL2Len = (j + 1) * BaseK > KL1Len ? KL1Len - j * BaseK : BaseK;
                kL2FineBlocks = KL2Len / 16;
                mmadParams.k = KL2Len;
                
                SplitA(j, a1Local);
                LocalTensor<half> a2Local = inQueueA2.DeQue<half>();
                // split matrix B into 2 parts, [32, 16] and [32, 16]
                for (int k = 0; k < 1; ++k) {
                    SplitB(b1Local, j, k);
                    Compute(a2Local, c1Local, mmadParams, j, k);
                    // PipeBarrier<PIPE_M>();
                }
                mmadParams.cmatrixInitVal = false;
                inQueueA2.FreeTensor(a2Local);
            }
            inQueueA1.FreeTensor(a1Local);
            inQueueB1.FreeTensor(b1Local);
        }
        outQueueCO1.EnQue<float>(c1Local);
        CopyOut(0);
        // PipeBarrier<PIPE_ALL>();
    }

private:
    __aicore__ inline void CopyND2NZ(const LocalTensor<half>& dst, const GlobalTensor<half>& src, const uint16_t height,
        const uint16_t width, const uint16_t TotalWidth, const int HeightOffset, const int WidthOffset){
        int srcOffset = HeightOffset + WidthOffset;
        int dstOffset = 0;
        for (int i = 0; i < width / 16; ++i) {
            // DataCopy(dst[dstOffset], src[srcOffset], { height, 1, uint16_t(width / 16 - 1), 0 });
            DataCopy(dst[dstOffset], src[srcOffset], { height, 1, uint16_t(TotalWidth / 16 - 1), 0 });
            srcOffset += 16;
            dstOffset += 16 * height;
        }
    }
    __aicore__ inline void CopyIn(int idx){
        LocalTensor<half> a1Local = inQueueA1.AllocTensor<half>();
        LocalTensor<half> b1Local = inQueueB1.AllocTensor<half>();

        int HeightOffset = idx > 1? (idx - 1) * N * TilingL1K : 0;
        int WidthOffset = idx > 1? (idx - 1) * BaseK * TilingL1K : 0;
        CopyND2NZ(a1Local, aGM[MOffset], MLen, KL1Len, K, HeightOffset, 0);
        CopyND2NZ(b1Local, bGM[NOffset], KL1Len, NLen, N, 0, WidthOffset);

        inQueueA1.EnQue(a1Local);
        inQueueB1.EnQue(b1Local);
    }
    __aicore__ inline void SplitA(int L1Splitidx, LocalTensor<half>& a1Local){
        // int srcOffset = L1Splitidx * 16 * 16 * mBlocks;
        int srcOffset = L1Splitidx * 16 * 16 * mBlocks;
        int dstOffset = 0;
        // LocalTensor<half> a1Local = inQueueA1.DeQue<half>();
        LocalTensor<half> a2Local = inQueueA2.AllocTensor<half>();

        LoadData2dParams loadDataParams;
        loadDataParams.repeatTimes = kL2FineBlocks;
        loadDataParams.srcStride = mBlocks;
        loadDataParams.ifTranspose = false;

        // transform nz to zz
        for (int i = 0; i < mBlocks; ++i) {
            LoadData(a2Local[dstOffset], a1Local[srcOffset], loadDataParams);
            srcOffset += 16 * 16;
            dstOffset += kL2FineBlocks * 16 * 16;
        }
        
        inQueueA2.EnQue<half>(a2Local);
        // inQueueA1.FreeTensor(a1Local);
    }
    __aicore__ inline void SplitB(const LocalTensor<half>& b1Local, const int L1Splitidx, const int L2SplitIdx){
        
        int srcOffset = L1Splitidx * 16 * 16 * kL2FineBlocks;
        // int srcOffset = L1Splitidx * 16 * 16 * kL2FineBlocks;
        int dstOffset = 0;

        LocalTensor<half> b2Local = inQueueB2.AllocTensor<half>();

        // transform Nz to Zn
        LoadData2dParams loadDataParams;
        loadDataParams.repeatTimes = nBlocks;
        // loadDataParams.repeatTimes = nBlocks / 2;
        loadDataParams.srcStride = kL2FineBlocks;
        loadDataParams.ifTranspose = true;

        for (int i = 0; i < kL2FineBlocks; i++){
            LoadData(b2Local[dstOffset], b1Local[srcOffset], loadDataParams);
            srcOffset += 16 * 16;
            dstOffset += nBlocks * 16 * 16;
            // LoadData(b2Local[i * nBlocks / 2 * 16 * 16], b1Local[bSplitIdx * bSize / 2 + i * 16 * 16], loadDataParams);
        }

        inQueueB2.EnQue<half>(b2Local);
    }
    __aicore__ inline void Compute(const LocalTensor<half>& a2Local, LocalTensor<float>& c1Local, MmadParams mmadParams, const int L1Splitidx, const int L2SplitIdx)
    {
        // int srcOffset = L1Splitidx * 16 * 16 * mBlocks;
        int dstOffset = 0;

        // LocalTensor<float> c1Local = outQueueCO1.AllocTensor<float>();
        LocalTensor<half> b2Local = inQueueB2.DeQue<half>();

        // Nz
        Mmad(c1Local, a2Local, b2Local, mmadParams);
        PipeBarrier<PIPE_M>();

        // outQueueCO1.EnQue<float>(c1Local);
        inQueueB2.FreeTensor(b2Local);
    }
    
    __aicore__ inline void CopyOut(const int bSplitIdx)
    {
        LocalTensor<float> c1Local = outQueueCO1.DeQue<float>();
        FixpipeParamsV220 fixpipeParams;
        // fixpipeParams.nSize = NLen / 2;
        fixpipeParams.nSize = NLen;
        fixpipeParams.mSize = MLen;
        fixpipeParams.srcStride = MLen;
        fixpipeParams.dstStride = N;
        fixpipeParams.ndNum = 1;
        Fixpipe(cGM[ResOffset + bSplitIdx * NLen], c1Local, fixpipeParams);
        // Fixpipe(cGM[bSplitIdx * NLen / 2], c1Local, fixpipeParams);

        outQueueCO1.FreeTensor(c1Local);
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

    uint16_t M, N, K;
    uint16_t BaseM, BaseN, BaseK;
    uint16_t coreM, coreN;
    uint16_t BlockNumM, BlockNumN, BlockNumK;
    uint16_t TilingL1K;
    uint32_t MLen, NLen, KL1Len, KL2Len;
    uint32_t MOffset, NOffset, ResOffset;
    // Attr attr;

    uint32_t aL1Size, bL1Size, aL2Size, bL2Size, cSize;
    uint16_t mBlocks, nBlocks, kBlocks, kL1FineBlocks, kL2FineBlocks;
    uint16_t TotalResBlock;
    int64_t coreidx;
};

// 直接传结构体会读不出来，可能需要片上构建
extern "C" __global__ __aicore__ void matmul_custom_m128_n256_k128(GM_ADDR A, GM_ADDR B, GM_ADDR C, 
                                                                   uint16_t M, uint16_t N, uint16_t K, 
                                                                   uint16_t BaseM, uint16_t BaseN, uint16_t BaseK,
                                                                   uint16_t BlockNumM, uint16_t BlockNumN, uint16_t BlockNumK,
                                                                   uint16_t TilingL1K,
                                                                   uint16_t TotalResBlock)
{
    KernelMatmul op(M, N, K);
    op.InitAttr(BaseM, BaseN, BaseK,
                BlockNumM, BlockNumN, BlockNumK,
                TilingL1K,
                TotalResBlock);
    op.Init(A, B, C);
    auto CoreIdx = GetBlockIdx();
    for (; CoreIdx < TotalResBlock; CoreIdx += 20)
    {
        op.GetOffset(CoreIdx);
        op.Process();
    }
}

#ifndef __CCE_KT_TEST__40
// call of kernel function
void matmul_custom_do(uint32_t blockDim, void* l2ctrl, void* stream, uint8_t* A, uint8_t* B, uint8_t* C, uint16_t M, uint16_t N, uint16_t K)
{

    // best: mem-comp 
    // 180, 180, K -- 90
    // 128, 256, K -- 85.3
    // 192, 160, K -- 87.3
    uint16_t BaseM = 128;
    uint16_t BaseN = 256;
    uint16_t BaseK = 128;
    // Attr attr_(BaseM, BaseN, BaseK);

    uint16_t block = 16;
    uint16_t height = 8;
    uint16_t width = 16;

    uint32_t L1BufferSize = 1 * 512 * 1024;
    uint32_t L0ABBufferSize = 64 * 1024;
    uint32_t L0CBufferSize = 64 * 1024;
    
    assert(uint32_t(BaseM) * uint32_t(BaseK) * 2 < L0ABBufferSize);
    assert(uint32_t(BaseN) * uint32_t(BaseK) * 2 < L0ABBufferSize);
    assert(uint32_t(BaseN) * uint32_t(BaseM) * 4 < L0CBufferSize);

    uint16_t BlockNumM = M / BaseM;
    uint16_t BlockNumN = N / BaseN;
    uint16_t BlockNumK = K / BaseK;

    // assert(false);

    assert(BlockNumM > 0);
    assert(BlockNumN > 0);
    assert(BlockNumK > 0);

    uint16_t TotalResBlock = BlockNumM * BlockNumN;
    if (blockDim > TotalResBlock) { blockDim = uint32_t(TotalResBlock); }
    // else {CHECK_ACL(false);}

    // uint16_t CoreTilingM = (BlockNumM - 1) / height;
    // uint16_t CoreTilingN = (BlockNumN - 1) / width;

    OpType optype = OpType::fp16; 

    uint16_t TilingL1K = L1BufferSize / (BaseM * BaseK + BaseN * BaseK) / optype;
    // uint16_t maxL1K = L1BufferSize / (BaseM * BaseK + BaseN * BaseK) / optype;
    // attr_.TilingL1K = L1BufferSize / (BaseM * BaseK + BaseN * BaseK) / optype;
    // attr_.BlockNumM = BlockNumM; attr_.BlockNumN = BlockNumN; attr_.BlockNumK = BlockNumK;

    matmul_custom_m128_n256_k128<<<blockDim, l2ctrl, stream>>>(A, B, C, 
                                                               M, N, K, 
                                                               BaseM, BaseN, BaseK,
                                                               BlockNumM, BlockNumN, BlockNumK,
                                                               TilingL1K,
                                                               TotalResBlock);
}
#endif
