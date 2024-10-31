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
    /*
    * @brief: 初始化基本参数
    * @param: TilingL1K确保是偶数或者为1（实际不可能为1）
    * @param: BaseM, BaseN需要确保是32的倍数，以使得能够在M、N维度进行双缓存切分（暂不考虑）
    * 本文件，我们优先做K轴的双缓存切分
    */
    __aicore__ inline void InitAttr(uint16_t BaseM_, uint16_t BaseN_, uint16_t BaseK_,
                                    uint16_t BlockNumM_, uint16_t BlockNumN_, uint16_t BlockNumK_,
                                    uint16_t TilingL1K_,
                                    uint16_t TotalResBlocks_,
                                    TPipe *pipe_){
        // attr = attr_;
        BaseM = BaseM_; BaseN = BaseN_; BaseK = BaseK_;
        BlockNumM = BlockNumM_; BlockNumN = BlockNumN_; BlockNumK = BlockNumK_;
        TilingL1K = TilingL1K_;
        aL1Size = BaseM * TilingL1K * BaseK;
        bL1Size = BaseN * TilingL1K * BaseK;
        aL0Size = BaseM * BaseK;
        bL0Size = BaseK * BaseN;
        cSize = BaseM * BaseN;
        TotalResBlocks = TotalResBlocks_;
        
        pipe = pipe_;
        pipe->InitBuffer(inQueueA1, 2, aL1Size * sizeof(half) / 2);
        pipe->InitBuffer(inQueueA2, 1, aL0Size * sizeof(half));
        // pipe->InitBuffer(inQueueA2, 2, aL0Size * sizeof(half) / 2);
        pipe->InitBuffer(inQueueB1, 2, bL1Size * sizeof(half) / 2);
        // pipe->InitBuffer(inQueueB2, 2, bL0Size * sizeof(half) / 2);
        pipe->InitBuffer(inQueueB2, 1, bL0Size * sizeof(half));
        // pipe->InitBuffer(outQueueCO1, 2, cSize * sizeof(float) / 2);
        pipe->InitBuffer(outQueueCO1, 1, cSize * sizeof(float));
    }
    /*
    * @brief: 获取每一个核的偏移，保证每一次读取到核上的地址为当前输入A、B矩阵及输出C矩阵的首地址；
    * 此外，计算出当前核对应的输出大小MLen、NLen
    */
    __aicore__ inline void GetOffset(int64_t coreidx_){
        coreidx = coreidx_;
        coreM = coreidx / BlockNumN;
        coreN = coreidx % BlockNumN;
        MOffset = coreM * BaseM * K;
        NOffset = coreN * BaseN;
        ResOffset = coreM * BaseM * N + coreN * BaseN;
        MLen = (coreM + 1) >= BlockNumM ? M - coreM * BaseM : BaseM;
        NLen = (coreN + 1) >= BlockNumN ? N - coreN * BaseN : BaseN;
        mBlocks = MLen / 16;
        nBlocks = NLen / 16;
    }
    /*
    * @brief: 获取double buffer的K长度
    */
    __aicore__ inline void GetL1DouBufLen(const int CurLen){
        
        int CurBaseNum = CurLen / BaseK;

        KL1DouBufBase[1] = CurBaseNum / 2;
        KL1DouBufBase[0] = CurBaseNum - KL1DouBufBase[1];
        KL1DouBuf[0] = KL1DouBufBase[0] * BaseK;
        KL1DouBuf[1] = CurLen - KL1DouBuf[0];
        KL1DouBufBlocks[0] = KL1DouBuf[0] / 16;
        KL1DouBufBlocks[1] = KL1DouBuf[1] / 16;

        // KL1DouBufBlocks[0] = kL1Blocks / 2;
        // KL1DouBufBlocks[1] = kL1Blocks - KL1DouBufBlocks[0];
        // KL1DouBuf[0] = KL1DouBufBlocks[0] * 16;
        // KL1DouBuf[1] = KL1DouBufBlocks[1] * 16;
        
    }
    /*
    * @brief: 初始化输入A、B矩阵及输出C矩阵的全局Buffer
    */
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
        outQueueCO1.EnQue<float>(c1Local);

        // split K by L1
        // 将K在L1级别进行切分，每次读取至多TilingL1K*BaseK个K
        for (int L2KIdx = 0; L2KIdx < BlockNumK; L2KIdx += TilingL1K) {
            // 当前的L1级别的K大小
            KL1Len = (L2KIdx + TilingL1K) > BlockNumK ? K - L2KIdx * BaseK : BaseK * TilingL1K;
            kL1Blocks = KL1Len / 16;
            
            // 记录L1级别tiling后, 双缓存中K的长度
            GetL1DouBufLen(KL1Len);

            // LocalTensor<half> a1Local = inQueueA1.AllocTensor<half>();
            // LocalTensor<half> b1Local = inQueueB1.AllocTensor<half>();
            
            // L1 Double buffer -- K dim
            for (int DouBuf = 0; DouBuf < 2; DouBuf++) {
                // ND2NZ, GM to L1
                CopyIn(L2KIdx, DouBuf); // a1Local, b1Local: Alloc & EnQueue
                LocalTensor<half> a1Local = inQueueA1.DeQue<half>();
                LocalTensor<half> b1Local = inQueueB1.DeQue<half>();
                
                // split K by L0, read 1 base K to L0B, up to TilingL1K itrations
                // L1的K需要被读取进L0的次数
                uint16_t TilingL0K = (KL1DouBuf[DouBuf] + BaseK - 1) / BaseK;
                for (int L1SplitIdx = 0; L1SplitIdx < TilingL0K; L1SplitIdx++){
                    
                    KL0Len = (L1SplitIdx + 1) * BaseK > KL1DouBuf[DouBuf] ? KL1DouBuf[DouBuf] - L1SplitIdx * BaseK : BaseK;
                    kL0Blocks = KL0Len / 16;
                    mmadParams.k = KL0Len;
                    
                    SplitA(a1Local, L1SplitIdx, DouBuf); // a2Local alloc, enque
                    LocalTensor<half> a2Local = inQueueA2.DeQue<half>();
                    // split matrix B into 2 parts, [32, 16] and [32, 16]
                    // for (int k = 0; k < 1; ++k) {
                    SplitB(b1Local, L1SplitIdx, DouBuf); // b2Local alloc, enque
                    Compute(a2Local, mmadParams, L1SplitIdx, DouBuf); // c1 alloc,  b2Local deque
                    PipeBarrier<PIPE_M>();
                    // }
                    mmadParams.cmatrixInitVal = false;
                    inQueueA2.FreeTensor(a2Local);
                }
                inQueueA1.FreeTensor(a1Local);
                inQueueB1.FreeTensor(b1Local);
            }
        }
        // outQueueCO1.EnQue<float>(c1Local);
        CopyOut(0);
    }

private:
    /*
    * @brief: 实际的搬运过程，一次搬运32Byte * height的数据，搬运width/16次
    */
    __aicore__ inline void CopyND2NZ(const LocalTensor<half>& dst, const GlobalTensor<half>& src, const uint16_t height,
        const uint16_t width, const uint16_t TotalWidth, const int HeightOffset, const int WidthOffset,
        const int DstOffset){
        int srcOffset = HeightOffset + WidthOffset;
        int dstOffset = DstOffset;
        for (int i = 0; i < width / 16; ++i) {
            // DataCopy(dst[dstOffset], src[srcOffset], { height, 1, uint16_t(width / 16 - 1), 0 });
            DataCopy(dst[dstOffset], src[srcOffset], { height, 1, uint16_t(TotalWidth / 16 - 1), 0 });
            srcOffset += 16;
            dstOffset += 16 * height;
        }
    }
    /*
    * @brief: 获取输入A、B矩阵的L1 Buffer，从GM搬运数据到L1 Buffer上，
    * 其分配的内存空间为当前的TilingL1K*BaseK*(BaseM+BaseN)*Type
    * 转换前为ND格式，转换后为Nz格式
    * 实际搬运的矩阵大小为A[MLen, KL1Len], B[KL1Len, NLen]
    * @param: L2KIdx 当前核的索引
    */
    __aicore__ inline void CopyIn(int L2KIdx, const int DouBuf){
        LocalTensor<half> a1Local = inQueueA1.AllocTensor<half>();
        LocalTensor<half> b1Local = inQueueB1.AllocTensor<half>();

        int WidthOffset = L2KIdx * BaseK;
        int HeightOffset = L2KIdx * BaseK * N;

        int MDouBufL1Offset = DouBuf * KL1DouBuf[0];
        int NDouBufL1Offset = DouBuf * KL1DouBuf[0] * N;

        int ML1DstOffset = DouBuf * aL1Size / 2;
        int NL1DstOffset = DouBuf * bL1Size / 2;

        CopyND2NZ(a1Local, aGM[MOffset + MDouBufL1Offset], MLen, KL1DouBuf[DouBuf], K, 0, WidthOffset, ML1DstOffset); 
        CopyND2NZ(b1Local, bGM[NOffset + NDouBufL1Offset], KL1DouBuf[DouBuf], NLen, N, HeightOffset, 0, NL1DstOffset);

        // int HeightOffset = L2KIdx > 1? (L2KIdx - 1) * N * TilingL1K : 0;
        // int WidthOffset = L2KIdx > 1? (L2KIdx - 1) * BaseK * TilingL1K : 0;
        // CopyND2NZ(a1Local, aGM[MOffset], MLen, KL1Len, K, HeightOffset, 0);
        // CopyND2NZ(b1Local, bGM[NOffset], KL1Len, NLen, N, 0, WidthOffset);

        inQueueA1.EnQue(a1Local);
        inQueueB1.EnQue(b1Local);
    }
    /* 
    * @brief: 将L1Buffer中的A矩阵数据[MLen, KL1Len]搬运到L0A中，实际搬运长度为[MLen, KL0Len]
    * 搬运前为Nz格式，搬运后为Zz格式
    */
    __aicore__ inline void SplitA(LocalTensor<half>& a1Local, const int L1SplitIdx, const int DouBuf){
        // int srcOffset = L1SplitIdx * 16 * 16 * mBlocks;

        int MDouBufL0Offset = DouBuf * aL1Size / 2;

        int srcOffset = L1SplitIdx * MLen * BaseK + MDouBufL0Offset;
        int dstOffset = 0;
        // int dstOffset = DouBuf * aL0Size / 2;
        // LocalTensor<half> a1Local = inQueueA1.DeQue<half>();
        LocalTensor<half> a2Local = inQueueA2.AllocTensor<half>();

        if (KL1DouBufBase[DouBuf] != 0){
            LoadData2dParams loadDataParams;
            loadDataParams.repeatTimes = kL0Blocks;
            // loadDataParams.repeatTimes = KL1DouBufBlock[DouBuf];
            loadDataParams.srcStride = mBlocks;
            loadDataParams.ifTranspose = false;

            // transform nz to zz
            for (int i = 0; i < mBlocks; ++i) {
                LoadData(a2Local[dstOffset], a1Local[srcOffset], loadDataParams);
                srcOffset += 16 * 16;
                dstOffset += kL0Blocks * 16 * 16;
            }
        }
        inQueueA2.EnQue<half>(a2Local);
        
        // inQueueA1.FreeTensor(a1Local);
    }
    /* 
    * @brief: 将L1Buffer中的B矩阵数据[KL1Len, NLen]搬运到L0A中，实际搬运长度为[KL0Len, NLen]
    * 搬运前为Nz格式，搬运后为Zn格式
    */
    __aicore__ inline void SplitB(const LocalTensor<half>& b1Local, const int L1SplitIdx, const int DouBuf){
        
        int NDouBufL0Offset = DouBuf * bL1Size / 2;

        int srcOffset = L1SplitIdx * 16 * BaseK + NDouBufL0Offset;
        // int srcOffset = L1SplitIdx * 16 * 16 * kL0Blocks;
        int dstOffset = 0;

        LocalTensor<half> b2Local = inQueueB2.AllocTensor<half>();

        // transform Nz to Zn
        LoadData2dParams loadDataParams;
        loadDataParams.repeatTimes = nBlocks;
        // loadDataParams.repeatTimes = nBlocks / 2;
        loadDataParams.srcStride = KL1DouBufBlocks[DouBuf];
        // loadDataParams.srcStride = kL0Blocks;
        loadDataParams.ifTranspose = true;

        for (int i = 0; i < kL0Blocks; i++){
            LoadData(b2Local[dstOffset], b1Local[srcOffset], loadDataParams);
            srcOffset += 16 * 16;
            dstOffset += nBlocks * 16 * 16;
            // LoadData(b2Local[i * nBlocks / 2 * 16 * 16], b1Local[bSplitIdx * bSize / 2 + i * 16 * 16], loadDataParams);
        }

        inQueueB2.EnQue<half>(b2Local);
    }
    /*
    @brief: 将L0A、L0B中的数据进行计算，计算结果存放在L0C中，实际计算长度为[MLen, NLen]
    */
    __aicore__ inline void Compute(const LocalTensor<half>& a2Local, MmadParams mmadParams, const int L1SplitIdx, const int L0SplitIdx)
    {
        // int srcOffset = L1SplitIdx * 16 * 16 * mBlocks;
        int dstOffset = 0;

        LocalTensor<float> c1Local = outQueueCO1.DeQue<float>();
        LocalTensor<half> b2Local = inQueueB2.DeQue<half>();

        // Nz
        Mmad(c1Local, a2Local, b2Local, mmadParams);
        // PipeBarrier<PIPE_M>();

        outQueueCO1.EnQue<float>(c1Local);
        inQueueB2.FreeTensor(b2Local);
    }
    /*
    @brief: 将L0C中的数据搬运到GM中，实际搬运长度为[MLen, NLen]
    */
    __aicore__ inline void CopyOut(const int bSplitIdx)
    {
        LocalTensor<float> c1Local = outQueueCO1.DeQue<float>();
        FixpipeParamsV220 fixpipeParams;
        // fixpipeParams.nSize = NLen / 2;
        fixpipeParams.nSize = NLen; // L0C中的N大小
        fixpipeParams.mSize = MLen; // L0C中的M大小
        fixpipeParams.srcStride = MLen; // 搬运MLen次，一次搬运NLen大小
        fixpipeParams.dstStride = N; // 每次搬运时的目标地址的偏移
        fixpipeParams.ndNum = 1;
        Fixpipe(cGM[ResOffset + bSplitIdx * NLen], c1Local, fixpipeParams);
        // Fixpipe(cGM[bSplitIdx * NLen / 2], c1Local, fixpipeParams);

        outQueueCO1.FreeTensor(c1Local);
    }

private:
    TPipe* pipe;

    TQue<QuePosition::A1, 1> inQueueA1;
    TQue<QuePosition::A2, 1> inQueueA2;
    TQue<QuePosition::B1, 1> inQueueB1;
    TQue<QuePosition::B2, 1> inQueueB2;
    // dst queue
    TQue<QuePosition::CO1, 1> outQueueCO1;
    TQue<QuePosition::CO2, 1> outQueueCO2;

    GlobalTensor<half> aGM, bGM;
    GlobalTensor<float> cGM;

    uint16_t M, N, K;
    uint16_t BaseM, BaseN, BaseK;
    uint16_t coreM, coreN;
    uint16_t BlockNumM, BlockNumN, BlockNumK;
    uint16_t TilingL1K;
    uint32_t MLen, NLen, KL1Len, KL0Len;
    uint32_t MOffset, NOffset, ResOffset;
    // Attr attr;

    uint32_t aL1Size, bL1Size, aL0Size, bL0Size, cSize;
    uint16_t mBlocks, nBlocks, kBlocks, kL1Blocks, kL0Blocks;
    uint16_t TotalResBlocks;
    int64_t coreidx;

    // uint32_t KDouBuf1, KDouBuf2;
    // uint16_t KDouBuf1Blocks, KDouBuf2BLocks;
    uint32_t KL1DouBufBlocks[2], KL1DouBufBase[2];
    uint16_t KL1DouBuf[2];
};

// 直接传结构体会读不出来，可能需要片上构建
extern "C" __global__ __aicore__ void matmul_custom_m128_n256_k128(GM_ADDR A, GM_ADDR B, GM_ADDR C, 
                                                                   uint16_t M, uint16_t N, uint16_t K, 
                                                                   uint16_t BaseM, uint16_t BaseN, uint16_t BaseK,
                                                                   uint16_t BlockNumM, uint16_t BlockNumN, uint16_t BlockNumK,
                                                                   uint16_t TilingL1K,
                                                                   uint16_t TotalResBlocks)
{
    KernelMatmul op(M, N, K);
    TPipe pipe;
    op.InitAttr(BaseM, BaseN, BaseK,
                BlockNumM, BlockNumN, BlockNumK,
                TilingL1K,
                TotalResBlocks,
                &pipe);
    op.Init(A, B, C);
    
    // div core, each core processes (TotalResBlocks / CoreNum, or (TotalResBlocks + CoreNum - 1) / CoreNum)
    // Base Blocks
    auto CoreIdx = GetBlockIdx();
    for (; CoreIdx < TotalResBlocks; CoreIdx += 20)
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

    // uint16_t block = 16;
    // uint16_t height = 8;
    // uint16_t width = 16;

    uint32_t L1BufferSize = 1 * 512 * 1024;
    uint32_t L0ABBufferSize = 64 * 1024;
    uint32_t L0CBufferSize = 64 * 1024;
    
    // assert(uint32_t(BaseM) * uint32_t(BaseK) * 2 < L0ABBufferSize);
    // assert(uint32_t(BaseN) * uint32_t(BaseK) * 2 < L0ABBufferSize);
    // assert(uint32_t(BaseN) * uint32_t(BaseM) * 4 < L0CBufferSize);

    uint16_t BlockNumM = (M + BaseM - 1) / BaseM;
    uint16_t BlockNumN = (N + BaseN - 1) / BaseN;
    uint16_t BlockNumK = (K + BaseK - 1) / BaseK;

    // assert(false);

    // assert(BlockNumM > 0);
    // assert(BlockNumN > 0);
    // assert(BlockNumK > 0);

    uint16_t TotalResBlocks = BlockNumM * BlockNumN;
    if (blockDim > TotalResBlocks) { blockDim = uint32_t(TotalResBlocks); }
    // else {CHECK_ACL(false);}

    // uint16_t CoreTilingM = (BlockNumM - 1) / height;
    // uint16_t CoreTilingN = (BlockNumN - 1) / width;

    OpType optype = OpType::fp16; 

    uint16_t TilingL1K = L1BufferSize / (BaseM * BaseK + BaseN * BaseK) / optype;
    TilingL1K -= TilingL1K % 2;
    TilingL1K = TilingL1K > 0 ? TilingL1K : 1;
    // uint16_t maxL1K = L1BufferSize / (BaseM * BaseK + BaseN * BaseK) / optype;
    // attr_.TilingL1K = L1BufferSize / (BaseM * BaseK + BaseN * BaseK) / optype;
    // attr_.BlockNumM = BlockNumM; attr_.BlockNumN = BlockNumN; attr_.BlockNumK = BlockNumK;

    matmul_custom_m128_n256_k128<<<blockDim, l2ctrl, stream>>>(A, B, C, 
                                                               M, N, K, 
                                                               BaseM, BaseN, BaseK,
                                                               BlockNumM, BlockNumN, BlockNumK,
                                                               TilingL1K,
                                                               TotalResBlocks);
}
#endif
