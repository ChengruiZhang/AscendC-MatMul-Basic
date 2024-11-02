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
        TotalResBlocks = TotalResBlocks_;
        pipe = pipe_;

        aL1Size = BaseM * TilingL1K * BaseK; // 读取TiliingL1K个basic block，这里是4
        bL1Size = BaseN * TilingL1K * BaseK;
        aL0Size = BaseM * BaseK;
        bL0Size = BaseK * BaseN;
        cSize = BaseM * BaseN;
        
        pipe->InitBuffer(inQueueA1, 2, aL1Size * sizeof(half) / 2);
        pipe->InitBuffer(inQueueA2, 2, aL0Size * sizeof(half));
        pipe->InitBuffer(inQueueB1, 2, bL1Size * sizeof(half) / 2);
        pipe->InitBuffer(inQueueB2, 2, bL0Size * sizeof(half));
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
    * @param: KL1DouBufBase[2]记录K共需要几次Compute计算
    * @param: KL1DouBuf[2]记录当前K的长度
    */
    __aicore__ inline void GetL1DouBufLen(const int CurLen){
        
        int CurBaseNum = CurLen / BaseK;

        KL1DouBufBase[1] = CurBaseNum / 2;
        KL1DouBufBase[0] = CurBaseNum - KL1DouBufBase[1];
        
        KL1DouBuf[0] = KL1DouBufBase[0] * BaseK;
        KL1DouBuf[1] = CurLen - KL1DouBuf[0];
        
        KL1DouBufBlocks[0] = KL1DouBuf[0] / 16;
        KL1DouBufBlocks[1] = KL1DouBuf[1] / 16;

    }
    /*
    * @brief: 初始化输入A、B矩阵及输出C矩阵的全局Buffer
    */
    __aicore__ inline void Init(GM_ADDR A, GM_ADDR B, GM_ADDR C)
    {
        aGM.SetGlobalBuffer((__gm__ half*)A);
        bGM.SetGlobalBuffer((__gm__ half*)B);
        cGM.SetGlobalBuffer((__gm__ float*)C);
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

            // L1 Double buffer -- K dim
            // L1中读取的次数
            for (int DouBufL1 = 0; DouBufL1 < 2; DouBufL1++) {
                // ND2NZ, GM to L1
                CopyIn(L2KIdx, DouBufL1); // a1Local, b1Local: Alloc & EnQueue
                LocalTensor<half> a1Local = inQueueA1.DeQue<half>();
                LocalTensor<half> b1Local = inQueueB1.DeQue<half>();
                
                // split K by L0, read 1 base K to L0B, up to TilingL1K itrations
                // L1的K需要被读取进L0的次数
                // L0 double buffer
                uint16_t TilingL0K = (KL1DouBuf[DouBufL1] + BaseK - 1) / BaseK;
                for (int DouBufL0 = 0; DouBufL0 < TilingL0K; DouBufL0++){
                    
                    KL0Len = (DouBufL0 + 1) * BaseK > KL1DouBuf[DouBufL1] ? KL1DouBuf[DouBufL1] - DouBufL0 * BaseK : BaseK;
                    kL0Blocks = KL0Len / 16;
                    mmadParams.k = KL0Len;
                    
                    SplitA(a1Local, DouBufL1, DouBufL0); // a2Local alloc, enque
                    SplitB(b1Local, DouBufL1, DouBufL0); // b2Local alloc, enque
                    Compute(mmadParams, DouBufL1, DouBufL0); // c1 alloc,  b2Local deque
                }
                inQueueA1.FreeTensor(a1Local);
                inQueueB1.FreeTensor(b1Local);
            }
        }
        CopyOut();
    }

private:
    /*
    * @brief: 实际的搬运过程，一次搬运32Byte * height的数据，搬运width/16次
    */
    __aicore__ inline void CopyND2NZ(const LocalTensor<half>& dst, const GlobalTensor<half>& src, const uint16_t height,
        const uint16_t width, const uint16_t TotalWidth, const int srcOffset_,
        const int dstOffset_){
        int srcOffset = srcOffset_;
        int dstOffset = dstOffset_;
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
    * @param: L2KIdx 当前核的K索引，为TilingL1K的倍数
    */
    __aicore__ inline void CopyIn(int L2KIdx, const int DouBuf){
        LocalTensor<half> a1Local = inQueueA1.AllocTensor<half>();
        LocalTensor<half> b1Local = inQueueB1.AllocTensor<half>();

        int MDouBufGMOffset = L2KIdx * BaseK;
        int MDouBufL1Offset = DouBuf * KL1DouBuf[0];

        int NDouBufGMOffset = L2KIdx * BaseK * N;
        int NDouBufL1Offset = DouBuf * KL1DouBuf[0] * N;

        int ML1DstOffset = 0;
        int NL1DstOffset = 0;

        // int ML1DstOffset = DouBuf * aL1Size / 2;
        // int NL1DstOffset = DouBuf * bL1Size / 2;

        CopyND2NZ(a1Local, aGM[MOffset], MLen, KL1DouBuf[DouBuf], K, MDouBufGMOffset + MDouBufL1Offset, ML1DstOffset); 
        CopyND2NZ(b1Local, bGM[NOffset], KL1DouBuf[DouBuf], NLen, N, NDouBufGMOffset + NDouBufL1Offset, NL1DstOffset);

        inQueueA1.EnQue(a1Local);
        inQueueB1.EnQue(b1Local);
    }
    /* 
    * @brief: 将L1Buffer中的A矩阵数据[MLen, KL1Len]搬运到L0A中，实际搬运长度为[MLen, KL0Len]
    * 搬运前为Nz格式，搬运后为Zz格式
    * @param: DouBufL0, 从L1搬向L0的block索引
    */
    __aicore__ inline void SplitA(LocalTensor<half>& a1Local, const int DouBufL1, const int DouBufL0){

        LocalTensor<half> a2Local = inQueueA2.AllocTensor<half>();

        // int MDouBufL1Offset = DouBufL1 * aL1Size / 2;
        int MDouBufL1Offset = 0;
        int srcOffset = MDouBufL1Offset + DouBufL0 * BaseK * MLen;
        // int dstOffset = DouBufL1 * aL0Size;
        int dstOffset = 0;

        LoadData2dParams loadDataParams;
        loadDataParams.repeatTimes = kL0Blocks;
        loadDataParams.srcStride = mBlocks;
        loadDataParams.ifTranspose = false;

        // transform nz to zz
        for (int i = 0; i < mBlocks; ++i) {
            LoadData(a2Local[dstOffset], a1Local[srcOffset], loadDataParams);
            srcOffset += 16 * 16;
            dstOffset += kL0Blocks * 16 * 16;
        }
        inQueueA2.EnQue<half>(a2Local);

    }
    /* 
    * @brief: 将L1Buffer中的B矩阵数据[KL1Len, NLen]搬运到L0A中，实际搬运长度为[KL0Len, NLen]
    * 搬运前为Nz格式，搬运后为Zn格式
    */
    __aicore__ inline void SplitB(const LocalTensor<half>& b1Local, const int DouBufL1, const int DouBufL0){
        
        LocalTensor<half> b2Local = inQueueB2.AllocTensor<half>();

        // int NDouBufL0Offset = DouBufL1 * bL1Size / 2;
        int NDouBufL0Offset = 0;
        int srcOffset = NDouBufL0Offset + DouBufL0 * 16 * BaseK;
        // int srcOffset = L1SplitIdx * 16 * 16 * kL0Blocks;
        // int dstOffset = DouBufL1 * bL0Size;
        int dstOffset = 0;

        // transform Nz to Zn
        LoadData2dParams loadDataParams;
        loadDataParams.repeatTimes = nBlocks;
        loadDataParams.srcStride = KL1DouBufBlocks[DouBufL1];
        loadDataParams.ifTranspose = true;

        for (int i = 0; i < kL0Blocks; i++){
            LoadData(b2Local[dstOffset], b1Local[srcOffset], loadDataParams);
            srcOffset += 16 * 16;
            dstOffset += nBlocks * 16 * 16;
        }

        inQueueB2.EnQue<half>(b2Local);
    }
    /*
    @brief: 将L0A、L0B中的数据进行计算，计算结果存放在L0C中，实际计算长度为[MLen, NLen]
    */
    __aicore__ inline void Compute(MmadParams& mmadParams, const int DouBufL1, const int DouBufL0)
    {
        LocalTensor<half> a2Local = inQueueA2.DeQue<half>();
        LocalTensor<half> b2Local = inQueueB2.DeQue<half>();
        LocalTensor<float> c1Local = outQueueCO1.DeQue<float>();

        // int srcOffset = L1SplitIdx * 16 * 16 * mBlocks;
        int dstOffset = 0;

        // Nz
        Mmad(c1Local, a2Local, b2Local, mmadParams);
        PipeBarrier<PIPE_M>();
        mmadParams.cmatrixInitVal = false;

        outQueueCO1.EnQue<float>(c1Local);
        inQueueB2.FreeTensor(b2Local);
        inQueueA2.FreeTensor(a2Local);
    }
    /*
    @brief: 将L0C中的数据搬运到GM中，实际搬运长度为[MLen, NLen]
    */
    __aicore__ inline void CopyOut()
    {
        LocalTensor<float> c1Local = outQueueCO1.DeQue<float>();
        FixpipeParamsV220 fixpipeParams;
        fixpipeParams.nSize = NLen; // L0C中的N大小
        fixpipeParams.mSize = MLen; // L0C中的M大小
        fixpipeParams.srcStride = MLen; // 搬运MLen次，一次搬运NLen大小
        fixpipeParams.dstStride = N; // 每次搬运时的目标地址的偏移
        fixpipeParams.ndNum = 1;
        Fixpipe(cGM[ResOffset], c1Local, fixpipeParams);

        outQueueCO1.FreeTensor(c1Local);
    }

private:
    TPipe* pipe;

    TQue<QuePosition::A1, 2> inQueueA1;
    TQue<QuePosition::A2, 2> inQueueA2;
    TQue<QuePosition::B1, 2> inQueueB1;
    TQue<QuePosition::B2, 2> inQueueB2;
    // dst queue
    TQue<QuePosition::CO1, 2> outQueueCO1;
    TQue<QuePosition::CO2, 2> outQueueCO2;

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
    uint16_t BaseK = 64;
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

    OpType optype = OpType::fp16; 

    // uint16_t TilingL1K = L1BufferSize / (BaseM * BaseK + BaseN * BaseK) / optype;
    // TilingL1K -= TilingL1K % 2;
    // TilingL1K = TilingL1K > 0 ? TilingL1K : 1;
    uint16_t TilingL1K = 4;

    matmul_custom_m128_n256_k128<<<blockDim, l2ctrl, stream>>>(A, B, C, 
                                                               M, N, K, 
                                                               BaseM, BaseN, BaseK,
                                                               BlockNumM, BlockNumN, BlockNumK,
                                                               TilingL1K,
                                                               TotalResBlocks);
}
#endif
