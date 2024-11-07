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
#include "matmul_custom.h"
// #include "data_utils.h"
using namespace AscendC;


enum OpType {
    fp32 = 4,
    fp16 = 2,
    int8 = 1,
};

// constexpr int64_t L1_PINGPONG_BUFFER_LEN = 64 * 1024 / sizeof(half);    // 64 KB
// constexpr int64_t L0AB_PINGPONG_BUFFER_LEN = 32 * 1024 / sizeof(half);   // 32 KB
// constexpr int64_t L0C_PINGPONG_BUFFER_LEN = 64 * 1024 / sizeof(float);    // 64 KB


// constexpr int64_t CUBE_M0 = 16;
// constexpr int64_t CUBE_N0 = 16;
// constexpr int64_t CUBE_K0 = 32 / sizeof(half);
// constexpr int64_t CUBE_MATRIX_SIZE = CUBE_K0 * CUBE_N0;

// constexpr int8_t EVENT_ID0 = 0;
// constexpr int8_t EVENT_ID1 = 1;
// constexpr int8_t EVENT_ID2 = 2;
// constexpr int8_t EVENT_ID3 = 3;

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
                                    uint16_t TotalResBlocks_){
        // attr = attr_;
        BaseM = BaseM_; BaseN = BaseN_; BaseK = BaseK_;
        BlockNumM = BlockNumM_; BlockNumN = BlockNumN_; BlockNumK = BlockNumK_;
        TilingL1K = TilingL1K_;
        TotalResBlocks = TotalResBlocks_;

        aL1Size = BaseM * BaseK * 2; // 读取2个basic block，这里是4: (128*256*2/1024=64)*2 double buffer
        bL1Size = BaseN * BaseK * 2;
        aL0Size = BaseM * BaseK;
        bL0Size = BaseK * BaseN; // 128*128*2/1024=32 double buffer
        cSize = BaseM * BaseN * 2;
        
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
    * @brief: 初始化输入A、B矩阵及输出C矩阵的全局Buffer
    */
    __aicore__ inline void Init(GM_ADDR A_, 
                                GM_ADDR B_, 
                                GM_ADDR C_)
    {
        gm_A = (__gm__ half* __restrict__) A_; 
        gm_B = (__gm__ half* __restrict__) B_; 
        gm_C = (__gm__ half* __restrict__) C_;
    }
    template <typename T = half>
    __aicore__ inline void SetTensorAddr(LocalTensor<T>& tensor, uint32_t dataLen, uint32_t bufferAddr, uint8_t logicPos){
        TBuffAddr TBuffAddr_;
        TBuffAddr_.dataLen = dataLen;
        TBuffAddr_.bufferAddr = bufferAddr;
        TBuffAddr_.logicPos = logicPos;
        tensor.SetAddr(TBuffAddr_);
    }
    /* @brief: 在L1， L0AB和L0C上均进行double buffer，假定base块为M0=128，N0=128，K0=128，则
    * 每一个core的每一次循环得出的res大小为M0 * N0，每两次循环完成一次double buffer，需满足M0*N0*2*type小于L0C的大小
    * 在L1上，每次读取L1DATA=(BaseK * 2) * (BaseM + BaseN)大小的数据，需满足L1DATA*2*type小于L1的大小，此处的2表示在L1进行double buffer
    * 在L0AB上，每次读取L0ABDATA=BaseK * (BaseM or BaseN)大小的数据，需满足L0ABDATA*2*type小于L0AB的大小，此处的2表示在L0进行double buffer
    * L0AB上的double buffer是针对L1的，会在L1的其中一个double buffer的数据中轮询读取
    * 此处的BaseMNK都是针对L1
    */
    __aicore__ inline void Process(){

        // LocalTensor<half> L1_tensor_a, L1_tensor_b, L0A_tensor, L0B_tensor;
        // LocalTensor<float> L0C_tensor;

        auto M0 = BaseM; auto N0 = BaseN; auto K0 = BaseK;

        int64_t lda = M; int64_t ldb = K; int64_t ldc = M;

        auto L1_base_a = reinterpret_cast<__cbuf__ __fp16 *>((uintptr_t)0);            // 128 KB 128*256*2/1024=64 double buffer
        auto L1_base_b = reinterpret_cast<__cbuf__ __fp16 *>((uintptr_t)(128 * 1024)); // 128 KB 256*128*2/1024=64 double buffer

        auto L0A_base = reinterpret_cast<__ca__ __fp16 *>((uintptr_t)0); // L0A 128*128*2/1024=32 double buffer
        auto L0B_base = reinterpret_cast<__cb__ __fp16 *>((uintptr_t)0); // L0B 128*128*2/1024=32 double buffer
        auto L0C_base = reinterpret_cast<__cc__ float *>((uintptr_t)0); // L0C 128*128*4/1024=64 double buffer

        int64_t m_loop = (M + BaseM - 1) / BaseM; // 在 M 方向分的核数
        int64_t n_loop = (N + BaseN - 1) / BaseN; // 在 N 方向分的核数
        int64_t k_loop = (K + BaseK - 1) / BaseK; // K 方向循环的次数
        int64_t loop = m_loop * n_loop; // 总需要的核数
        
        int64_t loop_ping_flag = 1;
        int64_t k_loop_ping_flag = 1;

        SetFlagImpl<HardEvent::FIX_M>(EVENT_ID0);
        SetFlagImpl<HardEvent::FIX_M>(EVENT_ID1);

        SetFlagImpl<HardEvent::MTE1_MTE2>(EVENT_ID0);
        SetFlagImpl<HardEvent::MTE1_MTE2>(EVENT_ID1);
        SetFlagImpl<HardEvent::MTE1_MTE2>(EVENT_ID2);
        SetFlagImpl<HardEvent::MTE1_MTE2>(EVENT_ID3);
        
        SetFlagImpl<HardEvent::M_MTE1>(EVENT_ID0);
        SetFlagImpl<HardEvent::M_MTE1>(EVENT_ID1);

        for (int64_t loop_idx = 0; loop_idx < TotalResBlocks; loop_idx++){
            
            // 该循环计算L1上B分块(N0, K0)和A分块(K0, M0)的矩阵乘法并加到L0C上

            // 平均分配给物理核心，不是自己的任务就略过到下一项
    
            if (loop_idx % GetBlockNum() != GetBlockIdx()) {
                continue;
            }
            
            auto L0C_buf = loop_ping_flag ? L0C_base + L0C_PINGPONG_BUFFER_LEN : L0C_base;
            auto LOOP_EVENT_ID = loop_ping_flag ? EVENT_ID0 : EVENT_ID1;

            // 让分块的id按照zN的方式排列 -- 在L1BaseBlock之后再次进行分块
            int64_t batch_idx = 0;
            int64_t m_idx, n_idx;
            
            constexpr int64_t N_COL = 16;
            int64_t in_batch_idx = loop_idx % (m_loop * n_loop);
            int64_t tile_block_loop = (n_loop + N_COL - 1) / N_COL; //行方向z的个数
            int64_t tile_block_idx = in_batch_idx / (N_COL * m_loop); //在行方向第几个z
            int64_t in_tile_block_idx = in_batch_idx % (N_COL * m_loop); //在行方向第几个z中的id
            int64_t n_col = N_COL; //最后一个z的实际宽度
            if(tile_block_idx == tile_block_loop - 1) {
                n_col = n_loop - N_COL * tile_block_idx;
            }
            m_idx = in_tile_block_idx / n_col;
            n_idx = tile_block_idx * N_COL + in_tile_block_idx % n_col;
            
            // 获取GM上的偏移
            // GetOffset(loop_idx); 

            // mmadParams.m = MLen;
            // mmadParams.n = NLen;
            // mmadParams.cmatrixInitVal = true;
            
            int64_t offset_a, offset_b;
            int64_t offset_c = m_idx * M0 + n_idx * N0 * ldc;

            int64_t m_actual = (m_idx == (m_loop - 1)) ? (M - m_idx * M0) : M0;
            int64_t n_actual = (n_idx == (n_loop - 1)) ? (N - n_idx * N0) : N0;
            int64_t m_round = m_actual;
            int64_t n_round = n_actual;

            int64_t mn_max = m_round > n_round ? m_round : n_round;
            int64_t L0AB_K0 = L0AB_PINGPONG_BUFFER_LEN / mn_max / 16 * 16;

            // 每一个核上进行K轴切分
            for (int k_idx = 0; k_idx < k_loop; k_idx++){

                offset_a = m_idx * M0 * K0 + k_idx * K0 * lda;
                offset_b = k_idx * K0 * N0  + n_idx * N0 * ldb;

                int64_t k_actual = (k_idx == (k_loop - 1)) ? (K - k_idx * K0) : K0;
                int64_t k_round = k_actual;
                int64_t L0AB_k_loop = (k_actual + L0AB_K0 - 1) / L0AB_K0;

                auto L1_buf_a = k_loop_ping_flag ? L1_base_a : L1_base_a + L1_PINGPONG_BUFFER_LEN;
                auto L1_buf_b = k_loop_ping_flag ? L1_base_b : L1_base_b + L1_PINGPONG_BUFFER_LEN;
                auto K_LOOP_EVENT_ID = k_loop_ping_flag ? EVENT_ID0 : EVENT_ID1;

                // load A from GM to L1 [MLen, KL1Len]
                // ND2NZ
                WaitFlagImpl(HardEvent::MTE1_MTE2, K_LOOP_EVENT_ID);
                ascblas_matrix_gm2cbuf_ND2nN(L1_buf_a, gm_A + offset_a, M0, K0, m_actual, k_actual, M0);
                SetFlagImpl<HardEvent::MTE2_MTE1>(K_LOOP_EVENT_ID);
                
                // load B from GM to L1 [KL1Len, NLen]
                WaitFlagImpl(HardEvent::MTE1_MTE2, K_LOOP_EVENT_ID + 2);
                ascblas_matrix_gm2cbuf_ND2nZ(L1_buf_b, gm_B + offset_b, K0, N0, k_actual, n_actual, K0);
                // ascblas_matrix_gm2cbuf_ND2nZ(L1_buf_b, gm_B + offset_b, BaseK, BaseN, KL1Len, NLen, BaseK);
                // CopyND2NZ(L1_tensor_b[L1_buf_b], bGM[offset_b], KL1Len, NLen, N, 0, 0);
                SetFlagImpl<HardEvent::MTE2_MTE1>(K_LOOP_EVENT_ID + 2);

                for (int L0AB_k_idx = 0; L0AB_k_idx < L0AB_k_loop; L0AB_k_idx++){

                    int64_t L0AB_k_round = (L0AB_k_idx < L0AB_k_loop - 1) ? L0AB_K0 : k_round - L0AB_k_idx * L0AB_K0;
                    int64_t L0AB_k_actual = (L0AB_k_idx < L0AB_k_loop - 1) ? L0AB_K0 : k_actual - L0AB_k_idx * L0AB_K0;
                    // KL0Blocks = L0AB_k_round / 16; 

                    auto mte1_mad_ping_flag = 1 - L0AB_k_idx % 2;
                    auto mte1_mad_event_id = mte1_mad_ping_flag ? EVENT_ID0 : EVENT_ID1;
                    auto L0A_buf = L0A_base + (L0AB_k_idx % 2) * L0AB_PINGPONG_BUFFER_LEN;
                    auto L0B_buf = L0B_base + (L0AB_k_idx % 2) * L0AB_PINGPONG_BUFFER_LEN;

                    // *** load matrix A from L1 to L0A [MLen, L0AB_k_round]
                    if (L0AB_k_idx == 0) {
                        WaitFlagImpl(HardEvent::MTE2_MTE1, K_LOOP_EVENT_ID);
                    }
                    WaitFlagImpl(HardEvent::M_MTE1, mte1_mad_event_id);
                    // load data
                    auto L1_src_a = L1_buf_a + L0AB_k_idx * L0AB_K0 * M0;
                    for (int i = 0; i < m_round / CUBE_M0; i++) {
                        load_cbuf_to_cb(
                            L0B_buf + i * CUBE_MATRIX_SIZE,
                            L1_src_a + i * CUBE_MATRIX_SIZE,
                            0,
                            L0AB_k_round / (CUBE_K0),
                            M0 / CUBE_M0,
                            m_round / CUBE_M0 - 1,
                            0,
                            true,
                            inc
                        );
                    }
                    if (L0AB_k_idx == L0AB_k_loop - 1) { // L1上的数据已经读取完毕，可以进行下一次 GM -> L1 了
                        SetFlagImpl<HardEvent::MTE1_MTE2>(K_LOOP_EVENT_ID);
                    }

                    // *** load matrix B from L1 to L0B
                    if (L0AB_k_idx == 0) {
                        WaitFlagImpl(HardEvent::MTE2_MTE1, K_LOOP_EVENT_ID + 2);
                    }
                    // load data -- Nz to Zn
                    auto L1_src_b = L1_buf_b + L0AB_k_idx * L0AB_K0 * N0;; // 当CUBE计算结束，才能拷贝到L0AB上
                    for (int i = 0; i < n_round / CUBE_N0; i++) {
                        load_cbuf_to_ca(
                            L0A_buf + i * L0AB_k_round * CUBE_N0,
                            L1_src_b + i * CUBE_MATRIX_SIZE,
                            0,
                            L0AB_k_round / CUBE_K0,
                            N0 / CUBE_N0,
                            0,
                            0,
                            false,
                            inc
                        );
                    }
                    if (L0AB_k_idx == L0AB_k_loop - 1) { // L1上的数据已经读取完毕，可以进行下一次 GM -> L1 了
                        SetFlagImpl<HardEvent::MTE1_MTE2>(K_LOOP_EVENT_ID + 2);
                    }

                    SetFlagImpl<HardEvent::MTE1_M>(mte1_mad_event_id);
                    WaitFlagImpl(HardEvent::MTE1_M, mte1_mad_event_id);

                    bool init_c = (k_idx == 0 && L0AB_k_idx == 0); // 第一次循环直接赋值即可
                    if (init_c) { // 同步L0C的写入和CUBE的计算
                        WaitFlagImpl(HardEvent::FIX_M, LOOP_EVENT_ID);
                    }
                    // compute
                    mad(L0C_buf,
                        L0A_buf,
                        L0B_buf,
                        n_round,
                        L0AB_k_actual,
                        m_round,
                        0,
                        1,
                        0,
                        init_c
                    );
                    pipe_barrier(PIPE_M);
                    // mmadParams.cmatrixInitVal = false;
                    SetFlagImpl<HardEvent::M_MTE1>(mte1_mad_event_id);
                }
                k_loop_ping_flag = 1 - k_loop_ping_flag; // 更换标记做双缓存
            }
            SetFlagImpl<HardEvent::M_FIX>(LOOP_EVENT_ID);
            WaitFlagImpl(HardEvent::M_FIX, LOOP_EVENT_ID);

            // load to GM from L0C -- not done
            copy_matrix_cc_to_gm(
                gm_C + offset_c,
                L0C_buf,
                0,
                m_actual,
                n_actual,  
                ldc,   
                n_round,
                0,
                F322F16,
                0,
                false,
                true
            );

            loop_ping_flag = 1 - loop_ping_flag; // 更换标记做双缓存

            SetFlagImpl<HardEvent::FIX_M>(LOOP_EVENT_ID);
        }

        WaitFlagImpl(HardEvent::M_MTE1, EVENT_ID0);
        WaitFlagImpl(HardEvent::M_MTE1, EVENT_ID1);

        WaitFlagImpl(HardEvent::MTE1_MTE2, EVENT_ID0);
        WaitFlagImpl(HardEvent::MTE1_MTE2, EVENT_ID1);
        WaitFlagImpl(HardEvent::MTE1_MTE2, EVENT_ID2);
        WaitFlagImpl(HardEvent::MTE1_MTE2, EVENT_ID3);

        WaitFlagImpl(HardEvent::FIX_M, EVENT_ID0);
        WaitFlagImpl(HardEvent::FIX_M, EVENT_ID1);
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

private:

    // GlobalTensor<half> aGM, bGM;
    // GlobalTensor<half> cGM;

    __gm__ __fp16 * __restrict__ gm_A = nullptr;
    __gm__ __fp16 * __restrict__ gm_B = nullptr;
    __gm__ __fp16 * __restrict__ gm_C = nullptr;

    int64_t M, N, K;
    int64_t BaseM, BaseN, BaseK;
    int64_t coreM, coreN;
    int64_t BlockNumM, BlockNumN, BlockNumK;
    int64_t TilingL1K;
    int64_t MLen, NLen, KL1Len, KL0Len;
    int64_t MOffset, NOffset, ResOffset;
    // Attr attr;

    int64_t aL1Size, bL1Size, aL0Size, bL0Size, cSize;
    int64_t mBlocks, nBlocks, kBlocks, KL1Blocks, KL0Blocks;
    int64_t TotalResBlocks;
    int64_t coreidx;

    // uint32_t KDouBuf1, KDouBuf2;
    // uint16_t KDouBuf1Blocks, KDouBuf2BLocks;
    uint32_t KL1DouBufBlocks[2], KL1DouBufBase[2];
    uint16_t KL1DouBuf[2];
};

// 直接传结构体会读不出来，可能需要片上构建
extern "C" __global__ __aicore__ void matmul_custom_m128_n256_k128(GM_ADDR A, 
                                                                   GM_ADDR B, 
                                                                   GM_ADDR C, 
                                                                   uint16_t M, uint16_t N, uint16_t K, 
                                                                   uint16_t BaseM, uint16_t BaseN, uint16_t BaseK,
                                                                   uint16_t BlockNumM, uint16_t BlockNumN, uint16_t BlockNumK,
                                                                   uint16_t TilingL1K,
                                                                   uint16_t TotalResBlocks)
{
    KernelMatmul op(M, N, K);
    // TPipe pipe;
    op.InitAttr(BaseM, BaseN, BaseK,
                BlockNumM, BlockNumN, BlockNumK,
                TilingL1K,
                TotalResBlocks);
    op.Init(A, B, C);
    op.Process();
}

#ifndef __CCE_KT_TEST__40
// call of kernel function
void matmul_custom_do(uint32_t blockDim, void* l2ctrl, void* stream, 
                    uint8_t* A, 
                    uint8_t* B, 
                    uint8_t* C,
                     uint16_t M, uint16_t N, uint16_t K)
{

    // best: mem-comp 
    // 180, 180, K -- 90
    // 128, 256, K -- 85.3
    // 192, 160, K -- 87.3
    uint16_t BaseM = 128;
    uint16_t BaseN = 128;
    uint16_t BaseK = 256;

    uint32_t L1BufferSize = 1 * 512 * 1024;
    uint32_t L0ABBufferSize = 64 * 1024;
    uint32_t L0CBufferSize = 64 * 1024;
    
    uint16_t BlockNumM = (M + BaseM - 1) / BaseM;
    uint16_t BlockNumN = (N + BaseN - 1) / BaseN;
    uint16_t BlockNumK = (K + BaseK - 1) / BaseK;

    uint16_t TotalResBlocks = BlockNumM * BlockNumN;
    if (blockDim > TotalResBlocks) { blockDim = uint32_t(TotalResBlocks); }

    OpType optype = OpType::fp16; 

    uint16_t TilingL1K = 1;

    matmul_custom_m128_n256_k128<<<blockDim, l2ctrl, stream>>>(A, B, C, 
                                                               M, N, K, 
                                                               BaseM, BaseN, BaseK,
                                                               BlockNumM, BlockNumN, BlockNumK,
                                                               TilingL1K,
                                                               TotalResBlocks);
}
#endif
