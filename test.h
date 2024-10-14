// #pragma once

// void RL1_WGM_do(uint32_t blockDim, void* l2ctrl, void* stream, uint8_t* a, uint8_t* b, uint8_t* c, uint16_t M, uint16_t N, uint16_t K);

// void RGM_WL1_do(uint32_t blockDim, void* l2ctrl, void* stream, uint8_t* a, uint8_t* b, uint8_t* c, uint16_t M, uint16_t N, uint16_t K);

// // using namespace AscendC;
// // constexpr int32_t SINGLE_CORE_OFFSET = 512;

// // class KernelMatmul {
// // public:
// //     __aicore__ inline KernelMatmul(uint16_t M, uint16_t N, uint16_t K)
// //     {
// //         m = M; n = N; k = K;
// //         aSize = m * k;
// //         bSize = k * n;
// //         cSize = m * m;
// //         mBlocks = m / 16;
// //         nBlocks = n / 16;
// //         kBlocks = k / 16;
// //     }
// //     __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c)
// //     {
// //         auto coreidx = GetBlockIdx();
// //         aGM.SetGlobalBuffer((__gm__ half*)a);
// //         bGM.SetGlobalBuffer((__gm__ half*)b);
// //         cGM.SetGlobalBuffer((__gm__ float*)c);
// //         pipe.InitBuffer(inQueueA1, 1, 1 * 512 * 1024);
// //         pipe.InitBuffer(inQueueA2, 1, 1 * 64 * 1024);
// //         pipe.InitBuffer(inQueueB2, 1, 1 * 64 * 1024);
// //         pipe.InitBuffer(outQueueCO1, 1, 1 * 128 * 1024);
// //     }

// //     __aicore__ inline void Process_RGM_WL1();
// //     __aicore__ inline void Process_RL1_WGM();

// //     private:
// //     TPipe pipe;

// //     TQue<QuePosition::A1, 1> inQueueA1;
// //     TQue<QuePosition::A2, 1> inQueueA2;
// //     TQue<QuePosition::B1, 1> inQueueB1;
// //     TQue<QuePosition::B2, 1> inQueueB2;
// //     // dst queue
// //     TQue<QuePosition::CO1, 1> outQueueCO1;
// //     // TQue<QuePosition::CO2, 1> outQueueCO2;

// //     GlobalTensor<half> aGM, bGM;
// //     GlobalTensor<float> cGM;

// //     uint16_t m, n, k;

// //     uint16_t aSize, bSize, cSize, mBlocks, nBlocks, kBlocks;
// // };