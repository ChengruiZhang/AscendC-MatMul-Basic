/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 * This file constains code of cpu debug and npu code.We read data from bin file
 * and write result to file.
 */
#include "data_utils.h"
#ifndef __CCE_KT_TEST__
#include "acl/acl.h"
extern void GM2L1_do(uint32_t coreDim, void* l2ctrl, void* stream,
    uint8_t *param1, uint8_t *param2, uint8_t *param3, uint16_t M, uint16_t N, uint16_t K);
extern void L12GM_do(uint32_t coreDim, void* l2ctrl, void* stream,
    uint8_t *param1, uint8_t *param2, uint8_t *param3, uint16_t M, uint16_t N, uint16_t K);
extern void GM2L0AB_do(uint32_t blockDim, void* l2ctrl, void* stream, 
    uint8_t* a, uint8_t* b, uint8_t* c, uint16_t M, uint16_t N, uint16_t K)
extern void L12L0AB_do(uint32_t blockDim, void* l2ctrl, void* stream, 
    uint8_t* a, uint8_t* b, uint8_t* c, uint16_t M, uint16_t N, uint16_t K)

#else
#include "tikicpulib.h"
// extern "C" void GM2L1(uint8_t *param1, uint8_t *param2, uint8_t *param3);
#endif

#include <chrono>
#include <algorithm>
#include <numeric>

// #include "test.h"

void execute_kernel(uint32_t blockDim, void* ptrs, aclrtStream stream, uint8_t* param1Device, 
                    uint8_t* param2Device, uint8_t* param3Device, uint16_t M, uint16_t N, uint16_t K, std::string type){
    // char expression;
    std::unordered_map<std::string, int> str2int_map = {
        {"GM-R", 0},
        {"GM-W", 1},
        {"L1-R", 2},
        {"L1-W", 3},
        {"L0AB-R", 4},
        {"L0AB-W-GM", 5},
        {"L0AB-W-L1", 6},
        {"L0C-R", 7},
        {"L0C-W", 8},
    };
    
    switch (str2int_map[type])
    {
    case 0: // GM-R
        GM2L1_do(blockDim, nullptr, stream, param1Device, param2Device, param3Device, M, N, K);
        break;
    case 1: // GM-W
        L12GM_do(blockDim, nullptr, stream, param1Device, param2Device, param3Device, M, N, K);
        break;
    case 2: // l1-R
        L12GM_do(blockDim, nullptr, stream, param1Device, param2Device, param3Device, M, N, K);
        break;
    case 3: // L1-W
        GM2L1_do(blockDim, nullptr, stream, param1Device, param2Device, param3Device, M, N, K);
        break;
    case 4: // L0AB-R
        /* code */
        break;
    case 5: // L0AB-W-GM
        GM2L0AB_do(blockDim, nullptr, stream, param1Device, param2Device, param3Device, M, N, K);
        break;
    case 6: // L0AB-W-L1
        L12L0AB_do(blockDim, nullptr, stream, param1Device, param2Device, param3Device, M, N, K);
        break;
    case 7: // L0C-R
        /* code */
        break;
    case 8: // L0C-W

        break;
    default:
        break;
    }
}

int32_t main(int32_t argc, char* argv[])
{
    uint16_t M, N, K;
    
    // M = 128;
    // N = 128;
    // K = 128;
    // int repeat = 5;
    // int batch = 20;
    
    // uint32_t blockDim = 1;

    M = std::stoi(argv[1]);
    N = std::stoi(argv[2]);
    K = std::stoi(argv[3]);
    int repeat = std::stoi(argv[4]);
    int batch = std::stoi(argv[5]);
    
    uint32_t blockDim = 1;
    if(argc > 6){
        blockDim = std::stoi(argv[6]);
    }

    std::string type = argv[7];

    size_t param1FileSize = M * K * sizeof(uint16_t);  // uint16_t represent half
    size_t param2FileSize = K * N * sizeof(uint16_t);  // uint16_t represent half
    size_t param3FileSize = M * N * sizeof(float);

#ifdef __CCE_KT_TEST__
    uint8_t *param1 = (uint8_t *)AscendC::GmAlloc(param1FileSize);
    uint8_t *param2 = (uint8_t *)AscendC::GmAlloc(param2FileSize);
    uint8_t *param3 = (uint8_t *)AscendC::GmAlloc(param3FileSize);

    ReadFile("./input/x1_gm.bin", param1FileSize, param1, param1FileSize);
    ReadFile("./input/x2_gm.bin", param2FileSize, param2, param2FileSize);

    ICPU_RUN_KF(matmul_custom, blockDim, param1, param2, param3);

    WriteFile("./output/output.bin", param3, param3FileSize);

    AscendC::GmFree((void *)param1);
    AscendC::GmFree((void *)param2);
    AscendC::GmFree((void *)param3);
#else
    CHECK_ACL(aclInit(nullptr));
    aclrtContext context;
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    // CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclrtCreateContext(&context, deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *param1Host;
    uint8_t *param1Device;
    CHECK_ACL(aclrtMallocHost((void**)(&param1Host), param1FileSize));
    CHECK_ACL(aclrtMalloc((void**)&param1Device, param1FileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ReadFile("./input/x1_gm.bin", param1FileSize, param1Host, param1FileSize);
    CHECK_ACL(aclrtMemcpy(param1Device, param1FileSize, param1Host, param1FileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *param2Host;
    uint8_t *param2Device;
    CHECK_ACL(aclrtMallocHost((void**)(&param2Host), param2FileSize));
    CHECK_ACL(aclrtMalloc((void**)&param2Device, param2FileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ReadFile("./input/x2_gm.bin", param2FileSize, param2Host, param2FileSize);
    CHECK_ACL(aclrtMemcpy(param2Device, param2FileSize, param2Host, param2FileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *param3Host;
    uint8_t *param3Device;
    CHECK_ACL(aclrtMallocHost((void**)(&param3Host), param3FileSize));
    CHECK_ACL(aclrtMalloc((void**)&param3Device, param3FileSize, ACL_MEM_MALLOC_HUGE_FIRST));

    std::vector<double> times;
    for(int i = 0; i < repeat; i++){
        auto start = std::chrono::high_resolution_clock::now();
        for(int j = 0; j < batch; j++){
            execute_kernel(blockDim, nullptr, stream, param1Device, param2Device, param3Device, M, N, K, type);
        }
        CHECK_ACL(aclrtSynchronizeStream(stream));
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::micro> elapsed = end - start;
        times.push_back(elapsed.count() / batch);
    }
    std::sort(times.begin(), times.end());
    if(repeat > 2){
        times.erase(times.begin());
        times.erase(times.end() - 1);
    }
    double average_time = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    std::cout << "operator time: " << average_time << " us" << std::endl;

    CHECK_ACL(aclrtMemcpy(param3Host, param3FileSize, param3Device, param3FileSize, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./output/output.bin", param3Host, param3FileSize);
    CHECK_ACL(aclrtFree(param3Device));
    CHECK_ACL(aclrtFreeHost(param3Host));

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtDestroyContext(context));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
#endif
    return 0;
}