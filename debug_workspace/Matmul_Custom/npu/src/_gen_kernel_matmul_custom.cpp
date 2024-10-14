#ifndef __MATMUL_CUSTOM__KERNEL_FUN_H__
#define __MATMUL_CUSTOM__KERNEL_FUN_H__

#undef __global__
#define __global__ inline
#include "/home/westhpc/RayCode/samples/cplusplus/level1_single_api/4_op_dev/6_ascendc_custom_op/kernel_invocation/MatMul_paral/main.cpp"
#undef __global__
#if __CCE_KT_TEST__
#define __global__
#else
#define __global__ __attribute__((cce_kernel))
#endif

extern "C" __global__ __aicore__ void auto_gen_matmul_custom_kernel(
    __gm__ uint8_t* x1_gm,
    __gm__ uint8_t* res)
{
    matmul_custom(x1_gm, res);
}

#endif
