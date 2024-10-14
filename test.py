import ascendebug
import numpy as np
import shutil
import os 

if __name__ == "__main__":
    
    debug_op = ascendebug.create_debug_op('Matmul_Custom', 'CubeCore', 'Ascend910B3')

    folder_path = "./debug_workspace"
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

    x1_gm = np.fromfile("./input/x1_gm.bin", dtype=np.float16)
    x2_gm = np.fromfile("./input/x2_gm.bin", dtype=np.float16)

    debug_op.list_tensor_input([("x1_gm", x1_gm, []), ("x2_gm", x2_gm, [])])

    debug_op.custom_output("res", "float32", [128, 128], "./output.bin")

    op_executor = ascendebug.create_op_executor(debug_op=debug_op)

    source_file = "/home/westhpc/RayCode/samples/cplusplus/level1_single_api/4_op_dev/6_ascendc_custom_op/kernel_invocation/MatMul_paral/main.cpp"
    kernel_name = "matmul_custom"
    header_files = ["/home/westhpc/RayCode/samples/cplusplus/level1_single_api/4_op_dev/6_ascendc_custom_op/kernel_invocation/MatMul_paral/matmul_custom.cpp",
                    "/usr/local/Ascend/ascend-toolkit/8.0.RC2/aarch64-linux/acl/acl.h"]
    opKernelInfo = ascendebug.OpKernelInfo(source_file, kernel_name, header_files)

    compileOps = ascendebug.CompileNpuOptions()

    op_executor.compile_call_kernel_npu(opKernelInfo, compileOps)

    simuops = ascendebug.RunSimuOptions()

    op_executor.run_camodel("./matmul_custom.o", simuops)

    