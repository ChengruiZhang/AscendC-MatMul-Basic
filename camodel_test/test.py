import ascendebug
import numpy as np

if __name__ == "__main__":
    
    debug_op = ascendebug.create_debug_op('Matmul_Custom', 'CubeCore', 'Ascend910B3')

    x1_gm = np.fromfile(".=./input/x1_gm.bin", dtype=np.float16)
    x2_gm = np.fromfile("../input/x2_gm.bin", dtype=np.float16)

    debug_op.list_tensor_input([("x1_gm", x1_gm, []), ("x1_gm", x1_gm, [])])

    debug_op.list_custom_output(["res", "float32", [128, 128], "./output.bin", []])

    op_executor = ascendebug.create_op_executor(debug_op=debug_op)

    op_executor.run_camodel("../matmul_custom_npu")

    simuops = ascendebug.RunSimuOptions()

