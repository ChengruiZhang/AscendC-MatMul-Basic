import os
import sys
import numpy as np

loss = 1e-3
minimum = 10e-10

def verify_result(real_result, golden):
    real_result = np.fromfile(real_result, dtype=np.float16)
    golden = np.fromfile(golden, dtype=np.float16)
    print(real_result)
    print(golden)
    result = np.abs(real_result - golden)
    deno = np.maximum(np.abs(real_result), np.abs(golden))
    result_atol = np.less_equal(result, loss)
    result_rtol = np.less_equal(result / np.add(deno, minimum), loss)
    if not result_rtol.all() and not result_atol.all():
        if np.sum(result_rtol == False) > real_result.size * loss and np.sum(result_atol == False) > real_result.size * loss:
            print("[ERROR] result error")
            return False
    print("test pass")
    return True

def verify():
    x1_gm = np.fromfile("./input/x1_gm.bin", dtype=np.float16)
    # print(x1_gm)
    # x1_gm_test = np.fromfile("./input/x1_gm_test.bin", dtype=np.float32)
    # print(x1_gm_test)
    x2_gm = np.fromfile("./input/x2_gm.bin", dtype=np.float16)
    # print(x2_gm)
    # x2_gm_test = np.fromfile("./input/x2_gm_test.bin", dtype=np.float32)
    # print(x2_gm_test)

    # print((x1_gm_test-x1_gm).sum())
    # print((x2_gm_test-x2_gm).sum())

def verify_gpt():
    x1_gm = np.fromfile("./input/matrix1.bin", dtype=np.float16).reshape([4096, 4096])
    x1_gm_32 = x1_gm.astype(np.float32)
    print(x1_gm)
    x2_gm = np.fromfile("./input/matrix2.bin", dtype=np.float16).reshape([4096, 4096])
    x2_gm_32 = x2_gm.astype(np.float32)
    print(x2_gm)
    res = np.fromfile("./input/product_matrix.bin", dtype=np.float32).reshape([4096, 4096])

    golden_res = np.matmul(x1_gm_32, x2_gm_32)    

    print(np.allclose(res, golden_res))

if __name__ == '__main__':
    # verify_result(sys.argv[1],sys.argv[2])
    
    # verify_gpt()
    verify()
    # verify_result("./output/output.bin", "./output/product_matrix.bin")
    verify_result("./output/output.bin", "./output/golden.bin")
