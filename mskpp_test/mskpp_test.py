from mskpp import mmad, Tensor, Chip
def my_mmad(gm_x, gm_y, gm_z):
    # 矩阵乘的基本数据通路：
    # 左矩阵A：GM-L1-L0A
    # 右矩阵B：GM-L1-L0B
    # 结果矩阵C： L0C(初始化)-GM
    l1_x = Tensor("L1")
    l1_y = Tensor("L1")
    l1_x.load(gm_x)
    l1_y.load(gm_y)
    x = Tensor("L0A")
    y = Tensor("L0B")
    x.load(l1_x)
    y.load(l1_y)
    z = Tensor("L0C", "FP32", [256, 256], format="ND")
    out = mmad(x, y, z, True)() # 对于输出需要返回传出
    z = out[0]
    return z

if __name__ == '__main__':
    with Chip("Ascend910B3") as chip:
        # chip.set_prof_summary_path("\
        #     /home/HwHiAiUser/project/add_custom/OPPROF_{timestamp}_XXX/PipeUtilization.csv")
        chip.enable_trace()    # 使能算子模拟流水图的功能，生成trace.json文件
        chip.enable_metrics()   # 使能单指令及分PIPE的流水信息，生成Instruction_statistic.csv和Pipe_statistic.csv文件
        # 这里进入了对数据切分逻辑的处理，对一大块GM的数据，如何经过拆分成小数据分批次搬入，如何对
        # 内存进行分片多buffer搬运，都是属于tiling策略的范畴，这里模拟了单buffer情况，
        # 将[160, 240]和[240, 80]的矩阵乘，切割为5个[32, 48], [48, 16]的小矩阵分批次进行运算的一个tiling策略
        for _ in range(16 * 16 * 16):
            in_x = Tensor("GM", "FP16", [256, 128], format="ND")
            in_y = Tensor("GM", "FP16", [128, 256], format="ND")
            in_z = Tensor("GM", "FP32", [256, 256], format="ND")
            out_z = my_mmad(in_x, in_y, in_z)
            in_z.load(out_z)