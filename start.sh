M=1024
N=1024
K=1024

repeat=5
batch=20
core_num=20

input_file1="./input/x1_gm.bin"
input_file2="./input/x2_gm.bin"
output_file="./output/output.bin"

# core_num=20
device_id=5

bash run.sh matmul_custom Ascend910B1 AiCore npu $M $N $K $repeat $batch $core_num

python matmul_custom.py $M $N $K

./matmul_custom_npu $M $N $K $repeat $batch $core_num $device_id $input_file1 $input_file2 $output_file

python verification.py