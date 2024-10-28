M=128
N=256
K=128

repeat=5
batch=20
core_num=20

# core_num=20
device_id=0

bash run.sh matmul_custom Ascend910B1 AiCore npu $M $N $K $repeat $batch $core_num

python matmul_custom.py $M $N $K
./matmul_custom_npu $M $N $K $repeat $batch $core_num $device_id

python verification.py