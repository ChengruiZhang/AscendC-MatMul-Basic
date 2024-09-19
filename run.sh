#!/bin/bash
clear;clear
rm -rf *_cpu *_sim *_npu build cceprint npuchk output/*.bin input/*.bin > /dev/null
export PRINT_TIK_MEM_ACCESS=FALSE

CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})
    pwd
); cd $CURRENT_DIR

declare -A VersionMap
VersionMap["Ascend910A"]="Ascend910A"
VersionMap["Ascend910B"]="Ascend910A"
VersionMap["Ascend910ProA"]="Ascend910A"
VersionMap["Ascend910ProB"]="Ascend910A"
VersionMap["Ascend910PremiumA"]="Ascend910A"
VersionMap["Ascend310B1"]="Ascend310B1"
VersionMap["Ascend310B2"]="Ascend310B1"
VersionMap["Ascend310B3"]="Ascend310B1"
VersionMap["Ascend310B4"]="Ascend310B1"
VersionMap["Ascend310P1"]="Ascend310P1"
VersionMap["Ascend310P3"]="Ascend310P1"
VersionMap["Ascend910B1"]="Ascend910B1"
VersionMap["Ascend910B2"]="Ascend910B1"
VersionMap["Ascend910B3"]="Ascend910B1"
VersionMap["Ascend910B4"]="Ascend910B1"
# legacy
VersionMap["ascend910"]="Ascend910A"
VersionMap["ascend310p"]="Ascend310P1"
VersionMap["ascend310B1"]="Ascend310B1"
VersionMap["ascend910B1"]="Ascend910B1"

if [ ! $ASCEND_HOME_DIR ]; then
    if [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
        export ASCEND_HOME_DIR=/usr/local/Ascend/ascend-toolkit/latest
    else
        export ASCEND_HOME_DIR=/usr/local/Ascend/ascend-toolkit/latest
    fi
fi

# 指定当前sample的算子文件名
FILE_NAME=matmul_custom

# 指定芯片版本: ascend910, ascend310p
SOC_VERSION=ascend910B1
if [ ${SOC_VERSION}"x" = "x" ]; then
    echo "ERROR: SOC_VERSION is not specified! please specify ascend910, ascend310p!"
    exit -1
fi

# 指定运行的核: AiCore, VectorCore
CORE_TYPE=AiCore
if [ ${CORE_TYPE}"x" = "x" ]; then
    echo "WARNING: CORE_TYPE is not specified, using AiCore as default."
    CORE_TYPE=AiCore
fi

RUN_MODE_LIST="cpu sim npu"
# 指定运行模式: cpu, sim, npu
RUN_MODE=$1

if [[ " $RUN_MODE_LIST " != *" $RUN_MODE "* ]]; then
    echo "ERROR: RUN_MODE error, This sample only support specify cpu, sim or npu!"
    exit -1
fi

if [ "$RUN_MODE" = "sim" ]; then
    export LD_LIBRARY_PATH=${ASCEND_HOME_DIR}/runtime/lib64/stub:${LD_LIBRARY_PATH}
    if [ ! $CAMODEL_LOG_PATH ]; then
        export CAMODEL_LOG_PATH=./sim_log
    fi
    rm -rf $CAMODEL_LOG_PATH
    mkdir -p $CAMODEL_LOG_PATH
fi



CUSTOM_TILING=NO_CUSTOM_TILING
# if [ ${CUSTOM_TILING}"x" = "x" ]; then
#     echo "WARNING: CUSTOM_TILING is not specified, using NO_CUSTOM_TILING as default."
#     CUSTOM_TILING=NO_CUSTOM_TILING
# fi

source $ASCEND_HOME_DIR/bin/setenv.bash

# 生成计算输入数据和对比用的真值数据
python3 $FILE_NAME.py

function compile_and_execute() {
    # 使用cmake编译cpu侧或者npu侧算子, SIMULATOR or ONBOARD
    mkdir -p build; cd build;                                \
    cmake ..                                                 \
        -Dsmoke_testcase=$1                                  \
        -DASCEND_PRODUCT_TYPE=${VersionMap[$SOC_VERSION]}    \
        -DASCEND_CORE_TYPE=$3                                \
        -DASCEND_RUN_MODE=${RUN_MODE}                        \
        -DASCEND_INSTALL_PATH=$ASCEND_HOME_DIR
    cmake --build . --target ${1}_${4}
    # cmake --build . --target ascendc_kernels && cmake --install . && cmake --build . --target ${1}_lib_${4}
    if [ ${CUSTOM_TILING} = "CUSTOM_TILING" ]; then
        cmake --build . --target ${1}_tiling
    fi
    cd -

    if [ ${CUSTOM_TILING} = "CUSTOM_TILING" ]; then
        ./${1}_tiling
    fi


    if [ $? -ne 0 ]; then
        echo "ERROR: compile op on failed!"
        return 1
    fi
    echo "INFO: compile op on ${RUN_MODE} succeed!"
    # 执行生成的可执行文件
    (export LD_LIBRARY_PATH=`pwd`:$ASCEND_HOME_DIR/tools/simulator/${VersionMap[$SOC_VERSION]}/lib:$LD_LIBRARY_PATH && ./${1}_${4})
    if [ $? -ne 0 ]; then
        echo "ERROR: execute op on ${RUN_MODE} failed!"
        return 1
    fi
    echo "INFO: execute op on ${RUN_MODE} succeed!"
}
compile_and_execute $FILE_NAME $SOC_VERSION $CORE_TYPE $RUN_MODE

# 验证计算结果
echo "md5sum: ";md5sum output/*.bin
  
