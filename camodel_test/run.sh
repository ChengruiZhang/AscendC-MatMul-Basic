op_config_json_file=./matmul_custom_test.json
repo_type=minimalist
chip_version=Ascend910B3
core_type=AiCore
work_dir=./res

ascendebug kernel --backend simulator --json-file ${op_config_json_file} --repo-type ${repo_type} \
                  --chip-version ${chip_version} --core-type ${core_type} \
                  --work-dir ${work_dir} --block-num 1 --timeout 1200