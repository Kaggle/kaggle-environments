# Just build the orchestrator container
path=$(dirname $0)
./$path/build_cpu.sh
./$path/build_gpu.sh