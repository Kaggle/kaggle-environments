# Just build the orchestrator container
path=$(dirname $0)
./$path/build_cpu.sh
# We don't use the GPU image yet, uncomment when used.
# ./$path/build_gpu.sh