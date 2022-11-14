CURRENT_PATH="${MICRO_AEVOL_SOURCE_DIRECTORY}/v0/bin"
cd $CURRENT_PATH

mkdir -p ../build
cd ../build 
cmake .. -DUSE_CUDA=on
make

mkdir -p ../experiments
cp micro_aevol_gpu ../experiments/micro_aevol_gpu
