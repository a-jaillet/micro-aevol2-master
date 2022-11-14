CURRENT_PATH="${MICRO_AEVOL_SOURCE_DIRECTORY}/v0/bin"
cd $CURRENT_PATH

mkdir -p ../build
cd ../build 
cmake ..
make

mkdir -p ../experiments

cp micro_aevol_cpu ../experiments/micro_aevol_cpu

cmake .. -DUSE_CUDA=on
make
cp micro_aevol_gpu ../experiments/micro_aevol_gpu
