CURRENT_PATH="${MICRO_AEVOL_PROJECT_DIRECTORY}/v3/bin"
echo $CURRENT_PATH

cd $CURRENT_PATH

mkdir -p ../build
cd ../build 
cmake .. -DUSE_CUDA=on
make

mkdir -p ../experiments
cp micro_aevol_gpu ../experiments/micro_aevol_gpu
