CURRENT_PATH="/home/ajaillet1/Documents/OT5/micro-aevol2-master/v1/bin"
cd $CURRENT_PATH

mkdir -p ../build
cd ../build 
cmake .. -DUSE_CUDA=on
make
cp micro_aevol_gpu ../experiments/micro_aevol_gpu
