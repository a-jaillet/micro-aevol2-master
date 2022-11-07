CURRENT_PATH="/home/ajaillet/Documents/5IF/OT5/micro-aevol2-master/v1/bin"
cd $CURRENT_PATH

mkdir -p ../build
cd ../build 
cmake .. -DUSE_CUDA=on
make

mkdir -p ../experiments
cp micro_aevol_gpu ../experiments/micro_aevol_gpu
