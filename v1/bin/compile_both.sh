CURRENT_PATH="/home/ajaillet1/Documents/OT5/micro-aevol2-master/v0/bin"
cd $CURRENT_PATH

cd ../build 
cmake ..
make
cp micro_aevol_cpu ../bin/micro_aevol_cpu
cmake .. -DUSE_CUDA=on
make
cp micro_aevol_gpu ../bin/micro_aevol_gpu