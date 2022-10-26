CURRENT_PATH="/home/ajaillet1/Documents/OT5/micro-aevol2-master/v1/bin"
cd $CURRENT_PATH

cd ../build 
cmake ..
make

cp micro_aevol_cpu ../experiments/micro_aevol_cpu
