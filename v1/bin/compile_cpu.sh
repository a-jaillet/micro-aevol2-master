CURRENT_PATH="/home/ajaillet/Documents/5IF/OT5/micro-aevol2-master/v1/bin"
cd $CURRENT_PATH

mkdir -p ../build
cd ../build 
cmake ..
make

mkdir -p ../experiments
cp micro_aevol_cpu ../experiments/micro_aevol_cpu
