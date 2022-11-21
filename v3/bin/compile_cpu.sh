CURRENT_PATH="${MICRO_AEVOL_PROJECT_DIRECTORY}/v3/bin"
cd $CURRENT_PATH

mkdir -p ../build
cd ../build 
cmake ..
make

mkdir -p ../experiments
cp micro_aevol_cpu ../experiments/micro_aevol_cpu
