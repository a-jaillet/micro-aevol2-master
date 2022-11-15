import os
import shutil
import subprocess
import sys

ABSOLUTE_PATH = os.environ.get('MICRO_AEVOL_PROJECT_DIRECTORY') + "/v2/result_comparison"


def prRed(skk): print("\033[91m {}\033[00m" .format(skk))

def prGreen(skk): print("\033[92m {}\033[00m" .format(skk))

def storeNUM_STEPS(default):
    if len(sys.argv) == 2:
        new_num = int(sys.argv[1])
        if new_num > 0:
            NUM_STEPS = new_num
        else:
            NUM_STEPS = default
    else:
        NUM_STEPS = default
    print("Num steps set to ", NUM_STEPS)
    return NUM_STEPS

def main():
    NUM_STEPS = storeNUM_STEPS(1000)

    os.chdir(ABSOLUTE_PATH)

    if os.path.exists("experiment_cpu_v0"):
        shutil.rmtree("experiment_cpu_v0")

    if os.path.exists("experiment_to_challenge"):
        shutil.rmtree("experiment_to_challenge")

    print("All old files are removed")

    os.mkdir("experiment_cpu_v0")
    os.mkdir("experiment_to_challenge")



    args = ("sh", "../bin/compile_cpu.sh")
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    popen.wait()

    args = ("cp", "../build/micro_aevol_cpu", "./experiment_cpu_v0/micro_aevol_cpu")
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    popen.wait()

    print("V0 CPU compiled and copied to experiment_cpu_v0")

    os.chdir("./experiment_cpu_v0")

    args = ("./micro_aevol_cpu", "-n", str(NUM_STEPS))
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    popen.wait()

    print("V0 CPU executed and fitness result written in ./experiment_cpu_v0/result_fitness.csv")

    os.chdir("..")

    args = ("sh", "../bin/compile_gpu.sh")
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    popen.wait()

    args = ("cp", "../build/micro_aevol_gpu", "./experiment_to_challenge/micro_aevol_gpu")
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    popen.wait()

    print("V1 GPU compiled and copied to experiment_to_challenge")

    os.chdir("./experiment_to_challenge")

    args = ("./micro_aevol_gpu", "-n", str(NUM_STEPS))
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    popen.wait()

    print("V1 GPU executed and fitness result written in ./experiment_to_challenge/result_fitness_gpu.csv")

    os.chdir("..")


    # comparing the results
    f = open("./experiment_cpu_v0/result_fitness.csv", "r")
    res1 = f.read()
    f.close()

    f = open("./experiment_to_challenge/result_fitness_gpu.csv", "r")
    res2 = f.read()
    f.close()

    if res1 == res2:
        prGreen("OK: We got the same results")
    else:
        prRed("Error: There might be an error")  

main()