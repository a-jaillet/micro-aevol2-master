import os
import shutil
import subprocess

CURRENT_PATH = "/home/ajaillet/Documents/5IF/OT5/micro-aevol2-master/result_comparison"

def main():
    os.chdir(CURRENT_PATH)

    if os.path.exists("experiment_cpu_v0"):
        shutil.rmtree("experiment_cpu_v0")

    # os.remove("experiment_to_challenge/*")

    print("All old files are removed")

    os.mkdir("experiment_cpu_v0")


    args = ("sh", "../v0/bin/compile_cpu.sh")
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    popen.wait()

    args = ("cp", "../v0/build/micro_aevol_cpu", "./experiment_cpu_v0/micro_aevol_cpu")
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    popen.wait()

    print("V0 CPU compiled and copied to experiment_cpu_v0")

    os.chdir("./experiment_cpu_v0")

    args = ("./micro_aevol_cpu")
    popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    popen.wait()

    # args = ("sh", "../v1/bin/compile_gpu.sh")
    # popen = subprocess.Popen(args, stdout=subprocess.PIPE)
    # popen.wait()

    # print("V1 GPU compiled")








main()