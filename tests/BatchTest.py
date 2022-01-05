import re
import subprocess
import numpy as np

def getTimeUs(str):
    matchObj = re.match(r'(.*)ms', str)
    if matchObj:
        return 1000 * float(matchObj.group(1))
    matchObj = re.match(r'(.*)us', str)
    if matchObj:
        return float(matchObj.group(1))
    matchObj = re.match(r'(.*)s', str)
    if matchObj:
        return 1000000 * float(matchObj.group(1))
    raise Exception('error occurs when parsing time, can\'t parse ' + str)

def getAvgGPUTime(str):
    lines = str.splitlines()
    isDataLine = False
    for line in lines:
        if line.find("GPU activities:") != -1:
            isDataLine = True 
        if isDataLine and line.find("CUDA memcpy") == -1 and line.find("assign_") == -1:
            item_list = list(filter(None, line.split(" ")))
            if len(item_list) < 4:
                raise Exception('cannot find desired contents in line')
            if line.find("GPU activities:") != -1:
                return item_list[5]
            else:
                return item_list[3]
        if isDataLine and (line.find("API calls:") != -1 or line.find("Error") != -1):
            raise Exception('cannot find desired line')
    print(lines)
    raise Exception('cannot find desired contents, did you actvate dace env?')

def runWithTwoArgs(cmd, arg_list_1, arg_list_2):
    print("\n------- start running \"" + cmd.format("<arg_1>","<arg_2>","<arg_3>") + "\" -------\n")
    for a1 in arg_list_1:
        for a2 in arg_list_2:
            formated_cmd = cmd.format(a1, a2)
            array = []
            for i in range(50):
                cmd_output = subprocess.getoutput(formated_cmd)
                time_string = getAvgGPUTime(cmd_output)
                time_us = getTimeUs(time_string)
                array.append(time_us)
            average = np.mean(array)
            std = np.std(array)
            delta = 1.96 * std / np.sqrt(50)
            print('{0:2d} {1:8d} {2:14f} {3:20f} {4:26f} {5:32f}'.format(a1, a2, average, std, average + delta, average - delta))
    print("\n------- finish running \"" + cmd.format("<arg_1>","<arg_2>","<arg_3>") + "\" -------\n")

def runWithThreeArgs(cmd, arg_list_1, arg_list_2, arg_list_3):
    print("\n------- start running \"" + cmd.format("<arg_1>","<arg_2>","<arg_3>") + "\" -------\n")
    for a1 in arg_list_1:
        for a2 in arg_list_2:
            for a3 in arg_list_3:
                formated_cmd = cmd.format(a1, a2, a3)
                array = []
                for i in range(50):
                    cmd_output = subprocess.getoutput(formated_cmd)
                    time_string = getAvgGPUTime(cmd_output)
                    time_us = getTimeUs(time_string)
                    array.append(time_us)
                average = np.mean(array)
                std = np.std(array)
                delta = 1.96 * std / np.sqrt(50)
                print('{0:2d} {1:8d} {2:14d} {3:20f} {4:26f} {5:32f} {6:38f}'.format(a1, a2, a3, average, std, average + delta, average - delta))
    print("\n------- finish running \"" + cmd.format("<arg_1>","<arg_2>","<arg_3>") + "\" -------\n")

if __name__ == '__main__':
    subprocess.getoutput("conda activate dace38")
    runWithThreeArgs("nvprof python tests/TestReduce2D.py {} {} {}", [9], range(4, 97, 4), [4096])
    # runWithThreeArgs("nvprof python tests/TestReduce2D.py {} {} {}", [6,8], range(1024, 1024*25, 1024), [1024])
    # runWithTwoArgs("nvprof /home/anqili/.conda/envs/dace/bin/python tests/compare.py {} {}", [1,2,3,4], [3214300,6428600,12857200,25714400,51428800,102857600])

 