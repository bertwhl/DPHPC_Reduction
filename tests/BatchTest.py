import re
import subprocess

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
    raise Exception('cannot find desired contents, did you actvate dace env?')

def runWithTwoArgs(cmd, arg_list_1, arg_list_2):
    for a1 in arg_list_1:
        for a2 in arg_list_2:
                formated_cmd = cmd.format(a1, a2)
                cmd_output = subprocess.getoutput(formated_cmd)
                time_string = getAvgGPUTime(cmd_output)
                time_us = getTimeUs(time_string)
                print('{0:2d} {1:4d} {2:8f}'.format(a1, a2, time_us))

def runWithThreeArgs(cmd, arg_list_1, arg_list_2, arg_list_3):
    print("\n------- start running \"" + cmd.format("<arg_1>","<arg_2>","<arg_3>") + "\" -------\n")
    for a1 in arg_list_1:
        for a2 in arg_list_2:
            for a3 in arg_list_3:
                formated_cmd = cmd.format(a1, a2, a3)
                cmd_output = subprocess.getoutput(formated_cmd)
                time_string = getAvgGPUTime(cmd_output)
                time_us = getTimeUs(time_string)
                print('{0:2d} {1:8d} {2:14d} {3:20f}'.format(a1, a2, a3, time_us))
    print("\n------- finish running \"" + cmd.format("<arg_1>","<arg_2>","<arg_3>") + "\" -------\n")

if __name__ == '__main__':
    subprocess.getoutput("conda activate dace")
    # runWithThreeArgs("nvprof python tests/TestReduce2D.py {} {} {}", [1,3,4], range(4, 81, 4), [4096])
    runWithThreeArgs("nvprof python tests/TestReduce2D.py {} {} {}", [6,7], range(1024, 4096, 512), [12])
    # runWithThreeArgs("nvprof python tests/TestReduce2D.py {} {} {}", [6,7], [10240], range(1,20))

 