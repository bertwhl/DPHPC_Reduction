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
    raise Exception('cannot find desired contents')

def getAvgGPUTime(str):
    lines = str.splitlines()
    isDataLine = False
    for line in lines:
        if line.find("GPU activities:") != -1:
            isDataLine = True 
        if isDataLine and line.find("CUDA memcpy") == -1 and line.find("assign_") == -1:
            item_list = list(filter(None, line.split(" ")))
            if len(item_list) < 4:
                raise Exception('cannot find desired contents')
            return item_list[3]
        if line.find("API calls:") != -1 or line.find("Error") != -1:
            raise Exception('cannot find desired contents')
    raise Exception('cannot find desired contents')

def runWithTwoArgs(cmd, arg_list_1, arg_list_2):
    for a1 in arg_list_1:
        for a2 in arg_list_2:
                formated_cmd = cmd.format(a1, a2)
                cmd_output = subprocess.getoutput(formated_cmd)
                time_string = getAvgGPUTime(cmd_output)
                time_us = getTimeUs(time_string)
                print('{0:2d} {1:4d} {2:8f}'.format(a1, a2, time_us))

def runWithThreeArgs(cmd, arg_list_1, arg_list_2, arg_list_3):
    for a1 in arg_list_1:
        for a2 in arg_list_2:
            for a3 in arg_list_3:
                formated_cmd = cmd.format(a1, a2, a3)
                cmd_output = subprocess.getoutput(formated_cmd)
                time_string = getAvgGPUTime(cmd_output)
                time_us = getTimeUs(time_string)
                print('{0:2d} {1:4d} {2:6d} {3:10f}'.format(a1, a2, a3, time_us))

if __name__ == '__main__':
    subprocess.getoutput("conda activate dace")
    runWithThreeArgs("nvprof python library/mwpr.py {} {} {}", [1], [128, 256], [512, 1024])

 