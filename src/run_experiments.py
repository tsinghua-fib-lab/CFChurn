import os
import time
import argparse
import numpy as np


def get_spare_gpu(min_mem):
    gpu_status = os.popen('nvidia-smi | grep %').read().split('|')
    gpu_info = ''
    for u in gpu_status:
        gpu_info += u
    gpu_info = gpu_info.strip().split('\n')
    prior_gpus = []
    spare_gpus = []
    spare_gpus_bak = []
    for i, gpus in enumerate(gpu_info):
        gpus = [u for u in gpus.split(' ') if len(u) > 0]
        # gpu_used_memory = int(gpus[6].split('M')[0])
        gpu_spare_memory = int(gpus[8].split('M')[0]) - int(gpus[6].split('M')[0])
        gpu_util = int(gpus[9].split('%')[0])
        if gpu_spare_memory > min_mem and gpu_util == 0:
            prior_gpus.append(i)
        if gpu_spare_memory > min_mem and 0 < gpu_util < 60:
            spare_gpus.append(i)
        if gpu_spare_memory > min_mem and gpu_util > 60:
            spare_gpus_bak.append(i)
    return (prior_gpus, spare_gpus, spare_gpus_bak)


def task_queue(cmd, min_mem, interval=5):
    for command in cmd:
        ret = 1
        retry = 0
        ori_cmd = command
        while ret != 0:
            device = []
            command = ori_cmd
            while len(device) == 0:
                time.sleep(interval)
                device, device_bak1, device_bak2 = get_spare_gpu(min_mem)
                if len(device) == 0:
                    device += device_bak1
                if len(device) == 0:
                    device += device_bak2
                if len(device) == 0:
                    raise NameError('no available cuda device！')

            cuda_idx = np.random.randint(0, len(device))
            cmd_dvs = " -dvs 'cuda:{}'".format(device[cuda_idx])
            command += cmd_dvs
            print(' ----- Executing task on GPU {} ----- '.format(device[cuda_idx]))
            time.sleep(1)  # 留一点人工操作的时间
            ret = os.system(command)
            ret >>= 8
            # 如果任务运行失败，则等10秒，然后重新申请资源，失败3次则返回退出
            time.sleep(10)
            retry += 1
            if retry >= 3:
                print(' -------------- Command failed -------------- ')
                print(command)
                return 0
    return 0


def get_experiments(path):
    cmd = []
    with open(path, 'r') as file:
        for line in file:
            if len(line) > 10:
                cmd.append(line.strip())
    return cmd


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiments')
    parser.add_argument('-p', '--config_path', type=str,
                        default='../exp_config/xxx', help='config path')
    parser.add_argument('-m', '--min_mem', type=int,
                        default=6000, help='min memory needed')
    args = parser.parse_args()

    cmd = get_experiments(args.config_path)
    task_queue(cmd, args.min_mem, interval=3)
    print(' -------------- all experiments down! -------------- ')
