
import os
import time
import numpy as np

from termcolor import colored
from . import util

from datetime import datetime


def parse_line(line):
    info_d = {}

    l1, l2 = line.split(') ')
    l1 = l1.replace('(', '')
    l1 = l1.split(', ')

    l2 = l2.replace('(', '')
    l2 = l2.split(' ')

    info_d = {}
    for s in l1:
        k, v = s.split(': ')
        if k in ['epoch', 'iters']:
            info_d[k] = int(v)
        else:
            info_d[k] = float(v)

    l2_keys = l2[0::2]
    l2_vals = l2[1::2]
    
    for k, v in zip(l2_keys, l2_vals):
        k = k.replace(':','')
        info_d[k] = float(v)

    return info_d


class Visualizer():
    def __init__(self, isTrain=True, name=None, tag_name=None):
        self.isTrain = isTrain
        self.gif_fps = 4

        if tag_name is not None:
            logs_dir = os.path.join('../logs_'+tag_name)
            results_dir = os.path.join('../results_'+tag_name)
        else:
            logs_dir = '../logs_home'
            results_dir = '../results_home'
        if name is None:
            name = 'txt2shape_' + datetime.now().strftime('%Y-%m-%dT%H-%M')
        else:
            name = name

        if self.isTrain:
            self.log_dir = os.path.join(logs_dir, name)
        else:
            self.log_dir = os.path.join(results_dir, name)

        self.img_dir = os.path.join(self.log_dir, 'images')
        self.name = name

    def setup_io(self):
        print('[*] create image directory:\n%s...' % os.path.abspath(self.img_dir) )
        util.mkdirs([self.img_dir])

        if self.isTrain:
            self.log_name = os.path.join(self.log_dir, 'loss_log.txt')
            with open(self.log_name, "w") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        self.saved = False

    def print_current_errors(self, current_iters, errors, t):
        self.gpu_ids_str = 0
        message = f"[{self.name}] (GPU: {self.gpu_ids_str}, iters: {current_iters}, time: {t:.3f}) "
        for k, v in errors.items():
            message += '%s: %.6f ' % (k, v)

        print(colored(message, 'magenta'))
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def print_current_metrics(self, current_iters, metrics, phase):
        self.gpu_ids_str = 0
        message = f'([{self.name}] [{phase}] GPU: {self.gpu_ids_str}, steps: {current_iters}) '
        for k, v in metrics.items():
            message += '%s: %.3f ' % (k, v)

        print(colored(message, 'yellow'))
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def print_current_errors(self, errors):
        message = errors
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def display_current_results(self, visuals, epoch, current_iters, phase='train'):
        # write images to disk
        vis_all = None
        img_path = os.path.join(self.img_dir, f'{phase}epoch_{epoch:05d}_step{current_iters:05d}.png')
        for label, image_numpy in visuals.items():
            if image_numpy.shape[2] == 3:
                alpha_channel = np.ones((image_numpy.shape[0], image_numpy.shape[1], 1), dtype=np.uint8) * 127
                image_numpy = np.concatenate([image_numpy, alpha_channel], axis=2)

            if vis_all is None:
                vis_all = image_numpy
            else:
                vis_all = np.concatenate([vis_all, image_numpy], axis=1)

        util.save_image(vis_all, img_path)
