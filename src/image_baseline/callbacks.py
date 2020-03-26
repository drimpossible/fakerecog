import os
from tensorboardX import SummaryWriter
import numpy as np


class Logger:
    def __init__(self, log_dir, n_logged_samples=10, summary_writer=SummaryWriter):
        self._log_dir = log_dir
        print('########################')
        print('logging outputs to ', log_dir)
        print('########################')
        self._n_logged_samples = n_logged_samples
        self._summ_writer = summary_writer(log_dir)

    def log_scalar(self, scalar, name, step_):
        self._summ_writer.add_scalar('{}'.format(name), scalar, step_)

    def flush(self):
       pass