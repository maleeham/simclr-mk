#! /usr/bin/env python

from __future__ import print_function

import sys
import os
import json

path=os.path.join(os.getenv("FABRIC_HOME"), "libs", "fabric.zip")
print('> fabric loaded from %s'%path)
sys.path.insert(0,path)

from callbacks import LoggerCallback
from emetrics import EMetrics
from elog import ELog

class EDTLoggerCallback(LoggerCallback):
    def __init__(self):
        self.gs =0
        self.test_metrics = []

    def log_train_metrics(self, loss, acc, completed_batch,  worker=0):
        acc = acc/100.0
        self.gs += 1
        with EMetrics.open() as em:
            em.record(EMetrics.TEST_GROUP,completed_batch,{'loss': loss, 'accuracy': acc})
        with ELog.open() as log:
            log.recordTrain("Train", completed_batch, self.gs, loss, acc, worker)
        self.test_metrics.append((self.gs, {"loss": float(loss)}))

    def log_test_metrics(self, loss, acc, completed_batch, worker=0):
        acc = acc/100.0
        #print ('kelvin: log_test_metrics acc: %d' %acc) 
        with ELog.open() as log:
            log.recordTest("Test", loss, acc, worker)

    def on_train_end(self):
        training_out =[]
        for test_metric in self.test_metrics:
            out = {'steps':test_metric[0]}
            #print ('kelvin: out: %s' %out)
            for (metric,value) in test_metric[1].items():
                out[metric] = value
                #print ('kelvin: value:  %s' %out[metric])
            training_out.append(out)
        with open('{}/val_dict_list.json'.format(os.environ['RESULT_DIR']), 'w') as f:
            json.dump(training_out, f)
