import os
import sys
import atexit
import source.lib.JMSLab as jms

sys.path.append('config')
sys.dont_write_bytecode = True # Don't write .pyc files

os.environ['PYTHONPATH'] = '.'
env = Environment(ENV = {'PATH' : os.environ['PATH']},)

env.Decider('MD5-timestamp') # Only computes hash if time-stamp changed
Export('env')

SConscript('source/derived/SConscript')
SConscript('source/analysis/SConscript')

