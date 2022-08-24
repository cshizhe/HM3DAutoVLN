from yacs.config import CfgNode as CN

import os
DATA_DIR = '../datasets'

_C = CN()

_C.SEED = 0
_C.OUTPUT_DIR = ''
_C.RESUME_FILE = None

_C.MODEL = CN()
_C.MODEL.TYPE = 'gpt2'
_C.MODEL.GPUID = 0
_C.MODEL.HIDDEN_SIZE = 768
_C.MODEL.ENC_LAYERS = 6
_C.MODEL.DROPOUT = 0.
_C.MODEL.GPT_FREEZE = False
_C.MODEL.MAX_TXT_LEN = 100 
_C.MODEL.OBJIMG_FT_SIZE = 768
_C.MODEL.OBJNAME_FT_SIZE = 300
_C.MODEL.VIEWIMG_FT_SIZE = 768
_C.MODEL.USE_VIEW_FT = True
_C.MODEL.USE_CTX_OBJS = True

_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 64
_C.TRAIN.NUM_EPOCH = 100
_C.TRAIN.LEARNING_RATE = 0.0001
_C.TRAIN.WARMUP_STEPS = 100
_C.TRAIN.EVAL_EVERY_EPOCH = 5

_C.DATA = CN()
_C.DATA.VIEW_FT_DIR = ''
_C.DATA.OBJ_FT_DIR = ''
_C.DATA.ANNO_DIR = ''
_C.DATA.OBJNAME_FT_NORM = False

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()