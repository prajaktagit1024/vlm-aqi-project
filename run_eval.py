import os, sys
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
sys.path.insert(0, os.path.dirname(__file__))
from src.evaluate import run_evaluation
run_evaluation()
