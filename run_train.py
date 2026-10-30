"""Runner script that trains and logs output to train_log.txt"""
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Redirect stdout/stderr to log file AND console
import io

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

log = open(os.path.join(os.path.dirname(__file__), "train_log.txt"), "w", buffering=1)
sys.stdout = Tee(sys.__stdout__, log)
sys.stderr = Tee(sys.__stderr__, log)

from src.train import train
train()

log.close()
print("Done. Check train_log.txt for full output.")
