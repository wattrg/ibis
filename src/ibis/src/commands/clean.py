import shutil
import os

from ibis_py_utils import read_defaults

def main():
    directories = read_defaults("directories.json") 
    for dir in ("config_dir", "grid_dir", "flow_dir"):
        if os.path.exists(directories[dir]):
            shutil.rmtree(directories[dir])

    if os.path.exists("__pycache__"):
        shutil.rmtree("__pycache__")
