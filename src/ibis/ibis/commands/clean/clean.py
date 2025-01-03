import shutil
import os

from ibis_py_utils import read_defaults

def main(lib_dir):
    directories = read_defaults(f"{lib_dir}/defaults",
                                "directories.json") 
    for dir in ("config_dir", "io_dir", "log_dir", "plot_dir"):
        if os.path.exists(directories[dir]):
            shutil.rmtree(directories[dir])

    if os.path.exists("__pycache__"):
        shutil.rmtree("__pycache__")
