import shutil

from ibis_py_utils import read_defaults

def main():
    directories = read_defaults("directories.json") 
    for dir in ("config_dir", "grid_dir"):
        try:
            shutil.rmtree(directories[dir])
        except:
            pass
