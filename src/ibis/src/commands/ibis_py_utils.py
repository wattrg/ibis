import json
import os

IBIS = os.environ.get("IBIS")

def read_defaults(file_name):
    with open(f"{IBIS}/resources/defaults/{file_name}", "r") as defaults:
        defaults = json.load(defaults)
    return defaults
