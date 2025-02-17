# ---------------------------------------------------------------- #

import os
import shutil

# ---------------------------------------------------------------- #

def copy_folder(source, destination):
    os.makedirs(destination, exist_ok=True)

    for item in os.listdir(source):
        source_path = os.path.join(source, item)
        destination_path = os.path.join(destination, item)

        if os.path.isdir(source_path):
            shutil.copytree(source_path, destination_path, dirs_exist_ok=True)
        else:
            shutil.copy2(source_path, destination_path)

def os_path_join(*args):
    return os.path.join(*[arg for arg in args if arg is not None])

# ---------------------------------------------------------------- #
