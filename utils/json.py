# ---------------------------------------------------------------- #

import os
import json

# ---------------------------------------------------------------- #

def json_append(file_path, dictionary):
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            content = json.load(file)

            if not isinstance(content, list):
                content = [content]

    else:
        content = []

    content.append(dictionary)

    with open(file_path, "w") as file:
        json.dump(content, file, indent=4)


def json_get_id(file_path, dictionary, asserting=True):
    if not os.path.exists(file_path): return

    with open(file_path, "r") as file:
        content = json.load(file)

    id = None

    for c in content:
        if dictionary == c["data"]:
            id = c["id"]

    if asserting: assert id is not None, (file_path, dictionary)

    return id

def json_modify_content(file_path, modify_content):
    with open(file_path, "r") as file:
        content = json.load(file)

        if not isinstance(content, list):
            content = [content]

    content = modify_content(content)

    with open(file_path, "w") as file:
        json.dump(content, file, indent=4)

def json_remove_by_id(file_path, id):
    if not os.path.exists(file_path): return

    modify_content = lambda content: \
        [ c for c in content if c["id"] != id]

    return json_modify_content(file_path, modify_content)

def json_remove_by_dict(file_path, dictionary):
    if not os.path.exists(file_path): return

    modify_content = lambda content: \
        [ c for c in content if c["data"] != dictionary]

    return json_modify_content(file_path, modify_content)

# ---------------------------------------------------------------- #
