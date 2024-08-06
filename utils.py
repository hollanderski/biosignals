from pathlib import Path

import yaml as yaml

"""
utils functions
"""


def load_config(config_path):
    try:
        with open(config_path, "r") as config_file:
            return yaml.safe_load(config_file)
    except yaml.YAMLError as e:
        print(e)


def create_folder_structure(date: str, config: yaml) -> Path:
    root_data_path = Path(Path(config['DATA_CAPTURE']['ROOT_DATA_PATH']) / date)
    print("creating the folder structure in {}".format(root_data_path))
    dir_list = config['DATA_CAPTURE']['DIRECTORIES']
    for i in dir_list:
        _dir = root_data_path / i
        # we'll create the directory if it doesn't exist yet, otherwise we'll have to check first
        try:
            Path.mkdir(Path(_dir), parents=True, exist_ok=False)
        except FileExistsError as e:
            res = input("Folder exists already, do you really want to overwrite it? Y/N")
            if res == "Y":
                Path.mkdir(Path(_dir), parents=True, exist_ok=True)
            else:
                continue
    return root_data_path
