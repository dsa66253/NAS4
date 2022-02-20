import os


def makeDir(save_folder, log_path):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    if not os.path.exists(log_path):
        os.makedirs(log_path)