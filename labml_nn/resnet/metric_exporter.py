import os
import time

import requests
from selenium.webdriver.common.keys import Keys
import sys
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions
from bs4 import BeautifulSoup
import bs4
import yaml

def main():
    if len(sys.argv) > 1:
        run_id = sys.argv[1]
    else:
        run_id = input("Please enter the run ID: ")
        if run_id.strip() == "":
            return

    base_path = os.path.join(os.getcwd(), "logs", "resnet")

    while True:
        path = os.path.join(base_path, run_id)
        if not os.path.exists(path):
            print(f"Error: Run with ID {run_id} not found!", file=sys.stderr)
            run_id = input("Please enter a valid run ID: ")
            if run_id.strip() == "":
                return
        else:
            break

    config_names = ["bottlenecks", "dataset_name", "device", "--- device_info", "epochs",
                    "first_kernel_size", "inner_iterations", "loss_func", "mode", "model",
                    "n_blocks", "n_channels", "optimizer", "--- amsgrad",
                    "--- betas", "--- eps", "--- learning_rate", "--- optimized_adam_update",
                    "--- optimizer", "--- weight_decay", "--- weight_decay_absolute",
                    "--- weight_decay_obj", "--- weight_decouple",
                    "train_batch_size", "train_dataset", "train_loader", "train_loader_shuffle",
                    "trainer",
                    "valid_batch_size", "valid_dataset", "valid_loader", "valid_loader_shuffle",
                    "validator"]

    configs = {key: '' for key in config_names}

    cfg = None
    with open(os.path.join(path, "configs.yaml"), 'r') as stream:
        try:
            # Converts yaml document to python object
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as e:
            print(e)

    print()


if __name__ == '__main__':
    main()
