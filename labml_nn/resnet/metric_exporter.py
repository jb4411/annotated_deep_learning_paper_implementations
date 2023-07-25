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


def output(config_dict, param_name, config_names):
    for temp in config_names:
        name = temp
        if '.' in temp:
            name = temp.split('.')[-1]
        print(f"{name}  {config_dict[temp][param_name]}")


def show_dataset(cfg, dataset_name):
    output(cfg, "computed", [dataset_name + "_batch_size"])
    output(cfg, "value", [dataset_name + "_dataset"])
    temp = cfg[dataset_name + "_dataset"]["computed"]
    t_idx = temp.find("Number of datapoints:")
    temp = temp[t_idx:].split("\n")[0]
    temp = temp.split(':')
    print(f"{temp[0].strip()}  {temp[1].strip()}")
    output(cfg, "computed", [dataset_name + "_loader_shuffle"])


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

    """base_url = "http://localhost:5005/run/"
    url = base_url + run_id
    response = requests.get(url)
    temp = response.text
    s_idx = response.text.find("name")
    e_idx = response.text[s_idx:].find("\",\"")
    title = response.text[s_idx:s_idx + e_idx]"""

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

    # output table
    print("\nConfigurations:\n")
    output(cfg, "computed", ["bottlenecks", "dataset_name", "device", "device.device_info", "epochs",
                             "first_kernel_size", "inner_iterations", "loss_func"])

    for temp in ["mode", "model"]:
        print(f"{temp}  {cfg[temp]['options'][0]}")

    output(cfg, "computed", ["n_blocks", "n_channels"])

    output(cfg, "value", ["optimizer.optimizer"])

    output(cfg, "computed", ["optimizer.amsgrad", "optimizer.betas", "optimizer.eps", "optimizer.learning_rate",
                             "optimizer.optimized_adam_update", "optimizer.weight_decay",
                             "optimizer.weight_decay_absolute", "optimizer.weight_decouple"])

    show_dataset(cfg, "train")
    show_dataset(cfg, "valid")


if __name__ == '__main__':
    main()
