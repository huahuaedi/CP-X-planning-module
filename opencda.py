# -*- coding: utf-8 -*-
"""
Script to run different scenarios.
"""

# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import argparse
import inspect
import importlib
import os
import sys
from omegaconf import OmegaConf

from opencda.version import __version__


def _load_scenario_config(config_yaml):
    """Load a scenario YAML with an optional local ``base_config`` overlay."""

    scene_dict = OmegaConf.load(config_yaml)
    base_config = scene_dict.pop("base_config", None)
    if not base_config:
        return scene_dict
    config_dir = os.path.dirname(config_yaml)
    base_path = os.path.join(config_dir, str(base_config))
    if not os.path.isfile(base_path):
        raise FileNotFoundError(
            "Scenario base_config not found: %s" % os.path.abspath(base_path)
        )
    return OmegaConf.merge(_load_scenario_config(base_path), scene_dict)


def arg_parse():
    # create an argument parser
    parser = argparse.ArgumentParser(description="OpenCDA scenario runner.")
    # add arguments to the parser
    parser.add_argument('-t', "--test_scenario", required=True, type=str,
                        help='Define the name of the scenario you want to test. The given name must'
                             'match one of the testing scripts(e.g. single_2lanefree_carla) in '
                             'opencda/scenario_testing/ folder'
                             ' as well as the corresponding yaml file in opencda/scenario_testing/config_yaml.')
    parser.add_argument("--record", action='store_true',
                        help='whether to record and save the simulation process to .log file')
    parser.add_argument("--apply_ml",
                        action='store_true',
                        help='whether ml/dl framework such as sklearn/pytorch is needed in the testing. '
                             'Set it to true only when you have installed the pytorch/sklearn package.')
    parser.add_argument('-v', "--version", type=str, default='0.9.11',
                        help='Specify the CARLA simulator version, default'
                             'is 0.9.11, 0.9.12 is also supported.')
    # parse the arguments and return the result
    opt = parser.parse_args()
    return opt


def main():
    # parse the arguments
    opt = arg_parse()
    #print(f'opt {opt}')
    # print the version of OpenCDA
    print("OpenCDA Version: %s" % __version__)
    # set the default yaml file
    default_yaml = config_yaml = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        'opencda/scenario_testing/config_yaml/default.yaml')
    # set the yaml file for the specific testing scenario
    config_yaml = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               'opencda/scenario_testing/config_yaml/%s.yaml' % opt.test_scenario)
    # load the default yaml file and the scenario yaml file as dictionaries
    default_dict = OmegaConf.load(default_yaml)
    scene_dict = _load_scenario_config(config_yaml)
    # merge the dictionaries
    scene_dict = OmegaConf.merge(default_dict, scene_dict)

    # PyTorch 1.10/MKL must initialize the YOLO model before Open3D is
    # imported by the scenario stack. Reversing this order can stall CPU
    # inference indefinitely in the legacy OpenCDA runtime.
    if opt.apply_ml:
        ml_manager_cls = getattr(importlib.import_module(
            "opencda.customize.ml_libs.ml_manager"), "MLManager")
        ml_manager_cls()

    # import the testing script
    testing_scenario = importlib.import_module(
        "opencda.scenario_testing.%s" % opt.test_scenario)
    
    #print(f'testing_scenario  {testing_scenario }')
    # check if the yaml file for the specific testing scenario exists
    if not os.path.isfile(config_yaml):
        print('Exited')
        sys.exit(
            "opencda/scenario_testing/config_yaml/%s.yaml not found!" % opt.test_cenario)

    # get the function for running the scenario from the testing script
    scenario_runner = getattr(testing_scenario, 'run_scenario')

    #print(f"============================scene_dict: {scene_dict}==============================")

    
    # run the scenario testing
    if "experiment_params" in inspect.signature(scenario_runner).parameters:
        scenario_runner(opt, scene_dict, experiment_params={})
    else:
        scenario_runner(opt, scene_dict)
    
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')
