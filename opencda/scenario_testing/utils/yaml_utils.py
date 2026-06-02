# -*- coding: utf-8 -*-
"""
Used to load and write yaml files
"""
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

import re
import yaml
from datetime import datetime
from omegaconf import OmegaConf
import os

def load_yaml(file):
    """
    Load yaml file and return a dictionary.
    Parameters
    ----------
    file : string
        yaml file path.

    Returns
    -------
    param : dict
        A dictionary that contains defined parameters.
    """

    stream = open(file, 'r')
    loader = yaml.Loader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    param = yaml.load(stream, Loader=loader)

    # load current time for data dumping and evaluation
    current_time = datetime.now()
    current_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")

    param['current_time'] = current_time

    return param


def add_current_time(params):
    """
    Add current time to the params dictionary.
    """
    # load current time for data dumping and evaluation
    current_time = datetime.now()
    current_time = current_time.strftime("%Y_%m_%d_%H_%M_%S")

    params['current_time'] = current_time

    return params


def save_yaml(data, save_name):
    """
    Save the dictionary into a yaml file.

    Parameters
    ----------
    data : dict
        The dictionary contains all data.

    save_name : string
        Full path of the output yaml file.
    """
    if isinstance(data, dict):
        with open(save_name, 'w') as outfile:
            yaml.dump(data, outfile, default_flow_style=False)
    else:
        with open(save_name, "w") as f:
            OmegaConf.save(data, f)



##=================== from openInfrax ============================================
def calculate_frame_rate(params):

    frame_rate = int(1/params['world']['fixed_delta_seconds'])
    params['world']['frame_rate'] = frame_rate

    return params

def modify_save_path(scenario_params,base_name):
    if "output_directory" not in scenario_params:
        scenario_params["output_directory"] = ""

    if not scenario_params["output_directory"]:
        # output_dir is empty
        # Set the output directory to the default location
        current_path = os.getcwd()
        output_dir = os.path.join(current_path, "data_dumping")
        data_dump_path = os.path.join(output_dir, base_name)
        # print(f"data_dump_path: {data_dump_path}")
        scenario_params["output_directory"] = output_dir  
    else:
        # output_dir is not empty
        # Use the specified output directory
        data_dump_path = os.path.join(scenario_params["output_directory"],base_name)
        
    return scenario_params, data_dump_path

def check_path(output_dir, config_data):
    generate_flag = False
    if os.path.exists(output_dir):
        files_in_dir = os.listdir(output_dir)
        if (len(files_in_dir) == 1 and "data_protocol.yaml" in files_in_dir):
            print("Output dir exists, only data_protocol.yaml is found, need to generate data")
            generate_flag = True
        else:
            existing_dirs = [d for d in files_in_dir if os.path.isdir(os.path.join(output_dir, d))]
            if existing_dirs:
                data_path = os.path.join(output_dir, existing_dirs[0])
                yaml_files = [file for file in os.listdir(data_path) if file.endswith('.yaml')]
                print("len(yaml_files):", len(yaml_files))
                print("Number that we need:", config_data['world']['time_period'] - 30)
                print("---")
                if len(yaml_files) < config_data['world']['time_period'] - 30:
                    print("Output dir exists, but not enough yaml files found, need to generate data")
                    generate_flag = True
                else:
                    print("Output dir exists, and enough yaml files found, skip generating data")
                    generate_flag = False
            else:
                print("Output dir exists, but no valid folder found, need to generate data")
                generate_flag = True
    else:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Creating output dir: {output_dir}")
        generate_flag = True
    print("---check_path---")
    return generate_flag

def modify_yaml(config_yaml, cam_key, lidar_key, sun_key,
                CAMERA_CONFIGS, LIDAR_CONFIGS, SUN_ANGLES,random_seed):
    """ Modify the base YAML file with new camera, LiDAR, and sun angle settings. """
    config_data = load_yaml(config_yaml)
    config_data["file_save"] = True
    if "output_directory" not in config_data:
        config_data["output_directory"] = ""
    else:
        config_data["output_directory"] = config_data["output_directory"]

    # config_data["world"]["time_period"] = 180

    # Update camera settings
    # Handle nested camera configs with indices
    # if key in CAMERA_CONFIGS[cam_key] is integer, then it is the new sensor presets.
    if isinstance(list(CAMERA_CONFIGS[cam_key].keys())[0], int) and isinstance(list(LIDAR_CONFIGS[lidar_key].keys())[0], int):
        print("This is the NEW sensor presets.")
        # This is the for the new sensor presets.
        for idx in CAMERA_CONFIGS[cam_key]:
            if idx >= len(config_data["rsu_list"]):
                # Copy the first RSU as template and update its ID and position
                new_rsu = config_data["rsu_list"][0].copy()
                new_rsu["id"] = str(-(int(config_data["rsu_list"][idx-1]["id"]) + idx)) # Generate unique negative ID for new RSU
                new_rsu["center_pos"] = config_data["rsu_list"][idx-1]["center_pos"] # Default position, should be updated based on camera positions
                config_data["rsu_list"].append(new_rsu)
            config_data["rsu_list"][idx]["sensors"]["cameras"]["positions"] = CAMERA_CONFIGS[cam_key][idx]["positions"]
            config_data["rsu_list"][idx]["sensors"]["cameras"]["num"] = CAMERA_CONFIGS[cam_key][idx]["num"]
            config_data["rsu_list"][idx]["sensors"]["cameras"]["parameters"] = CAMERA_CONFIGS[cam_key][idx]["parameters"]
            config_data["rsu_list"][idx]["sensors"]["cameras"]["save"] = CAMERA_CONFIGS[cam_key][idx]["save"]

        # Update LiDAR settings
        # Handle nested lidar configs with indices
        for idx in LIDAR_CONFIGS[lidar_key]:
            if idx >= len(config_data["rsu_list"]):
                # Copy the first RSU as template and update its ID and position
                new_rsu = config_data["rsu_list"][0].copy()
                new_rsu["id"] = str(-(int(config_data["rsu_list"][idx-1]["id"]) + idx)) # Generate unique negative ID for new RSU
                new_rsu["center_pos"] = config_data["rsu_list"][idx-1]["center_pos"] # Default position, should be updated based on lidar positions
                config_data["rsu_list"].append(new_rsu)
            config_data["rsu_list"][idx]["sensors"]["lidars"]["positions"] = LIDAR_CONFIGS[lidar_key][idx]["positions"]
            config_data["rsu_list"][idx]["sensors"]["lidars"]["num"] = LIDAR_CONFIGS[lidar_key][idx]["num"]
            config_data["rsu_list"][idx]["sensors"]["lidars"]["parameters"] = LIDAR_CONFIGS[lidar_key][idx]["parameters"]
            config_data["rsu_list"][idx]["sensors"]["lidars"]["save"] = LIDAR_CONFIGS[lidar_key][idx]["save"]
    else:
        print("This is the OLD sensor presets.")
        # This is the for the old sensor presets.
        config_data["rsu_list"][0]["sensors"]["cameras"]["positions"] = CAMERA_CONFIGS[cam_key]["positions"]
        config_data["rsu_list"][0]["sensors"]["cameras"]["num"] = CAMERA_CONFIGS[cam_key]["num"]
        config_data["rsu_list"][0]["sensors"]["cameras"]["parameters"] = CAMERA_CONFIGS[cam_key]["parameters"]
        config_data["rsu_list"][0]["sensors"]["cameras"]["save"] = CAMERA_CONFIGS[cam_key]["save"]
        
        # same for lidar
        config_data["rsu_list"][0]["sensors"]["lidars"]["positions"] = LIDAR_CONFIGS[lidar_key]["positions"]
        config_data["rsu_list"][0]["sensors"]["lidars"]["num"] = LIDAR_CONFIGS[lidar_key]["num"]
        config_data["rsu_list"][0]["sensors"]["lidars"]["parameters"] = LIDAR_CONFIGS[lidar_key]["parameters"]
        config_data["rsu_list"][0]["sensors"]["lidars"]["save"] = LIDAR_CONFIGS[lidar_key]["save"]
        
    # Update sun angle
    config_data["world"]["weather"]["sun_altitude_angle"] = SUN_ANGLES[sun_key]

    # Update random seed
    config_data["world"]["seed"] = random_seed

    # Generate modified YAML filename
    yaml_dir = os.path.dirname(config_yaml)
    # get the parent directory of the yaml_dir
    config_yaml_dir = os.path.dirname(yaml_dir)
    base_name = os.path.basename(config_yaml).replace(".yaml", "")
    # create a new folder for different yaml
    town_folder = os.path.join(config_yaml_dir, "batch_scenarios", base_name)
    # make town folder
    os.makedirs(town_folder, exist_ok=True)
    modified_yaml_path = os.path.join(town_folder, f"{base_name}_{cam_key}_{lidar_key}_{sun_key}_s{random_seed}.yaml")
    save_yaml(config_data, modified_yaml_path)

    return modified_yaml_path, config_data

def find_yaml_file(yaml_path, scenario_name):
    """
    Find the yaml absolute path based on the scenario_name
    This code will look up all the folders under the yaml_path and find the scenario_name.yaml file
    """

    for root, dirs, files in os.walk(yaml_path):
        for file in files:
            if file == f"{scenario_name}.yaml":
                yaml_file_path = os.path.join(root, file)
                print(f"Found {yaml_file_path}")
                return yaml_file_path

    return None

