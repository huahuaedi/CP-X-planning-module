import argparse
import importlib
import os
import sys
import shutil
import yaml
from itertools import product
# from opencda_infra.version import __version__
# from opencda_infra.utils.yaml_utils import load_yaml, modify_yaml, find_yaml_file, \
#                                         modify_save_path, check_path

from opencda.version import __version__
from opencda.scenario_testing.utils.yaml_utils import load_yaml, modify_yaml, find_yaml_file, modify_save_path, check_path


def single_n_batch_runner(opt,config_yaml,batch_mode=False, config_data=None, scenario_name=None):
    try:
        running_scenario = importlib.import_module("opencda_infra.opencda_mod.intersection_carla")
    except ModuleNotFoundError:
        sys.exit(f"ERROR: {opt.scenario}.py not found under opencda_infra/opencda_mod")

    base_name = os.path.basename(config_yaml).replace(".yaml", "")

    if not batch_mode:
        print("This is the SINGLE Scenario Mode!")
        assert config_data is None, "config_data should be None in single mode"
        assert scenario_name is None, "scenario_name should be None in single mode"
        
        config_data = load_yaml(config_yaml)
        config_data,output_dir = modify_save_path(config_data,base_name)
        generate_flag = check_path(output_dir, config_data)

        if generate_flag:
            print(f"Running scenario with {base_name}")
            scenario_runner = getattr(running_scenario, "test_scenario")
            scenario_runner(opt, config_yaml)    
    else:
        print("This is the BATCH Mode!")
        assert config_data is not None, "config_data should not be None in batch mode"
        assert scenario_name is not None, "scenario_name should not be None in batch mode"

        config_data,output_dir = modify_save_path(config_data,scenario_name)
        print(f"output_dir:{output_dir}")
        generate_flag = check_path(output_dir, config_data)
        if generate_flag:
            
            scenario_runner = getattr(running_scenario, "test_scenario")
            # here config_yaml is the modified yaml file
            scenario_runner(opt, config_yaml)

        if os.path.exists("output"):
            shutil.move("output", output_dir)        

def run_scenarios(opt):

    print("Sensor Configurator Version: %s" % __version__)

    if isinstance(opt.scenario_name, list) and len(opt.scenario_name) > 1:
        # This is multi-scenario mode
        assert opt.batch != True, "Multi-scenario mode can not support batch mode"
        for s_name in opt.scenario_name:
            
            print(f"Running scenario: {s_name}")
            config_yaml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            'opencda_infra/config_yaml')
            config_yaml = find_yaml_file(config_yaml_path, s_name)

            single_n_batch_runner(opt,config_yaml)            
    else:
        # This is single-scenario mode
        # Single-scenario mode include batch mode and single mode
        # Single mode is used to run one scenario with one sensor placement
        # Batch mode is used to run different scenarios with different sensor placements
        if not opt.batch:
            assert len(opt.scenario_name) == 1, "Batch mode only supports one scenario at a time"
            if len(opt.scenario_name) == 1:
                s_name = opt.scenario_name[0]
            config_yaml_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            'opencda_infra/config_yaml')

            config_yaml = find_yaml_file(config_yaml_path, s_name)
            print(f"config_yaml: {config_yaml}")
            single_n_batch_runner(opt,config_yaml)

        else:
            # The following are the batch processing of different scenarios
            if len(opt.scenario_name) == 1:
                s_name = opt.scenario_name[0]
            present_file = f'opencda_infra.config_yaml.sensor_presets.%s' % s_name   
            
            try:
                present_module = importlib.import_module(present_file)
                CAMERA_CONFIGS = present_module.CAMERA_CONFIGS
                LIDAR_CONFIGS = present_module.LIDAR_CONFIGS
                SUN_ANGLES = present_module.SUN_ANGLES
                if present_module.RANDOM_SEEDs:
                    RANDOM_SEEDs = present_module.RANDOM_SEEDs
                else:
                    # It depends on how many different scenarios you want to run
                    # Default is 5 if not specified
                    RANDOM_SEEDs = [0,5,10,15,20] 

            except ModuleNotFoundError:
                sys.exit(f"ERROR: {s_name}.py not found under opencda_infra/config_yaml/sensor_presets/")        
            
            base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                    'opencda_infra/config_yaml')
            base_yaml = find_yaml_file(base_path, s_name)

            if not os.path.exists(base_yaml):
                sys.exit(f"ERROR: {base_yaml} not found")

            for random_seed in RANDOM_SEEDs:
                for cam_key, lidar_key, sun_key in product(CAMERA_CONFIGS.keys(), LIDAR_CONFIGS.keys(), 
                                                        SUN_ANGLES.keys()):
                    # print(f"camera: {cam_key}, lidar: {lidar_key}, sun: {sun_key}")

                    modified_yaml, config_data = modify_yaml(base_yaml, cam_key, lidar_key, sun_key,
                                                            CAMERA_CONFIGS, LIDAR_CONFIGS, SUN_ANGLES,
                                                            random_seed)

                    base_name = os.path.basename(base_yaml).replace(".yaml", "")
                    # print(f"base_name: {base_name}")
                    scenario_name = f"{base_name}_{cam_key}_{lidar_key}_{sun_key}_s{random_seed}"

                    print(f"Running scenario with {cam_key}_{lidar_key}_{sun_key}")

                    single_n_batch_runner(opt,modified_yaml, batch_mode=True, 
                                                    config_data=config_data, 
                                                    scenario_name=scenario_name)

                print("All sensor placement completed!")
            print("All scenarios completed!")


def arg_parse():
    parser = argparse.ArgumentParser(description="OpenCDA scenario runner.")
    parser.add_argument("-s","--scenario_name", nargs='+',type=str, 
                        help='Define one or more scenario names you want to test. The given name must'
                             'match one of the yaml file(e.g. single_2lanefree_carla) in '
                             'opencda_infra/config_yaml/ folder')
    parser.add_argument("--record", action="store_true", help="Record the simulation process")
    parser.add_argument("--apply_ml", action="store_true", help="Use ML/DL framework if required")
    parser.add_argument("-v", "--version", type=str, default="0.9.12", help="CARLA simulator version")
    parser.add_argument("--batch", action="store_true", help="Run scenarios in batch mode")

    return parser.parse_args()

if __name__ == "__main__":
    try:
        opt = arg_parse()
        run_scenarios(opt)
    except KeyboardInterrupt:
        print(" - Exited by user.")