import carla
import argparse
import os
from opencda_infra.version import __version__
import opencda_infra.opencda_mod.sim_api as sim_api 
from opencda_infra.utils.yaml_utils import add_current_time, \
                                                save_yaml, load_yaml,\
                                                calculate_frame_rate, \
                                                modify_save_path
from opencda_infra.utils.dictionary_creator import find_sensor_groups, find_sensor_groups_v2
from opencda_infra.core.opencda_lib.cav_world import CavWorld
from opencda_infra.core.vehicle_manager import VehicleManager

def check_path(output_dir):
    pass

def test_scenario(opt, config_yaml):
    try:
        # Load scenario parameters
        scenario_params = load_yaml(config_yaml)

        # assign parameters
        mytown = scenario_params['world']['town']
        spectator_pos = scenario_params['world']['spectator_pos']
        time_period = scenario_params['world']['time_period']

        # Extract the correct naming format from YAML filename
        base_name = os.path.basename(config_yaml).replace(".yaml", "")
        
        # Ensure `carla_base` exists in scenario_params
        if "carla_base" not in scenario_params:
            scenario_params["carla_base"] = {} 

        # Add current time to scenario_params['world']
        add_current_time(scenario_params['world'])
        # Calculate frame rate based on parameters in scenario_params['world']
        scenario_params = calculate_frame_rate(scenario_params)

        scenario_params, data_dump_path = modify_save_path(scenario_params,base_name)
        
        scenario_params['world']['data_dump_path'] = data_dump_path
        # Ensure the output directory exists
        os.makedirs(data_dump_path, exist_ok=True)

        # Save modified YAML for debugging and reproducibility
        modified_yaml_path = os.path.join(data_dump_path, 'data_protocol.yaml')
        save_yaml(scenario_params, modified_yaml_path)

        # Ensure `scenario_manager` is defined before `finally`
        scenario_manager = None

        # Start simulation
        cav_world = CavWorld(opt.apply_ml)

        # ==== Initialize the simulation API ====
        scenario_manager = sim_api.ScenarioManager(
            scenario_params, opt.apply_ml, opt.version,
            town=mytown, cav_world=cav_world
        )

        if opt.record:
            save_log_name = os.path.join(data_dump_path,
                                        f'{base_name}.log')
            scenario_manager.client. \
                start_recorder(save_log_name, True)

        traffic_manager, bg_list = scenario_manager.create_bg_traffic()

        single_cav_list = \
            scenario_manager.create_vehicle_manager(application=['single'],
                                                    traffic_manager=traffic_manager,
                                                    save_time=base_name,
                                                    data_dump=scenario_params['file_save'])

        # Initialize intersection manager with correct save_time
        rsu_list = scenario_manager.create_intersection_manager(
            data_dump=scenario_params['file_save']
        )

        # groups = find_sensor_groups(scenario_params, distance_threshold=1.0)
        # groups = find_sensor_groups_v2(scenario_params, distance_threshold=1.0)

        if len(spectator_pos) == 6:
            # spectator_pos is set
            spectator = scenario_manager.world.get_spectator()
            spectator.set_transform(carla.Transform(
                carla.Location(
                x=spectator_pos[0],
                y=spectator_pos[1],
                z=spectator_pos[2]
            ),carla.Rotation(
                pitch=spectator_pos[3], 
                yaw=spectator_pos[4], 
                roll=spectator_pos[5]
            )))
        else:
            print("Separate spectator position not set")

        # Simulation loop
        while True:
            scenario_manager.tick()

            # if bg_list:
            #     for agent in bg_list:
            #         # determine if agent is a vehicle or pedestrian
            #         if agent.type_id.startswith("vehicle."):
            #             VehicleManager.update_vehicle_lights(agent, scenario_manager.my_weather)

            if single_cav_list:
                for single_cav in single_cav_list:
                    single_cav.update_info()
                    single_cav.run_step()

            if rsu_list:
                for rsu in rsu_list:
                    rsu.update_info()
                    rsu.run_step()

            print(scenario_manager.count)
            
            if scenario_manager.count >= time_period:
                break
            print("=======================================")

    except KeyError as e:
        print(f"ERROR: Missing key {e} in scenario_params")
    except Exception as e:
        print(f"ERROR: {e}")

    finally:
        if scenario_manager:
            if opt.record:
                scenario_manager.client.stop_recorder()
            scenario_manager.close()

            for r in rsu_list:
                r.destroy()
            for v in single_cav_list:
                v.destroy()
            for v in bg_list:
                v.destroy()