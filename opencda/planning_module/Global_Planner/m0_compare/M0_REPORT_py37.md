# M0 — dij vs astar global-route comparison

## astar setup diagnostics

```json
{
  "carla_file": "/home/umd-user/miniforge3/envs/opencda_planning/lib/python3.7/site-packages/carla/__init__.py",
  "carla_version": "?"
}
```

| verdict | count |
| --- | --- |
| DIJ-FAIL | 10 |

| case | verdict | detail | dij len | astar len | dij LC | astar LC | dij gapv |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gp_readme_town10_example | DIJ-FAIL | ValueError: unsupported pickle protocol: 5 | - | 147.66 | - | 1 | - |
| town10_cp_passing_merge | DIJ-FAIL | ValueError: unsupported pickle protocol: 5 | - | 428.1 | - | 1 | - |
| town10_cp_red_light_violator | DIJ-FAIL | ValueError: unsupported pickle protocol: 5 | - | 404.17 | - | 0 | - |
| town10_cp_roadway_object | DIJ-FAIL | ValueError: unsupported pickle protocol: 5 | - | 428.1 | - | 1 | - |
| town10_scenario_1 | DIJ-FAIL | ValueError: unsupported pickle protocol: 5 | - | 182.26 | - | 1 | - |
| town10_scenario_2 | DIJ-FAIL | ValueError: unsupported pickle protocol: 5 | - | 177.33 | - | 0 | - |
| town10_scenario_3 | DIJ-FAIL | ValueError: unsupported pickle protocol: 5 | - | 156.73 | - | 0 | - |
| town10_scenario_4 | DIJ-FAIL | ValueError: unsupported pickle protocol: 5 | - | 86.72 | - | 0 | - |
| town10_scenario_5 | DIJ-FAIL | ValueError: unsupported pickle protocol: 5 | - | 223.48 | - | 0 | - |
| town10_scenario_6 | DIJ-FAIL | ValueError: unsupported pickle protocol: 5 | - | 404.17 | - | 0 | - |

## Per-case detail (non-MATCH)

### gp_readme_town10_example — DIJ-FAIL
- ValueError: unsupported pickle protocol: 5
- dij   road_option_seq: `None`
- carla road_option_seq: `['LANEFOLLOW', 'STRAIGHT', 'LANEFOLLOW', 'CHANGELANELEFT', 'LANEFOLLOW']`
- dij   lane_seq: `None`
- carla lane_seq: `[[5, 0, 2], [516, 0, 2], [4, 0, 2], [8, 0, -2], [1, 0, -2], [1, 0, -1]]`

### town10_cp_passing_merge — DIJ-FAIL
- ValueError: unsupported pickle protocol: 5
- dij   road_option_seq: `None`
- carla road_option_seq: `['LANEFOLLOW', 'STRAIGHT', 'LANEFOLLOW', 'STRAIGHT', 'LANEFOLLOW', 'STRAIGHT', 'CHANGELANERIGHT', 'LANEFOLLOW']`
- dij   lane_seq: `None`
- carla lane_seq: `[[6, 0, -1], [89, 0, -1], [7, 0, -1], [17, 0, 1], [10, 0, 1], [0, 0, 1], [3, 0, 1], [565, 0, 1], [2, 0, 1], [676, 0, 1], [1, 0, 1], [1, 0, 2], [8, 0, 2]]`

### town10_cp_red_light_violator — DIJ-FAIL
- ValueError: unsupported pickle protocol: 5
- dij   road_option_seq: `None`
- carla road_option_seq: `['LANEFOLLOW', 'STRAIGHT', 'LANEFOLLOW', 'STRAIGHT', 'LANEFOLLOW']`
- dij   lane_seq: `None`
- carla lane_seq: `[[6, 0, 2], [735, 0, 2], [5, 0, 2], [516, 0, 2], [4, 0, 2], [8, 0, -2], [1, 0, -2]]`

### town10_cp_roadway_object — DIJ-FAIL
- ValueError: unsupported pickle protocol: 5
- dij   road_option_seq: `None`
- carla road_option_seq: `['LANEFOLLOW', 'STRAIGHT', 'LANEFOLLOW', 'STRAIGHT', 'LANEFOLLOW', 'STRAIGHT', 'CHANGELANERIGHT', 'LANEFOLLOW']`
- dij   lane_seq: `None`
- carla lane_seq: `[[6, 0, -1], [89, 0, -1], [7, 0, -1], [17, 0, 1], [10, 0, 1], [0, 0, 1], [3, 0, 1], [565, 0, 1], [2, 0, 1], [676, 0, 1], [1, 0, 1], [1, 0, 2], [8, 0, 2]]`

### town10_scenario_1 — DIJ-FAIL
- ValueError: unsupported pickle protocol: 5
- dij   road_option_seq: `None`
- carla road_option_seq: `['LANEFOLLOW', 'STRAIGHT', 'CHANGELANERIGHT', 'LANEFOLLOW']`
- dij   lane_seq: `None`
- carla lane_seq: `[[5, 0, -1], [736, 0, -1], [6, 0, -1], [6, 0, -2]]`

### town10_scenario_2 — DIJ-FAIL
- ValueError: unsupported pickle protocol: 5
- dij   road_option_seq: `None`
- carla road_option_seq: `['LANEFOLLOW', 'STRAIGHT', 'LANEFOLLOW']`
- dij   lane_seq: `None`
- carla lane_seq: `[[515, 0, -1], [5, 0, -1], [736, 0, -1], [6, 0, -1]]`

### town10_scenario_3 — DIJ-FAIL
- ValueError: unsupported pickle protocol: 5
- dij   road_option_seq: `None`
- carla road_option_seq: `['LANEFOLLOW', 'STRAIGHT', 'LANEFOLLOW']`
- dij   lane_seq: `None`
- carla lane_seq: `[[5, 0, -1], [736, 0, -1], [6, 0, -1]]`

### town10_scenario_4 — DIJ-FAIL
- ValueError: unsupported pickle protocol: 5
- dij   road_option_seq: `None`
- carla road_option_seq: `['LANEFOLLOW', 'RIGHT', 'LANEFOLLOW']`
- dij   lane_seq: `None`
- carla lane_seq: `[[344, 0, -1], [20, 0, -2], [848, 0, -1], [9, 0, 1]]`

### town10_scenario_5 — DIJ-FAIL
- ValueError: unsupported pickle protocol: 5
- dij   road_option_seq: `None`
- carla road_option_seq: `['LANEFOLLOW', 'STRAIGHT', 'LANEFOLLOW', 'RIGHT', 'LANEFOLLOW', 'LEFT', 'LANEFOLLOW', 'RIGHT', 'LANEFOLLOW']`
- dij   lane_seq: `None`
- carla lane_seq: `[[19, 0, -2], [268, 0, -1], [255, 0, -2], [20, 0, -2], [848, 0, -1], [9, 0, 1], [152, 0, 1], [11, 0, 1], [712, 0, 1], [1, 0, 2]]`

### town10_scenario_6 — DIJ-FAIL
- ValueError: unsupported pickle protocol: 5
- dij   road_option_seq: `None`
- carla road_option_seq: `['LANEFOLLOW', 'STRAIGHT', 'LANEFOLLOW', 'STRAIGHT', 'LANEFOLLOW']`
- dij   lane_seq: `None`
- carla lane_seq: `[[6, 0, 2], [735, 0, 2], [5, 0, 2], [516, 0, 2], [4, 0, 2], [8, 0, -2], [1, 0, -2]]`
