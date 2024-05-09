import os
import sys
import json
from extract_log_info import ExtractLogInfo
# with open('srunner/data/FellowVehicleStopGo.json', 'r') as fh:
#     json_data = json.load(fh)
#
# for town_dict in json_data['available_scenarios']:
#     for town_name in town_dict.keys():
#         print(town_name)
#         scenarios = town_dict[town_name]
#         print(scenarios)
#         for scenario in scenarios:
#             for event in scenario["available_event_configurations"]:
#                 print(event)
#                 if '_start_distance' in event:
#                     if event['_start_distance'] == "%1d":
#                         event['_start_distance'] = "2"

def file_name(file_dir):
    L=[]
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.json':
                L.append(os.path.join(root, file))
    return L

# with open("./news_json.json", "w") as f:
#     json.dump(json_data, f)
#     print("已生成news_json.json文件...")

if __name__ == '__main__':
    # PedestrianCrossing_3 EmergencyBraking_1 CutInWithObstacle_1 CutInWithObstacle_4 ThroughRedLight_1
    # EmergencyBraking_2 PedestrianCrossing_1 ThroughRedLight_4
    files = file_name("D:/carla/ScenarioVerifyDB（interfuser）/ThroughRedLight_1/log")
    num = 0
    for name in files:
        with open(name, 'r') as fh:
            json_data = json.load(fh)
        CollisionTest = json_data["CollisionTest"]

        if CollisionTest["test_status"] == "SUCCESS":
            continue
        else:
            print(name)
            num=num+1
    print(num)
