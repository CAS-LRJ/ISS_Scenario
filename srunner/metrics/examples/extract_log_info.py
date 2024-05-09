import os
import sys
import json
import math
import matplotlib.pyplot as plt
import os
import sys
import importlib
import inspect
import json
import argparse
from argparse import RawTextHelpFormatter
from numpy import *
from scipy.spatial.distance import cdist
import numpy as np
import carla
from srunner.metrics.tools.metrics_log import MetricsLog

from srunner.metrics.examples.basic_metric import BasicMetric


class ExtractLogInfo(object):

    def __init__(self, log, criteria, actor_id, host='127.0.0.1', port=2000):

        # Parse the arguments
        self.host = host
        self.port = port
        self._client = carla.Client(self.host, self.port)

        recorder_str = self._get_recorder(log)
        criteria_dict = self._get_criteria(criteria)

        # Get the correct world and load it
        map_name = self._get_recorder_map(recorder_str)
        world = self._client.load_world(map_name)
        self.debug = world.debug
        town_map = world.get_map()

        # change weather
        weather = world.get_weather()
        weather.sun_altitude_angle = 69.93514724377155

        weather.precipitation = -6.533753516809867
        weather.wind_intensity = 89.27755145828856
        weather.sun_azimuth_angle = 132.4021295538381
        weather.cloudiness = 32.05220456165719
        weather.fog_density = 61.493443445256936
        weather.wetness = 60.8736825294781
        weather.precipitation_deposits = 10.442199673310137

        world.set_weather(weather)

        # Instanciate the MetricsLog, used to querry the needed information
        log = MetricsLog(recorder_str)

        self.actor_id = actor_id
        self.town_map = town_map
        self.log = log
        self.criteria = criteria_dict

    def _get_recorder(self, log):
        """
        Parses the log argument into readable information
        """

        # Get the log information.
        self._client = carla.Client(self.host, self.port)
        recorder_file = log
        # print(recorder_file)
        # Check that the file is correct
        if recorder_file[-4:] != '.log':
            print("ERROR: The log argument has to point to a .log file")
            sys.exit(-1)
        if not os.path.exists(recorder_file):
            print("ERROR: The specified log file does not exist")
            sys.exit(-1)

        recorder_str = self._client.show_recorder_file_info(recorder_file, True)

        return recorder_str

    def _get_criteria(self, criteria_file):
        """
        Parses the criteria argument into a dictionary
        """
        if criteria_file:
            with open(criteria_file) as fd:
                criteria_dict = json.load(fd)
        else:
            criteria_dict = None

        return criteria_dict

    def _get_recorder_map(self, recorder_str):
        """
        Returns the name of the map the simulation took place in
        """

        header = recorder_str.split("\n")
        sim_map = header[1][5:]

        return sim_map

    def angle_rotation(self, pitch, yaw, roll):
        r1 = cos(radians(yaw)) * cos(radians(pitch))
        r2 = cos(radians(yaw)) * sin(radians(roll)) * sin(radians(pitch)) - cos(radians(roll)) * sin(radians(yaw))
        r3 = sin(radians(yaw)) * sin(radians(roll)) + cos(radians(yaw)) * cos(radians(roll)) * sin(radians(pitch))
        r4 = cos(radians(pitch)) * sin(radians(yaw))
        r5 = cos(radians(yaw)) * cos(radians(roll)) + sin(radians(yaw)) * sin(radians(roll)) * sin(radians(pitch))
        r6 = cos(radians(roll)) * sin(radians(yaw)) * sin(radians(pitch)) - cos(radians(yaw)) * sin(radians(roll))
        r7 = - sin(radians(pitch))
        r8 = cos(radians(pitch)) * sin(radians(roll))
        r9 = cos(radians(roll)) * cos(radians(pitch))
        R = np.array([r1, r2, r3, r4, r5, r6, r7, r8, r9]).reshape((3, 3))
        return R

    def generate_point2(self, x, y, z, a, b, c, num, pitch, yaw, roll):

        o = np.array([x, y, z]).reshape((3, 1))
        ponits = []
        # # 底面
        # x1 = []
        # y1 = []
        # x1 = linspace(0 - a, a, num)
        # y1 = linspace(0 - b, b, num)
        # z1 = [0] * num
        # for i in range(num):
        #     for j in range(num):
        #         vector = np.array([x1[i], y1[j], z1[i]]).reshape((3, 1))
        #         ponits.append(vector)
        #
        # # 顶面
        # x2 = []
        # y2 = []
        # z2 = []
        # x2 = linspace(0 - a, a, num)
        # y2 = linspace(0 - b, b, num)
        # z2 = [0 + 2 * c] * num
        # for i in range(num):
        #     for j in range(num):
        #         vector = np.array([x2[i], y2[j], z2[i]]).reshape((3, 1))
        #         ponits.append(vector)

        # 前
        x3 = []
        y3 = []
        z3 = []
        x3 = linspace(0 - a, 0 + a, num)
        y3 = [0 - b] * num
        z3 = linspace(0, 0 + 2 * c, num)
        for i in range(num):
            for j in range(num):
                vector = np.array([x3[i], y3[i], z3[j]]).reshape((3, 1))
                ponits.append(vector)

        # 后
        x4 = []
        y4 = []
        z4 = []
        x4 = linspace(0 - a, 0 + a, num)
        y4 = [0 + b] * num
        z4 = linspace(0, 0 + 2 * c, num)
        for i in range(num):
            for j in range(num):
                vector = np.array([x4[i], y4[i], z4[j]]).reshape((3, 1))
                ponits.append(vector)

        # 左
        x5 = []
        y5 = []
        z5 = []
        x5 = [0 - a] * num
        y5 = linspace(0 - b, 0 + b, num)
        z5 = linspace(0, 0 + 2 * c, num)
        for i in range(num):
            for j in range(num):
                vector = np.array([x5[i], y5[j], z5[i]]).reshape((3, 1))
                ponits.append(vector)

        # 右
        x6 = []
        y6 = []
        z6 = []
        x6 = [0 + a] * num
        y6 = linspace(0 - b, 0 + b, num)
        z6 = linspace(0, 0 + 2 * c, num)
        for i in range(num):
            for j in range(num):
                vector = np.array([x6[i], y6[j], z6[i]]).reshape((3, 1))
                ponits.append(vector)

        R = self.angle_rotation(pitch, yaw, roll)

        for i in range(4 * num * num):
            ponits[i] = np.dot(R, ponits[i])

        for i in range(4 * num * num):
            ponits[i] = ponits[i] + o

        return ponits

    def _min_distance(self, town_map, log, criteria, actor_id):
        """
        Implementation of the metric. This is an example to show how to use the recorder,
        accessed via the log.
        """
        # Get _percentage_route_completed
        # route_completion_test = criteria['RouteCompletionTest']
        # _percentage_route_completed = route_completion_test['_percentage_route_completed']
        # print("RouteCompletionTest:", _percentage_route_completed)

        # Get the ID of the two vehicles
        ego_id = log.get_ego_vehicle_id()
        adv_id = log.get_actor_ids_with_role_name("scenario")[actor_id]  # Could have also used its type_id

        dist_list = []
        frames_list = []

        # Get the frames both actors were alive
        start_ego, end_ego = log.get_actor_alive_frames(ego_id)
        # print(ego_id)

        start_adv, end_adv = log.get_actor_alive_frames(adv_id)
        # print(adv_id)
        start = max(start_ego, start_adv)
        end = min(end_ego, end_adv)

        # Get the distance between the two
        min_dis = 999
        num = 10
        for i in range(start, end):

            ego_transform = log.get_actor_transform(ego_id, i)
            ego_bounding_box = log.get_actor_bounding_box(ego_id)

            a = ego_bounding_box.extent.x
            b = ego_bounding_box.extent.y
            c = ego_bounding_box.extent.z

            x = ego_transform.location.x
            y = ego_transform.location.y
            z = ego_transform.location.z

            pitch = ego_transform.rotation.pitch
            yaw = ego_transform.rotation.yaw
            roll = ego_transform.rotation.roll
            # self.debug.draw_box(carla.BoundingBox(carla.Location(x=x, y=y, z=c), carla.Vector3D(a, b, c)),
            #                     carla.Rotation(pitch, yaw, roll), 0.05, carla.Color(255, 0, 0, 0), 0)
            ego_points = self.generate_point2(x, y, z, a, b, c, num, pitch, yaw, roll)
            # for j in range(num * num * 4):
            #     self.debug.draw_point(
            #         carla.Location(x=ego_points[j][0][0], y=ego_points[j][1][0], z=ego_points[j][2][0]),
            #         size=0.1, life_time=0)

            ############################################################################################################
            adv_transform = log.get_actor_transform(adv_id, i)
            adv_bounding_box = log.get_actor_bounding_box(adv_id)

            a = adv_bounding_box.extent.x
            b = adv_bounding_box.extent.y
            c = adv_bounding_box.extent.z

            x = adv_transform.location.x
            y = adv_transform.location.y
            z = adv_transform.location.z

            pitch = adv_transform.rotation.pitch
            yaw = adv_transform.rotation.yaw
            roll = adv_transform.rotation.roll
            # self.debug.draw_box(carla.BoundingBox(carla.Location(x=x, y=y, z=c), carla.Vector3D(a, b, c)),
            #                     carla.Rotation(pitch, yaw, roll), 0.05, carla.Color(255, 0, 0, 0), 0)
            adv_points = self.generate_point2(x, y, z, a, b, c, num, pitch, yaw, roll)
            # for j in range(num * num * 4):
            #     self.debug.draw_point(
            #         carla.Location(x=adv_points[j][0][0], y=adv_points[j][1][0], z=adv_points[j][2][0]),
            #         size=0.1, life_time=0)

            # part_dis = 999
            # for k in range(4 * num * num):
            #     for j in range(4 * num * num):
            #         dist_v = ego_points[k] - adv_points[j]
            #         dist = math.sqrt(dist_v[0] * dist_v[0] + dist_v[1] * dist_v[1] + dist_v[2] * dist_v[2])
            #         if dist < min_dis:
            #             min_dis = dist
            #         if dist < part_dis:
            #             part_dis = dist
            ego_points_ = np.array(ego_points).reshape(-1, 3)
            adv_points_ = np.array(adv_points).reshape(-1, 3)
            part_dis = cdist(ego_points_, adv_points_).min()
            if part_dis < min_dis:
                min_dis = part_dis
            # print(part_dis)
            dist_list.append(part_dis)
            frames_list.append(i)

        # 打印最短距离
        print("min_dis", min_dis)

        # Use matplotlib to show the results
        # plt.plot(frames_list, dist_list)
        # plt.ylabel('Distance [m]')
        # plt.xlabel('Frame number')
        # plt.title('Distance between the ego vehicle and the adversary over time')
        # plt.show()
        return min_dis

    def run_min_distance(self):
        min_dis = self._min_distance(self.town_map, self.log, self.criteria, self.actor_id)
        return min_dis

    def collision_replay(self, name, start, duration, follow_id):

        self._client.replay_file(name, start, duration, follow_id)

    def collision_info(self, name):
        print(self._client.show_recorder_collisions(name, "v", "a"))

    def client_timeout(self):
        self._client.set_timeout(1.0)


if __name__ == '__main__':
    # E:/ScenarioVerifyDB/ThroughRedLight_4/log/ThroughRedLight_1208.log 1396
    # ThroughRedLight_1494
    # PedestrianCrossing_1/log/PedestrianCrossing_899 LeftVehiclesGoStraight_9
    log = r"E:\results\GA\ThroughRedLight_1\log\ThroughRedLight_988.log"
    criteria = r"E:\results\GA\ThroughRedLight_1\log\ThroughRedLight_988.json"
    actor_id = 0

    ExtractLogInfo = ExtractLogInfo(log, criteria, actor_id)
    # ExtractLogInfo.run_min_distance()
    ExtractLogInfo.collision_info(log)
    ExtractLogInfo.collision_replay(log, 0, 50, 2867655)
    ExtractLogInfo.client_timeout()
