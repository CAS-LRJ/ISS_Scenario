from __future__ import print_function

import math

import py_trees

import carla

from agents.navigation.local_planner import RoadOption
from srunner.tools.scenario_helper import (get_waypoint_in_distance,
                                           get_location_in_distance_from_wp,
                                           generate_target_waypoint,
                                           get_crossing_point, return_target_waypoint_list,
                                           get_location_in_distance_from_wp_previous)
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorTransformSetter,
                                                                      LaneChange,
                                                                      WaypointFollower,
                                                                      AccelerateToCatchUp,
                                                                      SetTrafficLightGreen,
                                                                      SetTrafficLightRed,
                                                                      ActorDestroy, StopVehicle, KeepVelocity)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import DriveDistance, InTriggerDistanceToLocation


class LRBikeCrossIntersectionObliquely(BasicScenario):
    """
    LRBikeCrossIntersectionObliquely scenario:
    The ego vehicle at an intersection,
    while left and right bike_cross_intersection_obliquely
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=60):
        """
        Setup all relevant parameters and create scenario
        """
        self._wmap = CarlaDataProvider.get_map()
        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)
        self.debug = world.debug
        self._other_actor_transform = None
        self._other_actor_transform2 = None
        self._other_actor_transform3 = None

        # steer
        if config._new_steer_noise is not None:
            self._new_steer_noise = config._new_steer_noise
        else:
            self._new_steer_noise = -0.8
        if config._new_steer_noise2 is not None:
            self._new_steer_noise2 = config._new_steer_noise2
        else:
            self._new_steer_noise2 = 0.7


        # Trigger_distance from ego to intersection
        if config._trigger_distance is not None:
            self.trigger_distance = config._trigger_distance
        else:
            self.trigger_distance = 30

        # Travel speed of the actor
        if config._actor_vel is not None:
            self.velocity = config._actor_vel
        else:
            self.velocity = 2

        if config._actor_vel2 is not None:
            self.velocity2 = config._actor_vel2
        else:
            self.velocity2 = 2

        # Timeout of scenario in seconds
        self.timeout = timeout

        super(LRBikeCrossIntersectionObliquely, self).__init__("LRBikeCrossIntersectionObliquely",
                                                               ego_vehicles,
                                                               config,
                                                               world,
                                                               debug_mode,
                                                               criteria_enable=criteria_enable)

    def _calculate_left_pedestrian_transform(self, _start_distance, waypoint):
        """
        Calculate the transform of the actor
        :param float _start_distance: Initial position of the actor
        :param carla.waypoint waypoint: Position of the reference object
        :return: carla.Transform, carla.Rotation.yaw
        """
        lane_width = waypoint.lane_width

        flag = waypoint.lane_id
        while True:
            if flag * waypoint.lane_id > 0:
                wp_next = waypoint.get_left_lane()
                if wp_next.lane_type == carla.LaneType.Bidirectional:
                    if wp_next.lane_id * flag > 0:
                        wp_next = wp_next.get_left_lane()
                    else:
                        wp_next = wp_next.get_right_lane()
            else:
                break

            if wp_next is None or wp_next.lane_type == carla.LaneType.Sidewalk:
                break
            elif wp_next.lane_type == carla.LaneType.Shoulder or wp_next.lane_type == carla.LaneType.Parking:
                break

            else:
                waypoint = wp_next

        # Move to the most right lane
        while True:
            if waypoint.get_right_lane() is None or waypoint.get_right_lane().lane_type == carla.LaneType.Sidewalk:
                waypoint = waypoint
                break
            elif waypoint.get_right_lane().lane_type == carla.LaneType.Shoulder or waypoint.get_right_lane().lane_type == carla.LaneType.Parking:
                waypoint = waypoint
                break
            else:
                waypoint = waypoint.get_right_lane()

        location, _ = get_location_in_distance_from_wp_previous(waypoint, _start_distance)
        waypoint = self._wmap.get_waypoint(location)
        # self.debug.draw_point(waypoint.transform.location + carla.Location(z=0.5), size=0.5, life_time=0)
        offset = {"orientation": -90, "position": 90, "z": 0, "k": 1.0}
        position_yaw = waypoint.transform.rotation.yaw + offset['position']
        orientation_yaw = waypoint.transform.rotation.yaw + offset['orientation']
        offset_location = carla.Location(
            offset['k'] * lane_width * math.cos(math.radians(position_yaw)),
            offset['k'] * lane_width * math.sin(math.radians(position_yaw)))
        location += offset_location
        location.z += offset['z']
        return carla.Transform(location, carla.Rotation(yaw=orientation_yaw)), orientation_yaw

    def _calculate_right_pedestrian_transform(self, _start_distance, waypoint):
        """
        Calculate the transform of the actor
        :param float _start_distance: Initial position of the actor
        :param carla.waypoint waypoint: Position of the reference object
        :return: carla.Transform, carla.Rotation.yaw
        """
        lane_width = waypoint.lane_width

        location, _ = get_location_in_distance_from_wp(waypoint, _start_distance)
        waypoint = self._wmap.get_waypoint(location)
        # self.debug.draw_point(waypoint.transform.location + carla.Location(z=0.5), size=0.5, life_time=0)
        offset = {"orientation": -90, "position": 90, "z": 0, "k": 1.0}
        position_yaw = waypoint.transform.rotation.yaw + offset['position']
        orientation_yaw = waypoint.transform.rotation.yaw + offset['orientation']
        offset_location = carla.Location(
            offset['k'] * lane_width * math.cos(math.radians(position_yaw)),
            offset['k'] * lane_width * math.sin(math.radians(position_yaw)))
        location += offset_location
        location.z += offset['z']
        return carla.Transform(location, carla.Rotation(yaw=orientation_yaw)), orientation_yaw

    def _initialize_actors(self, config):
        """
        Custom initialization
        """

        # left bike
        waypoint = self._reference_waypoint
        self._other_actor_transform, orientation_yaw = self._calculate_left_pedestrian_transform(100, waypoint)
        ped_transform = carla.Transform(
            carla.Location(self._other_actor_transform.location.x,
                           self._other_actor_transform.location.y,
                           self._other_actor_transform.location.z),
            self._other_actor_transform.rotation)
        ped = CarlaDataProvider.request_new_actor('vehicle.bh.crossbike', ped_transform)
        ped.set_simulate_physics(True)
        self.other_actors.append(ped)

        # right bike
        waypoint = self._reference_waypoint
        self._other_actor_transform2, orientation_yaw = self._calculate_right_pedestrian_transform(100, waypoint)
        ped2_transform = carla.Transform(
            carla.Location(self._other_actor_transform2.location.x,
                           self._other_actor_transform2.location.y,
                           self._other_actor_transform2.location.z),
            self._other_actor_transform2.rotation)
        ped2 = CarlaDataProvider.request_new_actor('vehicle.diamondback.century', ped2_transform)
        ped2.set_simulate_physics(True)
        self.other_actors.append(ped2)

    def _create_behavior(self):
        """
        Order of sequence:
        - ActorTransformSetter: spawn car at a visible transform
        - start_condition: drive until in trigger distance to ego_vehicle
        - SetTrafficLightGreen: the light in front of the ego is green
        - SetTrafficLightRed: the light in front of the actor is red
        - continue_driving:  drive according to the plan and stop
        - wait: drive for a defined distance

        """
        # Find the location in front of the intersection
        location, _ = get_location_in_distance_from_wp(self._reference_waypoint, 150)

        # start condition
        start_condition = InTriggerDistanceToLocation(
            self.ego_vehicles[0],
            location,
            self.trigger_distance,
            name="Waiting for start position")

        continue_driving = py_trees.composites.Parallel(
            "ContinueDriving",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        ped_cross = py_trees.composites.Sequence()
        continue_driving_waypoints1 = KeepVelocity(self.other_actors[0], 0.2, duration=2.3,
                                                   _avoid_collision=True, steer=self._new_steer_noise)
        continue_driving_waypoints2 = KeepVelocity(self.other_actors[0], self.velocity, duration=10,
                                                   _avoid_collision=True, steer=0)
        ped_cross.add_child(continue_driving_waypoints1)
        ped_cross.add_child(continue_driving_waypoints2)
        ped_cross.add_child(StopVehicle(self.other_actors[0], 1.0))

        ped_cross2 = py_trees.composites.Sequence()
        continue_driving_waypoints3 = KeepVelocity(self.other_actors[1], 0.2, duration=2.3,
                                                   _avoid_collision=True, steer=self._new_steer_noise2)
        continue_driving_waypoints4 = KeepVelocity(self.other_actors[1], self.velocity2, duration=10,
                                                   _avoid_collision=True, steer=0)
        ped_cross2.add_child(continue_driving_waypoints3)
        ped_cross2.add_child(continue_driving_waypoints4)
        ped_cross2.add_child(StopVehicle(self.other_actors[1], 1.0))

        # Drive according to the plan
        continue_driving.add_child(ped_cross)
        continue_driving.add_child(ped_cross2)

        # wait the ego vehicle drove a specific distance
        wait = DriveDistance(
            self.ego_vehicles[0],
            10,
            name="DriveDistance")

        # Build behavior tree
        sequence = py_trees.composites.Sequence("Sequence Behavior")
        # ped
        sequence.add_child(ActorTransformSetter(self.other_actors[0], self._other_actor_transform))
        sequence.add_child(ActorTransformSetter(self.other_actors[1], self._other_actor_transform2))
        sequence.add_child(start_condition)
        sequence.add_child(SetTrafficLightGreen(self.ego_vehicles[0]))
        sequence.add_child(SetTrafficLightRed(self.other_actors[0]))
        sequence.add_child(continue_driving)
        sequence.add_child(StopVehicle(self.other_actors[1], 1.0))
        sequence.add_child(StopVehicle(self.other_actors[0], 1.0))
        sequence.add_child(ActorDestroy(self.other_actors[0]))
        sequence.add_child(ActorDestroy(self.other_actors[1]))
        sequence.add_child(wait)
        return sequence

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        collision_criteria = CollisionTest(self.ego_vehicles[0])
        criteria.append(collision_criteria)

        return criteria

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()
