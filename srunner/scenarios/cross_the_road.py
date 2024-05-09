from __future__ import print_function

import math
import py_trees
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorTransformSetter,
                                                                      ActorDestroy,
                                                                      AccelerateToVelocity,
                                                                      HandBrakeVehicle,
                                                                      KeepVelocity,
                                                                      StopVehicle)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (InTriggerDistanceToLocationAlongRoute,
                                                                               InTimeToArrivalToVehicle,
                                                                               DriveDistance,
                                                                               InTriggerDistanceToNextIntersection,
                                                                               InTriggerDistanceToLocation,
                                                                               InTriggerDistanceToVehicle)
from srunner.scenariomanager.timer import TimeOut
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.scenario_helper import get_location_in_distance_from_wp, get_location_in_distance_from_wp_previous
import numpy as np


class CrossTheRoad(BasicScenario):
    """
    This class holds everything required for a simple object crash
    without prior vehicle action involving a vehicle and a cyclist/pedestrian,
    The ego vehicle is driving along the road,
    And encounters a cyclist/pedestrian crossing the road.

    This is a single ego vehicle scenario

    """

    def __init__(self, world, ego_vehicles, config, randomize=False,
                 debug_mode=False, criteria_enable=True, adversary_type=False, timeout=60):
        """
        Setup all relevant parameters and create scenario
        """
        self._wmap = CarlaDataProvider.get_map()
        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)
        self.debug = world.debug
        self._ego_vehicle_distance_driven = 40
        self._num_lane_changes = 1
        self.transform = None
        self.timeout = timeout
        # actor parameters

        # choose model pedestrian (0) or cyclist (1)
        if config._model is not None:
            self._model = config._model
        else:
            self._model = 1
        if self._model == 1:
            self._adversary_type = True  # flag to select either pedestrian (False) or cyclist (True)
        else:
            self._adversary_type = False

        # actor's location
        if config._start_distance is not None:
            self._start_distance = config._start_distance
        else:
            self._start_distance = 20

        # actor's speed
        if config._actor_vel is not None:
            self._other_actor_target_velocity = config._actor_vel
        else:
            self._other_actor_target_velocity = 5

        # actor's brake
        self._other_actor_max_brake = 1.0

        # trigger_distance
        if config._trigger_distance is not None:
            self.dist_to_trigger = config._trigger_distance
        else:
            self.dist_to_trigger = 20

        # when distance between ego and target_location is _trigger_distance, the scenario start
        self.target_location = None

        super(CrossTheRoad, self).__init__("CrossTheRoad",
                                           ego_vehicles,
                                           config,
                                           world,
                                           debug_mode,
                                           criteria_enable=criteria_enable)

    def _calculate_base_transform(self, _start_distance, waypoint):
        # look forward put the actor
        lane_width = waypoint.lane_width
        location, _ = get_location_in_distance_from_wp_previous(waypoint, _start_distance, False)
        waypoint = self._wmap.get_waypoint(location)
        # print(_start_distance)
        # self.debug.draw_point(waypoint.transform.location + carla.Location(z=0.5), size=0.5,
        #                       life_time=0)
        if waypoint.get_right_lane() is None:
            waypoint = waypoint
        else:
            waypoint = waypoint.get_right_lane()
        offset = {"orientation": 270, "position": 90, "z": 0.6, "k": 1.0}
        position_yaw = waypoint.transform.rotation.yaw + offset['position']
        orientation_yaw = waypoint.transform.rotation.yaw + offset['orientation']
        offset_location = carla.Location(
            offset['k'] * lane_width * math.cos(math.radians(position_yaw)),
            offset['k'] * lane_width * math.sin(math.radians(position_yaw)))
        location += offset_location
        location.z += offset['z']
        return carla.Transform(location, carla.Rotation(yaw=orientation_yaw)), orientation_yaw

    def _spawn_adversary(self, transform, orientation_yaw):

        if self._adversary_type is False:
            walker = CarlaDataProvider.request_new_actor('walker.pedestrian.0001', transform)
            walker.set_simulate_physics(enabled=True)
            adversary = walker
        else:
            first_vehicle = CarlaDataProvider.request_new_actor('vehicle.diamondback.century', transform)
            first_vehicle.set_simulate_physics(enabled=True)
            adversary = first_vehicle

        return adversary

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        # cyclist transform
        # We start by getting and waypoint in the closest sidewalk.
        waypoint = self._reference_waypoint
        self.target_location, _ = get_location_in_distance_from_wp(waypoint, self._start_distance)
        # self.debug.draw_point(self.target_location, size=0.5, life_time=0)

        flag = waypoint.lane_id
        # move to the most left lane
        while True:
            if flag * waypoint.lane_id > 0:
                wp_next = waypoint.get_left_lane()
                self._num_lane_changes += 1
            else:
                wp_next = waypoint.get_right_lane()
                self._num_lane_changes += 1
            if wp_next is None or wp_next.lane_type == carla.LaneType.Sidewalk:
                break
            elif wp_next.lane_type == carla.LaneType.Shoulder:
                break
            else:
                waypoint = wp_next

        # self.debug.draw_point(waypoint.transform.location + carla.Location(z=0.5), size=0.5, life_time=0)

        # find position
        self.transform, orientation_yaw = self._calculate_base_transform(self._start_distance, waypoint)
        first_vehicle = self._spawn_adversary(self.transform, orientation_yaw)

        # Now that we found a possible position we just put the vehicle to the underground
        disp_transform = carla.Transform(
            carla.Location(self.transform.location.x,
                           self.transform.location.y,
                           self.transform.location.z - 500),
            self.transform.rotation)
        first_vehicle.set_transform(disp_transform)

        first_vehicle.set_simulate_physics(enabled=True)

        self.other_actors.append(first_vehicle)

    def _create_behavior(self):
        """
        After invoking this scenario, cyclist will wait for the user
        controlled vehicle to enter trigger distance location,
        the cyclist starts crossing the road once the condition meets,
        then after 60 seconds, a timeout stops the scenario
        """

        root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="OccludedObjectCrossing")
        lane_width = self._reference_waypoint.lane_width
        lane_width = 1.25 * lane_width * self._num_lane_changes

        start_condition = InTriggerDistanceToLocation(self.ego_vehicles[0],
                                                      self.target_location, self.dist_to_trigger)

        actor_remove = ActorDestroy(self.other_actors[0],
                                    name="Destroying walker")

        end_condition = DriveDistance(self.ego_vehicles[0],
                                      self._ego_vehicle_distance_driven,
                                      name="End condition ego drive distance")

        scenario_sequence = py_trees.composites.Sequence()
        scenario_sequence.add_child(ActorTransformSetter(self.other_actors[0], self.transform,
                                                         name='TransformSetterTS3walker', physics=False))
        scenario_sequence.add_child(start_condition)
        scenario_sequence.add_child(
            KeepVelocity(self.other_actors[0], self._other_actor_target_velocity, distance=lane_width,
                         _avoid_collision=True))
        # scenario_sequence.add_child(keep_velocity_other)
        # scenario_sequence.add_child(actor_stop_crossed_lane)
        scenario_sequence.add_child(actor_remove)
        scenario_sequence.add_child(end_condition)

        root.add_child(scenario_sequence)
        return root

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        collision_criterion = CollisionTest(self.ego_vehicles[0])
        criteria.append(collision_criterion)

        return criteria

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()
