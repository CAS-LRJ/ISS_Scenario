#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

from __future__ import print_function

import math
import sys
import py_trees
import carla

from agents.navigation.local_planner import RoadOption

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorTransformSetter,
                                                                      ActorDestroy,
                                                                      StopVehicle,
                                                                      SyncArrival,
                                                                      WaypointFollower,
                                                                      SetTrafficLightGreen,
                                                                      SetTrafficLightRed)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import DriveDistance, InTriggerDistanceToLocation
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.scenario_helper import (get_geometric_linear_intersection,
                                           get_crossing_point,
                                           generate_target_waypoint,
                                           get_location_in_distance_from_wp)


class SignalizedJunctionRightTurn(BasicScenario):
    """
    SignalizedJunctionRightTurn scenario:
    ego is turning right at a signalized intersection,
    while other actor coming straight from left intersection
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=80):
        """
        Setup all relevant parameters and create scenario
        """
        self._wmap = CarlaDataProvider.get_map()
        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)
        self.debug = world.debug
        self._brake_value = 0.5
        self._ego_distance = 40
        self._other_actor_transform = None

        # Travel speed of the actor
        if config._actor_vel is not None:
            self._target_vel = config._actor_vel
        else:
            self._target_vel = 10

        # Initial position of the actor
        if config._start_distance is not None:
            self._start_distance = config._start_distance
        else:
            self._start_distance = 15

        # Trigger_distance from ego to intersection
        if config._trigger_distance is not None:
            self.trigger_distance = config._trigger_distance
        else:
            self.trigger_distance = 10

        # Timeout of scenario in seconds
        self.timeout = timeout

        super(SignalizedJunctionRightTurn, self).__init__("SignalizedJunctionRightTurn",
                                                          ego_vehicles,
                                                          config,
                                                          world,
                                                          debug_mode,
                                                          criteria_enable=criteria_enable)

    def _calculate_base_transform(self, _start_distance, waypoint):
        """
        Calculate the transform of the actor
        :param float _start_distance: Initial position of the actor
        :param carla.waypoint waypoint: Position of the reference object
        :return: carla.Transform, carla.Rotation.yaw
        """
        lane_width = waypoint.lane_width
        # Get the waypoint of the left intersection
        flag = waypoint.lane_id
        if waypoint.get_left_lane() is not None and waypoint.get_left_lane().lane_id * flag > 0:
            waypoint = waypoint.get_left_lane()
        waypoint1 = generate_target_waypoint(waypoint, turn=-1)
        location, _ = get_location_in_distance_from_wp(waypoint1, self._start_distance, False)
        waypoint = self._wmap.get_waypoint(location)

        # Move to the right lane
        flag = waypoint.lane_id
        while True:
            if flag * waypoint.lane_id > 0:
                wp_next = waypoint.get_left_lane()
            else:
                break

            if wp_next is None or wp_next.lane_type == carla.LaneType.Sidewalk:
                break
            elif wp_next.lane_type == carla.LaneType.Shoulder or wp_next.lane_type == carla.LaneType.Parking:
                break
            elif wp_next.lane_type == carla.LaneType.Bidirectional:
                waypoint = wp_next.get_right_lane()
            else:
                waypoint = wp_next

        # Move to the right most lane
        while True:
            if waypoint.get_right_lane() is None or waypoint.get_right_lane().lane_type == carla.LaneType.Sidewalk:
                waypoint = waypoint
                break
            elif waypoint.get_right_lane().lane_type == carla.LaneType.Shoulder or waypoint.get_right_lane().lane_type == carla.LaneType.Parking:
                waypoint = waypoint
                break
            else:
                waypoint = waypoint.get_right_lane()

        location = waypoint.transform.location
        # self.debug.draw_point(waypoint.transform.location + carla.Location(z=0.5), size=0.5, life_time=0)
        offset = {"orientation": 0, "position": 0, "z": 0, "k": 1.0}
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
        waypoint = self._reference_waypoint
        self._other_actor_transform, orientation_yaw = self._calculate_base_transform(self._start_distance, waypoint)
        first_vehicle_transform = carla.Transform(
            carla.Location(self._other_actor_transform.location.x,
                           self._other_actor_transform.location.y,
                           self._other_actor_transform.location.z),
            self._other_actor_transform.rotation)
        first_vehicle = CarlaDataProvider.request_new_actor('vehicle.lincoln.mkz2017', first_vehicle_transform)
        first_vehicle.set_simulate_physics(False)
        self.other_actors.append(first_vehicle)

    def _create_behavior(self):
        """
        ego is turning right at a signalized intersection,
        while other actor coming straight from left intersection
        Order of sequence:
        - ActorTransformSetter: spawn car at a visible transform
        - SetTrafficLightGreen: the light in front of the ego is green
        - SetTrafficLightRed: the light in front of the actor is red
        - sync_arrival_parallel: Two vehicles driving towards the collision point at the same time
        - move_actor_parallel:  drive according to the plan
        - stop: stop the actor
        - end_condition: drive for a defined distance
        - ActorDestroy: remove the actor
        """
        location_of_collision_dynamic = get_geometric_linear_intersection(self.ego_vehicles[0], self.other_actors[0])
        crossing_point_dynamic = get_crossing_point(self.other_actors[0])
        sync_arrival = SyncArrival(
            self.other_actors[0], self.ego_vehicles[0], location_of_collision_dynamic)
        sync_arrival_stop = InTriggerDistanceToLocation(self.other_actors[0], crossing_point_dynamic, 4)
        sync_arrival_parallel = py_trees.composites.Parallel(
            "Synchronize arrival times",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        sync_arrival_parallel.add_child(sync_arrival)
        sync_arrival_parallel.add_child(sync_arrival_stop)

        # Find the location in front of the intersection
        location, _ = get_location_in_distance_from_wp(self._reference_waypoint, 150)

        # start condition
        start_condition = InTriggerDistanceToLocation(
            self.ego_vehicles[0],
            location,
            self.trigger_distance,
            name="Waiting for start position")

        # Selecting straight path at intersection
        target_waypoint = generate_target_waypoint(
            CarlaDataProvider.get_map().get_waypoint(self.other_actors[0].get_location()), 0)

        # Generating waypoint list till next intersection
        plan = []
        wp_choice = target_waypoint.next(1.0)
        while not wp_choice[0].is_intersection:
            target_waypoint = wp_choice[0]
            plan.append((target_waypoint, RoadOption.LANEFOLLOW))
            wp_choice = target_waypoint.next(1.0)

        # drive according to the plan
        move_actor = WaypointFollower(self.other_actors[0], self._target_vel, plan=plan, avoid_collision=True)
        waypoint_follower_end = InTriggerDistanceToLocation(
            self.other_actors[0], plan[-1][0].transform.location, 10)
        move_actor_parallel = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        move_actor_parallel.add_child(move_actor)
        move_actor_parallel.add_child(waypoint_follower_end)

        # stop other actor
        stop = StopVehicle(self.other_actors[0], self._brake_value)

        # end condition
        end_condition = DriveDistance(self.ego_vehicles[0], self._ego_distance)

        # Behavior tree
        sequence = py_trees.composites.Sequence()
        sequence.add_child(ActorTransformSetter(self.other_actors[0], self._other_actor_transform))
        sequence.add_child(SetTrafficLightGreen(self.other_actors[0], self.ego_vehicles[0]))
        sequence.add_child(SetTrafficLightRed(self.other_actors[0]))
        sequence.add_child(start_condition)
        sequence.add_child(move_actor_parallel)
        sequence.add_child(stop)
        sequence.add_child(ActorDestroy(self.other_actors[0]))
        sequence.add_child(end_condition)

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
        self._traffic_light = None
        self.remove_all_actors()
