from __future__ import print_function

import sys

from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (InTriggerDistanceToVehicle,
                                                                               InTriggerDistanceToNextIntersection,
                                                                               DriveDistance,
                                                                               StandStill)

from srunner.tools.scenario_helper import get_waypoint_in_distance, get_location_in_distance_from_wp_previous

from agents.navigation.local_planner import RoadOption

from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest, DrivenDistanceTest, MaxVelocityTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (InTriggerDistanceToLocation,
                                                                               InTriggerDistanceToNextIntersection,
                                                                               DriveDistance)

from srunner.tools.scenario_helper import (get_crossing_point,
                                           get_geometric_linear_intersection,
                                           generate_target_waypoint_list)
import random
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorTransformSetter,
                                                                      LaneChange,
                                                                      WaypointFollower,
                                                                      AccelerateToCatchUp)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import InTriggerDistanceToVehicle, DriveDistance
from srunner.scenarios.basic_scenario import BasicScenario

from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorTransformSetter,
                                                                      ActorDestroy,
                                                                      SyncArrival,
                                                                      KeepVelocity,
                                                                      StopVehicle)

from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorTransformSetter,
                                                                      ActorDestroy,
                                                                      StopVehicle,
                                                                      SyncArrival,
                                                                      WaypointFollower)

from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import DriveDistance, InTriggerDistanceToLocation

from srunner.tools.scenario_helper import (get_geometric_linear_intersection,
                                           get_crossing_point,
                                           generate_target_waypoint)
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
                                                                               DriveDistance)
from srunner.scenariomanager.timer import TimeOut
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.scenario_helper import get_location_in_distance_from_wp


class CrossingBehindVehicles(BasicScenario):
    """
    CrossingBehindVehicles scenario:
    Ego vehicle is driving straight on the road
    And encounters a cyclist/pedestrian crossing the road.
    """

    def __init__(self, world, ego_vehicles, config, randomize=False,
                 debug_mode=False, criteria_enable=True, adversary_type=False, timeout=60):
        """
        Setup all relevant parameters and create scenario
        """
        self._wmap = CarlaDataProvider.get_map()

        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)
        # end condition
        self._ego_vehicle_distance_driven = 40
        self._adversary_type = adversary_type  # flag to select either pedestrian (False) or cyclist (True)
        self._num_lane_changes = 1
        self.transform = None
        self.timeout = timeout
        self._trigger_location = config.trigger_points[0].location
        self.debug = world.debug

        # actor parameters
        # Initial position of the actor
        if config._start_distance is not None:
            self._start_distance = config._start_distance
        else:
            self._start_distance = 30

        # actor's speed
        if config._actor_vel is not None:
            self._other_actor_target_velocity = config._actor_vel
        else:
            self._other_actor_target_velocity = 1

        # wait for ego vehicle to enter trigger distance region,
        # the cyclist starts crossing the road
        if config._trigger_distance is not None:
            self.dist_to_trigger = config._trigger_distance
        else:
            self.dist_to_trigger = 15

        # actor's brake
        self._other_actor_max_brake = 1.0

        # obstacle parameters
        # Initial position of the obstacle1
        self._start_distance_obstacle = self._start_distance - 10
        self.transform_obstacle = None

        # Initial position of the obstacle2
        self._start_distance_obstacle2 = self._start_distance - 5
        self.transform_obstacle2 = None

        super(CrossingBehindVehicles, self).__init__("CrossingBehindVehicles",
                                                     ego_vehicles,
                                                     config,
                                                     world,
                                                     debug_mode,
                                                     criteria_enable=criteria_enable)

    def _calculate_base_transform(self, _start_distance, waypoint):
        """
        Calculate the transform of the actor
        :param (float) _start_distance: Initial position of the actor
        :param (carla.waypoint) waypoint: Position of the reference object
        :return: carla.Transform, carla.Rotation.yaw
        """
        if waypoint.lane_type == carla.LaneType.Parking:
            offset = {"orientation": 270, "position": 0, "z": 0.6, "k": 1.0}
        else:
            offset = {"orientation": 270, "position": 90, "z": 0.6, "k": 2.0}
        lane_width = waypoint.lane_width
        location, _ = get_location_in_distance_from_wp(waypoint, _start_distance, False)
        waypoint = self._wmap.get_waypoint(location)
        # self.debug.draw_point(waypoint.transform.location + carla.Location(z=0.5), size=0.5, life_time=0)
        position_yaw = waypoint.transform.rotation.yaw + offset['position']
        orientation_yaw = waypoint.transform.rotation.yaw + offset['orientation']
        offset_location = carla.Location(
            offset['k'] * lane_width * math.cos(math.radians(position_yaw)),
            offset['k'] * lane_width * math.sin(math.radians(position_yaw)))
        location += offset_location
        location.z = self._trigger_location.z + offset['z']
        return carla.Transform(location, carla.Rotation(yaw=orientation_yaw)), orientation_yaw

    def _calculate_obstacle_transform(self, _start_distance, waypoint):
        """
        Calculate the transform of the actor
        :param (float) _start_distance: Initial position of the actor
        :param (carla.waypoint) waypoint: Position of the reference object
        :return: carla.Transform, carla.Rotation.yaw
        """
        # if waypoint.get_right_lane() is None:
        #     waypoint = waypoint
        # else:
        #     waypoint = waypoint.get_right_lane()
        # self.debug.draw_point(waypoint.transform.location + carla.Location(z=0.5), size=0.5, life_time=0)
        if waypoint.lane_type == carla.LaneType.Parking:
            offset = {"orientation": 0, "position": 0, "z": 0, "k": 1.0}
        else:
            offset = {"orientation": 0, "position": 90, "z": 0, "k": 2.0}
        lane_width = waypoint.lane_width
        location, _ = get_location_in_distance_from_wp(waypoint, _start_distance, False)
        waypoint = self._wmap.get_waypoint(location)

        position_yaw = waypoint.transform.rotation.yaw + offset['position']
        orientation_yaw = waypoint.transform.rotation.yaw + offset['orientation']
        offset_location = carla.Location(
            offset['k'] * lane_width * math.cos(math.radians(position_yaw)),
            offset['k'] * lane_width * math.sin(math.radians(position_yaw)))
        location += offset_location
        location.z = self._trigger_location.z + offset['z']
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
        waypoint = self._reference_waypoint
        while True:
            wp_next = waypoint.get_right_lane()
            self._num_lane_changes += 1
            if wp_next is None or wp_next.lane_type == carla.LaneType.Sidewalk:
                break
            elif wp_next.lane_type == carla.LaneType.Shoulder or wp_next.lane_type == carla.LaneType.Parking:
                # Filter Parking considered as Shoulder
                waypoint = wp_next
                break
            else:
                waypoint = wp_next
        # self.debug.draw_point(waypoint.transform.location + carla.Location(z=0.5), size=0.5, life_time=0)
        self.transform, orientation_yaw = self._calculate_base_transform(self._start_distance, waypoint)
        first_vehicle = self._spawn_adversary(self.transform, orientation_yaw)
        disp_transform = carla.Transform(
            carla.Location(self.transform.location.x,
                           self.transform.location.y,
                           self.transform.location.z - 500),
            self.transform.rotation)
        first_vehicle.set_transform(disp_transform)
        first_vehicle.set_simulate_physics(enabled=True)
        self.other_actors.append(first_vehicle)

        # obstacle1
        self.transform_obstacle, _ = self._calculate_obstacle_transform(self._start_distance_obstacle, waypoint)
        obstacle_transform = carla.Transform(
            carla.Location(self.transform_obstacle.location.x,
                           self.transform_obstacle.location.y,
                           self.transform_obstacle.location.z - 500),
            self.transform_obstacle.rotation)
        obstacle = CarlaDataProvider.request_new_actor('vehicle.lincoln.mkz2017', obstacle_transform)
        obstacle.set_simulate_physics(False)
        self.other_actors.append(obstacle)

        # obstacle2
        self.transform_obstacle2, _ = self._calculate_obstacle_transform(self._start_distance_obstacle2, waypoint)
        obstacle_transform2 = carla.Transform(
            carla.Location(self.transform_obstacle2.location.x,
                           self.transform_obstacle2.location.y,
                           self.transform_obstacle2.location.z - 500),
            self.transform_obstacle.rotation)
        obstacle2 = CarlaDataProvider.request_new_actor('vehicle.lincoln.mkz2017', obstacle_transform2)
        obstacle2.set_simulate_physics(False)
        self.other_actors.append(obstacle2)

    def _create_behavior(self):
        """
        Ego vehicle is driving straight on the road
        And encounters a cyclist/pedestrian crossing the road.
        Order of sequence:
        - ActorTransformSetter: spawn cyclist, obstacle at visible transform
        - start_condition: triggered at a certain distance
        - keep_velocity: cyclist cross the road
        - StopVehicle: emergency brake
        - actor_remove: remove the actor
        - end condition: wait for ego drive a certain distance
        """

        root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="CrossingBehindVehicles")
        lane_width = self._reference_waypoint.lane_width
        lane_width = lane_width + (lane_width * self._num_lane_changes)

        # Triggered at a certain distance
        start_condition = InTriggerDistanceToVehicle(self.ego_vehicles[0], self.other_actors[0],
                                                     self.dist_to_trigger)

        # cyclist cross the road
        keep_velocity = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="keep velocity")
        actor_velocity = KeepVelocity(self.other_actors[0],
                                      self._other_actor_target_velocity,
                                      name="walker velocity")
        actor_drive = DriveDistance(self.other_actors[0],
                                    2 * lane_width,
                                    name="walker drive distance")
        keep_velocity.add_child(actor_velocity)
        keep_velocity.add_child(actor_drive)

        # remove the actor
        actor_remove = ActorDestroy(self.other_actors[0],
                                    name="Destroying walker")
        # end_condition
        end_condition = DriveDistance(self.ego_vehicles[0],
                                      self._ego_vehicle_distance_driven,
                                      name="End condition ego drive distance")

        scenario_sequence = py_trees.composites.Sequence()
        scenario_sequence.add_child(ActorTransformSetter(self.other_actors[0], self.transform,
                                                         name='TransformSetterTS3walker', physics=False))
        scenario_sequence.add_child(ActorTransformSetter(self.other_actors[1], self.transform_obstacle,
                                                         name='vehicle.lincoln.mkz2017', physics=False))
        scenario_sequence.add_child(ActorTransformSetter(self.other_actors[2], self.transform_obstacle2,
                                                         name='vehicle.lincoln.mkz2017', physics=False))
        scenario_sequence.add_child(start_condition)
        scenario_sequence.add_child(keep_velocity)
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
