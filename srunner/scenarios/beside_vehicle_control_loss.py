from __future__ import print_function

import sys

from srunner.tools.scenario_helper import get_waypoint_in_distance, get_location_in_distance_from_wp_previous

from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (InTriggerDistanceToLocation,
                                                                               InTriggerDistanceToNextIntersection,
                                                                               DriveDistance)

from srunner.tools.scenario_helper import (get_crossing_point,
                                           get_geometric_linear_intersection,
                                           generate_target_waypoint_list)
import random

from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import InTriggerDistanceToVehicle, DriveDistance

from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorTransformSetter,
                                                                      ActorDestroy,
                                                                      StopVehicle,
                                                                      SyncArrival,
                                                                      WaypointFollower, AddNoiseToVehicle, Idle)

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


class BesideVehicleControlLoss(BasicScenario):
    """
       BesideVehicleControlLoss scenario:
       Ego is driving straight along the road, and the vehicle beside control loss
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=600):
        self.timeout = timeout

        # add noise
        if config._new_steer_noise is not None:
            self._new_steer_noise = config._new_steer_noise
        else:
            self._new_steer_noise = 0.05

        if config._new_throttle_noise is not None:
            self._new_throttle_noise = config._new_throttle_noise
        else:
            self._new_throttle_noise = 0.2

        # actor parameters
        # actor's speed
        if config._actor_vel is not None:
            self._velocity = config._actor_vel
        else:
            self._velocity = 10

        # the distance between ego and actor are at a certain distance to trigger the behavior
        if config._trigger_distance is not None:
            self._trigger_distance = config._trigger_distance
        else:
            self._trigger_distance = 15

        # Initial position of the actor
        if config._start_distance is not None:
            self._start_distance = config._start_distance
        else:
            self._start_distance = 30

        # actor's brake
        if config._brake is not None:
            self.brake = config._brake
        else:
            self.brake = 0.5

        self.max_brake = 1.0
        #
        self.transform = None
        self._config = config
        self._wmap = CarlaDataProvider.get_map()
        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)
        # Ego drives the last distance to end the scene
        self.vehicle_distance_driven = 20

        super(BesideVehicleControlLoss, self).__init__("BesideVehicleControlLoss",
                                                       ego_vehicles,
                                                       config,
                                                       world,
                                                       debug_mode,
                                                       criteria_enable=criteria_enable)

        if randomize:
            self._velocity = random.randint(20, 60)
            self._trigger_distance = random.randint(10, 40)

    def _calculate_base_transform(self, _start_distance, waypoint):
        """
        Calculate the transform of the actor
        :param float _start_distance: Initial position of the actor
        :param carla.waypoint waypoint: Position of the reference object
        :return: carla.Transform, carla.Rotation.yaw
        """
        lane_width = waypoint.lane_width
        # If the vehicle can be placed on the left, put it on the left, otherwise put it on the right
        if waypoint.get_right_lane() is None or waypoint.get_right_lane().lane_type == carla.LaneType.Sidewalk:
            waypoint = waypoint.get_left_lane()
        elif waypoint.get_right_lane().lane_type == carla.LaneType.Shoulder or waypoint.get_right_lane().lane_type == carla.LaneType.Parking:
            waypoint = waypoint.get_left_lane()
        elif waypoint.get_right_lane().lane_type == carla.LaneType.Bidirectional:
            waypoint = waypoint.get_left_lane()
        else:
            waypoint = waypoint.get_right_lane()

        # put forward_ start_ distance m
        location, _ = get_location_in_distance_from_wp(waypoint, _start_distance, False)
        waypoint = self._wmap.get_waypoint(location)

        # self.debug.draw_point((waypoint.previous(20.0)[-1]).transform.location + carla.Location(z=0.5), size=0.5,
        #                       life_time=0)
        offset = {"orientation": 0, "position": 0, "z": 0.6, "k": 1.0}
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
        self.transform, _ = self._calculate_base_transform(self._start_distance, waypoint)
        first_vehicle_transform = carla.Transform(
            carla.Location(self.transform.location.x,
                           self.transform.location.y,
                           self.transform.location.z),
            self.transform.rotation)
        first_vehicle = CarlaDataProvider.request_new_actor('vehicle.lincoln.mkz2017', first_vehicle_transform)
        first_vehicle.set_simulate_physics(True)
        self.other_actors.append(first_vehicle)

    def _create_behavior(self):
        # car_visible debris_visible
        car_visible = ActorTransformSetter(self.other_actors[0], self.transform)

        trigger_distance = InTriggerDistanceToVehicle(self.ego_vehicles[0], self.other_actors[0],
                                                      self._trigger_distance)
        just_drive = py_trees.composites.Parallel(
            "DrivingStraight", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        keep_v = WaypointFollower(self.other_actors[0], self._velocity)
        just_drive.add_child(keep_v)
        just_drive.add_child(trigger_distance)

        # control_loss
        control_loss = py_trees.composites.Parallel(
            "DrivingStraight", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)
        control_loss.add_child(Idle(3))
        control_loss.add_child(AddNoiseToVehicle(self.other_actors[0], self._new_steer_noise, self._new_throttle_noise))

        # end condition
        end_condition = py_trees.composites.Parallel("Waiting for end position",
                                                     policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        end_condition.add_child(DriveDistance(self.ego_vehicles[0], self.vehicle_distance_driven))

        # build tree
        root = py_trees.composites.Sequence("Behavior")
        root.add_child(car_visible)
        # root.add_child(debris_visible)
        # root.add_child(obstacle_visible)
        root.add_child(just_drive)
        root.add_child(control_loss)
        root.add_child(TimeOut(5))
        root.add_child(ActorDestroy(self.other_actors[0]))
        root.add_child(end_condition)

        return root

    def _create_test_criteria(self):
        """
        A list of all test criteria is created, which is later used in the parallel behavior tree.
        """
        criteria = []

        collision_criterion = CollisionTest(self.ego_vehicles[0])

        criteria.append(collision_criterion)

        return criteria

    def __del__(self):
        """
        Remove all actors after deletion.
        """
        self.remove_all_actors()
