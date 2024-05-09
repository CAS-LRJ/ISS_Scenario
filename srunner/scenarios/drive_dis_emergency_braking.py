from __future__ import print_function

import math
import py_trees
import carla
import random

from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import InTriggerDistanceToVehicle, DriveDistance, \
    InTriggerDistanceToNextIntersection

from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorTransformSetter,
                                                                      ActorDestroy,
                                                                      StopVehicle,
                                                                      SyncArrival,
                                                                      WaypointFollower)

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


class DriveDisEmergencyBraking(BasicScenario):
    """
    EmergencyBraking scenario:
    ego vehicle follows a leading car driving down a given road.
    when two vehicles are at a certain distance,
    the leading car has to slow down and finally stop.
    """
    timeout = 1200

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=600):
        self.timeout = timeout
        self._config = config
        self._wmap = CarlaDataProvider.get_map()
        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)
        # end condition
        self.vehicle_distance_driven = 20

        # actor parameters
        # actor's speed
        if config._actor_vel is not None:
            self._velocity = config._actor_vel
        else:
            self._velocity = 10

        # actor slow down when two vehicles are at a certain distance
        if config._trigger_distance is not None:
            self._trigger_distance = config._trigger_distance
        else:
            self._trigger_distance = 30

        # Initial position of the actor
        if config._start_distance is not None:
            self._start_distance = config._start_distance
        else:
            self._start_distance = 40
        # drive dis
        if config._start_distance2 is not None:
            self._drive_distance = config._start_distance2
        else:
            self._drive_distance = 30

        # actor's break
        if config._brake is not None:
            self.brake = config._brake
        else:
            self.brake = 0.5

        self.max_brake = 1.0

        super(DriveDisEmergencyBraking, self).__init__("DriveDisEmergencyBraking",
                                                       ego_vehicles,
                                                       config,
                                                       world,
                                                       debug_mode,
                                                       criteria_enable=criteria_enable)

        if randomize:
            self._velocity = random.randint(20, 60)
            self._trigger_distance = random.randint(10, 40)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        # generate the front vehicle
        location, _ = get_location_in_distance_from_wp(self._reference_waypoint, self._start_distance)
        waypoint = self._wmap.get_waypoint(location)
        lane_width = waypoint.lane_width
        offset = {"orientation": 0, "position": 0, "z": 0.6, "k": 0}
        position_yaw = waypoint.transform.rotation.yaw + offset['position']
        orientation_yaw = waypoint.transform.rotation.yaw + offset['orientation']
        offset_location = carla.Location(
            offset['k'] * lane_width * math.cos(math.radians(position_yaw)),
            offset['k'] * lane_width * math.sin(math.radians(position_yaw)))
        location += offset_location
        location.z += offset['z']
        self.transform = carla.Transform(location, carla.Rotation(yaw=orientation_yaw))
        first_vehicle_transform = carla.Transform(
            carla.Location(self.transform.location.x,
                           self.transform.location.y,
                           self.transform.location.z - 500),
            self.transform.rotation)
        pedestrian = CarlaDataProvider.request_new_actor('vehicle.lincoln.mkz2017', first_vehicle_transform)
        pedestrian.set_simulate_physics(True)
        self.other_actors.append(pedestrian)

    def _create_behavior(self):
        """
        ego vehicle follows a leading car driving down a given road.
        when two vehicles are at a certain distance,
        the leading car has to slow down and finally stop.
        Order of sequence:
        - ActorTransformSetter: spawn car at a visible transform
        - just_drive: actor drive along the road, the scenario trigger in a certain distance
        - StopVehicle: actor slow down and stop
        - TimeOut: wait for remove actor
        - ActorDestroy: remove the actor
        - end_condition: wait ego vehicle drive a specific distance
        """
        # car_visible
        behaviour = py_trees.composites.Sequence("Sequence Behavior")
        car_visible = ActorTransformSetter(self.other_actors[0], self.transform)
        behaviour.add_child(car_visible)

        # trigger in a certain distance
        trigger_distance = InTriggerDistanceToVehicle(
            self.other_actors[0], self.ego_vehicles[0], self._trigger_distance)

        # drive along the road
        just_drive = py_trees.composites.Parallel(
            "DrivingStraight", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        keep_v = WaypointFollower(self.other_actors[0], self._velocity)
        just_drive.add_child(keep_v)
        just_drive.add_child(DriveDistance(self.other_actors[0], self._drive_distance))
        just_drive.add_child(InTriggerDistanceToNextIntersection(self.other_actors[0], 10))

        behaviour.add_child(just_drive)
        # stop for 8 secs
        behaviour.add_child(StopVehicle(self.other_actors[0], self.brake))
        behaviour.add_child(TimeOut(8))
        # end condition
        end_condition = py_trees.composites.Parallel("Waiting for end position",
                                                     policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        end_condition.add_child(DriveDistance(self.ego_vehicles[0], self.vehicle_distance_driven))
        # build tree
        root = py_trees.composites.Sequence("Behavior")
        root.add_child(behaviour)
        # remove actor
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
