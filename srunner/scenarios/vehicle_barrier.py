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
                                                                               DriveDistance)
from srunner.scenariomanager.timer import TimeOut
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.scenario_helper import get_location_in_distance_from_wp


class VehicleBarrier(BasicScenario):
    """
    VehicleBarrier scenario:
    The ego vehicle is driving on the road, and another car in front occupies part of the lane
    The ego vehicle may need to brake to avoid a collision
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=60):
        """
        Setup all relevant parameters and create scenario
        """
        self._wmap = CarlaDataProvider.get_map()
        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)
        # ego vehicle parameters
        self._ego_vehicle_distance_driven = 40
        # Timeout of scenario in seconds
        self.timeout = timeout
        # Other vehicle location
        if config._start_distance is not None:
            self._start_distance = config._start_distance
        else:
            self._start_distance = 30

        super(VehicleBarrier, self).__init__("VehicleBarrier",
                                             ego_vehicles,
                                             config,
                                             world,
                                             debug_mode,
                                             criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        # Look for a position in front of the adjacent lane to place the other vehicle
        front_location, _ = get_location_in_distance_from_wp(self._reference_waypoint, self._start_distance, False)
        front_waypoint = self._wmap.get_waypoint(front_location)
        flag = front_waypoint.lane_id
        waypoint = front_waypoint.get_left_lane()
        if waypoint.lane_id * flag > 0:
            offset = {"orientation": 20, "position": 90, "z": 0, "k": 0.5}
        else:
            offset = {"orientation": -30, "position": -90, "z": 0, "k": 0.5}
        lane_width = waypoint.lane_width
        location = waypoint.transform.location
        position_yaw = waypoint.transform.rotation.yaw + offset['position']
        orientation_yaw = waypoint.transform.rotation.yaw + offset['orientation']
        offset_location = carla.Location(
            0,
            offset['k'] * lane_width * math.sin(math.radians(position_yaw)))
        location += offset_location
        location.z += offset['z']
        self.transform = carla.Transform(location, carla.Rotation(yaw=orientation_yaw))
        obstacle_transform = carla.Transform(
            carla.Location(self.transform.location.x,
                           self.transform.location.y,
                           self.transform.location.z - 500),
            self.transform.rotation)
        obstacle = CarlaDataProvider.request_new_actor('vehicle.lincoln.mkz2017', obstacle_transform)
        obstacle.set_simulate_physics(True)
        self.other_actors.append(obstacle)

    def _create_behavior(self):
        """
        The ego vehicle is driving on the road, and another car in front occupies part of the lane
        The ego vehicle may need to brake to avoid a collision
        Order of sequence:
        - ActorTransformSetter: spawn car at a visible transform
        - actor_stand: Other actor stop in the middle of the road and wait
        - actor_removed: remove the other actor
        - end_condition: drive for a defined distance
        """
        # leaf nodes
        actor_stand = TimeOut(60)
        actor_removed = ActorDestroy(self.other_actors[0])
        end_condition = DriveDistance(self.ego_vehicles[0], self._ego_vehicle_distance_driven)

        # non leaf nodes
        root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        scenario_sequence = py_trees.composites.Sequence()

        # building tree
        root.add_child(scenario_sequence)
        scenario_sequence.add_child(ActorTransformSetter(self.other_actors[0], self.transform))
        scenario_sequence.add_child(actor_stand)
        scenario_sequence.add_child(actor_removed)
        scenario_sequence.add_child(end_condition)

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
