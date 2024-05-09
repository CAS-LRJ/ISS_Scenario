from __future__ import print_function

import random

from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import InTriggerDistanceToVehicle, DriveDistance

from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorTransformSetter,
                                                                      ActorDestroy,
                                                                      StopVehicle,
                                                                      SyncArrival,
                                                                      WaypointFollower)
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


class FollowPedestrian(BasicScenario):
    """
    FollowPedestrian scenario:
    ego is driving along the road, while there is a Pedestrian ahead
    """
    timeout = 1200

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=600):
        #
        self._config = config
        self._wmap = CarlaDataProvider.get_map()
        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)
        # end condition
        self.vehicle_distance_driven = 20
        self.timeout = timeout
        # actor parameters
        # choose cyclist(1)/pedestrian(0)
        if config._model is not None:
            self._model = config._model
        else:
            self._model = 1
        # actor's speed
        if config._actor_vel is not None:
            self._velocity = config._actor_vel
        else:
            self._velocity = 3

        # _trigger_distance wait for the ego vehicle to enter trigger distance region
        if config._trigger_distance is not None:
            self._trigger_distance = config._trigger_distance
        else:
            self._trigger_distance = 20

        # Initial position of the actor
        if config._start_distance is not None:
            self._start_distance = config._start_distance
        else:
            self._start_distance = 20

        super(FollowPedestrian, self).__init__("FollowPedestrian",
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
        location, _ = get_location_in_distance_from_wp(self._reference_waypoint, self._start_distance)
        waypoint = self._wmap.get_waypoint(location)
        lane_width = waypoint.lane_width
        offset = {"orientation": 0, "position": 10, "z": 0.6, "k": 1.0}
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
        if self._model == 1:
            pedestrian = CarlaDataProvider.request_new_actor('vehicle.diamondback.century', first_vehicle_transform)
        else:
            pedestrian = CarlaDataProvider.request_new_actor('walker.pedestrian.0001', first_vehicle_transform)
        pedestrian.set_simulate_physics(True)
        self.other_actors.append(pedestrian)

    def _create_behavior(self):
        """
        cyclist will wait for the ego vehicle to enter trigger distance region,
        the cyclist starts crossing the road once the condition meets,
        Order of sequence:
        - car_visible: spawn pedestrian at a visible transform
        - trigger_distance: ego vehicle enter trigger distance region
        - just_drive: pedestrian walk
        - StopVehicle: pedestrian stop
        - TimeOut: wait for secs
        - ActorDestroy: remove the actor
        - end_condition: drive for a defined distance
        """
        # car_visible
        behaviour = py_trees.composites.Sequence("Sequence Behavior")
        car_visible = ActorTransformSetter(self.other_actors[0], self.transform)
        behaviour.add_child(car_visible)

        # wait for ego vehicle enter trigger distance region
        trigger_distance = InTriggerDistanceToVehicle(
            self.other_actors[0], self.ego_vehicles[0], self._trigger_distance)

        # walk
        just_drive = py_trees.composites.Parallel(
            "DrivingStraight", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        keep_v = WaypointFollower(self.other_actors[0], self._velocity)
        wait = DriveDistance(self.other_actors[0], 30)
        just_drive.add_child(keep_v)
        just_drive.add_child(wait)

        behaviour.add_child(trigger_distance)
        behaviour.add_child(just_drive)
        # stop
        behaviour.add_child(StopVehicle(self.other_actors[0], 1.0))
        behaviour.add_child(TimeOut(5))

        # end condition
        end_condition = py_trees.composites.Parallel("Waiting for end position",
                                                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        end_condition.add_child(DriveDistance(self.ego_vehicles[0], self.vehicle_distance_driven))

        # build tree
        root = py_trees.composites.Sequence("Behavior", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        root.add_child(behaviour)
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
