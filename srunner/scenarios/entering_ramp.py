from __future__ import print_function

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


class EnteringRamp(BasicScenario):
    timeout = 1200
    """
      Ego is entering the ramp, and other vehicles cut in on the main road
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=600):
        self.timeout = timeout
        # actor parameters
        # actor's speed
        if config._actor_vel is not None:
            self._velocity = config._actor_vel
        else:
            self._velocity = 20

        # when the distance between ego and actor is _trigger_distance, the actor cut in
        if config._trigger_distance is not None:
            self._trigger_distance = config._trigger_distance
        else:
            self._trigger_distance = 20

        # actor's initial position
        if config._start_distance is not None:
            self._start_distance = config._start_distance
        else:
            self._start_distance = 5

        self._config = config
        self._wmap = CarlaDataProvider.get_map()
        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)
        self._other_vehicle_distance_driven = 40

        super(EnteringRamp, self).__init__("EnteringRamp",
                                           ego_vehicles,
                                           config,
                                           world,
                                           debug_mode,
                                           criteria_enable=criteria_enable)

        if randomize:
            self._velocity = random.randint(20, 60)
            self._trigger_distance = random.randint(10, 40)

    def _initialize_actors(self, config):
        # put the front main road vehicle
        location1, _ = get_location_in_distance_from_wp(self._reference_waypoint, self._start_distance)
        waypoint1 = self._wmap.get_waypoint(location1)
        waypoint = waypoint1.get_left_lane()
        lane_width = waypoint.lane_width
        location = waypoint.transform.location

        offset = {"orientation": 0, "position": 0, "z": 0, "k": 0.2}
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
        first_vehicle = CarlaDataProvider.request_new_actor('vehicle.chevrolet.impala', first_vehicle_transform)
        first_vehicle.set_simulate_physics(True)
        self.other_actors.append(first_vehicle)

    def _create_behavior(self):
        # car_visible
        behaviour = py_trees.composites.Sequence("Sequence Behavior")
        car_visible = ActorTransformSetter(self.other_actors[0], self.transform)
        behaviour.add_child(car_visible)

        # trigger in trigger distance
        just_drive = py_trees.composites.Parallel(
            "DrivingStraight", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        keepv = WaypointFollower(self.other_actors[0], self._velocity)
        trigger_distance = InTriggerDistanceToVehicle(
            self.other_actors[0], self.ego_vehicles[0], self._trigger_distance)
        just_drive.add_child(keepv)
        just_drive.add_child(trigger_distance)
        behaviour.add_child(just_drive)

        # lane_change，cut in
        lane_change = LaneChange(
            self.other_actors[0], speed=self._velocity, direction='right', distance_same_lane=10,
            distance_other_lane=50)
        behaviour.add_child(lane_change)

        # end condition
        end_condition = py_trees.composites.Parallel("Waiting for end position",
                                                     policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        end_condition.add_child(WaypointFollower(self.other_actors[0], self._other_vehicle_distance_driven))
        end_condition.add_child(DriveDistance(self.other_actors[0], 40))

        # build tree
        root = py_trees.composites.Sequence("Behavior", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        root.add_child(behaviour)
        root.add_child(end_condition)
        root.add_child(ActorDestroy(self.other_actors[0]))

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
