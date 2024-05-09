from __future__ import print_function

from srunner.tools.scenario_helper import get_waypoint_in_distance, get_location_in_distance_from_wp_previous

import random

from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorTransformSetter,
                                                                      ActorDestroy,
                                                                      StopVehicle,
                                                                      SyncArrival,
                                                                      WaypointFollower)

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


class MergeIntoTheMainRoad(BasicScenario):
    """
       Ego cut in the main road on the left, and there are other vehicles
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=600):
        self.timeout = timeout
        # actor parameters
        # actor's speed
        if config._actor_vel is not None:
            self._velocity = config._actor_vel
        else:
            self._velocity = 25
        # actor's initial position
        if config._start_distance is not None:
            self._start_distance = config._start_distance
        else:
            self._start_distance = 8
        #
        self._config = config
        self._wmap = CarlaDataProvider.get_map()
        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)
        self._other_vehicle_distance_driven = 100
        self.debug = world.debug

        super(MergeIntoTheMainRoad, self).__init__("MergeIntoTheMainRoad",
                                                   ego_vehicles,
                                                   config,
                                                   world,
                                                   debug_mode,
                                                   criteria_enable=criteria_enable)

        if randomize:
            self._velocity = random.randint(20, 60)
            self._trigger_distance = random.randint(10, 40)

    def _initialize_actors(self, config):

        # Find the waypoint in the left main road
        waypoint = self._reference_waypoint.get_left_lane()
        # self.debug.draw_point(waypoint.transform.location + carla.Location(z=0.5), size=0.5, life_time=0)
        # Find the waypoint previous
        location, _ = get_location_in_distance_from_wp_previous(waypoint, self._start_distance, False)
        waypoint = self._wmap.get_waypoint(location)

        lane_width = waypoint.lane_width
        # self.debug.draw_point(waypoint.transform.location + carla.Location(z=0.5), size=0.5, life_time=0)
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
                           self.transform.location.z),
            self.transform.rotation)
        first_vehicle = CarlaDataProvider.request_new_actor('vehicle.chevrolet.impala', first_vehicle_transform)
        first_vehicle.set_simulate_physics(True)
        self.other_actors.append(first_vehicle)

    def _create_behavior(self):
        # car_visible
        behaviour = py_trees.composites.Sequence("Sequence Behavior")
        car_visible = ActorTransformSetter(self.other_actors[0], self.transform)

        behaviour.add_child(car_visible)

        keep_v = WaypointFollower(self.other_actors[0], self._velocity, avoid_collision=True)
        drive = py_trees.composites.Parallel("Waiting for end position",
                                             policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        drive.add_child(DriveDistance(self.other_actors[0], self._other_vehicle_distance_driven))
        drive.add_child(keep_v)

        behaviour.add_child(drive)

        # build tree
        root = py_trees.composites.Sequence("Behavior", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        root.add_child(behaviour)

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
