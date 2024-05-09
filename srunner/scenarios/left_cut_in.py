from __future__ import print_function

from srunner.tools.scenario_helper import get_waypoint_in_distance, get_location_in_distance_from_wp_previous

import math
import py_trees
import carla
import random

from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorTransformSetter,
                                                                      LaneChange,
                                                                      WaypointFollower,
                                                                      AccelerateToCatchUp)
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import InTriggerDistanceToVehicle, DriveDistance
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
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.scenario_helper import get_location_in_distance_from_wp


class FrontLeftCutIn(BasicScenario):
    """
    FrontLeftCutIn scenario:
    The ego runs straight on the road. When it is close to a front vehicle in the left lane,
    the vehicle cuts in
    """
    timeout = 1200

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=600):
        self._config = config
        self._wmap = CarlaDataProvider.get_map()
        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)
        # end condition
        self._other_vehicle_distance_driven = 60
        self.timeout = timeout

        # actor parameters
        # actor's speed
        if config._actor_vel is not None:
            self._velocity = config._actor_vel
        else:
            self._velocity = 6

        # _trigger_distance
        if config._trigger_distance is not None:
            self._trigger_distance = config._trigger_distance
        else:
            self._trigger_distance = 20

        # Initial position of the actor
        if config._start_distance is not None:
            self._start_distance = config._start_distance
        else:
            self._start_distance = 25

        super(FrontLeftCutIn, self).__init__("FrontLeftCutIn",
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
        # get the waypoint in front
        location1, _ = get_location_in_distance_from_wp(self._reference_waypoint, self._start_distance, False)
        waypoint1 = self._wmap.get_waypoint(location1)
        # get the left waypoint
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
        """
        The ego runs straight on the road. When it is close to a front vehicle in the left lane,
        the vehicle cuts in
        Order of sequence:
        - car_visible: spawn car at a visible transform
        - just_drive: ego go straight, until ego and actor are in a certain distance
        - lane_change: actor lane_change
        - end_condition: ego drive for a defined distance
        - ActorDestroy: remove the actor
        """
        # car_visible
        behaviour = py_trees.composites.Sequence("Sequence Behavior")
        car_visible = ActorTransformSetter(self.other_actors[0], self.transform)
        behaviour.add_child(car_visible)

        # _trigger_distance
        just_drive = py_trees.composites.Parallel(
            "DrivingStraight", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        keepv = WaypointFollower(self.other_actors[0], self._velocity)
        trigger_distance = InTriggerDistanceToVehicle(
            self.other_actors[0], self.ego_vehicles[0], self._trigger_distance)
        just_drive.add_child(keepv)
        just_drive.add_child(trigger_distance)
        behaviour.add_child(just_drive)

        # lane_change
        lane_change = LaneChange(
            self.other_actors[0], speed=self._velocity, direction='right', distance_same_lane=10,
            distance_other_lane=40)
        behaviour.add_child(lane_change)

        # end condition
        end_condition = py_trees.composites.Parallel("Waiting for end position",
                                                     policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        end_condition.add_child(DriveDistance(self.other_actors[0], self._other_vehicle_distance_driven))
        end_condition.add_child(keepv)

        # build tree
        root = py_trees.composites.Sequence("Behavior")
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


class FrontLeftCutInWithObstacleAvoid(BasicScenario):
    """
    FrontLeftCutIn scenario:
    The ego runs straight on the road. When it is close to a front vehicle in the left lane,
    the vehicle cuts in, the actor has obstacle avoidance function
    """
    timeout = 1200

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=600):
        self._config = config
        self._wmap = CarlaDataProvider.get_map()
        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)
        # end condition
        self._other_vehicle_distance_driven = 40
        self.timeout = timeout

        # actor parameters
        # actor's speed
        if config._actor_vel is not None:
            self._velocity = config._actor_vel
        else:
            self._velocity = 6

        # _trigger_distance
        if config._trigger_distance is not None:
            self._trigger_distance = config._trigger_distance
        else:
            self._trigger_distance = 20

        # Initial position of the actor
        if config._start_distance is not None:
            self._start_distance = config._start_distance
        else:
            self._start_distance = 25

        super(FrontLeftCutInWithObstacleAvoid, self).__init__("FrontLeftCutInWithObstacleAvoid",
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
        # get the waypoint in front
        location1, _ = get_location_in_distance_from_wp(self._reference_waypoint, self._start_distance, False)
        waypoint1 = self._wmap.get_waypoint(location1)
        # get the left waypoint
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
        """
        The ego runs straight on the road. When it is close to a front vehicle in the left lane,
        the vehicle cuts in
        Order of sequence:
        - car_visible: spawn car at a visible transform
        - just_drive: ego go straight, until ego and actor are in a certain distance
        - lane_change: actor lane_change
        - end_condition: ego drive for a defined distance
        - ActorDestroy: remove the actor
        """
        # car_visible
        behaviour = py_trees.composites.Sequence("Sequence Behavior")
        car_visible = ActorTransformSetter(self.other_actors[0], self.transform)
        behaviour.add_child(car_visible)

        # _trigger_distance
        just_drive = py_trees.composites.Parallel(
            "DrivingStraight", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        keepv = WaypointFollower(self.other_actors[0], self._velocity)
        trigger_distance = InTriggerDistanceToVehicle(
            self.other_actors[0], self.ego_vehicles[0], self._trigger_distance)
        just_drive.add_child(keepv)
        just_drive.add_child(trigger_distance)
        behaviour.add_child(just_drive)

        # lane_change
        lane_change = LaneChange(
            self.other_actors[0], speed=self._velocity, direction='right', distance_same_lane=10,
            distance_other_lane=40, avoid_collision=True)
        behaviour.add_child(lane_change)

        # end condition
        end_condition = py_trees.composites.Parallel("Waiting for end position",
                                                     policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        end_condition.add_child(DriveDistance(self.other_actors[0], self._other_vehicle_distance_driven))
        end_condition.add_child(keepv)

        # build tree
        root = py_trees.composites.Sequence("Behavior")
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


class BehindLeftCutIn(BasicScenario):
    """
    BehindRightCutIn scenario:
    The ego is going  straight on the road,
    while another vehicle behind the ego is accelerating and changing lane from left
    """

    timeout = 1200

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=600):
        self.timeout = timeout
        self._config = config
        self.transform = None
        self._wmap = CarlaDataProvider.get_map()
        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)
        # end condition
        self._actor_distance = 60

        # actor's speed
        if config._actor_vel is not None:
            self._velocity = config._actor_vel
        else:
            self._velocity = 10

        # actor accelerate to self._velocity+self._delta_velocity
        self._delta_velocity = 10

        # _trigger_distance
        if config._trigger_distance is not None:
            self._trigger_distance = config._trigger_distance
        else:
            self._trigger_distance = 20

        # Initial position of the actor
        if config._start_distance is not None:
            self._start_distance = config._start_distance
        else:
            self._start_distance = 25

        super(BehindLeftCutIn, self).__init__("BehindLeftCutIn",
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
        # get the previous waypoint
        location1, _ = get_location_in_distance_from_wp_previous(self._reference_waypoint, self._start_distance)
        waypoint1 = self._wmap.get_waypoint(location1)
        lane_width = self._reference_waypoint.lane_width
        # get the left waypoint
        waypoint = waypoint1.get_left_lane()
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
        """
        Order of sequence:
        - car_visible: spawn car at a visible transform
        - just_drive: drive until in trigger distance to ego_vehicle
        - accelerate: accelerate to catch up distance to ego_vehicle
        - lane_change: change the lane
        - end condition: drive for a defined distance
        """

        # car_visible
        behaviour = py_trees.composites.Sequence("BehindLeftCutIn")
        car_visible = ActorTransformSetter(self.other_actors[0], self.transform)
        behaviour.add_child(car_visible)

        # just_drive
        just_drive = py_trees.composites.Parallel(
            "DrivingStraight", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        car_driving = WaypointFollower(self.other_actors[0], self._velocity)
        just_drive.add_child(car_driving)

        trigger_distance = InTriggerDistanceToVehicle(
            self.other_actors[0], self.ego_vehicles[0], self._trigger_distance)
        just_drive.add_child(trigger_distance)
        behaviour.add_child(just_drive)

        # accelerate
        accelerate = AccelerateToCatchUp(self.other_actors[0], self.ego_vehicles[0], throttle_value=1,
                                         delta_velocity=self._delta_velocity, trigger_distance=5, max_distance=500)
        behaviour.add_child(accelerate)

        # lane_change
        lane_change = LaneChange(
            self.other_actors[0], speed=None, direction='right', distance_same_lane=5,
            distance_other_lane=self._actor_distance)
        behaviour.add_child(lane_change)

        # end condition
        end_condition = DriveDistance(self.ego_vehicles[0], 20)

        # build tree
        root = py_trees.composites.Sequence("Behavior")
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
