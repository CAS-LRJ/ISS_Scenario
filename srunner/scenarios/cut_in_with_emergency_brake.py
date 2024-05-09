from __future__ import print_function

from srunner.tools.scenario_helper import get_waypoint_in_distance, get_location_in_distance_from_wp_previous

import random
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorTransformSetter,
                                                                      LaneChange,
                                                                      WaypointFollower,
                                                                      AccelerateToCatchUp)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
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


class FrontCutInWithBrake(BasicScenario):
    """
    FrontCutInWithBrake scenario:
    Ego vehicle is driving straight on the road
    Another car is cutting just in front, coming from left or right lane and taking emergency brake
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=600):
        """
        Setup all relevant parameters and create scenario
        """
        self.timeout = timeout
        self._config = config
        # Determine where to generate the vehicle
        self.flag = 0
        self._wmap = CarlaDataProvider.get_map()
        # trigger's waypoint
        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)

        # actor parameters
        # Travel speed of the actor
        if config._actor_vel is not None:
            self._velocity = config._actor_vel
        else:
            self._velocity = 6

        # trigger_distance
        if config._trigger_distance is not None:
            self._trigger_distance = config._trigger_distance
        else:
            self._trigger_distance = 20

        # Initial position of the actor
        if config._start_distance is not None:
            self._start_distance = config._start_distance
        else:
            self._start_distance = 25

        # actor's brake
        if config._brake is not None:
            self.brake = config._brake
        else:
            self.brake = 1.0

        super(FrontCutInWithBrake, self).__init__("FrontCutInWithBrake",
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
        # Generate other vehicle on the left if it can be placed on the left, or on the right
        waypoint = self._reference_waypoint
        if waypoint.get_right_lane() is None or waypoint.get_right_lane().lane_type == carla.LaneType.Sidewalk:
            waypoint = waypoint.get_left_lane()
        elif waypoint.get_right_lane().lane_type == carla.LaneType.Shoulder or waypoint.get_right_lane().lane_type == carla.LaneType.Parking:
            waypoint = waypoint.get_left_lane()
        elif waypoint.get_right_lane().lane_type == carla.LaneType.Bidirectional:
            waypoint = waypoint.get_left_lane()
        else:
            waypoint = waypoint.get_right_lane()
            self.flag = 1

        front_location, _ = get_location_in_distance_from_wp(waypoint, self._start_distance, False)
        waypoint = self._wmap.get_waypoint(front_location)
        lane_width = waypoint.lane_width
        location = waypoint.transform.location

        # Calculate the transform of the actor
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
        first_vehicle = CarlaDataProvider.request_new_actor('vehicle.lincoln.mkz2017', first_vehicle_transform)
        first_vehicle.set_simulate_physics(True)
        self.other_actors.append(first_vehicle)

    def _create_behavior(self):
        """
        Ego vehicle is driving straight on the road
        Another car is cutting just in front, coming from left or right lane and taking emergency brake
        Order of sequence:
        - ActorTransformSetter: spawn car at a visible transform
        - just_drive: Triggered at a certain distance
        - LaneChange: If the vehicle is on the left, cut in from the left, otherwise cut in from the right
        - StopVehicle: emergency brake
        - end condition: wait
        - ActorDestroy: remove the actor
        """
        # car_visible
        behaviour = py_trees.composites.Sequence("Sequence Behavior")
        car_visible = ActorTransformSetter(self.other_actors[0], self.transform)
        behaviour.add_child(car_visible)

        # Triggered at a certain distance
        just_drive = py_trees.composites.Parallel(
            "DrivingStraight", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        keep_v = WaypointFollower(self.other_actors[0], self._velocity)
        trigger_distance = InTriggerDistanceToVehicle(
            self.other_actors[0], self.ego_vehicles[0], self._trigger_distance)
        just_drive.add_child(keep_v)
        just_drive.add_child(trigger_distance)
        behaviour.add_child(just_drive)

        # lane_change,If the vehicle is on the left, cut in from the left, otherwise cut in from the right
        if self.flag == 0:
            lane_change = LaneChange(
                self.other_actors[0], speed=self._velocity, direction='right', distance_same_lane=10,
                distance_other_lane=20)
        else:
            lane_change = LaneChange(
                self.other_actors[0], speed=self._velocity, direction='left', distance_same_lane=10,
                distance_other_lane=20)
        behaviour.add_child(lane_change)

        # emergency brake
        stop = StopVehicle(self.other_actors[0], self.brake)
        behaviour.add_child(stop)

        # end condition
        end_condition = TimeOut(5)

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


class BehindCutInWithBrake(BasicScenario):
    """
    BehindCutInWithBrake scenario:
    Ego vehicle is driving straight on the road
    Another car behind the ego is  overtaking and cut in front,
    coming from left or right lane and taking emergency brake
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=600):
        """
        Setup all relevant parameters and create scenario
        """
        self.timeout = timeout
        # Determine where to generate the vehicle
        self.flag = 0
        self._config = config
        self._direction = None
        self.transform = None
        self._wmap = CarlaDataProvider.get_map()
        # trigger's waypoint
        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)

        # Travel speed of the actor
        if config._actor_vel is not None:
            self._velocity = config._actor_vel
        else:
            self._velocity = 20

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

        # actor's brake
        if config._brake is not None:
            self.brake = config._brake
        else:
            self.brake = 1.0

        super(BehindCutInWithBrake, self).__init__("BehindCutInWithBrake",
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
        # Generate other vehicle on the left if it can be placed on the left, or on the right
        waypoint = self._reference_waypoint
        if waypoint.get_right_lane() is None or waypoint.get_right_lane().lane_type == carla.LaneType.Sidewalk:
            waypoint = waypoint.get_left_lane()
        elif waypoint.get_right_lane().lane_type == carla.LaneType.Shoulder or waypoint.get_right_lane().lane_type == carla.LaneType.Parking:
            waypoint = waypoint.get_left_lane()
        elif waypoint.get_right_lane().lane_type == carla.LaneType.Bidirectional:
            waypoint = waypoint.get_left_lane()
        else:
            waypoint = waypoint.get_right_lane()
            self.flag = 1
        behind_location, _ = get_location_in_distance_from_wp_previous(waypoint, self._start_distance)
        waypoint = self._wmap.get_waypoint(behind_location)
        lane_width = self._reference_waypoint.lane_width
        location = waypoint.transform.location

        # Calculate the transform of the actor
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
        Ego vehicle is driving straight on the road
        Another car behind the ego is  overtaking and cut in front,
        coming from left or right lane and taking emergency brake
        Order of sequence:
        - car_visible: spawn car at a visible transform
        - just_drive: drive until in trigger distance to ego_vehicle
        - accelerate: accelerate to catch up distance to ego_vehicle
        - lane_change: change the lane
        - StopVehicle: emergency brake
        - end_condition: drive for a defined distance
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

        # lane_change,If the vehicle is on the left, cut in from the left, otherwise cut in from the right
        if self.flag == 0:
            lane_change = LaneChange(
                self.other_actors[0], speed=self._velocity, direction='right', distance_same_lane=10,
                distance_other_lane=15)
        else:
            lane_change = LaneChange(
                self.other_actors[0], speed=self._velocity, direction='left', distance_same_lane=10,
                distance_other_lane=15)
        behaviour.add_child(lane_change)

        # emergency
        stop = StopVehicle(self.other_actors[0], self.brake)
        behaviour.add_child(stop)
        behaviour.add_child(TimeOut(5))

        # end condition
        end_condition = DriveDistance(self.ego_vehicles[0], 20)

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

