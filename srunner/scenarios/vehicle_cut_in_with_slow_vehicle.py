from __future__ import print_function

import random
import math
import py_trees
import carla

from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorTransformSetter,
                                                                      LaneChange,
                                                                      WaypointFollower,
                                                                      AccelerateToCatchUp, SetTrafficLightGreen)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
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
from srunner.scenariomanager.timer import TimeOut
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.scenario_helper import get_location_in_distance_from_wp, get_location_in_distance_from_wp_previous


class VehicleCutInWithSlowVehicle(BasicScenario):
    """
    VehicleCutInWithSlowVehicle scenario:
    Ego vehicle is driving straight on the road
    beside vehicle cut in because of the front slow car
    """
    timeout = 1200

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=600):
        """
        Setup all relevant parameters and create scenario
        """
        self.timeout = timeout
        self._config = config

        # Determine where to generate the vehicle
        self.flag = 1
        self._wmap = CarlaDataProvider.get_map()
        # ego's waypoint
        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)

        # actor parameters
        # Travel speed of the actor
        if config._actor_vel is not None:
            self._velocity = config._actor_vel
        else:
            self._velocity = 10
        if config._actor_vel2 is not None:
            self._velocity2 = config._actor_vel2
        else:
            self._velocity2 = 12

        # trigger_distance
        if config._trigger_distance is not None:
            self._trigger_distance = config._trigger_distance
        else:
            self._trigger_distance = 15

        # Initial position of the actor
        if config._start_distance is not None:
            self._start_distance = config._start_distance
        else:
            self._start_distance = 90
        if config._start_distance2 is not None:
            self._start_distance2 = config._start_distance2
        else:
            self._start_distance2 = 70

        # actor's brake
        if config._brake is not None:
            self.brake = config._brake
        else:
            self.brake = 1.0

        self._delta_velocity = 10

        super(VehicleCutInWithSlowVehicle, self).__init__("VehicleCutInWithSlowVehicle",
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
        :param (float) _start_distance: Initial position of the actor
        :param (carla.waypoint) waypoint: Position of the reference object
        :return: carla.Transform, carla.Rotation.yaw
        """
        lane_width = waypoint.lane_width
        location, _ = get_location_in_distance_from_wp(waypoint, _start_distance, False)
        waypoint = self._wmap.get_waypoint(location)
        # self.debug.draw_point(waypoint.transform.location + carla.Location(z=0.5), size=0.5, life_time=0)
        offset = {"orientation": 0, "position": 0, "z": 0, "k": 2.0}
        position_yaw = waypoint.transform.rotation.yaw + offset['position']
        orientation_yaw = waypoint.transform.rotation.yaw + offset['orientation']
        offset_location = carla.Location(
            offset['k'] * lane_width * math.cos(math.radians(position_yaw)),
            offset['k'] * lane_width * math.sin(math.radians(position_yaw)))
        location += offset_location
        location.z += offset['z']
        return carla.Transform(location, carla.Rotation(yaw=orientation_yaw)), orientation_yaw

    def _calculate_behind_transform(self, _start_distance, waypoint):
        """
        Calculate the transform of the actor
        :param (float) _start_distance: Initial position of the actor
        :param (carla.waypoint) waypoint: Position of the reference object
        :return: carla.Transform, carla.Rotation.yaw
        """
        # move to the left lane or right lane
        if waypoint.get_left_lane() is None or \
                waypoint.get_left_lane().lane_type == carla.LaneType.Sidewalk or \
                waypoint.get_left_lane().lane_type == carla.LaneType.Shoulder or \
                waypoint.get_left_lane().lane_type == carla.LaneType.Parking or \
                waypoint.get_left_lane().lane_type == carla.LaneType.Bidirectional:
            waypoint = waypoint.get_right_lane()
        else:
            waypoint = waypoint.get_left_lane()

        lane_width = waypoint.lane_width
        location, _ = get_location_in_distance_from_wp_previous(waypoint, _start_distance, False)
        waypoint = self._wmap.get_waypoint(location)
        # self.debug.draw_point(waypoint.transform.location + carla.Location(z=0.5), size=0.5, life_time=0)
        offset = {"orientation": 0, "position": 0, "z": 0, "k": 2.0}
        position_yaw = waypoint.transform.rotation.yaw + offset['position']
        orientation_yaw = waypoint.transform.rotation.yaw + offset['orientation']
        offset_location = carla.Location(
            offset['k'] * lane_width * math.cos(math.radians(position_yaw)),
            offset['k'] * lane_width * math.sin(math.radians(position_yaw)))
        location += offset_location
        location.z += offset['z']
        return carla.Transform(location, carla.Rotation(yaw=orientation_yaw)), orientation_yaw

    def _calculate_front_transform(self, _start_distance, waypoint):
        """
        Calculate the transform of the actor
        :param (float) _start_distance: Initial position of the actor
        :param (carla.waypoint) waypoint: Position of the reference object
        :return: carla.Transform, carla.Rotation.yaw
        """
        # move to the left lane or right lane
        if waypoint.get_left_lane() is None or \
                waypoint.get_left_lane().lane_type == carla.LaneType.Sidewalk or \
                waypoint.get_left_lane().lane_type == carla.LaneType.Shoulder or \
                waypoint.get_left_lane().lane_type == carla.LaneType.Parking or \
                waypoint.get_left_lane().lane_type == carla.LaneType.Bidirectional:
            waypoint = waypoint.get_right_lane()
        else:
            waypoint = waypoint.get_left_lane()

        lane_width = waypoint.lane_width
        location, _ = get_location_in_distance_from_wp(waypoint, _start_distance, False)
        waypoint = self._wmap.get_waypoint(location)
        # self.debug.draw_point(waypoint.transform.location + carla.Location(z=0.5), size=0.5, life_time=0)
        offset = {"orientation": 0, "position": 0, "z": 0, "k": 2.0}
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
        if waypoint.get_left_lane() is None or \
                waypoint.get_left_lane().lane_type == carla.LaneType.Sidewalk or \
                waypoint.get_left_lane().lane_type == carla.LaneType.Shoulder or \
                waypoint.get_left_lane().lane_type == carla.LaneType.Parking or \
                waypoint.get_left_lane().lane_type == carla.LaneType.Bidirectional:
            self.flag = 0

        # front vehicle
        self.transform, _ = self._calculate_front_transform(self._start_distance, self._reference_waypoint)
        first_vehicle_transform = carla.Transform(
            carla.Location(self.transform.location.x,
                           self.transform.location.y,
                           self.transform.location.z - 500),
            self.transform.rotation)
        first_vehicle = CarlaDataProvider.request_new_actor('vehicle.carlamotors.carlacola', first_vehicle_transform)
        first_vehicle.set_simulate_physics(False)
        self.other_actors.append(first_vehicle)

        # behind vehicle
        self.transform_obstacle, _ = self._calculate_front_transform(self._start_distance2, self._reference_waypoint)
        obstacle_transform = carla.Transform(
            carla.Location(self.transform_obstacle.location.x,
                           self.transform_obstacle.location.y,
                           self.transform_obstacle.location.z - 500),
            self.transform_obstacle.rotation)
        obstacle = CarlaDataProvider.request_new_actor('vehicle.tesla.model3', obstacle_transform)
        obstacle.set_simulate_physics(False)
        self.other_actors.append(obstacle)

    def _create_behavior(self):
        """
        Ego vehicle is driving straight on the road
        Another car is cutting in the road behind a vehicle
        Order of sequence:
        - car_visible: spawn car at a visible transform
        - behaviour: If the vehicle is on the left, cut in from the left, otherwise cut in from the right
        - StopVehicle: emergency brake
        - end condition: wait
        - ActorDestroy: remove the actor
        """
        # car_visible
        car_visible = ActorTransformSetter(self.other_actors[0], self.transform)
        car2_visible = ActorTransformSetter(self.other_actors[1], self.transform_obstacle)

        behaviour = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        sequence = py_trees.composites.Sequence()

        # trigger
        just_drive = py_trees.composites.Parallel(
            "DrivingStraight", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        just_drive.add_child(
            InTriggerDistanceToVehicle(self.other_actors[1], self.other_actors[0], self._trigger_distance))
        just_drive.add_child(WaypointFollower(self.other_actors[1], self._velocity2, avoid_collision=True))
        sequence.add_child(just_drive)

        # cut_in, If flag=1, cut in from the left, otherwise from the right
        if self.flag == 1:
            cut_in = LaneChange(
                self.other_actors[1], speed=self._velocity2, direction='right', distance_same_lane=10,
                distance_other_lane=80, avoid_collision=True)
        else:
            cut_in = LaneChange(
                self.other_actors[1], speed=self._velocity2, direction='left', distance_same_lane=10,
                distance_other_lane=80, avoid_collision=True)
        sequence.add_child(cut_in)

        behaviour.add_child(sequence)
        behaviour.add_child(WaypointFollower(self.other_actors[0], self._velocity, avoid_collision=True))

        # end condition
        end_condition = TimeOut(2)
        # build tree
        root = py_trees.composites.Sequence()
        root.add_child(car_visible)
        root.add_child(car2_visible)
        root.add_child(behaviour)
        # emergency brake
        root.add_child(StopVehicle(self.other_actors[0], self.brake))
        root.add_child(StopVehicle(self.other_actors[1], self.brake))
        root.add_child(end_condition)
        root.add_child(ActorDestroy(self.other_actors[0]))
        root.add_child(ActorDestroy(self.other_actors[1]))

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
