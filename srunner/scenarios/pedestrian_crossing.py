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
                                                                      StopVehicle, SetTrafficLightGreen)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (InTriggerDistanceToLocationAlongRoute,
                                                                               InTimeToArrivalToVehicle,
                                                                               DriveDistance,
                                                                               InTriggerDistanceToNextIntersection,
                                                                               InTriggerDistanceToLocation)
from srunner.scenariomanager.timer import TimeOut
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.scenario_helper import get_location_in_distance_from_wp, get_location_in_distance_from_wp_previous


class PedestrianCrossing(BasicScenario):
    """
    PedestrianCrossing scenario:
    This class holds everything required for a simple object crash
    without prior vehicle action involving a vehicle and a cyclist/pedestrian,
    The ego vehicle is passing through the intersection,
    And encounters a cyclist/pedestrian crossing the intersection from right.

    This is a single ego vehicle scenario
    """

    def __init__(self, world, ego_vehicles, config, randomize=False,
                 debug_mode=False, criteria_enable=True, adversary_type=True, timeout=60):
        """
        Setup all relevant parameters and create scenario
        """
        self._wmap = CarlaDataProvider.get_map()
        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)
        self._num_lane_changes = 1
        self.transform = None
        self.timeout = timeout
        self._trigger_location = config.trigger_points[0].location
        self.debug = world.debug
        # end condition
        self._ego_vehicle_distance_driven = 40

        # choose cyclist(1)/pedestrian(0)
        if config._model is not None:
            self._model = config._model
        else:
            self._model = 1
        if self._model == 1:
            self._adversary_type = adversary_type  # flag to select either pedestrian (False) or cyclist (True)
        else:
            self._adversary_type = False

        # actor's speed
        if config._actor_vel is not None:
            self._other_actor_target_velocity = config._actor_vel
        else:
            self._other_actor_target_velocity = 2

        # actor's brake
        self._other_actor_max_brake = 1.0

        # _trigger_distance
        if config._trigger_distance is not None:
            self.dist_to_trigger = config._trigger_distance
        else:
            self.dist_to_trigger = 25

        super(PedestrianCrossing, self).__init__("PedestrianCrossing",
                                                 ego_vehicles,
                                                 config,
                                                 world,
                                                 debug_mode,
                                                 criteria_enable=criteria_enable)

    def _calculate_base_transform(self, _start_distance, waypoint):
        """
        Calculate the transform of the actor
        :param float _start_distance: Initial position of the actor
        :param carla.waypoint waypoint: Position of the reference object
        :return: carla.Transform, carla.Rotation.yaw
        """
        lane_width = waypoint.lane_width
        # Find the location at the intersection
        location, _ = get_location_in_distance_from_wp(waypoint, _start_distance)
        waypoint = self._wmap.get_waypoint(location)
        # self.debug.draw_point(waypoint.transform.location + carla.Location(z=0.5), size=0.5, life_time=0)
        offset = {"orientation": 270, "position": 90, "z": 0.2, "k": 1.0}
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
        # cyclist transform
        _start_distance = 500
        # move to the rightmost lane
        waypoint = self._reference_waypoint
        while True:
            wp_next = waypoint.get_right_lane()
            self._num_lane_changes += 1
            if wp_next is None or wp_next.lane_type == carla.LaneType.Sidewalk:
                break
            elif wp_next.lane_type == carla.LaneType.Shoulder:
                # Filter Parkings considered as Shoulders
                break
            else:
                waypoint = wp_next

        self.transform, orientation_yaw = self._calculate_base_transform(_start_distance, waypoint)
        first_vehicle = self._spawn_adversary(self.transform, orientation_yaw)

        disp_transform = carla.Transform(
            carla.Location(self.transform.location.x,
                           self.transform.location.y,
                           self.transform.location.z),
            self.transform.rotation)
        first_vehicle.set_transform(disp_transform)

        first_vehicle.set_simulate_physics(enabled=True)

        self.other_actors.append(first_vehicle)

    def _create_behavior(self):
        """
        cyclist will wait for the ego vehicle to enter trigger distance region,
        the cyclist starts crossing the road once the condition meets,
        Order of sequence:
        - ActorTransformSetter: spawn car at a visible transform
        - SetTrafficLightGreen: the light in front of the ego is green
        - start_condition: ego vehicle enter trigger distance region
        - keep_velocity: pedestrian crossing the road
        - actor_remove: remove the actor
        - end_condition: drive for a defined distance
        """
        root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="OccludedObjectCrossing")
        lane_width = self._reference_waypoint.lane_width
        lane_width = lane_width + (1.25 * lane_width * self._num_lane_changes)

        # find the location of the intersection
        location, _ = get_location_in_distance_from_wp(self._reference_waypoint, 100)

        # start condition
        # wait for the ego vehicle to enter trigger distance region
        start_condition = InTriggerDistanceToLocation(
            self.ego_vehicles[0],
            location,
            self.dist_to_trigger,
            name="Waiting for start position")

        actor_velocity = KeepVelocity(self.other_actors[0],
                                      self._other_actor_target_velocity,
                                      name="walker velocity",
                                      _avoid_collision=True)
        actor_drive = DriveDistance(self.other_actors[0],
                                    lane_width,
                                    name="walker drive distance")
        actor_start_cross_lane = AccelerateToVelocity(self.other_actors[0],
                                                      1.0,
                                                      self._other_actor_target_velocity,
                                                      name="walker crossing lane accelerate velocity")
        actor_cross_lane = DriveDistance(self.other_actors[0],
                                         lane_width,
                                         name="walker drive distance for lane crossing ")
        actor_stop_crossed_lane = StopVehicle(self.other_actors[0],
                                              self._other_actor_max_brake,
                                              name="walker stop")
        ego_pass_machine = DriveDistance(self.ego_vehicles[0],
                                         5,
                                         name="ego vehicle passed prop")
        actor_remove = ActorDestroy(self.other_actors[0],
                                    name="Destroying walker")

        end_condition = DriveDistance(self.ego_vehicles[0],
                                      self._ego_vehicle_distance_driven,
                                      name="End condition ego drive distance")

        # non leaf nodes
        keep_velocity_other = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="keep velocity other")
        keep_velocity = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="keep velocity")

        # building tree
        keep_velocity.add_child(actor_velocity)
        keep_velocity.add_child(actor_drive)
        keep_velocity_other.add_child(actor_start_cross_lane)
        keep_velocity_other.add_child(actor_cross_lane)
        keep_velocity_other.add_child(ego_pass_machine)

        scenario_sequence = py_trees.composites.Sequence()
        scenario_sequence.add_child(ActorTransformSetter(self.other_actors[0], self.transform,
                                                         name='TransformSetterTS3walker', physics=False))
        scenario_sequence.add_child(SetTrafficLightGreen(self.ego_vehicles[0]))
        scenario_sequence.add_child(start_condition)
        scenario_sequence.add_child(keep_velocity)
        scenario_sequence.add_child(keep_velocity_other)
        scenario_sequence.add_child(actor_stop_crossed_lane)
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


class LeftPedestrianCrossing(BasicScenario):
    """
    This class holds everything required for a simple object crash
    without prior vehicle action involving a vehicle and a cyclist/pedestrian,
    The ego vehicle is passing through a road,
    And encounters a cyclist/pedestrian crossing the intersection from the left.

    This is a single ego vehicle scenario
    """

    def __init__(self, world, ego_vehicles, config, randomize=False,
                 debug_mode=False, criteria_enable=True, adversary_type=False, timeout=60):
        """
        Setup all relevant parameters and create scenario
        """
        self._wmap = CarlaDataProvider.get_map()
        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)
        self.debug = world.debug
        self._ego_vehicle_distance_driven = 40  # end condition
        self._adversary_type = adversary_type  # flag to select either pedestrian (False) or cyclist (True)
        self._num_lane_changes = 1
        self.transform = None
        self.timeout = timeout
        self._trigger_location = config.trigger_points[0].location

        # actor's speed
        if config._actor_vel is not None:
            self._other_actor_target_velocity = config._actor_vel
        else:
            self._other_actor_target_velocity = 5

        # actor's brake
        self._other_actor_max_brake = 1.0

        # _trigger_distance
        if config._trigger_distance is not None:
            self.dist_to_trigger = config._trigger_distance
        else:
            self.dist_to_trigger = 25

        super(LeftPedestrianCrossing, self).__init__("LeftPedestrianCrossing",
                                                     ego_vehicles,
                                                     config,
                                                     world,
                                                     debug_mode,
                                                     criteria_enable=criteria_enable)

    def _calculate_base_transform(self, _start_distance, waypoint):
        """
        Calculate the transform of the actor
        :param float _start_distance: Initial position of the actor
        :param carla.waypoint waypoint: Position of the reference object
        :return: carla.Transform, carla.Rotation.yaw
        """
        # put the actor at the intersection
        lane_width = waypoint.lane_width
        location, _ = get_location_in_distance_from_wp_previous(waypoint, _start_distance)
        waypoint = self._wmap.get_waypoint(location)
        # self.debug.draw_point((waypoint.previous(20.0)[-1]).transform.location + carla.Location(z=0.5), size=0.5,
        #                       life_time=0)
        if waypoint.get_right_lane() is None:
            waypoint = waypoint
        else:
            waypoint = waypoint.get_right_lane()
        offset = {"orientation": 270, "position": 90, "z": 0.6, "k": 1.0}
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
        # cyclist transform
        _start_distance = 1000
        waypoint = self._reference_waypoint
        flag = waypoint.lane_id
        # move to the left most lane
        while True:
            if flag * waypoint.lane_id > 0:
                wp_next = waypoint.get_left_lane()
                self._num_lane_changes += 1
            else:
                wp_next = waypoint.get_right_lane()
                self._num_lane_changes += 1
            if wp_next is None or wp_next.lane_type == carla.LaneType.Sidewalk:
                break
            elif wp_next.lane_type == carla.LaneType.Shoulder:
                break
            else:
                waypoint = wp_next

        # self.debug.draw_point(waypoint.transform.location + carla.Location(z=0.5), size=0.5, life_time=0)

        self.transform, orientation_yaw = self._calculate_base_transform(_start_distance, waypoint)
        first_vehicle = self._spawn_adversary(self.transform, orientation_yaw)

        # Now that we found a possible position we just put the vehicle to the underground
        disp_transform = carla.Transform(
            carla.Location(self.transform.location.x,
                           self.transform.location.y,
                           self.transform.location.z - 500),
            self.transform.rotation)
        first_vehicle.set_transform(disp_transform)

        first_vehicle.set_simulate_physics(enabled=True)

        self.other_actors.append(first_vehicle)

    def _create_behavior(self):
        """
        After invoking this scenario, cyclist will wait for the ego vehicle to enter trigger distance region,
        the cyclist starts crossing the road once the condition meets,
        Order of sequence:
        - ActorTransformSetter: spawn car at a visible transform
        - SetTrafficLightGreen: the light in front of the ego is green
        - start_condition: ego vehicle enter trigger distance region
        - keep_velocity: pedestrian crossing the road
        - actor_remove: remove the actor
        - end_condition: drive for a defined distance
        """

        root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="OccludedObjectCrossing")
        lane_width = self._reference_waypoint.lane_width
        lane_width = lane_width + (1.25 * lane_width * self._num_lane_changes)

        # find the location of the intersection
        location, _ = get_location_in_distance_from_wp(self._reference_waypoint, 100)

        # start condition, wait for the ego vehicle enter trigger distance region
        start_condition = InTriggerDistanceToLocation(
            self.ego_vehicles[0],
            location,
            self.dist_to_trigger,
            name="Waiting for start position")

        actor_velocity = KeepVelocity(self.other_actors[0],
                                      self._other_actor_target_velocity,
                                      name="walker velocity")
        actor_drive = DriveDistance(self.other_actors[0],
                                    0.5 * lane_width,
                                    name="walker drive distance")
        actor_start_cross_lane = AccelerateToVelocity(self.other_actors[0],
                                                      1.0,
                                                      self._other_actor_target_velocity,
                                                      name="walker crossing lane accelerate velocity")
        actor_cross_lane = DriveDistance(self.other_actors[0],
                                         lane_width,
                                         name="walker drive distance for lane crossing ")
        actor_stop_crossed_lane = StopVehicle(self.other_actors[0],
                                              self._other_actor_max_brake,
                                              name="walker stop")
        ego_pass_machine = DriveDistance(self.ego_vehicles[0],
                                         5,
                                         name="ego vehicle passed prop")
        actor_remove = ActorDestroy(self.other_actors[0],
                                    name="Destroying walker")

        end_condition = DriveDistance(self.ego_vehicles[0],
                                      self._ego_vehicle_distance_driven,
                                      name="End condition ego drive distance")

        # non leaf nodes
        keep_velocity_other = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="keep velocity other")
        keep_velocity = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="keep velocity")

        # building tree
        keep_velocity.add_child(actor_velocity)
        keep_velocity.add_child(actor_drive)
        keep_velocity_other.add_child(actor_start_cross_lane)
        keep_velocity_other.add_child(actor_cross_lane)
        keep_velocity_other.add_child(ego_pass_machine)

        scenario_sequence = py_trees.composites.Sequence()
        scenario_sequence.add_child(ActorTransformSetter(self.other_actors[0], self.transform,
                                                         name='TransformSetterTS3walker', physics=False))
        scenario_sequence.add_child(SetTrafficLightGreen(self.ego_vehicles[0]))
        scenario_sequence.add_child(start_condition)
        scenario_sequence.add_child(keep_velocity)
        scenario_sequence.add_child(keep_velocity_other)
        scenario_sequence.add_child(actor_stop_crossed_lane)
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
