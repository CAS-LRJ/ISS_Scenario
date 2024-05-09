from __future__ import print_function

import math

import py_trees

import carla

from agents.navigation.local_planner import RoadOption
from srunner.tools.scenario_helper import (get_waypoint_in_distance,
                                           get_location_in_distance_from_wp,
                                           generate_target_waypoint,
                                           get_crossing_point, return_target_waypoint_list,
                                           get_location_in_distance_from_wp_previous)
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorTransformSetter,
                                                                      LaneChange,
                                                                      WaypointFollower,
                                                                      AccelerateToCatchUp,
                                                                      SetTrafficLightGreen,
                                                                      SetTrafficLightRed,
                                                                      ActorDestroy, StopVehicle, KeepVelocity, Idle)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import DriveDistance, \
    InTriggerDistanceToLocation, InTriggerDistanceToVehicle


class OvertakingEncounterVehicle(BasicScenario):
    """
    OvertakingEncounterVehicle scenario:
    The ego vehicle is following a slow vehicle on a two-way street,
    ego want to overtake the slow car, while encounter another car
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=60):
        """
        Setup all relevant parameters and create scenario
        """
        self._wmap = CarlaDataProvider.get_map()
        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)
        self.debug = world.debug
        self._other_actor_transform = None
        self._other_actor_transform2 = None
        self._other_actor_transform3 = None

        # Initial position of the actor
        if config._start_distance is not None:
            self._start_distance = config._start_distance
        else:
            self._start_distance = 15

        if config._start_distance2 is not None:
            self._start_distance2 = config._start_distance2
        else:
            self._start_distance2 = 80

        # Trigger_distance from ego to intersection
        if config._trigger_distance is not None:
            self.trigger_distance = config._trigger_distance
        else:
            self.trigger_distance = 30

        # Travel speed of the actor
        if config._actor_vel is not None:
            self.velocity = config._actor_vel
        else:
            self.velocity = 5

        if config._actor_vel2 is not None:
            self.velocity2 = config._actor_vel2
        else:
            self.velocity2 = 10

        # Timeout of scenario in seconds
        self.timeout = timeout

        super(OvertakingEncounterVehicle, self).__init__("OvertakingEncounterVehicle",
                                                         ego_vehicles,
                                                         config,
                                                         world,
                                                         debug_mode,
                                                         criteria_enable=criteria_enable)

    def _calculate_front_transform(self, _start_distance, waypoint):
        """
        Calculate the transform of the actor
        :param (float) _start_distance: Initial position of the actor
        :param (carla.waypoint) waypoint: Position of the reference object
        :return: carla.Transform, carla.Rotation.yaw
        """
        lane_width = waypoint.lane_width
        # self.debug.draw_point(waypoint.transform.location + carla.Location(z=0.5), size=1, life_time=0)
        location, _ = get_location_in_distance_from_wp(waypoint, _start_distance, False)
        waypoint = self._wmap.get_waypoint(location)
        # self.debug.draw_point(waypoint.transform.location + carla.Location(z=0.5), size=1, life_time=0)
        offset = {"orientation": 0, "position": 0, "z": 0, "k": 1.0}
        position_yaw = waypoint.transform.rotation.yaw + offset['position']
        orientation_yaw = waypoint.transform.rotation.yaw + offset['orientation']
        offset_location = carla.Location(
            0,
            offset['k'] * lane_width * math.sin(math.radians(position_yaw)))
        location += offset_location
        location.z += offset['z']
        return carla.Transform(location, carla.Rotation(yaw=orientation_yaw)), orientation_yaw

    def _calculate_left_transform(self, _start_distance, waypoint):
        """
        Calculate the transform of the actor
        :param float _start_distance: Initial position of the actor
        :param carla.waypoint waypoint: Position of the reference object
        :return: carla.Transform, carla.Rotation.yaw
        """

        lane_width = waypoint.lane_width
        # self.debug.draw_point(waypoint1.transform.location + carla.Location(z=0.5), size=0.5, life_time=0)
        location, _ = get_location_in_distance_from_wp(waypoint, _start_distance, False)
        waypoint = self._wmap.get_waypoint(location)
        flag = waypoint.lane_id
        # Move to the left lane
        while True:
            if flag * waypoint.lane_id > 0:
                wp_next = waypoint.get_left_lane()
                if wp_next.lane_type == carla.LaneType.Bidirectional:
                    if wp_next.lane_id * flag > 0:
                        wp_next = wp_next.get_left_lane()
                    else:
                        wp_next = wp_next.get_right_lane()
            else:
                break

            if wp_next is None or wp_next.lane_type == carla.LaneType.Sidewalk:
                break
            elif wp_next.lane_type == carla.LaneType.Shoulder or wp_next.lane_type == carla.LaneType.Parking:
                break

            else:
                waypoint = wp_next

        location = waypoint.transform.location
        # self.debug.draw_point(waypoint.transform.location + carla.Location(z=0.5), size=2, life_time=0)
        offset = {"orientation": 0, "position": 0, "z": 0, "k": 1.0}
        position_yaw = waypoint.transform.rotation.yaw + offset['position']
        orientation_yaw = waypoint.transform.rotation.yaw + offset['orientation']
        offset_location = carla.Location(
            0,
            offset['k'] * lane_width * math.sin(math.radians(position_yaw)))
        location += offset_location
        location.z += offset['z']
        return carla.Transform(location, carla.Rotation(yaw=orientation_yaw)), orientation_yaw

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        # front slow vehicle
        waypoint = self._reference_waypoint
        self._other_actor_transform, orientation_yaw = self._calculate_front_transform(self._start_distance, waypoint)
        first_vehicle_transform = carla.Transform(
            carla.Location(self._other_actor_transform.location.x,
                           self._other_actor_transform.location.y,
                           self._other_actor_transform.location.z),
            self._other_actor_transform.rotation)
        first_vehicle = CarlaDataProvider.request_new_actor('vehicle.carlamotors.carlacola', first_vehicle_transform)
        first_vehicle.set_simulate_physics(False)
        self.other_actors.append(first_vehicle)

        # left vehicle
        waypoint = self._reference_waypoint
        self._other_actor_transform2, orientation_yaw = self._calculate_left_transform(
            self._start_distance2,
            waypoint)
        sec_vehicle_transform = carla.Transform(
            carla.Location(self._other_actor_transform2.location.x,
                           self._other_actor_transform2.location.y,
                           self._other_actor_transform2.location.z),
            self._other_actor_transform2.rotation)
        sce_vehicle = CarlaDataProvider.request_new_actor('vehicle.tesla.model3', sec_vehicle_transform)
        sce_vehicle.set_simulate_physics(False)
        self.other_actors.append(sce_vehicle)

    def _create_behavior(self):
        """
        Order of sequence:
        - ActorTransformSetter: spawn car at a visible transform
        - start_condition: drive until in trigger distance to ego_vehicle
        - SetTrafficLightGreen: the light in front of the ego is green
        - SetTrafficLightRed: the light in front of the actor is red
        - continue_driving:  drive according to the plan and stop
        - wait: drive for a defined distance

        """
        # start condition
        start_condition = InTriggerDistanceToVehicle(self.ego_vehicles[0], self.other_actors[0], self.trigger_distance)

        # left vehicle and front vehicle Drive follow the road
        continue_driving = py_trees.composites.Parallel(
            "ContinueDriving",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        #########################################################################################################
        front_vehicle = py_trees.composites.Sequence()
        continue_driving_waypoints = WaypointFollower(
            self.other_actors[0], self.velocity)
        front_vehicle.add_child(continue_driving_waypoints)

        ##########################################################################################################
        left_vehicle = py_trees.composites.Sequence()
        continue_driving_waypoints2 = WaypointFollower(
            self.other_actors[1], self.velocity2)
        left_vehicle.add_child(continue_driving_waypoints2)

        ##########################################################################################################
        # Drive according to the plan
        continue_driving.add_child(front_vehicle)
        continue_driving.add_child(left_vehicle)
        continue_driving.add_child(Idle(30))

        # wait the ego vehicle drove a specific distance
        wait = DriveDistance(
            self.ego_vehicles[0],
            10,
            name="DriveDistance")

        # Build behavior tree
        sequence = py_trees.composites.Sequence("Sequence Behavior")
        # front vehicle
        sequence.add_child(ActorTransformSetter(self.other_actors[0], self._other_actor_transform))
        # left vehicle
        sequence.add_child(ActorTransformSetter(self.other_actors[1], self._other_actor_transform2))
        sequence.add_child(start_condition)
        sequence.add_child(continue_driving)
        sequence.add_child(ActorDestroy(self.other_actors[0]))
        sequence.add_child(ActorDestroy(self.other_actors[1]))
        sequence.add_child(wait)
        return sequence

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        collision_criteria = CollisionTest(self.ego_vehicles[0])
        criteria.append(collision_criteria)

        return criteria

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()
