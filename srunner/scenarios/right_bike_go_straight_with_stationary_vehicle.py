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
                                                                      ActorDestroy, StopVehicle, KeepVelocity)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import DriveDistance, InTriggerDistanceToLocation


class RightBikeGoStraightWithStationVehicle(BasicScenario):
    """
    RightBikeGoStraightWithStationVehicle scenario:
    The ego vehicle is turing right at an T intersection, a ped is crossing the road
    while a cyclist is driving straight from right intersection and
    there is a stationary vehicle at right intersection
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
            self._start_distance = 25

        if config._start_distance2 is not None:
            self._start_distance2 = config._start_distance2
        else:
            self._start_distance2 = 5

        # Trigger_distance from ego to intersection
        if config._trigger_distance is not None:
            self.trigger_distance = config._trigger_distance
        else:
            self.trigger_distance = 30

        # Travel speed of the actor
        if config._actor_vel is not None:
            self.velocity = config._actor_vel
        else:
            self.velocity = 10

        if config._actor_vel2 is not None:
            self.velocity2 = config._actor_vel2
        else:
            self.velocity2 = 2

        # Timeout of scenario in seconds
        self.timeout = timeout

        super(RightBikeGoStraightWithStationVehicle, self).__init__("RightBikeGoStraightWithStationVehicle",
                                                                    ego_vehicles,
                                                                    config,
                                                                    world,
                                                                    debug_mode,
                                                                    criteria_enable=criteria_enable)

    def _calculate_base_transform(self, _start_distance, waypoint):
        """
        Calculate the transform of the actor
        :param (float) _start_distance: Initial position of the actor
        :param (carla.waypoint) waypoint: Position of the reference object
        :return: carla.Transform, carla.Rotation.yaw
        """
        lane_width = waypoint.lane_width
        # move to the rightmost lane to find the waypoint of the right intersection
        while True:
            if waypoint.get_right_lane() is None or waypoint.get_right_lane().lane_type == carla.LaneType.Sidewalk:
                waypoint = waypoint
                break
            elif waypoint.get_right_lane().lane_type == carla.LaneType.Shoulder or waypoint.get_right_lane().lane_type == carla.LaneType.Parking:
                waypoint = waypoint
                break
            else:
                waypoint = waypoint.get_right_lane()

        # Get the waypoint of the right intersection
        waypoint1 = generate_target_waypoint(waypoint, turn=1)
        # self.debug.draw_point(waypoint1.transform.location + carla.Location(z=0.5), size=0.5, life_time=0)
        location, _ = get_location_in_distance_from_wp(waypoint1, _start_distance, False)
        waypoint = self._wmap.get_waypoint(location)
        flag = waypoint.lane_id

        # # Move to the right lane of the right intersection
        # while True:
        #     if flag * waypoint.lane_id > 0:
        #         wp_next = waypoint.get_left_lane()
        #     else:
        #         break
        #
        #     if wp_next is None or wp_next.lane_type == carla.LaneType.Sidewalk:
        #         break
        #     elif wp_next.lane_type == carla.LaneType.Shoulder or wp_next.lane_type == carla.LaneType.Parking:
        #         break
        #     elif wp_next.lane_type == carla.LaneType.Bidirectional:
        #         waypoint = wp_next.get_right_lane()
        #     else:
        #         waypoint = wp_next
        #
        # # Move to the most right lane of the right intersection
        # while True:
        #     if waypoint.get_right_lane() is None or waypoint.get_right_lane().lane_type == carla.LaneType.Sidewalk:
        #         waypoint = waypoint
        #         break
        #     elif waypoint.get_right_lane().lane_type == carla.LaneType.Shoulder or waypoint.get_right_lane().lane_type == carla.LaneType.Parking:
        #         waypoint = waypoint
        #         break
        #     else:
        #         waypoint = waypoint.get_right_lane()

        location = waypoint.transform.location
        offset = {"orientation": 0, "position": 0, "z": 0, "k": 1.0}
        position_yaw = waypoint.transform.rotation.yaw + offset['position']
        orientation_yaw = waypoint.transform.rotation.yaw + offset['orientation']
        offset_location = carla.Location(
            offset['k'] * lane_width * math.cos(math.radians(position_yaw)),
            offset['k'] * lane_width * math.sin(math.radians(position_yaw)))
        location += offset_location
        location.z += offset['z']
        return carla.Transform(location, carla.Rotation(yaw=orientation_yaw)), orientation_yaw

    def _calculate_right_transform(self, _start_distance, waypoint):
        """
        Calculate the transform of the actor
        :param (float) _start_distance: Initial position of the actor
        :param (carla.waypoint) waypoint: Position of the reference object
        :return: carla.Transform, carla.Rotation.yaw
        """
        lane_width = waypoint.lane_width
        # move to the rightmost lane to find the waypoint of the right intersection
        while True:
            if waypoint.get_right_lane() is None or waypoint.get_right_lane().lane_type == carla.LaneType.Sidewalk:
                waypoint = waypoint
                break
            elif waypoint.get_right_lane().lane_type == carla.LaneType.Shoulder or waypoint.get_right_lane().lane_type == carla.LaneType.Parking:
                waypoint = waypoint
                break
            else:
                waypoint = waypoint.get_right_lane()

        # Get the waypoint of the right intersection
        waypoint1 = generate_target_waypoint(waypoint, turn=1)
        # self.debug.draw_point(waypoint1.transform.location + carla.Location(z=0.5), size=0.5, life_time=0)
        location, _ = get_location_in_distance_from_wp(waypoint1, _start_distance, False)
        waypoint = self._wmap.get_waypoint(location)
        flag = waypoint.lane_id

        # Move to the right lane of the right intersection
        while True:
            if flag * waypoint.lane_id > 0:
                wp_next = waypoint.get_left_lane()
            else:
                break

            if wp_next is None or wp_next.lane_type == carla.LaneType.Sidewalk:
                break
            elif wp_next.lane_type == carla.LaneType.Shoulder or wp_next.lane_type == carla.LaneType.Parking:
                break
            elif wp_next.lane_type == carla.LaneType.Bidirectional:
                waypoint = wp_next.get_right_lane()
            else:
                waypoint = wp_next

        # Move to the most right lane of the right intersection
        while True:
            if waypoint.get_right_lane() is None or waypoint.get_right_lane().lane_type == carla.LaneType.Sidewalk:
                waypoint = waypoint
                break
            elif waypoint.get_right_lane().lane_type == carla.LaneType.Shoulder or waypoint.get_right_lane().lane_type == carla.LaneType.Parking:
                waypoint = waypoint
                break
            else:
                waypoint = waypoint.get_right_lane()

        location = waypoint.transform.location
        offset = {"orientation": 0, "position": 0, "z": 0, "k": 1.0}
        position_yaw = waypoint.transform.rotation.yaw + offset['position']
        orientation_yaw = waypoint.transform.rotation.yaw + offset['orientation']
        offset_location = carla.Location(
            offset['k'] * lane_width * math.cos(math.radians(position_yaw)),
            offset['k'] * lane_width * math.sin(math.radians(position_yaw)))
        location += offset_location
        location.z += offset['z']
        return carla.Transform(location, carla.Rotation(yaw=orientation_yaw)), orientation_yaw

    def _calculate_pedestrian_transform(self, _start_distance, waypoint):
        """
        Calculate the transform of the actor
        :param float _start_distance: Initial position of the actor
        :param carla.waypoint waypoint: Position of the reference object
        :return: carla.Transform, carla.Rotation.yaw
        """
        lane_width = waypoint.lane_width

        flag = waypoint.lane_id
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

        # Move to the most right lane of the left intersection
        while True:
            if waypoint.get_right_lane() is None or waypoint.get_right_lane().lane_type == carla.LaneType.Sidewalk:
                waypoint = waypoint
                break
            elif waypoint.get_right_lane().lane_type == carla.LaneType.Shoulder or waypoint.get_right_lane().lane_type == carla.LaneType.Parking:
                waypoint = waypoint
                break
            else:
                waypoint = waypoint.get_right_lane()

        location, _ = get_location_in_distance_from_wp_previous(waypoint, _start_distance)
        # self.debug.draw_point(waypoint.transform.location + carla.Location(z=0.5), size=0.5, life_time=0)
        offset = {"orientation": -90, "position": 90, "z": 0.3, "k": 1.0}
        position_yaw = waypoint.transform.rotation.yaw + offset['position']
        orientation_yaw = waypoint.transform.rotation.yaw + offset['orientation']
        offset_location = carla.Location(
            offset['k'] * lane_width * math.cos(math.radians(position_yaw)),
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
        # first move to the left lane
        flag = waypoint.lane_id
        if waypoint.get_left_lane() is None or waypoint.get_left_lane().lane_type == carla.LaneType.Sidewalk:
            waypoint = waypoint
        elif waypoint.get_left_lane().lane_type == carla.LaneType.Shoulder or waypoint.get_left_lane().lane_type == carla.LaneType.Parking:
            waypoint = waypoint
        elif waypoint.get_left_lane() is not None and waypoint.get_left_lane().lane_id * flag > 0:
            waypoint = waypoint.get_left_lane()
        # Get the waypoint of the left intersection
        waypoint1 = generate_target_waypoint(waypoint, turn=-1)
        # self.debug.draw_point(waypoint1.transform.location + carla.Location(z=0.5), size=0.5, life_time=0)
        location, _ = get_location_in_distance_from_wp(waypoint1, _start_distance, False)
        waypoint = self._wmap.get_waypoint(location)
        flag = waypoint.lane_id

        # Move to the right lane of the left intersection
        while True:
            if flag * waypoint.lane_id > 0:
                wp_next = waypoint.get_left_lane()
            else:
                break

            if wp_next is None or wp_next.lane_type == carla.LaneType.Sidewalk:
                break
            elif wp_next.lane_type == carla.LaneType.Shoulder or wp_next.lane_type == carla.LaneType.Parking:
                break
            elif wp_next.lane_type == carla.LaneType.Bidirectional:
                waypoint = wp_next.get_right_lane()
            else:
                waypoint = wp_next

        # Move to the most right lane of the left intersection
        # while True:
        #     if waypoint.get_right_lane() is None or waypoint.get_right_lane().lane_type == carla.LaneType.Sidewalk:
        #         waypoint = waypoint
        #         break
        #     elif waypoint.get_right_lane().lane_type == carla.LaneType.Shoulder or waypoint.get_right_lane().lane_type == carla.LaneType.Parking:
        #         waypoint = waypoint
        #         break
        #     else:
        #         waypoint = waypoint.get_right_lane()

        location = waypoint.transform.location
        # self.debug.draw_point(waypoint.transform.location + carla.Location(z=0.5), size=0.5, life_time=0)
        offset = {"orientation": 0, "position": 0, "z": 0, "k": 1.0}
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
        # right bike
        waypoint = self._reference_waypoint
        self._other_actor_transform, orientation_yaw = self._calculate_right_transform(self._start_distance, waypoint)
        first_vehicle_transform = carla.Transform(
            carla.Location(self._other_actor_transform.location.x,
                           self._other_actor_transform.location.y,
                           self._other_actor_transform.location.z),
            self._other_actor_transform.rotation)
        first_vehicle = CarlaDataProvider.request_new_actor('vehicle.bh.crossbike', first_vehicle_transform)
        first_vehicle.set_simulate_physics(False)
        self.other_actors.append(first_vehicle)

        # right stationary vehicle
        waypoint = self._reference_waypoint
        self._other_actor_transform2, orientation_yaw = self._calculate_base_transform(
            self._start_distance2,
            waypoint)
        sec_vehicle_transform = carla.Transform(
            carla.Location(self._other_actor_transform2.location.x,
                           self._other_actor_transform2.location.y,
                           self._other_actor_transform2.location.z),
            self._other_actor_transform2.rotation)
        sce_vehicle = CarlaDataProvider.request_new_actor('vehicle.carlamotors.carlacola', sec_vehicle_transform)
        sce_vehicle.set_simulate_physics(False)
        self.other_actors.append(sce_vehicle)

        # pedestrian
        waypoint = self._reference_waypoint
        self._other_actor_transform3, orientation_yaw = self._calculate_pedestrian_transform(100, waypoint)
        ped_transform = carla.Transform(
            carla.Location(self._other_actor_transform3.location.x,
                           self._other_actor_transform3.location.y,
                           self._other_actor_transform3.location.z),
            self._other_actor_transform3.rotation)
        ped = CarlaDataProvider.request_new_actor('walker.pedestrian.0007', ped_transform)
        ped.set_simulate_physics(False)
        self.other_actors.append(ped)

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
        # Find the location in front of the intersection
        location, _ = get_location_in_distance_from_wp(self._reference_waypoint, 150)

        # start condition
        start_condition = InTriggerDistanceToLocation(
            self.ego_vehicles[0],
            location,
            self.trigger_distance,
            name="Waiting for start position")

        # Selecting straight path at intersection     # right vehicle's plan
        plan = return_target_waypoint_list(
            CarlaDataProvider.get_map().get_waypoint(self.other_actors[0].get_location()), 0)
        # Generating waypoint list till next intersection
        waypoint = plan[-1][0]
        wp_choice = waypoint.next(1.0)
        length = 0
        while not wp_choice[0].is_intersection:
            waypoint = wp_choice[0]
            plan.append((waypoint, RoadOption.LANEFOLLOW))
            wp_choice = waypoint.next(1.0)
            length = length + 1

        # # Selecting straight path at intersection   # left vehicle's plan
        # plan2 = return_target_waypoint_list(
        #     CarlaDataProvider.get_map().get_waypoint(self.other_actors[1].get_location()), 0)
        # # Generating waypoint list till next intersection
        # waypoint = plan2[-1][0]
        # wp_choice = waypoint.next(1.0)
        # length = 0
        # while not wp_choice[0].is_intersection:
        #     waypoint = wp_choice[0]
        #     plan2.append((waypoint, RoadOption.LANEFOLLOW))
        #     wp_choice = waypoint.next(1.0)
        #     length = length + 1

        # left vehicle and right vehicle Drive according to the plan
        continue_driving = py_trees.composites.Parallel(
            "ContinueDriving",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)

        #########################################################################################################
        right_vehicle_turn_right = py_trees.composites.Sequence()
        continue_driving_waypoints = WaypointFollower(
            self.other_actors[0], self.velocity, plan=plan)
        right_vehicle_turn_right.add_child(continue_driving_waypoints)
        right_vehicle_turn_right.add_child(StopVehicle(self.other_actors[0], 1.0))
        # right_vehicle_turn_right.add_child(ActorDestroy(self.other_actors[0]))

        ##########################################################################################################
        ped_cross = py_trees.composites.Sequence()
        continue_driving_waypoints2 = KeepVelocity(self.other_actors[2], self.velocity2, duration=6,
                                                   _avoid_collision=True)
        ped_cross.add_child(continue_driving_waypoints2)
        ped_cross.add_child(StopVehicle(self.other_actors[2], 1.0))
        # left_vehicle_turn_left.add_child(ActorDestroy(self.other_actors[1]))

        ##########################################################################################################
        # Drive according to the plan
        continue_driving.add_child(right_vehicle_turn_right)
        continue_driving.add_child(ped_cross)

        # wait the ego vehicle drove a specific distance
        wait = DriveDistance(
            self.ego_vehicles[0],
            10,
            name="DriveDistance")

        # Build behavior tree
        sequence = py_trees.composites.Sequence("Sequence Behavior")
        # right bike
        sequence.add_child(ActorTransformSetter(self.other_actors[0], self._other_actor_transform))
        # right station vehicle
        sequence.add_child(ActorTransformSetter(self.other_actors[1], self._other_actor_transform2))
        # ped
        sequence.add_child(ActorTransformSetter(self.other_actors[2], self._other_actor_transform3))
        sequence.add_child(start_condition)
        sequence.add_child(SetTrafficLightGreen(self.ego_vehicles[0]))
        sequence.add_child(SetTrafficLightRed(self.other_actors[0]))
        sequence.add_child(SetTrafficLightRed(self.other_actors[1]))
        sequence.add_child(continue_driving)
        sequence.add_child(ActorDestroy(self.other_actors[0]))
        sequence.add_child(ActorDestroy(self.other_actors[2]))
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
