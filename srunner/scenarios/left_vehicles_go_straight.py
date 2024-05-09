#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


from six.moves.queue import Queue  # pylint: disable=relative-import
import math
import py_trees
import carla
from agents.navigation.local_planner import RoadOption

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorTransformSetter,
                                                                      ActorDestroy,
                                                                      ActorSource,
                                                                      ActorSink,
                                                                      WaypointFollower,
                                                                      SetTrafficLightGreen, SetTrafficLightRed)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import DriveDistance, \
    InTriggerDistanceToNextIntersection, InTriggerDistanceToLocation
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.scenario_helper import generate_target_waypoint, get_location_in_distance_from_wp, \
    generate_target_waypoint_list


class LeftVehiclesGoStraight(BasicScenario):
    """
    LeftVehiclesGoStraight scenario:
    ego vehicle at an intersection,
    while other vehicle is driving straight from the left intersection
    """
    timeout = 80  # Timeout of scenario in seconds

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=80):
        """
        Setup all relevant parameters and create scenario
        """
        self._world = world
        self._wmap = CarlaDataProvider.get_map()
        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)
        self._ego_distance = 100
        self._traffic_light = None
        self._other_actor_transform = None
        self._blackboard_queue_name = 'LeftVehiclesGoStraight/actor_flow_queue'
        self._queue = py_trees.blackboard.Blackboard().set(self._blackboard_queue_name, Queue())
        self._initialized = True
        self.debug = world.debug

        # actor's speed
        if config._actor_vel is not None:
            self._target_vel = config._actor_vel
        else:
            self._target_vel = 15

        # Trigger_distance from ego to intersection
        if config._trigger_distance is not None:
            self.trigger_distance = config._trigger_distance
        else:
            self.trigger_distance = 30

        # Initial position of the actor
        if config._start_distance is not None:
            self._start_distance = config._start_distance
        else:
            self._start_distance = 15

        super(LeftVehiclesGoStraight, self).__init__("LeftVehiclesGoStraight",
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
        # Get the waypoint of the left intersection
        plan, _ = generate_target_waypoint_list(
            waypoint, -1)
        # self.debug.draw_point(plan[-1][0].transform.location + carla.Location(z=0.5), size=0.5, life_time=0)
        waypoint1 = plan[-1][0]
        # self.debug.draw_point(waypoint1.transform.location + carla.Location(z=0.5), size=0.5, life_time=0)
        location, _ = get_location_in_distance_from_wp(waypoint1, self._start_distance, False)
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
        waypoint = self._reference_waypoint
        self._other_actor_transform, orientation_yaw = self._calculate_base_transform(self._start_distance, waypoint)
        first_vehicle_transform = carla.Transform(
            carla.Location(self._other_actor_transform.location.x,
                           self._other_actor_transform.location.y,
                           self._other_actor_transform.location.z),
            self._other_actor_transform.rotation)
        first_vehicle = CarlaDataProvider.request_new_actor('vehicle.lincoln.mkz2017', first_vehicle_transform)
        first_vehicle.set_simulate_physics(False)
        self.other_actors.append(first_vehicle)

    def _create_behavior(self):
        """
        ego vehicle at a signalized intersection,
        while other vehicle is driving straight from the left intersection
        Order of sequence:
        - ActorTransformSetter: spawn car at a visible transform
        - SetTrafficLightGreen: the light in front of the ego is green
        - SetTrafficLightRed: the light in front of the ego is red
        - behavior:  flow of actors drive according to the plan
        - ActorDestroy: remove the actor
        """
        sequence = py_trees.composites.Sequence("Sequence Behavior")

        # Find the location in front of the intersection
        location, _ = get_location_in_distance_from_wp(self._reference_waypoint, 100)

        # start condition
        start_condition = InTriggerDistanceToLocation(
            self.ego_vehicles[0],
            location,
            self.trigger_distance,
            name="Waiting for start position")

        # Selecting right path at intersection
        plan, _ = generate_target_waypoint_list(
            CarlaDataProvider.get_map().get_waypoint(self.other_actors[0].get_location()), 0)
        # self.debug.draw_point(plan[-1][0].transform.location + carla.Location(z=0.5), size=0.5, life_time=0)
        # Generating waypoint list
        waypoint = plan[-1][0]
        wp_choice = waypoint.next(1.0)
        while not wp_choice[0].is_intersection:
            waypoint = wp_choice[0]
            plan.append((waypoint, RoadOption.LANEFOLLOW))
            wp_choice = waypoint.next(0.1)

        # adding flow of actors
        actor_source = ActorSource(
            ['vehicle.tesla.model3', 'vehicle.audi.tt'],
            self._other_actor_transform, 5, self._blackboard_queue_name)

        # destroying flow of actors
        actor_sink = ActorSink(plan[-1][0].transform.location, 10)
        # self.debug.draw_point(plan[-1][0].transform.location + carla.Location(z=0.5), size=0.5, life_time=0)

        # follow waypoints
        move_actor = WaypointFollower(self.other_actors[0], self._target_vel, plan=plan,
                                      blackboard_queue_name=self._blackboard_queue_name, avoid_collision=True)

        # wait
        wait = DriveDistance(self.ego_vehicles[0], self._ego_distance)

        # Behavior tree
        root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        # root.add_child(wait)
        # root.add_child(actor_source)
        # root.add_child(actor_sink)
        root.add_child(move_actor)

        sequence.add_child(ActorTransformSetter(self.other_actors[0], self._other_actor_transform))
        # Set the traffic light at the intersection in front of ego
        # If there is no traffic light at the intersection, it can also be executed
        sequence.add_child(SetTrafficLightGreen(self.ego_vehicles[0]))
        sequence.add_child(SetTrafficLightRed(self.other_actors[0]))
        sequence.add_child(start_condition)
        sequence.add_child(root)
        sequence.add_child(ActorDestroy(self.other_actors[0]))

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
        self._traffic_light = None
        self.remove_all_actors()
