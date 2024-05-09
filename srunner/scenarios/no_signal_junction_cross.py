from __future__ import print_function

import math
import random
import py_trees
import carla

from srunner.tools.scenario_helper import get_waypoint_in_distance, get_location_in_distance_from_wp

from agents.navigation.local_planner import RoadOption

from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (InTriggerDistanceToLocation,
                                                                               InTriggerDistanceToNextIntersection,
                                                                               DriveDistance)
from srunner.scenariomanager.timer import TimeOut

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import InTriggerDistanceToVehicle, DriveDistance
from srunner.scenarios.basic_scenario import BasicScenario

from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorTransformSetter,
                                                                      ActorDestroy,
                                                                      StopVehicle,
                                                                      SyncArrival,
                                                                      WaypointFollower)

from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import DriveDistance, InTriggerDistanceToLocation

from srunner.tools.scenario_helper import (get_geometric_linear_intersection,
                                           get_crossing_point,
                                           generate_target_waypoint)


class NoSignalJunctionCross(BasicScenario):
    """
    NoSignalJunctionCross scenario:
    ego vehicle at an intersection,
    while other vehicle is driving straight from the left or right intersection
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=60):
        """
        Setup all relevant parameters and create scenario
        """
        # Timeout of scenario in seconds
        self.timeout = timeout
        self._wmap = CarlaDataProvider.get_map()
        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)
        self._num_lane_changes = 1
        self.debug = world.debug
        self._other_actor_transform = None

        # Initial position of the actor
        if config._start_distance is not None:
            print(config._start_distance)
            self._start_distance = config._start_distance
        else:
            self._start_distance = 15

        # wait for ego enter the trigger distance region
        if config._trigger_distance is not None:
            self.trigger_distance = config._trigger_distance
        else:
            self.trigger_distance = 70

        self.syn_distance = 5

        # actor's speed
        if config._actor_vel is not None:
            self.velocity = config._actor_vel
        else:
            self.velocity = 10

        super(NoSignalJunctionCross, self).__init__("NoSignalJunctionCross",
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
        # if ego is in the rightmost lane, the vehicle running the red light is generated from the right,
        # otherwise it is generated from the left
        if waypoint.get_right_lane() is None or waypoint.get_right_lane().lane_type == carla.LaneType.Sidewalk:
            waypoint1 = generate_target_waypoint(waypoint, turn=1)
        elif waypoint.get_right_lane().lane_type == carla.LaneType.Shoulder or waypoint.get_right_lane().lane_type == carla.LaneType.Parking:
            waypoint1 = generate_target_waypoint(waypoint, turn=1)
        else:
            waypoint1 = generate_target_waypoint(waypoint, turn=-1)
        # waypoint1 = generate_target_waypoint(waypoint, turn=-1)

        location, _ = get_location_in_distance_from_wp(waypoint1, self._start_distance, False)
        waypoint = self._wmap.get_waypoint(location)
        flag = waypoint.lane_id
        while True:
            if flag * waypoint.lane_id > 0:
                wp_next = waypoint.get_left_lane()
                self._num_lane_changes += 1
            else:
                self._num_lane_changes += 1
                break
            if wp_next is None or wp_next.lane_type == carla.LaneType.Sidewalk:
                break
            elif wp_next.lane_type == carla.LaneType.Shoulder:
                # print("Dsadsadasd")
                break
            else:
                waypoint = wp_next

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
        # self._other_actor_transform = config.other_actors[0].transform
        # first_vehicle_transform = carla.Transform(
        #     carla.Location(config.other_actors[0].transform.location.x,
        #                    config.other_actors[0].transform.location.y,
        #                    config.other_actors[0].transform.location.z+500),
        #     config.other_actors[0].transform.rotation)
        # first_vehicle = CarlaDataProvider.request_new_actor(config.other_actors[0].model, first_vehicle_transform)
        # first_vehicle.set_simulate_physics(enabled=False)
        # self.other_actors.append(first_vehicle)

    def _create_behavior(self):
        """
        ego vehicle at an intersection,
        while other vehicle is driving straight from the left intersection
        Order of sequence:
        - ActorTransformSetter: spawn car at a visible transform
        - start_condition: trigger in a certain distance
        - sync_arrival_parallel:  two cars driving to the intersection synchronously
        - continue_driving: actor continue driving
        - end_condition: wait ego vehicle drive a specific distance
        - ActorDestroy: remove the actor
        """
        # crossing_point_dynamic = get_crossing_point(self.other_actors[0])

        # start condition, trigger in a certain distance
        start_condition = InTriggerDistanceToVehicle(
            self.ego_vehicles[0],
            self.other_actors[0],
            self.trigger_distance,
            name="Waiting for start position")

        # Two cars driving to the intersection synchronously
        sync_arrival_parallel = py_trees.composites.Parallel(
            "Synchronize arrival times",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        location_of_collision_dynamic = get_geometric_linear_intersection(self.ego_vehicles[0], self.other_actors[0])
        sync_arrival = SyncArrival(
            self.other_actors[0], self.ego_vehicles[0], location_of_collision_dynamic)
        sync_arrival_stop = InTriggerDistanceToNextIntersection(self.other_actors[0],
                                                                self.syn_distance)
        sync_arrival_parallel.add_child(sync_arrival)
        sync_arrival_parallel.add_child(sync_arrival_stop)

        # Selecting straight path at intersection
        target_waypoint = generate_target_waypoint(
            CarlaDataProvider.get_map().get_waypoint(self.other_actors[0].get_location()), 0)
        # Generating waypoint list till next intersection
        plan = []
        wp_choice = target_waypoint.next(1.0)
        while not wp_choice[0].is_intersection:
            target_waypoint = wp_choice[0]
            plan.append((target_waypoint, RoadOption.LANEFOLLOW))
            wp_choice = target_waypoint.next(1.0)

        # actor continue_driving
        continue_driving = py_trees.composites.Parallel(
            "ContinueDriving",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        continue_driving_waypoints = WaypointFollower(
            self.other_actors[0], self.velocity, plan=plan, avoid_collision=False)
        continue_driving_distance = DriveDistance(
            self.other_actors[0],
            30,
            name="Distance")
        continue_driving_timeout = TimeOut(3)
        continue_driving.add_child(continue_driving_waypoints)
        continue_driving.add_child(continue_driving_distance)
        continue_driving.add_child(continue_driving_timeout)

        # end condition, wait ego vehicle drive a specific distance
        end_condition = DriveDistance(
            self.ego_vehicles[0],
            10,
            name="DriveDistance")

        # Build behavior tree
        sequence = py_trees.composites.Sequence("Sequence Behavior")
        sequence.add_child(ActorTransformSetter(self.other_actors[0], self._other_actor_transform))
        sequence.add_child(start_condition)
        sequence.add_child(sync_arrival_parallel)
        sequence.add_child(continue_driving)
        sequence.add_child(end_condition)
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
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()
