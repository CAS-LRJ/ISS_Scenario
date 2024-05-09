#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


from __future__ import print_function

import sys


from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (InTriggerDistanceToVehicle,
                                                                               InTriggerDistanceToNextIntersection,
                                                                               DriveDistance,
                                                                               StandStill)

from srunner.tools.scenario_helper import get_waypoint_in_distance


from agents.navigation.local_planner import RoadOption



from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest, DrivenDistanceTest, MaxVelocityTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (InTriggerDistanceToLocation,
                                                                               InTriggerDistanceToNextIntersection,
                                                                               DriveDistance)
from srunner.scenariomanager.timer import TimeOut

from srunner.tools.scenario_helper import (get_crossing_point,
                                           get_geometric_linear_intersection,
                                           generate_target_waypoint_list)
import random
import py_trees
import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
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

#自行车横穿马路

class test(BasicScenario):

    """
    This class holds a scenario similar to FollowLeadingVehicle
    but there is an obstacle in front of the leading vehicle

    This is a single ego vehicle scenario
    """

    timeout = 120            # Timeout of scenario in seconds

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True):
        """
        Setup all relevant parameters and create scenario
        """
        self._map = CarlaDataProvider.get_map()
        self._first_actor_location = 25
        self._second_actor_location = self._first_actor_location + 41
        self._first_actor_speed = 10
        self._second_actor_speed = 5
        self._reference_waypoint = self._map.get_waypoint(config.trigger_points[0].location)
        self._other_actor_max_brake = 1.0
        self._first_actor_transform = None
        self._second_actor_transform = None

        super(test, self).__init__("test",
                                    ego_vehicles,
                                    config,
                                    world,
                                    debug_mode,
                                    criteria_enable=criteria_enable)
        if randomize:
            self._ego_other_distance_start = random.randint(4, 8)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """

        # first_actor_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._first_actor_location)
        second_actor_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._second_actor_location)

        # first_actor_transform = carla.Transform(
        #     carla.Location(first_actor_waypoint.transform.location.x,
        #                    first_actor_waypoint.transform.location.y,
        #                    first_actor_waypoint.transform.location.z - 500),
        #     first_actor_waypoint.transform.rotation)

        # self._first_actor_transform = carla.Transform(
        #     carla.Location(first_actor_waypoint.transform.location.x,
        #                    first_actor_waypoint.transform.location.y,
        #                    first_actor_waypoint.transform.location.z + 1),
        #     first_actor_waypoint.transform.rotation)

        yaw_1 = second_actor_waypoint.transform.rotation.yaw + 90

        # second_actor_transform = carla.Transform(
        #     carla.Location(second_actor_waypoint.transform.location.x,
        #                    second_actor_waypoint.transform.location.y,
        #                    second_actor_waypoint.transform.location.z - 500),
        #     carla.Rotation(second_actor_waypoint.transform.rotation.pitch, yaw_1,
        #                    second_actor_waypoint.transform.rotation.roll))

        self._second_actor_transform = carla.Transform(
            carla.Location(second_actor_waypoint.transform.location.x,
                           second_actor_waypoint.transform.location.y-10,
                           second_actor_waypoint.transform.location.z),
            carla.Rotation(second_actor_waypoint.transform.rotation.pitch, yaw_1,
                           second_actor_waypoint.transform.rotation.roll))

        # first_actor = CarlaDataProvider.request_new_actor(
        #     'vehicle.nissan.patrol', self._first_actor_transform)
        second_actor = CarlaDataProvider.request_new_actor(
            'vehicle.diamondback.century', self._second_actor_transform)

        # first_actor.set_simulate_physics(enabled=False)
        second_actor.set_simulate_physics(enabled=False)
        # self.other_actors.append(first_actor)
        self.other_actors.append(second_actor)

    def _create_behavior(self):
        """
        The scenario defined after is a "follow leading vehicle" scenario. After
        invoking this scenario, it will wait for the user controlled vehicle to
        enter the start region, then make the other actor to drive towards obstacle.
        Once obstacle clears the road, make the other actor to drive towards the
        next intersection. Finally, the user-controlled vehicle has to be close
        enough to the other actor to end the scenario.
        If this does not happen within 60 seconds, a timeout stops the scenario
        """

        # let the other actor drive until next intersection
        driving_to_next_intersection = py_trees.composites.Parallel(
            "Driving towards Intersection",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        obstacle_clear_road = py_trees.composites.Parallel("Obstalce clearing road",
                                                           policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        obstacle_clear_road.add_child(DriveDistance(self.other_actors[0], 10))
        obstacle_clear_road.add_child(KeepVelocity(self.other_actors[0], self._second_actor_speed))

        # stop_near_intersection = py_trees.composites.Parallel(
        #     "Waiting for end position near Intersection",
        #     policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        # stop_near_intersection.add_child(WaypointFollower(self.other_actors[0], 10))
        # stop_near_intersection.add_child(InTriggerDistanceToNextIntersection(self.other_actors[0], 20))

        # driving_to_next_intersection.add_child(WaypointFollower(self.other_actors[0], self._first_actor_speed))
        driving_to_next_intersection.add_child(InTriggerDistanceToVehicle(self.other_actors[0],
                                                                          self.ego_vehicles[0], 55))

        # end condition
        endcondition = py_trees.composites.Parallel("Waiting for end position",
                                                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)
        endcondition_part1 = InTriggerDistanceToNextIntersection(self.ego_vehicles[0], 20)
        endcondition_part2 = StandStill(self.ego_vehicles[0], name="FinalSpeed", duration=1)
        endcondition.add_child(endcondition_part1)
        endcondition.add_child(endcondition_part2)

        # Build behavior tree
        sequence = py_trees.composites.Sequence("Sequence Behavior")
        # sequence.add_child(ActorTransformSetter(self.other_actors[0], self._first_actor_transform))
        sequence.add_child(ActorTransformSetter(self.other_actors[0], self._second_actor_transform))
        sequence.add_child(driving_to_next_intersection)
        # sequence.add_child(StopVehicle(self.other_actors[0], self._other_actor_max_brake))
        # sequence.add_child(TimeOut(3))
        sequence.add_child(obstacle_clear_road)
        # sequence.add_child(stop_near_intersection)
        # sequence.add_child(StopVehicle(self.other_actors[0], self._other_actor_max_brake))
        sequence.add_child(endcondition)
        # sequence.add_child(ActorDestroy(self.other_actors[0]))
        sequence.add_child(ActorDestroy(self.other_actors[0]))

        return sequence

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

# 跟车行驶急刹
class FollowVehicle(BasicScenario):

    """
    This class holds everything required for a simple "Follow a leading vehicle"
    scenario involving two vehicles.  (Traffic Scenario 2)

    This is a single ego vehicle scenario
    """

    timeout = 120            # Timeout of scenario in seconds

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=60):
        """
        Setup all relevant parameters and create scenario

        If randomize is True, the scenario parameters are randomized
        """

        self._map = CarlaDataProvider.get_map()
        self._first_vehicle_location = 20
        self._first_vehicle_speed = 7
        self._reference_waypoint = self._map.get_waypoint(config.trigger_points[0].location)
        self._other_actor_max_brake = 1.0
        self._other_actor_stop_in_front_intersection = 10
        self._other_actor_transform = None
        # Timeout of scenario in seconds
        self.timeout = timeout

        super(FollowVehicle, self).__init__("FollowVehicle",
                                                   ego_vehicles,
                                                   config,
                                                   world,
                                                   debug_mode,
                                                   criteria_enable=criteria_enable)

        if randomize:
            self._ego_other_distance_start = random.randint(4, 8)

            # Example code how to randomize start location
            # distance = random.randint(20, 80)
            # new_location, _ = get_location_in_distance(self.ego_vehicles[0], distance)
            # waypoint = CarlaDataProvider.get_map().get_waypoint(new_location)
            # waypoint.transform.location.z += 39
            # self.other_actors[0].set_transform(waypoint.transform)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """

        first_vehicle_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._first_vehicle_location)
        self._other_actor_transform = carla.Transform(
            carla.Location(first_vehicle_waypoint.transform.location.x,
                           first_vehicle_waypoint.transform.location.y,
                           first_vehicle_waypoint.transform.location.z),
            first_vehicle_waypoint.transform.rotation)
        # first_vehicle_transform = carla.Transform(
        #     carla.Location(self._other_actor_transform.location.x,
        #                    self._other_actor_transform.location.y,
        #                    self._other_actor_transform.location.z - 500),
        #     self._other_actor_transform.rotation)
        first_vehicle = CarlaDataProvider.request_new_actor('vehicle.nissan.patrol',  self._other_actor_transform)
        first_vehicle.set_simulate_physics(enabled=False)
        self.other_actors.append(first_vehicle)

    def _create_behavior(self):
        """
        The scenario defined after is a "follow leading vehicle" scenario. After
        invoking this scenario, it will wait for the user controlled vehicle to
        enter the start region, then make the other actor to drive until reaching
        the next intersection. Finally, the user-controlled vehicle has to be close
        enough to the other actor to end the scenario.
        If this does not happen within 60 seconds, a timeout stops the scenario
        """

        # to avoid the other actor blocking traffic, it was spawed elsewhere
        # reset its pose to the required one
        start_transform = ActorTransformSetter(self.other_actors[0], self._other_actor_transform)

        # keep_velocity = py_trees.composites.Parallel(
        #     policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="keep velocity")
        # actor_velocity = KeepVelocity(self.other_actors[0],
        #                               5,
        #                               name="walker velocity")
        #
        # actor_drive = DriveDistance(self.other_actors[0],
        #                             40,
        #                             name="drive distance")
        # keep_velocity.add_child(actor_velocity)
        # keep_velocity.add_child(actor_drive)

        # let the other actor drive until next intersection
        # @todo: We should add some feedback mechanism to respond to ego_vehicle behavior
        driving_to_next_intersection = py_trees.composites.Parallel(
            "DrivingTowardsIntersection",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        driving_to_next_intersection.add_child(WaypointFollower(self.other_actors[0], self._first_vehicle_speed))
        driving_to_next_intersection.add_child(InTriggerDistanceToNextIntersection(
            self.other_actors[0], self._other_actor_stop_in_front_intersection))

        # stop vehicle
        stop = StopVehicle(self.other_actors[0], self._other_actor_max_brake)

        # end condition
        endcondition = py_trees.composites.Parallel("Waiting for end position",
                                                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)
        endcondition_part1 = InTriggerDistanceToVehicle(self.other_actors[0],
                                                        self.ego_vehicles[0],
                                                        distance=20,
                                                        name="FinalDistance")
        endcondition_part2 = StandStill(self.ego_vehicles[0], name="StandStill", duration=1)
        endcondition.add_child(endcondition_part1)
        endcondition.add_child(endcondition_part2)

        # Build behavior tree
        sequence = py_trees.composites.Sequence("Sequence Behavior")
        sequence.add_child(start_transform)
        #sequence.add_child(keep_velocity)
        sequence.add_child(driving_to_next_intersection)
        sequence.add_child(stop)
        sequence.add_child(endcondition)
        sequence.add_child(ActorDestroy(self.other_actors[0]))

        return sequence

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

# 闯红灯车辆
class OppositeVehicleRunningRedLight(BasicScenario):
    """
    This class holds everything required for a scenario,
    in which an other vehicle takes priority from the ego
    vehicle, by running a red traffic light (while the ego
    vehicle has green) (Traffic Scenario 7)

    This is a single ego vehicle scenario
    """

    # ego vehicle parameters
    _ego_max_velocity_allowed = 20  # Maximum allowed velocity [m/s]
    _ego_avg_velocity_expected = 4  # Average expected velocity [m/s]
    _ego_expected_driven_distance = 70  # Expected driven distance [m]
    _ego_distance_to_traffic_light = 32  # Trigger distance to traffic light [m]
    _ego_distance_to_drive = 40  # Allowed distance to drive

    # other vehicle
    _other_actor_target_velocity = 10  # Target velocity of other vehicle
    _other_actor_max_brake = 1.0  # Maximum brake of other vehicle
    _other_actor_distance = 50  # Distance the other vehicle should drive

    _traffic_light = None

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=180):
        """
        Setup all relevant parameters and create scenario
        and instantiate scenario manager
        """

        self._other_actor_transform = None
        self._other_actor_transform2 = None

        # Timeout of scenario in seconds
        self.timeout = timeout

        super(OppositeVehicleRunningRedLight, self).__init__("OppositeVehicleRunningRedLight",
                                                             ego_vehicles,
                                                             config,
                                                             world,
                                                             debug_mode,
                                                             criteria_enable=criteria_enable)

        self._traffic_light = CarlaDataProvider.get_next_traffic_light(self.other_actors[1], False)

        if self._traffic_light is None:
            print("No traffic light for the given location of the ego vehicle found")
            sys.exit(-1)

        self._traffic_light.set_state(carla.TrafficLightState.Green)
        self._traffic_light.set_green_time(self.timeout)

        # other vehicle's traffic light
        traffic_light_other = CarlaDataProvider.get_next_traffic_light(self.other_actors[0], False)

        if traffic_light_other is None:
            print("No traffic light for the given location of the other vehicle found")
            sys.exit(-1)

        traffic_light_other.set_state(carla.TrafficLightState.Red)
        traffic_light_other.set_red_time(self.timeout)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        # 汽车
        self._other_actor_transform = config.other_actors[0].transform
        # 碎石
        self._other_actor_transform2 = config.other_actors[1].transform
        first_vehicle_transform = carla.Transform(
            carla.Location(config.other_actors[0].transform.location.x,
                           config.other_actors[0].transform.location.y,
                           config.other_actors[0].transform.location.z),
            config.other_actors[0].transform.rotation)
        first_vehicle = CarlaDataProvider.request_new_actor(config.other_actors[0].model, first_vehicle_transform)
        first_debris = CarlaDataProvider.request_new_actor('static.prop.dirtdebris01', self._other_actor_transform2,
                                                           'prop')
        self.other_actors.append(first_vehicle)
        self.other_actors.append(first_debris)

    def _create_behavior(self):
        """
        Scenario behavior:
        The other vehicle waits until the ego vehicle is close enough to the
        intersection and that its own traffic light is red. Then, it will start
        driving and 'illegally' cross the intersection. After a short distance
        it should stop again, outside of the intersection. The ego vehicle has
        to avoid the crash, but continue driving after the intersection is clear.

        If this does not happen within 120 seconds, a timeout stops the scenario
        """
        crossing_point_dynamic = get_crossing_point(self.other_actors[0])

        # start condition
        startcondition = InTriggerDistanceToLocation(
            self.ego_vehicles[0],
            crossing_point_dynamic,
            self._ego_distance_to_traffic_light,
            name="Waiting for start position")

        sync_arrival_parallel = py_trees.composites.Parallel(
            "Synchronize arrival times",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        location_of_collision_dynamic = get_geometric_linear_intersection(self.ego_vehicles[0], self.other_actors[0])

        sync_arrival = SyncArrival(
            self.other_actors[0], self.ego_vehicles[0], location_of_collision_dynamic)
        sync_arrival_stop = InTriggerDistanceToNextIntersection(self.other_actors[0],
                                                                5)
        sync_arrival_parallel.add_child(sync_arrival)
        sync_arrival_parallel.add_child(sync_arrival_stop)

        # Generate plan for WaypointFollower
        turn = 0  # drive straight ahead
        plan = []

        # generating waypoints until intersection (target_waypoint)
        plan, target_waypoint = generate_target_waypoint_list(
            CarlaDataProvider.get_map().get_waypoint(self.other_actors[0].get_location()), turn)

        # Generating waypoint list till next intersection
        wp_choice = target_waypoint.next(5.0)
        while len(wp_choice) == 1:
            target_waypoint = wp_choice[0]
            plan.append((target_waypoint, RoadOption.LANEFOLLOW))
            wp_choice = target_waypoint.next(5.0)

        continue_driving = py_trees.composites.Parallel(
            "ContinueDriving",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        continue_driving_waypoints = WaypointFollower(
            self.other_actors[0], self._other_actor_target_velocity, plan=plan, avoid_collision=False)

        continue_driving_distance = DriveDistance(
            self.other_actors[0],
            self._other_actor_distance,
            name="Distance")

        continue_driving_timeout = TimeOut(10)

        continue_driving.add_child(continue_driving_waypoints)
        continue_driving.add_child(continue_driving_distance)
        continue_driving.add_child(continue_driving_timeout)

        # finally wait that ego vehicle drove a specific distance
        wait = DriveDistance(
            self.ego_vehicles[0],
            self._ego_distance_to_drive,
            name="DriveDistance")

        # Build behavior tree
        sequence = py_trees.composites.Sequence("Sequence Behavior")
        sequence.add_child(ActorTransformSetter(self.other_actors[0], self._other_actor_transform))
        sequence.add_child(ActorTransformSetter(self.other_actors[1], self._other_actor_transform2))
        sequence.add_child(startcondition)
        sequence.add_child(sync_arrival_parallel)
        sequence.add_child(continue_driving)
        sequence.add_child(wait)
        sequence.add_child(ActorDestroy(self.other_actors[0]))

        return sequence

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        max_velocity_criterion = MaxVelocityTest(
            self.ego_vehicles[0],
            self._ego_max_velocity_allowed,
            optional=True)
        collision_criterion = CollisionTest(self.ego_vehicles[0])
        driven_distance_criterion = DrivenDistanceTest(
            self.ego_vehicles[0],
            self._ego_expected_driven_distance)

        criteria.append(max_velocity_criterion)
        criteria.append(collision_criterion)
        criteria.append(driven_distance_criterion)

        # Add the collision and lane checks for all vehicles as well
        for vehicle in self.other_actors:
            collision_criterion = CollisionTest(vehicle)
            criteria.append(collision_criterion)

        return criteria

    def __del__(self):
        """
        Remove all actors and traffic lights upon deletion
        """
        self._traffic_light = None
        self.remove_all_actors()

# 后车切入
class LeftCutIn(BasicScenario):
    """
    The ego vehicle is driving on a highway and another car is cutting in just in front.
    This is a single ego vehicle scenario
    """

    timeout = 1200

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=600):

        self.timeout = timeout
        self._map = CarlaDataProvider.get_map()
        self._reference_waypoint = self._map.get_waypoint(config.trigger_points[0].location)

        self._velocity = 21
        self._delta_velocity = 5
        self._trigger_distance = 15

        # get direction from config name
        self._config = config
        self._direction = None
        self._transform_visible = None

        super(LeftCutIn, self).__init__("LeftCutIn",
                                        ego_vehicles,
                                        config,
                                        world,
                                        debug_mode,
                                        criteria_enable=criteria_enable)

        if randomize:
            self._velocity = random.randint(20, 60)
            self._trigger_distance = random.randint(10, 40)

    def _initialize_actors(self, config):

        # direction of lane, on which other_actor is driving before lane change

        # add actors from xml file
        for actor in config.other_actors:
            vehicle = CarlaDataProvider.request_new_actor(actor.model, actor.transform)
            self.other_actors.append(vehicle)
            vehicle.set_simulate_physics(enabled=False)

        # transform visible
        other_actor_transform = self.other_actors[0].get_transform()
        self._transform_visible = carla.Transform(
            carla.Location(other_actor_transform.location.x,
                           other_actor_transform.location.y,
                           other_actor_transform.location.z),
            other_actor_transform.rotation)

    def _create_behavior(self):
        """
        Order of sequence:
        - car_visible: spawn car at a visible transform
        - just_drive: drive until in trigger distance to ego_vehicle
        - accelerate: accelerate to catch up distance to ego_vehicle
        - lane_change: change the lane
        - endcondition: drive for a defined distance
        """

        # car_visible
        behaviour = py_trees.composites.Sequence("Sequence Behavior")
        car_visible = ActorTransformSetter(self.other_actors[0], self._transform_visible)
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
            self.other_actors[0], speed=None, direction='right', distance_same_lane=5, distance_other_lane=300)
        behaviour.add_child(lane_change)

        # endcondition
        endcondition = DriveDistance(self.other_actors[0], 200)

        # build tree
        root = py_trees.composites.Sequence("Behavior", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        root.add_child(behaviour)
        root.add_child(endcondition)
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

# 后车切入刹车
class BehindLeftCutInStop(BasicScenario):
    """
    The ego vehicle is driving on a highway and another car is cutting in just in front.
    This is a single ego vehicle scenario
    """

    timeout = 1200

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=600):

        self.timeout = timeout
        self._map = CarlaDataProvider.get_map()
        self._reference_waypoint = self._map.get_waypoint(config.trigger_points[0].location)

        self._velocity = 10
        self._delta_velocity = 5
        self._trigger_distance = 10

        # get direction from config name
        self._config = config
        self._direction = None
        self._transform_visible = None

        super(BehindLeftCutInStop, self).__init__("BehindLeftCutInStop",
                                                  ego_vehicles,
                                                  config,
                                                  world,
                                                  debug_mode,
                                                  criteria_enable=criteria_enable)

        if randomize:
            self._velocity = random.randint(20, 60)
            self._trigger_distance = random.randint(10, 40)

    def _initialize_actors(self, config):

        # direction of lane, on which other_actor is driving before lane change

        # add actors from xml file
        for actor in config.other_actors:
            vehicle = CarlaDataProvider.request_new_actor(actor.model, actor.transform)
            self.other_actors.append(vehicle)
            vehicle.set_simulate_physics(enabled=False)

        # transform visible
        other_actor_transform = self.other_actors[0].get_transform()
        self._transform_visible = carla.Transform(
            carla.Location(other_actor_transform.location.x,
                           other_actor_transform.location.y,
                           other_actor_transform.location.z),
            other_actor_transform.rotation)

    def _create_behavior(self):
        """
        Order of sequence:
        - car_visible: spawn car at a visible transform
        - just_drive: drive until in trigger distance to ego_vehicle
        - accelerate: accelerate to catch up distance to ego_vehicle
        - lane_change: change the lane
        - endcondition: drive for a defined distance
        """

        # car_visible
        behaviour = py_trees.composites.Sequence("Sequence Behavior")
        car_visible = ActorTransformSetter(self.other_actors[0], self._transform_visible)
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

        # accelerate,后车加速
        accelerate = AccelerateToCatchUp(self.other_actors[0], self.ego_vehicles[0], throttle_value=1,
                                         delta_velocity=self._delta_velocity, trigger_distance=5, max_distance=500)
        behaviour.add_child(accelerate)

        # lane_change，换道切入
        lane_change = LaneChange(
            self.other_actors[0], speed=10, direction='right', distance_same_lane=5, distance_other_lane=50)
        behaviour.add_child(lane_change)

        # 直行一段距离
        # driveDistance = DriveDistance(self.other_actors[0], 10)
        # 急刹车
        stop = StopVehicle(self.other_actors[0], 1)
        # behaviour.add_child(driveDistance)
        behaviour.add_child(stop)

        # end condition
        endcondition = py_trees.composites.Parallel("Waiting for end position",
                                                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)
        endcondition_part1 = InTriggerDistanceToVehicle(self.other_actors[0],
                                                        self.ego_vehicles[0],
                                                        distance=20,
                                                        name="FinalDistance")
        endcondition_part2 = StandStill(self.ego_vehicles[0], name="StandStill", duration=1)
        endcondition.add_child(endcondition_part1)
        endcondition.add_child(endcondition_part2)

        # build tree
        root = py_trees.composites.Sequence("Behavior", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        root.add_child(behaviour)
        root.add_child(endcondition)
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

# 左前车切入刹车
class FrontLeftCutInStop(BasicScenario):
    """
    The ego vehicle is driving on a highway and another car is cutting in just in front.
    This is a single ego vehicle scenario
    """

    timeout = 1200

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=600):

        self.timeout = timeout
        self._map = CarlaDataProvider.get_map()
        self._reference_waypoint = self._map.get_waypoint(config.trigger_points[0].location)

        self._velocity = 8
        self._delta_velocity = 10
        self._trigger_distance = 18

        # get direction from config name
        self._config = config
        self._direction = None
        self._transform_visible = None

        super(FrontLeftCutInStop, self).__init__("FrontLeftCutInStop",
                                                 ego_vehicles,
                                                 config,
                                                 world,
                                                 debug_mode,
                                                 criteria_enable=criteria_enable)

        if randomize:
            self._velocity = random.randint(20, 60)
            self._trigger_distance = random.randint(10, 40)

    def _initialize_actors(self, config):

        # direction of lane, on which other_actor is driving before lane change

        # add actors from xml file
        for actor in config.other_actors:
            vehicle = CarlaDataProvider.request_new_actor(actor.model, actor.transform)
            self.other_actors.append(vehicle)
            vehicle.set_simulate_physics(enabled=False)

        # transform visible
        other_actor_transform = self.other_actors[0].get_transform()
        self._transform_visible = carla.Transform(
            carla.Location(other_actor_transform.location.x,
                           other_actor_transform.location.y,
                           other_actor_transform.location.z),
            other_actor_transform.rotation)

    def _create_behavior(self):

        # car_visible
        behaviour = py_trees.composites.Sequence("Sequence Behavior")
        car_visible = ActorTransformSetter(self.other_actors[0], self._transform_visible)
        behaviour.add_child(car_visible)

        # 一定距离触发
        just_drive = py_trees.composites.Parallel(
            "DrivingStraight", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        # keepv=WaypointFollower(self.other_actors[0],self._velocity)

        trigger_distance = InTriggerDistanceToVehicle(
            self.other_actors[0], self.ego_vehicles[0], self._trigger_distance)

        # just_drive.add_child(keepv)

        just_drive.add_child(trigger_distance)

        behaviour.add_child(just_drive)

        # lane_change，左方前车换道切入
        lane_change = LaneChange(
            self.other_actors[0], speed=5, direction='right', distance_same_lane=10, distance_other_lane=20)
        behaviour.add_child(lane_change)

        # 急刹车
        stop = StopVehicle(self.other_actors[0], 1)
        behaviour.add_child(stop)

        # end condition
        endcondition = py_trees.composites.Parallel("Waiting for end position",
                                                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)
        endcondition_part1 = InTriggerDistanceToVehicle(self.other_actors[0],
                                                        self.ego_vehicles[0],
                                                        distance=20,
                                                        name="FinalDistance")
        endcondition_part2 = StandStill(self.ego_vehicles[0], name="StandStill", duration=1)
        endcondition.add_child(endcondition_part1)
        endcondition.add_child(endcondition_part2)

        # build tree
        root = py_trees.composites.Sequence("Behavior", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        root.add_child(behaviour)
        root.add_child(endcondition)
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

# 无信号灯穿过马路
class nosignalJunctionCrossing(BasicScenario):
    """
    Implementation class for
    'Non-signalized junctions: crossing negotiation' scenario,
    (Traffic Scenario 10).

    This is a single ego vehicle scenario
    """

    # ego vehicle parameters
    _ego_vehicle_max_velocity = 20
    _ego_vehicle_driven_distance = 105

    # other vehicle
    _other_actor_max_brake = 1.0
    _other_actor_target_velocity = 15

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=60):
        """
        Setup all relevant parameters and create scenario
        """

        self._other_actor_transform = None

        # Timeout of scenario in seconds
        self.timeout = timeout

        super(nosignalJunctionCrossing, self).__init__("nosignalJunctionCrossing",
                                                       ego_vehicles,
                                                       config,
                                                       world,
                                                       debug_mode,
                                                       criteria_enable=criteria_enable)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        self._other_actor_transform = config.other_actors[0].transform
        first_vehicle_transform = carla.Transform(
            carla.Location(config.other_actors[0].transform.location.x,
                           config.other_actors[0].transform.location.y,
                           config.other_actors[0].transform.location.z),
            config.other_actors[0].transform.rotation)
        first_vehicle = CarlaDataProvider.request_new_actor(config.other_actors[0].model, first_vehicle_transform)
        first_vehicle.set_simulate_physics(enabled=False)
        self.other_actors.append(first_vehicle)

    def _create_behavior(self):
        crossing_point_dynamic = get_crossing_point(self.other_actors[0])

        # start condition
        startcondition = InTriggerDistanceToLocation(
            self.ego_vehicles[0],
            crossing_point_dynamic,
            70,
            name="Waiting for start position")

        sync_arrival_parallel = py_trees.composites.Parallel(
            "Synchronize arrival times",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        location_of_collision_dynamic = get_geometric_linear_intersection(self.ego_vehicles[0], self.other_actors[0])

        sync_arrival = SyncArrival(
            self.other_actors[0], self.ego_vehicles[0], location_of_collision_dynamic)
        sync_arrival_stop = InTriggerDistanceToNextIntersection(self.other_actors[0],
                                                                5)
        sync_arrival_parallel.add_child(sync_arrival)
        sync_arrival_parallel.add_child(sync_arrival_stop)

        # Generate plan for WaypointFollower
        turn = 0  # drive straight ahead
        plan = []

        # generating waypoints until intersection (target_waypoint)
        plan, target_waypoint = generate_target_waypoint_list(
            CarlaDataProvider.get_map().get_waypoint(self.other_actors[0].get_location()), turn)

        # Generating waypoint list till next intersection
        wp_choice = target_waypoint.next(5.0)
        while len(wp_choice) == 1:
            target_waypoint = wp_choice[0]
            plan.append((target_waypoint, RoadOption.LANEFOLLOW))
            wp_choice = target_waypoint.next(5.0)

        continue_driving = py_trees.composites.Parallel(
            "ContinueDriving",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        continue_driving_waypoints = WaypointFollower(
            self.other_actors[0], 10, plan=plan, avoid_collision=False)

        continue_driving_distance = DriveDistance(
            self.other_actors[0],
            30,
            name="Distance")

        continue_driving_timeout = TimeOut(10)

        continue_driving.add_child(continue_driving_waypoints)
        continue_driving.add_child(continue_driving_distance)
        continue_driving.add_child(continue_driving_timeout)

        # finally wait that ego vehicle drove a specific distance
        wait = DriveDistance(
            self.ego_vehicles[0],
            20,
            name="DriveDistance")

        # Build behavior tree
        sequence = py_trees.composites.Sequence("Sequence Behavior")
        sequence.add_child(ActorTransformSetter(self.other_actors[0], self._other_actor_transform))
        sequence.add_child(startcondition)
        sequence.add_child(sync_arrival_parallel)
        sequence.add_child(continue_driving)
        sequence.add_child(wait)
        sequence.add_child(ActorDestroy(self.other_actors[0]))

        return sequence

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        collison_criteria = CollisionTest(self.ego_vehicles[0])
        criteria.append(collison_criteria)

        return criteria

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()

# 有信号灯路口右转
class signalizedJunctionRightTurn(BasicScenario):
    """
    Implementation class for Hero
    Vehicle turning right at signalized junction scenario,
    Traffic Scenario 09.

    This is a single ego vehicle scenario
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=80):
        """
        Setup all relevant parameters and create scenario
        """
        self._target_vel = 6.9
        self._brake_value = 0.5
        self._ego_distance = 40
        self._traffic_light = None
        self._other_actor_transform = None
        self._other_actor_transform2 = None
        # Timeout of scenario in seconds
        self.timeout = timeout
        super(signalizedJunctionRightTurn, self).__init__("signalizedJunctionRightTurn",
                                                          ego_vehicles,
                                                          config,
                                                          world,
                                                          debug_mode,
                                                          criteria_enable=criteria_enable)

        self._traffic_light = CarlaDataProvider.get_next_traffic_light(self.other_actors[1], False)
        if self._traffic_light is None:
            print("No traffic light for the given location of the ego vehicle found")
            sys.exit(-1)
        self._traffic_light.set_state(carla.TrafficLightState.Red)
        self._traffic_light.set_red_time(self.timeout)
        # other vehicle's traffic light
        traffic_light_other = CarlaDataProvider.get_next_traffic_light(self.other_actors[0], False)
        if traffic_light_other is None:
            print("No traffic light for the given location of the other vehicle found")
            sys.exit(-1)
        traffic_light_other.set_state(carla.TrafficLightState.Green)
        traffic_light_other.set_green_time(self.timeout)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        self._other_actor_transform = config.other_actors[0].transform
        first_vehicle_transform = carla.Transform(
            carla.Location(config.other_actors[0].transform.location.x,
                           config.other_actors[0].transform.location.y,
                           config.other_actors[0].transform.location.z),
            config.other_actors[0].transform.rotation)
        first_vehicle = CarlaDataProvider.request_new_actor(config.other_actors[0].model, first_vehicle_transform)
        first_vehicle.set_simulate_physics(enabled=False)
        self.other_actors.append(first_vehicle)

        self._other_actor_transform2 = config.other_actors[1].transform
        first_debris = CarlaDataProvider.request_new_actor('static.prop.dirtdebris01', self._other_actor_transform2,
                                                           'prop')
        self.other_actors.append(first_debris)

    def _create_behavior(self):
        """
        Hero vehicle is turning right in an urban area,
        at a signalized intersection, while other actor coming straight
        from left.The hero actor may turn right either before other actor
        passes intersection or later, without any collision.
        After 80 seconds, a timeout stops the scenario.
        """

        location_of_collision_dynamic = get_geometric_linear_intersection(self.ego_vehicles[0], self.other_actors[0])
        crossing_point_dynamic = get_crossing_point(self.other_actors[0])
        sync_arrival = SyncArrival(
            self.other_actors[0], self.ego_vehicles[0], location_of_collision_dynamic)
        sync_arrival_stop = InTriggerDistanceToLocation(self.other_actors[0], crossing_point_dynamic, 5)

        sync_arrival_parallel = py_trees.composites.Parallel(
            "Synchronize arrival times",
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
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

        move_actor = WaypointFollower(self.other_actors[0], self._target_vel, plan=plan)
        waypoint_follower_end = InTriggerDistanceToLocation(
            self.other_actors[0], plan[-1][0].transform.location, 10)

        move_actor_parallel = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        move_actor_parallel.add_child(move_actor)
        move_actor_parallel.add_child(waypoint_follower_end)
        # stop other actor
        stop = StopVehicle(self.other_actors[0], self._brake_value)
        # end condition
        end_condition = DriveDistance(self.ego_vehicles[0], self._ego_distance)

        # Behavior tree
        sequence = py_trees.composites.Sequence()
        sequence.add_child(ActorTransformSetter(self.other_actors[0], self._other_actor_transform))
        sequence.add_child(ActorTransformSetter(self.other_actors[1], self._other_actor_transform2))
        sequence.add_child(sync_arrival_parallel)
        sequence.add_child(move_actor_parallel)
        sequence.add_child(stop)
        sequence.add_child(end_condition)
        sequence.add_child(ActorDestroy(self.other_actors[0]))

        return sequence

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []

        collison_criteria = CollisionTest(self.ego_vehicles[0])
        criteria.append(collison_criteria)

        return criteria

    def __del__(self):
        self._traffic_light = None
        self.remove_all_actors()

# 前车切出
class FollowVehicleCutoutLeft(BasicScenario):
    timeout = 1200

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=600):

        self.timeout = timeout
        self._map = CarlaDataProvider.get_map()
        self._reference_waypoint = self._map.get_waypoint(config.trigger_points[0].location)

        self._velocity = 8
        self._delta_velocity = 10
        self._trigger_distance = 25

        # get direction from config name
        self._config = config
        self._direction = None
        self._transform_visible = None
        self._transform_visible1 = None
        super(FollowVehicleCutoutLeft, self).__init__("FollowVehicleCutoutLeft",
                                                      ego_vehicles,
                                                      config,
                                                      world,
                                                      debug_mode,
                                                      criteria_enable=criteria_enable)

        if randomize:
            self._velocity = random.randint(20, 60)
            self._trigger_distance = random.randint(10, 40)

    def _initialize_actors(self, config):

        # direction of lane, on which other_actor is driving before lane change

        # add actors from xml file
        for actor in config.other_actors:
            vehicle = CarlaDataProvider.request_new_actor(actor.model, actor.transform)
            self.other_actors.append(vehicle)
            vehicle.set_simulate_physics(enabled=False)

        # transform visible
        other_actor_transform = self.other_actors[0].get_transform()
        self._transform_visible = carla.Transform(
            carla.Location(other_actor_transform.location.x,
                           other_actor_transform.location.y,
                           other_actor_transform.location.z),
            other_actor_transform.rotation)

        other_actor_transform1 = self.other_actors[1].get_transform()
        self._transform_visible1 = carla.Transform(
            carla.Location(other_actor_transform1.location.x,
                           other_actor_transform1.location.y,
                           other_actor_transform1.location.z),
            other_actor_transform1.rotation)

    def _create_behavior(self):
        """
        Order of sequence:
        - car_visible: spawn car at a visible transform
        - just_drive: drive until in trigger distance to ego_vehicle
        - accelerate: accelerate to catch up distance to ego_vehicle
        - lane_change: change the lane
        - endcondition: drive for a defined distance
        """

        # car_visible
        behaviour = py_trees.composites.Sequence("Sequence Behavior")
        car_visible = ActorTransformSetter(self.other_actors[0], self._transform_visible)
        car_visible2 = ActorTransformSetter(self.other_actors[1], self._transform_visible1)
        behaviour.add_child(car_visible)
        behaviour.add_child(car_visible2)

        # 一定距离触发
        just_drive = py_trees.composites.Parallel(
            "DrivingStraight", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        keepv = WaypointFollower(self.other_actors[0], self._velocity)

        trigger_distance = InTriggerDistanceToVehicle(
            self.other_actors[0], self.other_actors[1], self._trigger_distance)

        just_drive.add_child(keepv)

        just_drive.add_child(trigger_distance)

        behaviour.add_child(just_drive)

        # lane_change，前车切出
        lane_change = LaneChange(
            self.other_actors[0], speed=8, distance_same_lane=5, distance_other_lane=20)
        behaviour.add_child(lane_change)

        # end condition
        endcondition = py_trees.composites.Parallel("Waiting for end position",
                                                    policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)
        endcondition_part1 = InTriggerDistanceToVehicle(self.other_actors[0],
                                                        self.ego_vehicles[0],
                                                        distance=20,
                                                        name="FinalDistance")
        endcondition_part2 = StandStill(self.ego_vehicles[0], name="StandStill", duration=1)
        endcondition.add_child(endcondition_part1)
        endcondition.add_child(endcondition_part2)

        # build tree
        root = py_trees.composites.Sequence("Behavior", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        root.add_child(behaviour)
        root.add_child(endcondition)
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

#!/usr/bin/env python




