import random

import py_trees

import carla
from navigation.local_planner import RoadOption

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorTransformSetter,
                                                                      ActorDestroy,
                                                                      KeepVelocity,
                                                                      StopVehicle,
                                                                      WaypointFollower, SetTrafficLightGreen,
                                                                      SetTrafficLightRed)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (InTriggerDistanceToVehicle,
                                                                               InTriggerDistanceToNextIntersection,
                                                                               DriveDistance,
                                                                               StandStill, InTriggerDistanceToLocation)
from srunner.scenariomanager.timer import TimeOut
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.scenario_helper import get_waypoint_in_distance, generate_target_waypoint, \
    get_location_in_distance_from_wp, return_target_waypoint_list


class NewScenario(BasicScenario):

    timeout = 120  # Timeout of scenario in seconds

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=60):
        """
        Setup all relevant parameters and create scenario

        If randomize is True, the scenario parameters are randomized
        """

        self._map = CarlaDataProvider.get_map()
        self.debug = world.debug
        self._start_distance = 10
        self._first_vehicle_speed = 10
        self._reference_waypoint = self._map.get_waypoint(config.trigger_points[0].location)
        self._other_actor_max_brake = 1.0
        self._trigger_distance = 30
        self._other_actor_transform = None
        # Timeout of scenario in seconds
        self.timeout = timeout

        super(NewScenario, self).__init__("NewScenario",
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
        waypoint = self._reference_waypoint
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
        self.debug.draw_point(waypoint.transform.location + carla.Location(z=0.5), size=0.5, life_time=0)
        waypoint1 = generate_target_waypoint(waypoint, turn=1)
        self.debug.draw_point(waypoint1.transform.location + carla.Location(z=0.5), size=0.5, life_time=0)
        # move forward _start_distance m
        location, _ = get_location_in_distance_from_wp(waypoint1, self._start_distance, False)
        waypoint = self._map.get_waypoint(location)
        self.debug.draw_point(waypoint.transform.location + carla.Location(z=0.5), size=0.5, life_time=0)
        # get right lane
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
            else:
                waypoint = wp_next
        self.debug.draw_point(waypoint.transform.location + carla.Location(z=0.5), size=0.5, life_time=0)
        # first_vehicle_waypoint, _ = get_waypoint_in_distance(self._reference_waypoint, self._first_vehicle_location)
        self._other_actor_transform = carla.Transform(
            carla.Location(waypoint.transform.location.x,
                           waypoint.transform.location.y,
                           waypoint.transform.location.z),
            waypoint.transform.rotation)
        first_vehicle_transform = carla.Transform(
            carla.Location(self._other_actor_transform.location.x,
                           self._other_actor_transform.location.y,
                           self._other_actor_transform.location.z),
            self._other_actor_transform.rotation)
        first_vehicle = CarlaDataProvider.request_new_actor('vehicle.nissan.patrol', first_vehicle_transform)
        first_vehicle.set_simulate_physics(enabled=False)
        self.other_actors.append(first_vehicle)

    def _create_behavior(self):

        # Find the location in front of the intersection
        location, _ = get_location_in_distance_from_wp(self._reference_waypoint, 150)

        # start condition
        start_condition = InTriggerDistanceToLocation(
            self.ego_vehicles[0],
            location,
            self._trigger_distance,
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

        right_vehicle_go_straight = WaypointFollower(
            self.other_actors[0], self._first_vehicle_speed, plan=plan, avoid_collision=True)

        # Build behavior tree
        sequence = py_trees.composites.Sequence("Sequence Behavior")
        sequence.add_child(ActorTransformSetter(self.other_actors[0], self._other_actor_transform))
        sequence.add_child(start_condition)
        sequence.add_child(SetTrafficLightGreen(self.ego_vehicles[0]))
        sequence.add_child(SetTrafficLightRed(self.other_actors[0]))
        sequence.add_child(right_vehicle_go_straight)
        sequence.add_child(ActorDestroy(self.other_actors[0]))
        # sequence.add_child(driving_to_next_intersection)
        # sequence.add_child(stop)
        # sequence.add_child(endcondition)
        #
        #
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
