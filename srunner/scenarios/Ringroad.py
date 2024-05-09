from __future__ import print_function
import math
import sys

from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (InTriggerDistanceToVehicle,
                                                                               InTriggerDistanceToNextIntersection,
                                                                               DriveDistance,
                                                                               StandStill)

from srunner.tools.scenario_helper import get_waypoint_in_distance, get_location_in_distance_from_wp

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


class RingRoad(BasicScenario):
    """
    RingRoad scenario:
         Ego enters the circular lane from one of the lanes and meets other vehicles
         at the next exit to enter the circular lane
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=600):

        self.debug = world.debug
        self.timeout = timeout
        self._map = CarlaDataProvider.get_map()
        self._reference_waypoint = self._map.get_waypoint(config.trigger_points[0].location)

        # actor's speed
        if config._actor_vel is not None:
            self.velocity = config._actor_vel
        else:
            self.velocity = 7

        # Initial position of the actor
        if config._start_distance is not None:
            self._start_distance = config._start_distance
        else:
            self._start_distance = 30

        # Trigger_distance from ego to intersection
        if config._trigger_distance is not None:
            self.trigger_distance = config._trigger_distance
        else:
            self.trigger_distance = 20

        self.target_location = None
        self._config = config
        self._other_actor_transform = None
        super(RingRoad, self).__init__("RingRoad",
                                       ego_vehicles,
                                       config,
                                       world,
                                       debug_mode,
                                       criteria_enable=criteria_enable)

        if randomize:
            self._velocity = random.randint(20, 60)
            self._trigger_distance = random.randint(10, 40)

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
        waypoint = self._map.get_waypoint(location)
        flag = waypoint.lane_id

        # Move to the right lane of the right intersection
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
            0,
            offset['k'] * lane_width * math.sin(math.radians(position_yaw)))
        location += offset_location
        location.z += offset['z']
        return carla.Transform(location, carla.Rotation(yaw=orientation_yaw)), orientation_yaw

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        waypoint = self._reference_waypoint
        # ego's location as target_location
        self.target_location, _ = get_location_in_distance_from_wp(waypoint, 0)
        # self.debug.draw_point(self.target_location + carla.Location(z=0.5), size=0.5, life_time=0)
        # Get actor's transform
        self._other_actor_transform, orientation_yaw = self._calculate_right_transform(self._start_distance,
                                                                                       waypoint)
        first_vehicle_transform = carla.Transform(
            carla.Location(self._other_actor_transform.location.x,
                           self._other_actor_transform.location.y,
                           self._other_actor_transform.location.z),
            self._other_actor_transform.rotation)
        first_vehicle = CarlaDataProvider.request_new_actor('vehicle.lincoln.mkz2017', first_vehicle_transform)
        first_vehicle.set_simulate_physics(False)
        self.other_actors.append(first_vehicle)

    def _create_behavior(self):

        # car_visible
        behaviour = py_trees.composites.Sequence("Sequence Behavior")
        car_visible = ActorTransformSetter(self.other_actors[0], self._other_actor_transform)
        behaviour.add_child(car_visible)

        # trigger
        trigger = InTriggerDistanceToLocation(self.ego_vehicles[0],
                                              self.target_location, self.trigger_distance)
        behaviour.add_child(trigger)

        # actor enters the circular lane along the road
        just_drive = py_trees.composites.Parallel("DrivingStraight",
                                                  policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        waypoint_follow = WaypointFollower(self.other_actors[0], self.velocity, avoid_collision=True)
        drive_distance = DriveDistance(self.other_actors[0], 100)
        just_drive.add_child(waypoint_follow)
        just_drive.add_child(drive_distance)
        behaviour.add_child(just_drive)
        behaviour.add_child(StopVehicle(self.other_actors[0], 1))

        # wait for end
        end_condition = py_trees.composites.Parallel("Waiting for end position",
                                                     policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)
        end_condition.add_child(TimeOut(5))

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
