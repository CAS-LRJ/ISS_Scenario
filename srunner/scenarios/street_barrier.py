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
                                                                      StopVehicle)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (InTriggerDistanceToLocationAlongRoute,
                                                                               InTimeToArrivalToVehicle,
                                                                               DriveDistance)
from srunner.scenariomanager.timer import TimeOut
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.scenario_helper import get_location_in_distance_from_wp, generate_target_waypoint


class StreetBarrier(BasicScenario):
    """
    StreetBarrier scenario:
    The ego vehicle is driving on the road, and meets obstacle in front of the road
    The ego vehicle may need to brake to avoid a collision
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=60):
        """
        Setup all relevant parameters and create scenario
        """
        self._wmap = CarlaDataProvider.get_map()
        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)
        # ego vehicle parameters
        self._ego_vehicle_distance_driven = 40
        # Timeout of scenario in seconds
        self.timeout = timeout
        # barrier location
        if config._start_distance is not None:
            self._start_distance = config._start_distance
        else:
            self._start_distance = 50

        super(StreetBarrier, self).__init__("StreetBarrier",
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
        # If the ego is in the rightmost lane, the other actor is generated from the right,
        # otherwise it is generated from the left
        if waypoint.get_right_lane() is None or waypoint.get_right_lane().lane_type == carla.LaneType.Sidewalk:
            waypoint1 = generate_target_waypoint(waypoint, turn=1)
        elif waypoint.get_right_lane().lane_type == carla.LaneType.Shoulder or waypoint.get_right_lane().lane_type == carla.LaneType.Parking:
            waypoint1 = generate_target_waypoint(waypoint, turn=1)
        else:
            print(waypoint.get_right_lane().lane_type)
            waypoint1 = generate_target_waypoint(waypoint, turn=-1)

        # Move to the right lane
        # location, _ = get_location_in_distance_from_wp(waypoint1, self._start_distance, False)
        # waypoint = self._wmap.get_waypoint(location)
        # flag = waypoint.lane_id
        # while True:
        #     if flag * waypoint.lane_id > 0:
        #         wp_next = waypoint.get_left_lane()
        #     else:
        #         break
        #     if wp_next is None or wp_next.lane_type == carla.LaneType.Sidewalk:
        #         break
        #     elif wp_next.lane_type == carla.LaneType.Shoulder:
        #         break
        #     else:
        #         waypoint = wp_next
        waypoint = waypoint1
        location = waypoint.transform.location
        # self.debug.draw_point(waypoint.transform.location + carla.Location(z=0.5), size=0.5, life_time=0)
        offset = {"orientation": 0, "position": 0, "z": 0, "k": 0}
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
        # Look for a position in front of the road to place the other vehicle
        lane_width = self._reference_waypoint.lane_width
        location, _ = get_location_in_distance_from_wp(self._reference_waypoint, self._start_distance, False)
        waypoint = self._wmap.get_waypoint(location)

        # waypoint = self._reference_waypoint
        # self._other_actor_transform, orientation_yaw = self._calculate_base_transform(self._start_distance, waypoint)
        # location = waypoint.transform.location

        offset = {"orientation": 270, "position": 180, "z": 0.4, "k": 0}
        position_yaw = waypoint.transform.rotation.yaw + offset['position']
        orientation_yaw = waypoint.transform.rotation.yaw + offset['orientation']
        offset_location = carla.Location(
            offset['k'] * lane_width * math.cos(math.radians(position_yaw)),
            offset['k'] * lane_width * math.sin(math.radians(position_yaw)))
        location += offset_location
        location.z += offset['z']
        self.transform = carla.Transform(location, carla.Rotation(yaw=orientation_yaw))
        print(self.transform)
        obstacle_transform = carla.Transform(
            carla.Location(self.transform.location.x,
                           self.transform.location.y,
                           self.transform.location.z - 500),
            self.transform.rotation)
        obstacle = CarlaDataProvider.request_new_actor('static.prop.streetbarrier', obstacle_transform)
        obstacle.set_simulate_physics(True)
        self.other_actors.append(obstacle)

    def _create_behavior(self):
        """
        The ego vehicle is driving on the road, and meets obstacle in front of the road
        The ego vehicle may need to brake to avoid a collision
        Order of sequence:
        - ActorTransformSetter: spawn the barrier at a visible transform
        - actor_stand: barrier on the road for a period of time
        - actor_removed: remove the barrier
        - end_condition: drive for a defined distance
        """
        # leaf nodes
        actor_stand = TimeOut(8)
        actor_removed = ActorDestroy(self.other_actors[0])
        end_condition = DriveDistance(self.ego_vehicles[0], self._ego_vehicle_distance_driven)

        # non leaf nodes
        root = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        scenario_sequence = py_trees.composites.Sequence()

        # building tree
        root.add_child(scenario_sequence)
        scenario_sequence.add_child(ActorTransformSetter(self.other_actors[0], self.transform))
        scenario_sequence.add_child(actor_stand)
        scenario_sequence.add_child(actor_removed)
        scenario_sequence.add_child(end_condition)

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
