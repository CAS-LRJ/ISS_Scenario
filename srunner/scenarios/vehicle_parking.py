from __future__ import print_function

from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (InTriggerDistanceToLocation,
                                                                               InTriggerDistanceToNextIntersection,
                                                                               DriveDistance)

import random

from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import InTriggerDistanceToVehicle, DriveDistance

from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorTransformSetter,
                                                                      ActorDestroy,
                                                                      StopVehicle,
                                                                      SyncArrival,
                                                                      WaypointFollower, LaneChange)
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
from srunner.tools.scenario_helper import get_location_in_distance_from_wp


class VehicleParking(BasicScenario):
    """
    VehicleReverse scenario:
    ego is driving straight on the road
    while another vehicle in front is parking
    """

    timeout = 1200

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=600):
        self.timeout = timeout
        self._config = config
        self._wmap = CarlaDataProvider.get_map()
        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)
        # end condition
        self.vehicle_distance_driven = 10

        # actor parameters
        # actor's speed
        if config._actor_vel is not None:
            self._velocity = config._actor_vel
        else:
            self._velocity = 15

        # _trigger_distance
        if config._trigger_distance is not None:
            self._trigger_distance = config._trigger_distance
        else:
            self._trigger_distance = 20

        # Initial position of the actor
        if config._start_distance is not None:
            self._start_distance = config._start_distance
        else:
            self._start_distance = 20

        # actor's brake
        if config._brake is not None:
            self.brake = config._brake
        else:
            self.brake = 0.5
        self.max_brake = 1.0
        # Judge whether there is a place to park
        self.flag = 0
        super(VehicleParking, self).__init__("VehicleParking",
                                             ego_vehicles,
                                             config,
                                             world,
                                             debug_mode,
                                             criteria_enable=criteria_enable)

        if randomize:
            self._velocity = random.randint(20, 60)
            self._trigger_distance = random.randint(10, 40)

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        location1, _ = get_location_in_distance_from_wp(self._reference_waypoint, self._start_distance)
        waypoint = self._wmap.get_waypoint(location1)
        lane_width = waypoint.lane_width
        location = waypoint.transform.location
        offset = {"orientation": 0, "position": 0, "z": 0.6, "k": 1.0}
        position_yaw = waypoint.transform.rotation.yaw + offset['position']
        orientation_yaw = waypoint.transform.rotation.yaw + offset['orientation']
        offset_location = carla.Location(
            offset['k'] * lane_width * math.cos(math.radians(position_yaw)),
            offset['k'] * lane_width * math.sin(math.radians(position_yaw)))
        location += offset_location
        location.z += offset['z']
        self.transform = carla.Transform(location, carla.Rotation(yaw=orientation_yaw))
        first_vehicle_transform = carla.Transform(
            carla.Location(self.transform.location.x,
                           self.transform.location.y,
                           self.transform.location.z - 500),
            self.transform.rotation)
        pedestrian = CarlaDataProvider.request_new_actor('vehicle.lincoln.mkz2017', first_vehicle_transform)
        pedestrian.set_simulate_physics(True)
        self.other_actors.append(pedestrian)

    def _create_behavior(self):
        """
        Ego is driving straight on the road
        while another vehicle in front is parking
        Order of sequence:
        - car_visible: spawn car at a visible transform
        - trigger_distance: trigger in a certain distance
        - keep_velocity: drive along the road
        - StopVehicle: stop the actor
        - TimeOut: wait to remove the actor
        - lane_change: change lane and parking
        - ActorDestroy: remove the actor
        - end condition: drive for a defined distance
        """

        # behavior
        behaviour = py_trees.composites.Sequence("Sequence Behavior")

        # car_visible
        car_visible = ActorTransformSetter(self.other_actors[0], self.transform)

        # trigger in a certain distance
        trigger_distance = InTriggerDistanceToVehicle(
            self.other_actors[0], self.ego_vehicles[0], self._trigger_distance)

        # drive along the road
        keep_velocity = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE, name="keep velocity")
        actor_velocity = WaypointFollower(self.other_actors[0],
                                          self._velocity,
                                          name="velocity",
                                          )
        actor_drive = DriveDistance(self.other_actors[0],
                                    self._trigger_distance - 5,
                                    name="drive distance")
        trigger = InTriggerDistanceToVehicle(
            self.other_actors[0], self.ego_vehicles[0], 15)
        keep_velocity.add_child(actor_velocity)
        keep_velocity.add_child(trigger)

        # parking
        # Judge whether there is a place to park
        waypoint = CarlaDataProvider.get_map().get_waypoint(self.other_actors[0].get_location())
        if waypoint.get_right_lane() is None or waypoint.get_right_lane().lane_type == carla.LaneType.Sidewalk:
            self.flag = 0
        elif waypoint.get_right_lane().lane_type == carla.LaneType.Shoulder:
            self.flag = 0
        elif waypoint.get_right_lane().lane_type == carla.LaneType.Bidirectional:
            self.flag = 0
        else:
            self.flag = 1
        lane_change = LaneChange(
            self.other_actors[0], speed=self._velocity, direction='right', distance_same_lane=10,
            distance_other_lane=5, distance_lane_change=5)

        behaviour.add_child(car_visible)
        behaviour.add_child(trigger_distance)
        behaviour.add_child(keep_velocity)
        behaviour.add_child(StopVehicle(self.other_actors[0], self.max_brake))
        behaviour.add_child(TimeOut(2))
        if self.flag == 1:
            behaviour.add_child(lane_change)
            behaviour.add_child(TimeOut(2))
        # end condition
        end_condition = py_trees.composites.Parallel("Waiting for end position",
                                                     policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        end_condition.add_child(DriveDistance(self.ego_vehicles[0], self.vehicle_distance_driven))
        # build tree
        root = py_trees.composites.Sequence("Behavior")
        root.add_child(behaviour)
        root.add_child(ActorDestroy(self.other_actors[0]))
        root.add_child(end_condition)

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
