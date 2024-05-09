from __future__ import print_function

import math
import py_trees
import carla
import random

from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (InTriggerDistanceToLocation,
                                                                               InTriggerDistanceToNextIntersection,
                                                                               DriveDistance)

from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorTransformSetter,
                                                                      ActorDestroy,
                                                                      StopVehicle,
                                                                      SyncArrival,
                                                                      WaypointFollower,
                                                                      AddNoiseToVehicle,
                                                                      Idle)
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


class VehicleControlLoss(BasicScenario):
    """
    Control Loss Vehicle scenario:

    The scenario realizes that the vehicle looses control due to
    bad road conditions, etc. and checks to see if the vehicle
    regains control and corrects it's course.
    """
    timeout = 1200

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=600):
        self.timeout = timeout
        self.transform = None
        self.transform_obstacle = None
        self.first_loc = None
        self.first_loc_prev = None
        self._config = config
        self._wmap = CarlaDataProvider.get_map()
        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)
        self.debug = world.debug
        # end condition
        self.vehicle_distance_driven = 20

        # add noise
        if config._new_steer_noise is not None:
            self._new_steer_noise = config._new_steer_noise
        else:
            self._new_steer_noise = 0.1

        if config._new_throttle_noise is not None:
            self._new_throttle_noise = config._new_throttle_noise
        else:
            self._new_throttle_noise = 0.05

        # actor parameters
        # Initial position of the actor
        if config._start_distance is not None:
            self._start_distance = config._start_distance
        else:
            self._start_distance = 50

        super(VehicleControlLoss, self).__init__("VehicleControlLoss",
                                                 ego_vehicles,
                                                 config,
                                                 world,
                                                 debug_mode,
                                                 criteria_enable=criteria_enable)

        if randomize:
            self._velocity = random.randint(20, 60)
            self._trigger_distance = random.randint(10, 40)

    def _calculate_base_transform(self, _start_distance, waypoint):
        """
        Calculate the transform of the actor
        :param float _start_distance: Initial position of the actor
        :param carla.waypoint waypoint: Position of the reference object
        :return: carla.Transform, carla.Rotation.yaw
        """
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
        return carla.Transform(location, carla.Rotation(yaw=orientation_yaw)), orientation_yaw

    def _initialize_actors(self, config):
        """
        Custom initialization
        """
        # generate debris
        self.first_loc_prev, _ = get_location_in_distance_from_wp(self._reference_waypoint, self._start_distance)
        # waypoint = self._wmap.get_waypoint(self.first_loc_prev)
        # self.debug.draw_point(waypoint.transform.location + carla.Location(z=0.5), size=0.5,
        #                       life_time=0)
        self.first_transform = carla.Transform(self.first_loc_prev)
        self.first_transform = carla.Transform(carla.Location(self.first_loc_prev.x,
                                                              self.first_loc_prev.y,
                                                              self.first_loc_prev.z))
        first_debris = CarlaDataProvider.request_new_actor('static.prop.dirtdebris01', self.first_transform, 'prop')
        first_debris.set_transform(self.first_transform)
        first_debris.set_simulate_physics(False)
        self.other_actors.append(first_debris)

    def _create_behavior(self):
        """
        Ego vehicle is driving straight on the road
        Another car is cutting just in front, coming from left or right lane and taking emergency brake
        Order of sequence:
        - ActorTransformSetter: spawn car at a visible transform
        - control_loss: control_loss
        - StopVehicle: emergency brake
        - ActorDestroy: remove the actor
        - end condition: drive for a defined distance
        """

        # debris_visible
        debris_visible = ActorTransformSetter(self.other_actors[0], self.first_transform, physics=False)

        # add noise when approaching
        jitter = py_trees.composites.Sequence("Jitter Behavior")
        # jitter.add_child(InTriggerDistanceToLocation(self.ego_vehicles[0], self.first_loc_prev, self._start_distance))
        jitter.add_child(AddNoiseToVehicle(self.ego_vehicles[0], self._new_steer_noise, self._new_throttle_noise))

        # control_loss
        control_loss = py_trees.composites.Parallel(
            "DrivingStraight", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)
        control_loss.add_child(jitter)
        control_loss.add_child(Idle(1.5))

        # end condition
        end_condition = py_trees.composites.Parallel("Waiting for end position",
                                                     policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        end_condition.add_child(DriveDistance(self.ego_vehicles[0], self.vehicle_distance_driven))

        # build tree
        root = py_trees.composites.Sequence("Behavior")
        root.add_child(debris_visible)
        root.add_child(InTriggerDistanceToLocation(self.ego_vehicles[0], self.first_loc_prev, 2))
        root.add_child(control_loss)
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
