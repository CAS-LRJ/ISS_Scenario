from __future__ import print_function


import random
import math
import py_trees
import carla

from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import InTriggerDistanceToVehicle, DriveDistance

from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorTransformSetter,
                                                                      ActorDestroy,
                                                                      StopVehicle,
                                                                      SyncArrival,
                                                                      WaypointFollower, AddNoiseToVehicle, Idle)



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


class ControlWithEmergencyBraking(BasicScenario):
    """
    ControlWithEmergencyBraking scenario:
    Ego vehicle is driving straight on the road
    Another car front the ego emergency braking
    and ego control loss
    """
    timeout = 1200

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True,
                 timeout=600):
        self.timeout = timeout
        self.transform = None
        self.transform_obstacle = None
        self.first_loc = None
        self._config = config
        self._wmap = CarlaDataProvider.get_map()
        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)
        # end condition
        self.vehicle_distance_driven = 20

        # add noise
        if config._new_steer_noise is not None:
            self._new_steer_noise = config._new_steer_noise
        else:
            self._new_steer_noise = -0.05

        if config._new_throttle_noise is not None:
            self._new_throttle_noise = config._new_throttle_noise
        else:
            self._new_throttle_noise = 0.4

        # actor parameters
        # actor's speed
        if config._actor_vel is not None:
            self._velocity = config._actor_vel
        else:
            self._velocity = 10

        # trigger in a certain distance
        if config._trigger_distance is not None:
            self._trigger_distance = config._trigger_distance
        else:
            self._trigger_distance = 15

        # Initial position of the actor
        if config._start_distance is not None:
            self._start_distance = config._start_distance
        else:
            self._start_distance = 20

        if config._start_distance2 is not None:
            self._start_distance2 = config._start_distance2
        else:
            self._start_distance2 = 50

        if config._start_distance3 is not None:
            self._start_distance3 = config._start_distance3
        else:
            self._start_distance3 = 70

        # actor's brake
        if config._brake is not None:
            self.brake = config._brake
        else:
            self.brake = 0.5

        self.max_brake = 1.0

        super(ControlWithEmergencyBraking, self).__init__("ControlWithEmergencyBraking",
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
        # self.debug.draw_point((waypoint.previous(20.0)[-1]).transform.location + carla.Location(z=0.5), size=0.5,
        #                       life_time=0)
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
        # generate front vehicle
        location, _ = get_location_in_distance_from_wp(self._reference_waypoint, self._start_distance)
        waypoint = self._wmap.get_waypoint(location)
        lane_width = waypoint.lane_width
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
        first_vehicle = CarlaDataProvider.request_new_actor('vehicle.lincoln.mkz2017', first_vehicle_transform)
        first_vehicle.set_simulate_physics(True)
        self.other_actors.append(first_vehicle)

        # generate debris
        self.first_loc_prev, _ = get_location_in_distance_from_wp(self._reference_waypoint, self._start_distance2)
        self.first_transform = carla.Transform(self.first_loc_prev)
        self.first_transform = carla.Transform(carla.Location(self.first_loc_prev.x,
                                                              self.first_loc_prev.y,
                                                              self.first_loc_prev.z))
        first_debris = CarlaDataProvider.request_new_actor('static.prop.dirtdebris01', self.first_transform, 'prop')
        first_debris.set_transform(self.first_transform)
        first_debris.set_simulate_physics(False)
        self.other_actors.append(first_debris)

        # generate sec_vehicle
        location, _ = get_location_in_distance_from_wp(self._reference_waypoint, self._start_distance3)
        waypoint = self._wmap.get_waypoint(location)
        self.transform2, _ = self._calculate_base_transform(self._start_distance3, waypoint)
        sec_vehicle_transform = carla.Transform(
            carla.Location(self.transform2.location.x,
                           self.transform2.location.y,
                           self.transform2.location.z - 500),
            self.transform2.rotation)
        sec_vehicle = CarlaDataProvider.request_new_actor('vehicle.lincoln.mkz2017', sec_vehicle_transform)
        sec_vehicle.set_simulate_physics(True)
        self.other_actors.append(sec_vehicle)

    def _create_behavior(self):
        """
        Ego vehicle is driving straight on the road
        Another car front the ego emergency braking
        and ego control loss
        Order of sequence:
        - car_visible: spawn car at a visible transform
        - debris_visible: spawn debris at a visible transform
        - obstacle_visible: spawn obstacle at a visible transform
        - control_loss_with_emergency_brake: car front the ego emergency braking and ego control loss
        - ActorDestroy: remove the actor
        - end condition: drive for a defined distance
        """

        # car_visible debris_visible obstacle_visible
        car_visible = ActorTransformSetter(self.other_actors[0], self.transform)
        debris_visible = ActorTransformSetter(self.other_actors[1], self.first_transform, physics=False)
        obstacle_visible = ActorTransformSetter(self.other_actors[2], self.transform2)

        # trigger in a certain distance
        trigger_distance = InTriggerDistanceToVehicle(
            self.other_actors[0], self.other_actors[2], self._trigger_distance)

        # actor drive along the road
        just_drive = py_trees.composites.Parallel(
            "DrivingStraight", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        keep_v = WaypointFollower(self.other_actors[0], self._velocity)
        just_drive.add_child(keep_v)
        just_drive.add_child(trigger_distance)

        # add noise
        jitter = py_trees.composites.Parallel(
            "jitter", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)
        jitter.add_child(StopVehicle(self.other_actors[0], self.brake))
        jitter.add_child(AddNoiseToVehicle(self.ego_vehicles[0], self._new_steer_noise, self._new_throttle_noise))
        jitter.add_child((Idle(3)))

        # trigger in certain distance, actor make emergency brake ,and stop in a while
        behaviour = py_trees.composites.Sequence("Sequence Behavior")
        behaviour.add_child(just_drive)
        behaviour.add_child(jitter)
        behaviour.add_child(TimeOut(5))
        behaviour.add_child(ActorDestroy(self.other_actors[0]))

        # # add noise
        # jitter = py_trees.composites.Sequence("Jitter Behavior")
        # jitter.add_child(InTriggerDistanceToVehicle(self.ego_vehicles[0], self.first_loc_prev, 2))
        # jitter.add_child(AddNoiseToVehicle(self.ego_vehicles[0], self._new_steer_noise, self._new_throttle_noise))

        # control_loss_with_emergency_brake
        control_loss_with_emergency_brake = py_trees.composites.Parallel(
            "control_loss_with_emergency_brake", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)
        control_loss_with_emergency_brake.add_child(behaviour)
        # control_loss_with_emergency_brake.add_child(jitter)

        # end condition
        end_condition = py_trees.composites.Parallel("Waiting for end position",
                                                     policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
        end_condition.add_child(DriveDistance(self.ego_vehicles[0], self.vehicle_distance_driven))

        # build tree
        root = py_trees.composites.Sequence("Behavior")
        root.add_child(car_visible)
        root.add_child(debris_visible)
        root.add_child(obstacle_visible)
        root.add_child(control_loss_with_emergency_brake)
        root.add_child(ActorDestroy(self.other_actors[2]))
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
