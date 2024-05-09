#!/usr/bin/env python

# Copyright (c) 2019-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides Challenge routes as standalone scenarios
"""

from __future__ import print_function

import math
import traceback
import xml.etree.ElementTree as ET
import numpy.random as random

import py_trees

import carla

from agents.navigation.local_planner import RoadOption

# pylint: disable=line-too-long
from srunner.scenarioconfigs.scenario_configuration import ScenarioConfiguration, ActorConfigurationData
# pylint: enable=line-too-long
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import Idle, ScenarioTriggerer
from srunner.scenarios.beside_vehicle_control_loss import BesideVehicleControlLoss
from srunner.scenarios.bike_cross_intersection_obliquely import BikeCrossIntersectionObliquely
from srunner.scenarios.catch_cut_in_cut_out_with_slow_vehicle import CatchCutInCutOutWithSlowVehicle
from srunner.scenarios.cross_the_road_obliquely import CrossTheRoadObliquely
from srunner.scenarios.cut_in_with_emergency_brake import FrontCutInWithBrake, BehindCutInWithBrake
from srunner.scenarios.cut_in_with_left_vehicles_turn_left import CutInWithLeftVehiclesTurnLeft
from srunner.scenarios.cut_in_with_obstacle import CutInWithObstacle
from srunner.scenarios.cut_out_with_cut_in import CutOutWithCutIn
from srunner.scenarios.cut_out_with_slow_vehicle import CutOutWithSlowVehicle
from srunner.scenarios.fellow_front_vehicle_turn_r_with_pedcross import FellowFrontVehicleTurnRWithPedCross
from srunner.scenarios.follow_leading_vehicle import FollowLeadingVehicle
from srunner.scenarios.front_vehicle_turn_r_with_bike_cross_intersection_obliquely import \
    FrontVehicleTurnRWithBikeCrossIntersectionObliquely
from srunner.scenarios.front_vehicle_turn_r_with_stationarybike import FrontVehicleTurnRWithStationBike
from srunner.scenarios.l_vehicle_straight_r_vehicle_left_with_pedcrossl import \
    LVehicleStraightRightVehicleTLWithPedCrossL
from srunner.scenarios.left_and_opposite_vehicle_go_straight import LeftOppositeVehicleGoStraight
from srunner.scenarios.left_vehicle_go_straight_right_motors_turn_r_go_straight import \
    LeftVehicleGoStraightRightMotorsTurnRGoStraight
from srunner.scenarios.lr_bike_cross_intersection_obliquely import LRBikeCrossIntersectionObliquely
from srunner.scenarios.overtaking_encounter_vehicles import OvertakingEncounterVehicle
from srunner.scenarios.right_bike_go_straight_with_stationary_vehicle import RightBikeGoStraightWithStationVehicle
from srunner.scenarios.left_right_opposite_vehicle_go_straight import LeftRightOppositeVehicleGoStraight
from srunner.scenarios.left_right_opposite_vehicle_turn_left import LeftRightOppositeVehicleTurnLeft
from srunner.scenarios.left_vehicle_turnl_right_vehicle_turnr import LeftVehicleTurnLRightVehicleTurnR
from srunner.scenarios.left_vehicles_go_straight import LeftVehiclesGoStraight
from srunner.scenarios.merge_into_the_main_road import MergeIntoTheMainRoad

from srunner.scenarios.basic_scenario import BasicScenario
from srunner.scenarios.control_loss_with_emergency_brake import ControlWithEmergencyBraking
from srunner.scenarios.cross_the_road import CrossTheRoad
from srunner.scenarios.cut_out import CutOut
from srunner.scenarios.cut_out_with_obstacle import CutOutWithObstacle
from srunner.scenarios.occupy_the_middle_line import OccupyTheMiddleLine
from srunner.scenarios.emergency_braking import EmergencyBraking
from srunner.scenarios.entering_ramp import EnteringRamp
from srunner.scenarios.fellow_vehicle_stop_go import FellowVehicleStopGo
from srunner.scenarios.follow_pedestrian import FollowPedestrian
from srunner.scenarios.left_cut_in import FrontLeftCutIn, BehindLeftCutIn, FrontLeftCutInWithObstacleAvoid
from srunner.scenarios.left_vehicles_turn_left import LeftVehiclesTurnLeft
from srunner.scenarios.left_vehicles_turn_right import LeftVehiclesTurnRight
from srunner.scenarios.object_crash_intersection_retrograde import VehicleTurningRouteWithRetrograde
from srunner.scenarios.opposit_go_straight_vehicles import OppositeGoStraightVehicles
from srunner.scenarios.opposit_left_turning_vehicles import OppositeLeftTurningVehicles
from srunner.scenarios.opposit_right_turning_vehicles import OppositeRightTurningVehicles
from srunner.scenarios.pedestrian_cross_both_side import PedestrianCrossBothSide
from srunner.scenarios.pedestrian_cross_with_left_vehicles_turn_left import PedCrossLeftVehiclesTurnLeft
from srunner.scenarios.bike_go_straight_both_intersection import PedestrianGoStraightBothIntersection
from srunner.scenarios.retrograde import Retrograde
from srunner.scenarios.right_and_opposite_vehicle_go_straight import RightOppositeVehicleGoStraight
from srunner.scenarios.right_cut_in import FrontRightCutIn, BehindRightCutIn
from srunner.scenarios.right_motors_turn_r_go_straight import RightMotorsTurnRGoStraight
from srunner.scenarios.right_vehicles_go_straight import RightVehiclesGoStraight
from srunner.scenarios.right_vehicles_turn_left import RightVehiclesTurnLeft
from srunner.scenarios.right_vehicles_turn_right import RightVehiclesTurnRight
from srunner.scenarios.two_vehicles_front_cut_in import TwoVehiclesFrontCutIn
from srunner.scenarios.vehicle_both_side_turn_left import VehicleBothSideTurnLeft
from srunner.scenarios.vehicle_control_loss import VehicleControlLoss
from srunner.scenarios.vehicle_cross_both_side import VehicleCrossBothSide
from srunner.scenarios.vehicle_cross_both_side_with_pedcross import VehicleCrossBothSideWithPedCross
from srunner.scenarios.vehicle_cut_in_cut_out_with_slow_vehicle import VehicleCutInCutOutWithSlowVehicle
from srunner.scenarios.vehicle_cut_in_with_slow_vehicle import VehicleCutInWithSlowVehicle
from srunner.scenarios.vehicle_reverse import VehicleReverse
from srunner.tools.route_parser import RouteParser, TRIGGER_THRESHOLD, TRIGGER_ANGLE_THRESHOLD
from srunner.tools.route_manipulation import interpolate_trajectory
from srunner.tools.py_trees_port import oneshot_behavior

from srunner.scenarios.control_loss import ControlLoss
from srunner.scenarios.object_crash_vehicle import DynamicObjectCrossing
from srunner.scenarios.object_crash_intersection import VehicleTurningRoute
from srunner.scenarios.other_leading_vehicle import OtherLeadingVehicle
from srunner.scenarios.maneuver_opposite_direction import ManeuverOppositeDirection
from srunner.scenarios.junction_crossing_route import SignalJunctionCrossingRoute, NoSignalJunctionCrossingRoute

from srunner.scenarios.Ringroad import RingRoad
from srunner.scenarios.signalized_junction_left_turn import SignalizedJunctionLeftTurn
from srunner.scenarios.signalized_junction_right_turn import SignalizedJunctionRightTurn
from srunner.scenarios.no_signal_junction_cross import NoSignalJunctionCross
from srunner.scenarios.street_barrier import StreetBarrier
from srunner.scenarios.vehicle_barrier import VehicleBarrier
from srunner.scenarios.through_a_red_light import ThroughRedLight
from srunner.scenarios.pedestrian_crossing import PedestrianCrossing, LeftPedestrianCrossing
from srunner.scenarios.crossing_behind_vehicles import CrossingBehindVehicles

from srunner.scenariomanager.scenarioatomics.atomic_criteria import (CollisionTest,
                                                                     InRouteTest,
                                                                     RouteCompletionTest,
                                                                     OutsideRouteLanesTest,
                                                                     RunningRedLightTest,
                                                                     RunningStopTest,
                                                                     ActorSpeedAboveThresholdTest)

SECONDS_GIVEN_PER_METERS = 10

NUMBER_CLASS_TRANSLATION = {
    "Scenario1": ControlLoss,
    "Scenario2": FollowLeadingVehicle,
    "Scenario3": DynamicObjectCrossing,
    "Scenario4": VehicleTurningRoute,
    "Scenario5": OtherLeadingVehicle,
    "Scenario6": ManeuverOppositeDirection,
    "Scenario7": SignalJunctionCrossingRoute,
    "Scenario8": SignalJunctionCrossingRoute,
    "Scenario9": SignalJunctionCrossingRoute,
    "Scenario10": NoSignalJunctionCrossingRoute,

    "Scenario11": LeftRightOppositeVehicleGoStraight,
    "Scenario12": RightOppositeVehicleGoStraight,
    "Scenario13": CutOutWithCutIn,
    "Scenario14": CrossTheRoadObliquely,
    "Scenario15": VehicleBothSideTurnLeft,
    "Scenario16": VehicleCrossBothSide,
    "Scenario17": RingRoad,
    "Scenario18": NoSignalJunctionCross,
    "Scenario19": SignalizedJunctionLeftTurn,
    "Scenario20": SignalizedJunctionRightTurn,
    "Scenario21": StreetBarrier,
    "Scenario22": VehicleBarrier,
    "Scenario23": ThroughRedLight,
    "Scenario24": PedestrianCrossing,
    "Scenario25": CrossingBehindVehicles,
    "Scenario26": LeftPedestrianCrossing,
    "Scenario27": FollowPedestrian,
    "Scenario28": CrossTheRoad,
    "Scenario29": CutOutWithObstacle,
    "Scenario30": FrontLeftCutInWithObstacleAvoid,
    "Scenario31": BehindLeftCutIn,
    "Scenario32": FrontRightCutIn,
    "Scenario33": BehindRightCutIn,
    "Scenario34": FellowVehicleStopGo,
    "Scenario35": CutOut,
    "Scenario36": OccupyTheMiddleLine,
    "Scenario37": MergeIntoTheMainRoad,
    "Scenario38": EnteringRamp,
    "Scenario39": EmergencyBraking,
    "Scenario40": Retrograde,
    "Scenario41": VehicleTurningRouteWithRetrograde,
    "Scenario42": ControlWithEmergencyBraking,
    "Scenario43": OppositeRightTurningVehicles,
    "Scenario44": OppositeLeftTurningVehicles,
    "Scenario45": OppositeGoStraightVehicles,
    "Scenario46": LeftVehiclesTurnRight,
    "Scenario47": LeftVehiclesTurnLeft,
    "Scenario48": LeftVehiclesGoStraight,
    "Scenario49": RightVehiclesGoStraight,
    "Scenario50": RightVehiclesTurnLeft,
    "Scenario51": RightVehiclesTurnRight,
    "Scenario52": FrontCutInWithBrake,
    "Scenario53": BehindCutInWithBrake,
    "Scenario54": VehicleControlLoss,
    "Scenario55": BesideVehicleControlLoss,
    "Scenario56": VehicleReverse,
    "Scenario57": CutInWithObstacle,
    "Scenario58": PedCrossLeftVehiclesTurnLeft,
    "Scenario59": CutInWithLeftVehiclesTurnLeft,
    "Scenario60": PedestrianCrossBothSide,
    "Scenario61": VehicleCrossBothSideWithPedCross,
    "Scenario62": LeftRightOppositeVehicleTurnLeft,
    "Scenario63": CutOutWithSlowVehicle,
    "Scenario64": CatchCutInCutOutWithSlowVehicle,
    "Scenario65": LVehicleStraightRightVehicleTLWithPedCrossL,
    "Scenario66": LeftVehicleTurnLRightVehicleTurnR,
    "Scenario67": PedestrianGoStraightBothIntersection,
    "Scenario68": RightBikeGoStraightWithStationVehicle,
    "Scenario69": BikeCrossIntersectionObliquely,
    "Scenario70": FrontVehicleTurnRWithBikeCrossIntersectionObliquely,
    "Scenario71": RightMotorsTurnRGoStraight,
    "Scenario72": LeftVehicleGoStraightRightMotorsTurnRGoStraight,
    "Scenario73": OvertakingEncounterVehicle,
    "Scenario74": LRBikeCrossIntersectionObliquely,
    "Scenario75": LeftOppositeVehicleGoStraight,
    "Scenario76": TwoVehiclesFrontCutIn,
    "Scenario77": VehicleCutInWithSlowVehicle,
    "Scenario78": VehicleCutInCutOutWithSlowVehicle,
    "Scenario79": FrontVehicleTurnRWithStationBike,
    "Scenario80": FellowFrontVehicleTurnRWithPedCross,

}


def convert_json_to_float(actor_dict):
    """
    Convert a JSON string to float
    """
    return float(actor_dict)


def convert_json_to_start_distance(actor_dict):
    """
    Convert a JSON string to start_distance
    """
    return float(actor_dict)


def convert_json_to_start_distance2(actor_dict):
    """
    Convert a JSON string to start_distance
    """
    return float(actor_dict)


def convert_json_to_start_distance3(actor_dict):
    """
    Convert a JSON string to start_distance
    """
    return float(actor_dict)


def convert_json_to_actor_vel(actor_dict):
    """
    Convert a JSON string to actor_vel
    """
    # # print(actor_dict['_actor_vel'])
    # if actor_dict['_actor_vel'] is not None:
    return float(actor_dict)


def convert_json_to_trigger_distance(actor_dict):
    """
    Convert a JSON string to trigger_distance
    """
    return float(actor_dict)


def convert_json_to_brake(actor_dict):
    """
    Convert a JSON string to brake
    """
    return float(actor_dict)


def convert_json_to_model(actor_dict):
    """
    Convert a JSON string to model
    """
    return float(actor_dict)


def convert_json_to_new_steer_noise(actor_dict):
    """
    Convert a JSON string to _new_steer_noise
    """
    return float(actor_dict)


def convert_json_to_new_throttle_noise(actor_dict):
    """
    Convert a JSON string to _new_throttle_noise
    """
    return float(actor_dict)


def convert_json_to_weather(actor_dict):
    """
    Convert a JSON string to weather
    """
    weather = carla.WeatherParameters()
    if 'cloudiness' in actor_dict:
        weather.cloudiness = float(actor_dict['cloudiness'])
    if 'precipitation' in actor_dict:
        weather.precipitation = float(actor_dict['precipitation'])
    if 'precipitation_deposits' in actor_dict:
        weather.precipitation_deposits = float(actor_dict['precipitation_deposits'])
    if 'wind_intensity' in actor_dict:
        weather.wind_intensity = float(actor_dict['wind_intensity'])
    if 'sun_azimuth_angle' in actor_dict:
        weather.sun_azimuth_angle = float(actor_dict['sun_azimuth_angle'])
    if 'sun_altitude_angle' in actor_dict:
        weather.sun_altitude_angle = float(actor_dict['sun_altitude_angle'])
    if 'wetness' in actor_dict:
        weather.wetness = float(actor_dict['wetness'])
    if 'fog_distance' in actor_dict:
        weather.fog_distance = float(actor_dict['fog_distance'])
    if 'fog_density' in actor_dict:
        weather.fog_density = float(actor_dict['fog_density'])
    return weather


def convert_json_to_transform(actor_dict):
    """
    Convert a JSON string to a CARLA transform
    """
    return carla.Transform(location=carla.Location(x=float(actor_dict['x']), y=float(actor_dict['y']),
                                                   z=float(actor_dict['z'])),
                           rotation=carla.Rotation(roll=0.0, pitch=0.0, yaw=float(actor_dict['yaw'])))


def convert_json_to_actor(actor_dict):
    """
    Convert a JSON string to an ActorConfigurationData dictionary
    """
    node = ET.Element('waypoint')
    node.set('x', actor_dict['x'])
    node.set('y', actor_dict['y'])
    node.set('z', actor_dict['z'])
    node.set('yaw', actor_dict['yaw'])

    return ActorConfigurationData.parse_from_node(node, 'simulation')


def convert_transform_to_location(transform_vec):
    """
    Convert a vector of transforms to a vector of locations
    """
    location_vec = []
    for transform_tuple in transform_vec:
        location_vec.append((transform_tuple[0].location, transform_tuple[1]))

    return location_vec


def compare_scenarios(scenario_choice, existent_scenario):
    """
    Compare function for scenarios based on distance of the scenario start position
    """

    def transform_to_pos_vec(scenario):
        """
        Convert left/right/front to a meaningful CARLA position
        """
        position_vec = [scenario['trigger_position']]
        if scenario['other_actors'] is not None:
            if 'left' in scenario['other_actors']:
                position_vec += scenario['other_actors']['left']
            if 'front' in scenario['other_actors']:
                position_vec += scenario['other_actors']['front']
            if 'right' in scenario['other_actors']:
                position_vec += scenario['other_actors']['right']

        return position_vec

    # put the positions of the scenario choice into a vec of positions to be able to compare

    choice_vec = transform_to_pos_vec(scenario_choice)
    existent_vec = transform_to_pos_vec(existent_scenario)
    for pos_choice in choice_vec:
        for pos_existent in existent_vec:

            dx = float(pos_choice['x']) - float(pos_existent['x'])
            dy = float(pos_choice['y']) - float(pos_existent['y'])
            dz = float(pos_choice['z']) - float(pos_existent['z'])
            dist_position = math.sqrt(dx * dx + dy * dy + dz * dz)
            dyaw = float(pos_choice['yaw']) - float(pos_choice['yaw'])
            dist_angle = math.sqrt(dyaw * dyaw)
            if dist_position < TRIGGER_THRESHOLD and dist_angle < TRIGGER_ANGLE_THRESHOLD:
                return True

    return False


class RouteScenario(BasicScenario):
    """
    Implementation of a RouteScenario, i.e. a scenario that consists of driving along a pre-defined route,
    along which several smaller scenarios are triggered
    """

    def __init__(self, world, config, debug_mode=False, criteria_enable=True, timeout=300):
        """
        Setup all relevant parameters and create scenarios along route
        """

        self.config = config
        self.route = None
        self.sampled_scenarios_definitions = None

        self._update_route(world, config, debug_mode)

        ego_vehicle = self._update_ego_vehicle()

        self.list_scenarios = self._build_scenario_instances(world,
                                                             ego_vehicle,
                                                             self.sampled_scenarios_definitions,
                                                             scenarios_per_tick=5,
                                                             timeout=self.timeout,
                                                             debug_mode=debug_mode)

        super(RouteScenario, self).__init__(name=config.name,
                                            ego_vehicles=[ego_vehicle],
                                            config=self.config,
                                            world=world,
                                            debug_mode=False,
                                            terminate_on_failure=False,
                                            criteria_enable=criteria_enable)

    def _update_route(self, world, config, debug_mode):
        """
        Update the input route, i.e. refine waypoint list, and extract possible scenario locations

        Parameters:
        - world: CARLA world
        - config: Scenario configuration (RouteConfiguration)
        """

        # Transform the scenario file into a dictionary
        world_annotations = RouteParser.parse_annotations_file(config.scenario_file)

        # prepare route's trajectory (interpolate and add the GPS route)
        gps_route, route = interpolate_trajectory(world, config.trajectory)

        potential_scenarios_definitions, _ = RouteParser.scan_route_for_scenarios(config.town, route, world_annotations)

        self.route = route
        CarlaDataProvider.set_ego_vehicle_route(convert_transform_to_location(self.route))

        if config.agent is not None:
            config.agent.set_global_plan(gps_route, self.route)

        # Sample the scenarios to be used for this route instance.
        self.sampled_scenarios_definitions = self._scenario_sampling(potential_scenarios_definitions)

        # Timeout of scenario in seconds
        self.timeout = self._estimate_route_timeout()

        # Print route in debug mode
        if debug_mode:
            self._draw_waypoints(world, self.route, vertical_shift=1.0, persistency=50000.0)

    def _update_ego_vehicle(self):
        """
        Set/Update the start position of the ego_vehicle
        """
        # move ego to correct position
        elevate_transform = self.route[0][0]
        elevate_transform.location.z += 0.5

        ego_vehicle = CarlaDataProvider.request_new_actor('vehicle.lincoln.mkz2017',
                                                          elevate_transform,
                                                          rolename='hero')

        return ego_vehicle

    def _estimate_route_timeout(self):
        """
        Estimate the duration of the route
        """
        route_length = 0.0  # in meters

        prev_point = self.route[0][0]
        for current_point, _ in self.route[1:]:
            dist = current_point.location.distance(prev_point.location)
            route_length += dist
            prev_point = current_point

        return int(SECONDS_GIVEN_PER_METERS * route_length)

    # pylint: disable=no-self-use
    def _draw_waypoints(self, world, waypoints, vertical_shift, persistency=-1):
        """
        Draw a list of waypoints at a certain height given in vertical_shift.
        """
        for w in waypoints:
            wp = w[0].location + carla.Location(z=vertical_shift)

            size = 0.2
            if w[1] == RoadOption.LEFT:  # Yellow
                color = carla.Color(255, 255, 0)
            elif w[1] == RoadOption.RIGHT:  # Cyan
                color = carla.Color(0, 255, 255)
            elif w[1] == RoadOption.CHANGELANELEFT:  # Orange
                color = carla.Color(255, 64, 0)
            elif w[1] == RoadOption.CHANGELANERIGHT:  # Dark Cyan
                color = carla.Color(0, 64, 255)
            elif w[1] == RoadOption.STRAIGHT:  # Gray
                color = carla.Color(128, 128, 128)
            else:  # LANEFOLLOW
                color = carla.Color(0, 255, 0)  # Green
                size = 0.1

            world.debug.draw_point(wp, size=size, color=color, life_time=persistency)

        world.debug.draw_point(waypoints[0][0].location + carla.Location(z=vertical_shift), size=0.2,
                               color=carla.Color(0, 0, 255), life_time=persistency)
        world.debug.draw_point(waypoints[-1][0].location + carla.Location(z=vertical_shift), size=0.2,
                               color=carla.Color(255, 0, 0), life_time=persistency)

    def _scenario_sampling(self, potential_scenarios_definitions, random_seed=0):
        """
        The function used to sample the scenarios that are going to happen for this route.
        """

        # fix the random seed for reproducibility
        rng = random.RandomState(random_seed)

        def position_sampled(scenario_choice, sampled_scenarios):
            """
            Check if a position was already sampled, i.e. used for another scenario
            """
            for existent_scenario in sampled_scenarios:
                # If the scenarios have equal positions then it is true.
                if compare_scenarios(scenario_choice, existent_scenario):
                    return True

            return False

        # The idea is to randomly sample a scenario per trigger position.
        sampled_scenarios = []
        for trigger in potential_scenarios_definitions.keys():
            possible_scenarios = potential_scenarios_definitions[trigger]

            scenario_choice = rng.choice(possible_scenarios)
            del possible_scenarios[possible_scenarios.index(scenario_choice)]
            # We keep sampling and testing if this position is present on any of the scenarios.
            while position_sampled(scenario_choice, sampled_scenarios):
                if possible_scenarios is None or not possible_scenarios:
                    scenario_choice = None
                    break
                scenario_choice = rng.choice(possible_scenarios)
                del possible_scenarios[possible_scenarios.index(scenario_choice)]

            if scenario_choice is not None:
                sampled_scenarios.append(scenario_choice)

        return sampled_scenarios

    def _build_scenario_instances(self, world, ego_vehicle, scenario_definitions,
                                  scenarios_per_tick=5, timeout=300, debug_mode=False):
        """
        Based on the parsed route and possible scenarios, build all the scenario classes.
        """
        scenario_instance_vec = []

        if debug_mode:
            for scenario in scenario_definitions:
                loc = carla.Location(scenario['trigger_position']['x'],
                                     scenario['trigger_position']['y'],
                                     scenario['trigger_position']['z']) + carla.Location(z=2.0)
                world.debug.draw_point(loc, size=0.3, color=carla.Color(255, 0, 0), life_time=100000)
                world.debug.draw_string(loc, str(scenario['name']), draw_shadow=False,
                                        color=carla.Color(0, 0, 255), life_time=100000, persistent_lines=True)

        for scenario_number, definition in enumerate(scenario_definitions):
            # Get the class possibilities for this scenario number
            scenario_class = NUMBER_CLASS_TRANSLATION[definition['name']]

            # Create the other actors that are going to appear
            if definition['other_actors'] is not None:
                list_of_actor_conf_instances = self._get_actors_instances(definition['other_actors'])
            else:
                list_of_actor_conf_instances = []
            # Create an actor configuration for the ego-vehicle trigger position

            egoactor_trigger_position = convert_json_to_transform(definition['trigger_position'])
            scenario_configuration = ScenarioConfiguration()
            # It is used to control the position of the actor generation, for example
            # (in the scene where there is a vehicle on the opposite side of the intersection when turning left,
            # the distance from other vehicles to the intersection is _start_distance)
            #
            if definition['_start_distance'] is not None:
                scenario_configuration._start_distance = convert_json_to_start_distance(definition['_start_distance'])
            else:
                scenario_configuration._start_distance = None
            if definition['_start_distance2'] is not None:
                scenario_configuration._start_distance2 = convert_json_to_start_distance2(
                    definition['_start_distance2'])
            else:
                scenario_configuration._start_distance2 = None
            if definition['_start_distance3'] is not None:
                scenario_configuration._start_distance3 = convert_json_to_start_distance3(
                    definition['_start_distance3'])
            else:
                scenario_configuration._start_distance3 = None

            # actor's speed
            if definition['_actor_vel'] is not None:
                scenario_configuration._actor_vel = convert_json_to_actor_vel(definition['_actor_vel'])
            else:
                scenario_configuration._actor_vel = None

            if definition['_actor_vel2'] is not None:
                scenario_configuration._actor_vel2 = convert_json_to_actor_vel(definition['_actor_vel2'])
            else:
                scenario_configuration._actor_vel2 = None

            if definition['_actor_vel3'] is not None:
                scenario_configuration._actor_vel3 = convert_json_to_float(
                    definition['_actor_vel3'])
            else:
                scenario_configuration._actor_vel3 = None

            # _trigger_distance
            if definition['_trigger_distance'] is not None:
                scenario_configuration._trigger_distance = convert_json_to_trigger_distance(
                    definition['_trigger_distance'])
            else:
                scenario_configuration._trigger_distance = None
            if definition['_trigger_distance2'] is not None:
                scenario_configuration._trigger_distance2 = convert_json_to_float(
                    definition['_trigger_distance2'])
            else:
                scenario_configuration._trigger_distance2 = None

            # actor's braking coefficient
            if definition['_brake'] is not None:
                scenario_configuration._brake = convert_json_to_brake(
                    definition['_brake'])
            else:
                scenario_configuration._brake = None

            # model ped or bike
            if definition['_model'] is not None:
                scenario_configuration._model = convert_json_to_model(definition['_model'])
            else:
                scenario_configuration._model = None

            # noise
            if definition['_new_steer_noise'] is not None:
                scenario_configuration._new_steer_noise = convert_json_to_new_steer_noise(
                    definition['_new_steer_noise'])
            else:
                scenario_configuration._new_steer_noise = None
            if definition['_new_steer_noise2'] is not None:
                scenario_configuration._new_steer_noise2 = convert_json_to_new_steer_noise(
                    definition['_new_steer_noise2'])
            else:
                scenario_configuration._new_steer_noise2 = None
            if definition['_new_throttle_noise'] is not None:
                scenario_configuration._new_throttle_noise = convert_json_to_new_throttle_noise(
                    definition['_new_throttle_noise'])
            else:
                scenario_configuration._new_throttle_noise = None

            # weather
            if definition['weather'] is not None:
                scenario_configuration.weather = convert_json_to_weather(
                    definition['weather'])
                self.config.weather = convert_json_to_weather(
                    definition['weather'])
            else:
                scenario_configuration.weather = carla.WeatherParameters(sun_altitude_angle=70)

            # _traffic_distance
            if definition['_traffic_distance'] is not None:
                scenario_configuration._traffic_distance = convert_json_to_float(
                    definition['_traffic_distance'])
            else:
                scenario_configuration._traffic_distance = None

            scenario_configuration.other_actors = list_of_actor_conf_instances
            scenario_configuration.trigger_points = [egoactor_trigger_position]
            scenario_configuration.subtype = definition['scenario_type']
            scenario_configuration.ego_vehicles = [ActorConfigurationData('vehicle.lincoln.mkz2017',
                                                                          ego_vehicle.get_transform(),
                                                                          'hero')]
            route_var_name = "ScenarioRouteNumber{}".format(scenario_number)
            scenario_configuration.route_var_name = route_var_name

            try:
                scenario_instance = scenario_class(world, [ego_vehicle], scenario_configuration,
                                                   criteria_enable=False, timeout=timeout)
                # Do a tick every once in a while to avoid spawning everything at the same time
                if scenario_number % scenarios_per_tick == 0:
                    if CarlaDataProvider.is_sync_mode():
                        world.tick()
                    else:
                        world.wait_for_tick()

                scenario_number += 1
            except Exception as e:  # pylint: disable=broad-except
                if debug_mode:
                    traceback.print_exc()
                print("Skipping scenario '{}' due to setup error: {}".format(definition['name'], e))
                continue

            scenario_instance_vec.append(scenario_instance)

        return scenario_instance_vec

    def _get_actors_instances(self, list_of_antagonist_actors):
        """
        Get the full list of actor instances.
        """

        def get_actors_from_list(list_of_actor_def):
            """
                Receives a list of actor definitions and creates an actual list of ActorConfigurationObjects
            """
            sublist_of_actors = []
            for actor_def in list_of_actor_def:
                sublist_of_actors.append(convert_json_to_actor(actor_def))

            return sublist_of_actors

        list_of_actors = []
        # Parse vehicles to the left
        if 'front' in list_of_antagonist_actors:
            list_of_actors += get_actors_from_list(list_of_antagonist_actors['front'])

        if 'left' in list_of_antagonist_actors:
            list_of_actors += get_actors_from_list(list_of_antagonist_actors['left'])

        if 'right' in list_of_antagonist_actors:
            list_of_actors += get_actors_from_list(list_of_antagonist_actors['right'])

        return list_of_actors

    # pylint: enable=no-self-use

    def _initialize_actors(self, config):
        """
        Set other_actors to the superset of all scenario actors
        """

        # Create the background activity of the route
        # 3，120 4,200 6,150 7,110 1,120 2,100
        town_amount = {
            'Town01': 0,
            'Town02': 0,
            'Town03': 0,
            'Town04': 0,
            'Town05': 0,
            'Town06': 0,
            'Town07': 0,
            'Town08': 180,
            'Town09': 300,
            'Town10': 120,
        }

        amount = town_amount[config.town] if config.town in town_amount else 0

        new_actors = CarlaDataProvider.request_new_batch_actors('vehicle.*',
                                                                amount,
                                                                carla.Transform(),
                                                                autopilot=True,
                                                                random_location=True,
                                                                rolename='background')

        if new_actors is None:
            raise Exception("Error: Unable to add the background activity, all spawn points were occupied")

        for _actor in new_actors:
            self.other_actors.append(_actor)

        # Add all the actors of the specific scenarios to self.other_actors
        for scenario in self.list_scenarios:
            self.other_actors.extend(scenario.other_actors)

    def _create_behavior(self):
        """
        Basic behavior do nothing, i.e. Idle
        """
        scenario_trigger_distance = 1.5  # Max trigger distance between route and scenario

        behavior = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

        subbehavior = py_trees.composites.Parallel(name="Behavior",
                                                   policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)

        scenario_behaviors = []
        blackboard_list = []

        for i, scenario in enumerate(self.list_scenarios):
            if scenario.scenario.behavior is not None:
                route_var_name = scenario.config.route_var_name
                if route_var_name is not None:
                    # print(route_var_name)
                    scenario_behaviors.append(scenario.scenario.behavior)
                    blackboard_list.append([scenario.config.route_var_name,
                                            scenario.config.trigger_points[0].location])
                else:
                    name = "{} - {}".format(i, scenario.scenario.behavior.name)
                    oneshot_idiom = oneshot_behavior(name,
                                                     behaviour=scenario.scenario.behavior,
                                                     name=name)
                    scenario_behaviors.append(oneshot_idiom)

        # Add behavior that manages the scenarios trigger conditions
        scenario_triggerer = ScenarioTriggerer(
            self.ego_vehicles[0],
            self.route,
            blackboard_list,
            scenario_trigger_distance,
            repeat_scenarios=False
        )

        subbehavior.add_child(scenario_triggerer)  # make ScenarioTriggerer the first thing to be checked
        subbehavior.add_children(scenario_behaviors)
        subbehavior.add_child(Idle())  # The behaviours cannot make the route scenario stop
        behavior.add_child(subbehavior)

        return behavior

    def _create_test_criteria(self):
        """
        """

        criteria = []

        route = convert_transform_to_location(self.route)

        collision_criterion = CollisionTest(self.ego_vehicles[0], terminate_on_failure=False)

        route_criterion = InRouteTest(self.ego_vehicles[0],
                                      route=route,
                                      offroad_max=30,
                                      terminate_on_failure=True)

        completion_criterion = RouteCompletionTest(self.ego_vehicles[0], route=route)

        outsidelane_criterion = OutsideRouteLanesTest(self.ego_vehicles[0], route=route)

        red_light_criterion = RunningRedLightTest(self.ego_vehicles[0])

        stop_criterion = RunningStopTest(self.ego_vehicles[0])

        blocked_criterion = ActorSpeedAboveThresholdTest(self.ego_vehicles[0],
                                                         speed_threshold=0.1,
                                                         below_threshold_max_time=90.0,
                                                         terminate_on_failure=True)

        criteria.append(completion_criterion)
        criteria.append(collision_criterion)
        criteria.append(route_criterion)
        criteria.append(outsidelane_criterion)
        criteria.append(red_light_criterion)
        criteria.append(stop_criterion)
        criteria.append(blocked_criterion)

        return criteria

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()
