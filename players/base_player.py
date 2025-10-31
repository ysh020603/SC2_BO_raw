from sc2.bot_ai import BotAI
from sc2.units import Units
from sc2.unit import Unit
from sc2.units import Units
from sc2.position import Point2
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId

import time
import os
import json
import math
import pandas as pd
import random

from tools.logger import setup_logger
from tools.format import extract_code, extract_first_number
from tools.ops import IterativeMean
import numpy as np
from sklearn.cluster import HDBSCAN, DBSCAN

class TargetType:
    NONE = "None"
    POINT = "Point"
    UNIT = "Unit"
    POINT_OR_UNIT = "PointOrUnit"


def load_knowledge():
    TerranAbilityData = pd.read_csv("knowledge/TerranAbility.csv")
    with open("knowledge/data.json", "r") as f:
        game_data = json.load(f)

    TerranAbility = {}
    for idx, item in TerranAbilityData.iterrows():
        ability = item["ability"]
        description = item["description"]
        ability_data = [item for item in game_data["Ability"] if item["name"] == ability]
        if len(ability_data) == 0:
            target = TargetType.NONE
        else:
            target = ability_data[0]["target"]
        if not isinstance(target, str):
            if "Build" in target:
                target = TargetType.POINT
            elif "BuildOnUnit" in target or "Unit" in target:
                target = TargetType.UNIT
            else:
                target = TargetType.NONE
        TerranAbility[ability] = {
            "enabled": item["enabled"],
            "description": description,
            "target": target,  # None, Point, Unit, PointOrUnit
        }
    return TerranAbility


TerranAbility = load_knowledge()


class BasePlayer(BotAI):
    def __init__(self, config, player_name, model_name, generation_config, llm_client, log_path="logs", enable_logging=True):
        super().__init__()

        self.config = config
        self.player_name = player_name
        self.model_name = model_name
        self.generation_config = generation_config
        self.llm_client = llm_client
        map_size = str(extract_first_number(config.map_name))
        self.map_name = f"{map_size}x{map_size}"

        time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        self.real_model_name = self.model_name.split("/")[-1]
        
        self.enable_logging = enable_logging
        if enable_logging:
            self.log_path = f"{log_path}/{self.real_model_name}/{time_str}"
            self.log_path = self.log_path.replace(':', '-')
            os.makedirs(f"{self.log_path}/observation", exist_ok=True)
            self.logger = setup_logger(f"{player_name}_{self.real_model_name}", log_dir=self.log_path)

        self.entity_id_name = {}
        self._tag_to_id = {}
        self._id_to_tag = {}
        self._id_to_abilities = {}
        self.next_id = 1

        self.last_action = []
        self.trace = {}
        self.tag_to_health = {}

        self.sbr = IterativeMean()
        self.resource_cost = 0
        
        self.miner_units = ["SCV", "Probe", "Drone"]
        self.basic_battle_units = ["Marine", "Zealot", "Zergling"]

        self.mineral_locations = [(102.5, 126.5), (75.5, 126.5), (127.5, 22.5), (126.5, 102.5), (126.5, 74.5), (24.5, 101.5), (25.5, 45.5), (24.5, 22.5), (127.5, 46.5), (49.5, 126.5), (76.5, 21.5), (102.5, 21.5), (49.5, 21.5), (25.5, 73.5), (24.5, 125.5), (127.5, 125.5)]
        self.scouted_locations = {}
        for mineral in self.mineral_locations:
            self.scouted_locations[mineral] = False
        self.enemy_locations = {}
    
    def logging(self, key: str, value, level="info", save_trace=False, save_file=False, print_log=True):
        if not self.enable_logging:
            return
        idx = self.state.game_loop // 4
        if level in ["info", "warning", "error"] and print_log:
            text = f"({idx}) {key}: {str(value)}"
            if level == "info":
                self.logger.info(text)
            elif level == "warning":
                self.logger.warning(text)
            elif level == "error":
                self.logger.error(text)

        if save_trace:
            if idx not in self.trace:
                self.trace[idx] = {}
            self.trace[idx][key] = value
            if idx % 500 == 0:
                with open(f"{self.log_path}/trace.json", "w", encoding="utf-8") as f:
                    json.dump(self.trace, f, indent=2, ensure_ascii=False)

        if save_file:
            with open(f"{self.log_path}/observation/{idx}-{key}.txt", "w", encoding="utf-8") as f:
                if isinstance(value, list) or isinstance(value, dict):
                    value = json.dumps(value, indent=2, ensure_ascii=False)
                f.write(value)

    async def on_end(self, game_result):
        game_result = game_result.name
        self.logging("game_result", game_result, save_trace=True)
        self.logging("SBR", round(self.sbr.mean, 4), save_trace=True)
        
        time_cost = self.time_formatted.split(":")
        time_cost = int(time_cost[0]) * 60 + int(time_cost[1])
        self.logging("time_cost", time_cost, save_trace=True)
        self.logging("RUR", round(self.resource_cost / time_cost, 4), save_trace=True)

        with open(f"{self.log_path}/trace.json", "w", encoding="utf-8") as f:
            json.dump(self.trace, f, indent=2, ensure_ascii=False)

    def update_tag_to_health(self):
        self.tag_to_health = {unit.tag: unit.health for unit in self.units}
        self.tag_to_health.update({unit.tag: unit.health for unit in self.structures})
    
    def get_lowest_health_enemy(self, units: Units):
        """Get the enemy unit with the lowest health."""
        if not units.exists:
            return None
        return min(units, key=lambda unit: unit.health + unit.shield)

    def _can_build(self, unit_type):
        """辅助函数，检查是否可以且尚未开始建造某个单位/建筑。"""
        return self.can_afford(unit_type) and not self.already_pending(unit_type)

    def get_total_amount(self, unit_type: UnitTypeId):
        """获取指定单位类型的总数量，包括正在建造的和已完成的。"""
        unit_amount = self.units(unit_type).amount
        structures_amount = self.structures(unit_type).amount
        pending_amount = self.already_pending(unit_type)
        return unit_amount + structures_amount + pending_amount
    async def on_step(self, iteration: int):
        if len(self.units) == 0 or len(self.townhalls) == 0:
            return
        self.sbr.update(int(self.supply_used == self.supply_cap))

        #### before run
        await self.run(iteration)
        #### after run

        if iteration % 15 == 0:
            self.update_tag_to_health()

    async def run(self, iteration: int):
        raise NotImplementedError

    def verify_actions(self, actions):
        if isinstance(actions, str):
            try:
                actions = json.loads(extract_code(actions))
            except json.JSONDecodeError:
                return False, "Action must be a json list wrapped with triple backticks and without comments"
        if not isinstance(actions, list):
            return False, "Action must be a list"

        errors = []
        cost_minerals, cost_vespene, cost_supply = 0, 0, 0
        for action in actions:
            ok, message_or_cost = self.check_action(action)
            if not ok:
                errors.append(json.dumps(action, indent=2, ensure_ascii=False) + "\n>> Error: " + message_or_cost)
            else:
                cost_minerals += message_or_cost[0]
                cost_vespene += message_or_cost[1]
                cost_supply += message_or_cost[2]
        
        if self.minerals < cost_minerals:
            errors.append(">>>> Total actions error: minerals is not enough for executing all actions")
        if self.vespene < cost_vespene:
            errors.append(">>>> Total actions error: vespene is not enough for executing all actions")
        if self.supply_left < cost_supply:
            errors.append(">>>> Total actions error: supply is not enough for executing all actions")
            
        if errors:
            return False, "\n\n".join(errors)
        return True, ""

    def check_action(self, action: dict):
        if not isinstance(action, dict):
            return False, "Action must be a dictionary"

        ### required keys checks
        # action check
        base_keys = ["action", "units"]
        for key in base_keys:
            if key not in action:
                return False, f"Missing required key: {key}"
        action_name = action["action"]
        if action_name not in TerranAbility:
            return False, f"Unknown action: {action['action']}"
        # target check
        target_type = TerranAbility[action_name]["target"]
        if target_type == TargetType.NONE:
            required_keys = base_keys
        elif target_type == TargetType.POINT:
            required_keys = base_keys + ["target_position"]
        elif target_type == TargetType.UNIT:
            required_keys = base_keys + ["target_unit"]

        if target_type != TargetType.POINT_OR_UNIT:
            unused_keys = [key for key in action.keys() if key not in required_keys]
            for key in required_keys:
                if key not in action:
                    return False, f"Missing required key: {key}"
        else:
            if "target_position" in action and "target_unit" in action:
                return False, "Cannot have both `target_position` and `target_unit`"
            if "target_position" not in action and "target_unit" not in action:
                return False, "Missing required key: target_position or target_unit"
            unused_keys = [key for key in action.keys() if key not in base_keys + ["target_position", "target_unit"]]
        ### unused keys check
        if unused_keys:
            # unused_keys = [key for key in action.keys() if key not in ["action", "units"]]
            return False, f"Unused keys: {unused_keys}"

        ### value check
        if not isinstance(action_name, str):
            return False, "`action` must be a string"
        if not (isinstance(action["units"], list) and len(action["units"]) > 0):
            return False, "`units` must be a non-empty list of integers"
        if "target_position" in action:
            if not (isinstance(action["target_position"], list) and len(action["target_position"]) == 2 and all(isinstance(i, int) for i in action["target_position"])):
                return False, "`target_position` must be a list of two integers"
        if "target_unit" in action:
            if not isinstance(action["target_unit"], int):
                return False, "`target_unit` must be an integer"
            if action["target_unit"] not in self._id_to_tag:
                return False, f"Unit with id {action['target_unit']} not found"
            target_unit = self.get_unit_by_id(action["target_unit"])
            if target_unit is None:
                return False, f"Unit with id {action['target_unit']} not found"

        ### unit checks
        if len(action["units"]) == 0:
            return False, "`units` must not be an empty list"
        for unit_id in action["units"]:
            if not isinstance(unit_id, int):
                return False, "`units` must be a list of integers"
            if unit_id not in self._id_to_tag:
                return False, f"Unit with id {unit_id} not found"
            if unit_id not in self._id_to_abilities:
                return False, f"Unit with id {unit_id} not found"
            unit = self.get_unit_by_id(unit_id)
            if not unit:
                return False, f"Unit {unit_id} doesn't exist"
            if not unit.is_mine:
                return False, f"Unit {unit_id} is not mine"
            if action_name not in self._id_to_abilities[unit_id]:
                return False, f"[{unit_id}]{unit.name} cannot perform action {action_name}"
            if unit.is_constructing_scv:
                return False, f"[{unit_id}]{unit.name} is constructing, cannot perform other actions"

        ### action_name check
        building_units = self.get_building_units()
        building_units = [name.lower() for name in building_units]
        # if action_name == "TERRANBUILD_SUPPLYDEPOT":
        #     if self.supply_cap - self.supply_used >= 7:
        #         return False, "There is still space for supply depot, no need to build new Supply Depot."
        
        ### resource check
        cost = self.calculate_cost(AbilityId[action_name]) * len(action["units"])
        if self.minerals < cost.minerals:
            return False, f"Minerals is not enough for action {action_name}"
        if self.vespene < cost.vespene:
            return False, f"Vespene is not enough for action {action_name}"
        try:
            supply_cost = self.calculate_supply_cost(AbilityId[action_name]) * len(action["units"])
            if self.supply_left < supply_cost:
                return False, f"Supply is not enough for action {action_name}"
        except KeyError:
            return True, [cost.minerals, cost.vespene, 0]
        
        ### Protoss Pylon check
        if self.config.own_race == "Protoss":
            if action_name == "PROTOSSBUILD_PYLON":
                pylons = self.units(UnitTypeId.PYLON)
                close_pylons = pylons.closer_than(5, action["target_position"])
 
                if close_pylons:
                    close_pylons_pos = [f"({int(p.position.x)}, {int(p.position.y)})" for p in close_pylons]
                    return False, f"Too close to other Pylons at positions: " + ", ".join(close_pylons_pos)

        return True, [cost.minerals, cost.vespene, supply_cost]

    def get_building_units(self):
        building_units = []
        for unit in self.units:
            if unit.build_progress < 1.0:
                building_units.append(unit)
        return [unit.name for unit in building_units]

    ################ tag id mapping
    def tag_to_id(self, tag: int):
        if tag not in self._tag_to_id:
            next_id = tag % 1000
            while next_id in self._id_to_tag:
                next_id = (next_id + 1) % 1000
            self._tag_to_id[tag] = next_id
            self._id_to_tag[next_id] = tag
        return self._tag_to_id[tag]

    def id_to_tag(self, _id: int):
        return self._id_to_tag[_id]

    def get_unit_by_tag(self, tag: int):
        unit = self.all_units.find_by_tag(tag)
        return unit

    def get_unit_by_id(self, _id: int):
        tag = self.id_to_tag(_id)
        return self.get_unit_by_tag(tag)

    async def find_placement(
        self,
        building,
        near: Point2,
        max_distance: int = 20,
        random_alternative: bool = True,
        placement_step: int = 2,
        addon_place: bool = False,
    ):
        """Finds a placement location for building.
        Example::
            if self.townhalls:
                cc = self.townhalls[0]
                depot_position = await self.find_placement(UnitTypeId.SUPPLYDEPOT, near=cc)
        :param building:
        :param near:
        :param max_distance:
        :param random_alternative:
        :param placement_step:
        :param addon_place:"""
        assert isinstance(building, (AbilityId, UnitTypeId))
        assert isinstance(near, Point2), f"{near} is no Point2 object"
        if isinstance(building, UnitTypeId):
            building_ability = self.game_data.units[building.value].creation_ability.id
        else:
            building_ability = building
        # 【修复第一步】: 如果需要检查附加建筑，提前获取补给站的建造AbilityId
        # 这是一个标准的2x2建筑，非常适合用来检查附加建筑的空位
        addon_check_ability = None
        if addon_place:
            addon_check_ability = self.game_data.units[UnitTypeId.SUPPLYDEPOT.value].creation_ability.id
        # 检查'near'点本身
        if await self.can_place_single(building_ability, near):
            if not addon_place or await self.can_place_single(addon_check_ability, near.offset((2.5, -0.5))):
                return near
        if max_distance == 0:
            return None
        # 在一个方形螺旋中搜索
        for distance in range(placement_step, max_distance, placement_step):
            possible_positions = [
                Point2(p).offset(near).to2
                for p in (
                    [(dx, -distance) for dx in range(-distance, distance + 1, placement_step)]
                    + [(dx, distance) for dx in range(-distance, distance + 1, placement_step)]
                    + [(-distance, dy) for dy in range(-distance, distance + 1, placement_step)]
                    + [(distance, dy) for dy in range(-distance, distance + 1, placement_step)]
                )
            ]
            
            # 过滤出可以放置主建筑的位置
            res = await self.client._query_building_placement_fast(building_ability, possible_positions)
            possible = [p for r, p in zip(res, possible_positions) if r]
            if not possible:
                continue
            if addon_place:
                # 【修复第二步】: 使用正确的AbilityId来检查附加建筑的位置
                res = await self.client._query_building_placement_fast(
                    addon_check_ability,  # <-- 使用正确的ID
                    [p.offset((2.5, -0.5)) for p in possible],
                )
                possible = [p for r, p in zip(res, possible) if r]
            if not possible:
                continue
            if random_alternative:
                return random.choice(possible)
            return min(possible, key=lambda p: p.distance_to_point2(near))
            
        return None

    ################ run actions
    async def run_actions(self, actions):
        for action in actions:
            try:
                action_check_result, action_check_msg = self.check_action(action)
                if not action_check_result:
                    action["is_valid"] = False
                    action["error"] = action_check_msg
                else:
                    for unit_id in action["units"]:
                        ability = AbilityId[action["action"]]
                        target = None
                        curr_unit = self.get_unit_by_id(unit_id)
                        available_abilities = await self.get_available_abilities([curr_unit])
                        assert ability in available_abilities[0], f"Unit {unit_id} cannot perform action {action['action']}"
                        if "target_unit" in action:
                            target = self.get_unit_by_id(action["target_unit"])
                            assert target is not None, f"Unit with id {action['target_unit']} not found"
                        elif "target_position" in action:
                            target = Point2(action["target_position"])
                            need_addon = ability in [AbilityId.TERRANBUILD_BARRACKS, AbilityId.TERRANBUILD_FACTORY, AbilityId.TERRANBUILD_STARPORT]
                            if "BUILD_" in ability.name:
                                target = await self.find_placement(ability, target, max_distance=100, random_alternative=False, addon_place=need_addon)
                            assert target is not None, f"Invalid target position: {action['target_position']}"
                        ###### run action
                        run_state = curr_unit(ability=ability, target=target)
                        assert run_state, "Failed to execute command due to unknown error."
                        ###### run action end
                        # Chat send
                        target_str = "None"
                        if isinstance(target, Unit):
                            target_str = target.name
                        elif isinstance(target, Point2):
                            target_str = f"({int(target.x)}, {int(target.y)})"
                        await self.chat_send(f"{action['action']}({curr_unit.name} -> {target_str})")
                        # update cost
                        cost = self.calculate_cost(ability)
                        self.resource_cost += cost.minerals + cost.vespene
            except Exception as e:
                action["is_valid"] = False
                action["error"] = str(e)

        valid_actions = [action for action in actions if action.get("is_valid", True)]
        self.logging("valid_actions", valid_actions, save_trace=True, print_log=False)
        self.logging("valid_actions", "\n" + json.dumps(valid_actions, indent=2, ensure_ascii=False))

        valid_actions = [json.dumps(action, ensure_ascii=False) for action in valid_actions]
        self.last_action.extend(valid_actions)
    
    def get_mineral_locations(self):
        possible_locations = list(self.expansion_locations.keys())
        start_location = self.start_location
        
        def distance(p1, p2):
            return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

        structures = self.structures
        for structure in structures:
            if structure.name in ['CommandCenter', 'OrbitalCommand', 'PlanetaryFortress']:
                for idx, loc in enumerate(possible_locations):
                    if distance(loc, structure.position) < 5:
                        del possible_locations[idx]
                        break
        
        sorted_coordinates = sorted(possible_locations, key=lambda p: distance(p, start_location))

        return [(int(c[0]), int(c[1])) for c in sorted_coordinates]
    
    def history_prompt(self):
        for mineral in self.mineral_locations:
            if self.scouted_locations[mineral] == True:
                continue
            
            if self.is_visible(Point2(mineral)):
                self.scouted_locations[mineral] = True
        
        new_enemy_locations = {}
        # 如果视野范围内有这个地点，但是没有这个单位，就把它从这个 list 中移除
        for key in self.enemy_locations.keys():
            enemy_struct = self.enemy_locations[key]
            # if self.state.visibility.data_numpy[int(enemy_struct['coor'][0])][int(enemy_struct['coor'][1])] == 0:
            if not self.is_visible(Point2(enemy_struct['coor'])):
                new_enemy_locations[key] = enemy_struct
            else:
                unitID_list = []
                for structure in self.enemy_structures:
                    unitID = self.tag_to_id_new(structure.tag)
                    unitID_list.append(unitID)

                if key in unitID_list:
                    new_enemy_locations[key] = enemy_struct
                else:
                    print(f"{enemy_struct['name']} destroyed.")
        
        for structure in self.enemy_structures:
            unitID = self.tag_to_id_new(structure.tag)
            if unitID not in new_enemy_locations.keys():
                new_enemy_locations[unitID] = {
                    'name': structure.name,
                    'coor': structure.position
                }
        
        self.enemy_locations = new_enemy_locations

        self.enemy_location_minerals = {}
        for mineral in self.mineral_locations:
            self.enemy_location_minerals[mineral] = []
        
        def distance(p1, p2):
            return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

        # 计算单位距离哪个矿最近
        for unitID in self.enemy_locations.keys():
            dist_min = 2000
            mineral_min = None
            for mineral in self.mineral_locations:
                dis = distance(mineral, self.enemy_locations[unitID]['coor'])
                if dis < dist_min:
                    dist_min = dis
                    mineral_min = mineral
            self.enemy_location_minerals[mineral_min].append(unitID)
        
        prompt_scout = f"Coordinate you can scout about: {[ (int(loc[0]), int(loc[1])) for loc in self.mineral_locations]}. \n"

        prompt_scouted = f"Coordinate you have scouted: ["
        for mineral in self.mineral_locations:
            if self.scouted_locations[mineral]:
                prompt_scouted += f"{(int(mineral[0]), int(mineral[1]))}, "
        prompt_scouted += "]\n"
        print(prompt_scouted)
        prompt_enemy = f"Coordinate having enemy's structures: "
        for mineral in self.enemy_location_minerals.keys():
            if len(self.enemy_location_minerals[mineral]) == 0:
                continue
            else:
                prompt_enemy += f"coordinate {mineral} has {len(self.enemy_location_minerals[mineral])} enemy's structures; "
        
        return prompt_scout + prompt_scouted + prompt_enemy

    ################ obs to text
    async def obs_to_text(self):
        scout_info = self.history_prompt()
        cc_locations = self.get_mineral_locations()
        self._record_entity_ids()
        
        if len(cc_locations):
            cc_loc = "\nHere are the coordinates you can consider for building your Command Center. These coordinate can only be used to build Command Center. " + str(cc_locations)
        else:
            cc_loc = ""

        obs = {}
        obs["Round state"] = self.round_state_to_text()
        obs["Own units"] = await self.units_to_text(self.units)
        obs["Unit abilities"] = await self.abilities_to_text(self.units)
        obs["Own structures"] = await self.structures_to_text(self.structures)
        obs["Structure abilities"] = await self.abilities_to_text(self.structures)
        obs["Visible enemy units"] = await self.units_to_text(self.enemy_units)
        obs["Visible enemy structures"] = await self.structures_to_text(self.enemy_structures)
        obs["Scout info"] = scout_info
        obs["Action history"] = self.action_history_to_text()
        obs["Map information"] = self.miner_to_text() + "\n" + self.gas_to_text() + cc_loc
        obs["Ability description"] = self.get_ability_desc(obs["Unit abilities"] + obs["Structure abilities"])
        obs_text = "\n\n".join([f"# {key}\n{value}" for key, value in obs.items()])

        self.logging("obs", obs, save_trace=True, print_log=False)
        if self.enable_logging:
            self.logging("obs_text", obs_text, save_file=True, print_log=False)
        return obs_text

    def get_ability_desc(self, text: str):
        desc = []
        for action in TerranAbility:
            if TerranAbility[action].get("enabled", False) and action in text:
                action_desc = TerranAbility[action]["description"]
                action_keys = TerranAbility[action]["target"]
                desc.append(f"{action}(target: {action_keys}): {action_desc}")
                try:
                    cost = self.units[0]._bot_object.game_data.calculate_ability_cost(AbilityId[action])
                    if cost.minerals and cost.vespene:
                        desc[-1] += f" Cost: {cost.minerals} minerals, {cost.vespene} vespene."
                    elif cost.vespene:
                        desc[-1] += f" Cost: {cost.vespene} vespene."
                    elif cost.minerals:
                        desc[-1] += f" Cost: {cost.minerals} minerals."
                except Exception as e:
                    pass
        return "\n".join(desc)

    def round_state_to_text(self):
        text = ""
        text += "Time: {}\n".format(self.time_formatted)
        text += "Race: {}\n".format(self.race.name)
        text += "Minerals: {}\n".format(self.minerals)
        text += "Vespene: {}\n".format(self.vespene)
        text += "Supply used: {}\n".format(self.supply_used)
        text += "Supply army: {}\n".format(self.supply_army)
        text += "Supply workers: {}\n".format(self.supply_workers)
        text += "Supply unused: {}\n".format(self.supply_cap - self.supply_used)
        text += "Map size: {}".format(self.map_name)

        return text.strip()

    def action_history_to_text(self):
        if len(self.last_action) == 0:
            return "[Empty]"
        return "\n".join(self.last_action[-10:])

    # async def units_to_text(self, units: Units):
    #     if len(units) == 0:
    #         return "[Empty]"
        
    #     units_text = []
        
    #     other_units = units
    #     for mining_type in self.miner_units:
    #         mining_judge = lambda unit: unit.name == mining_type and not (unit.is_constructing_scv or unit.is_repairing or unit.is_attacking)
    #         mining_units = units.filter(mining_judge)
    #         if len(mining_units) > 0:
    #             mining_ids = [self.tag_to_id(unit.tag) for unit in mining_units]
    #             mining_ids = ", ".join(map(str, mining_ids))
    #             mining_text = f"[{mining_ids}]{mining_type}\nState: collecting resources automatically"
    #             units_text.append(mining_text)
    #         other_units = [unit for unit in other_units if not mining_judge(unit)]
            
    #         attacking_judge = lambda unit: unit.name == mining_type and unit.is_attacking
    #         attacking_units = units.filter(attacking_judge)
    #         if len(attacking_units) > 0:
    #             attacking_ids = [self.tag_to_id(unit.tag) for unit in attacking_units]
    #             attacking_ids = ", ".join(map(str, attacking_ids))
    #             attacking_text = f"[{attacking_ids}]{mining_type}\nState: attacking enemies automatically"
    #             units_text.append(attacking_text)

    #     distance_to_start = lambda unit: int((unit.position.x - self.start_location.x) ** 2 + (unit.position.y - self.start_location.y) ** 2) // 4
    #     other_units = sorted(other_units, key=lambda unit: (distance_to_start(unit), unit.name))
    #     units_text += [await self.unit_to_text(unit) for unit in other_units]
    #     units_text = "\n".join(units_text)
    #     return units_text
    
    async def cluster(self, units_text, battle_type, units, type_name):
        PARAMS = {
            # 'min_cluster_size': min(len(units), 3),
            'min_cluster_size': 2,
            'store_centers': "centroid",
            'cluster_selection_epsilon': 2
        }

        if len(units) == 0:
            return units_text
        elif len(units) == 1:
            units_text += [await self.unit_to_text(unit) for unit in units]
            return units_text
        else:
            clusterer = HDBSCAN(**PARAMS)
            # clusterer = HDBSCAN(, store_centers='centroid')
            positions = np.array([unit.position_tuple for unit in units])
            labels = clusterer.fit_predict(positions)
            centers = clusterer.centroids_
            for label in set(labels):
                cluster_units = [units[i] for i in range(len(units)) if labels[i] == label]
                if label != -1:
                    ids = [self.tag_to_id_new(unit.tag) for unit in cluster_units]
                    ids = ", ".join(map(str, ids))
                    cluster_center = tuple(centers[label])
                    text = f"[{ids}]{battle_type}\nGroup Center: {tuple([int(p) for p in cluster_center])}\nState: {type_name}"
                    units_text.append(text)
                else:     # all single unit(label == -1)
                    for unit in cluster_units:
                        id = self.tag_to_id_new(unit.tag)
                        position = unit.position_tuple
                        text = f"[{id}]{battle_type}\nPosition: {tuple([int(p) for p in position])}\nState: {type_name}"
                        units_text.append(text)
        return units_text

    async def units_to_text(self, units: Units):
        if len(units) == 0:
            return "[Empty]"

        units_text = []

        other_units = units
        for mining_type in self.miner_units:
            mining_judge = lambda unit: unit.name == mining_type and unit.is_mine and not (unit.is_constructing_scv or unit.is_repairing or unit.is_attacking)
            not_mining_judge = lambda unit: not (unit.name == mining_type and unit.is_mine and not (unit.is_constructing_scv or unit.is_repairing or unit.is_attacking))
            mining_units = other_units.filter(mining_judge)
            if len(mining_units) > 0:
                mining_ids = [self.tag_to_id_new(unit.tag) for unit in mining_units]
                mining_ids = ", ".join(map(str, mining_ids))
                mining_text = f"[{mining_ids}]{mining_type}\nState: collecting resources automatically"
                units_text.append(mining_text)
            other_units = other_units.filter(not_mining_judge)

            attacking_judge = lambda unit: unit.name == mining_type and unit.is_mine and unit.is_attacking
            not_attacking_judge = lambda unit: not (unit.name == mining_type and unit.is_mine and unit.is_attacking)
            attacking_units = other_units.filter(attacking_judge)
            if len(attacking_units) > 0:
                attacking_ids = [self.tag_to_id_new(unit.tag) for unit in attacking_units]
                attacking_ids = ", ".join(map(str, attacking_ids))
                attacking_text = f"[{attacking_ids}]{mining_type}\nState: attacking enemies automatically"
                units_text.append(attacking_text)
            other_units = other_units.filter(not_attacking_judge)
        
        for battle_type in self.basic_battle_units:
            attacking_judge = lambda unit: unit.name == battle_type and unit.is_mine and unit.is_attacking
            not_attacking_judge = lambda unit: not (unit.name == battle_type and unit.is_mine and unit.is_attacking)
            attacking_units = other_units.filter(attacking_judge)
            units_text = await self.cluster(units_text, battle_type, attacking_units, 'attacking')
            other_units = other_units.filter(not_attacking_judge)
            
            moving_judge = lambda unit: unit.name == battle_type and unit.is_mine and unit.is_moving
            not_moving_judge = lambda unit: not (unit.name == battle_type and unit.is_mine and unit.is_moving)
            moving_units = other_units.filter(moving_judge)
            units_text = await self.cluster(units_text, battle_type, moving_units, 'moving')
            other_units = other_units.filter(not_moving_judge)

            idle_judge = lambda unit: unit.name == battle_type and unit.is_mine and unit.is_idle
            not_idle_judge = lambda unit: not (unit.name == battle_type and unit.is_mine and unit.is_idle)
            idle_units = other_units.filter(idle_judge)
            units_text = await self.cluster(units_text, battle_type, idle_units, 'idle')
            other_units = other_units.filter(not_idle_judge)

        distance_to_start = lambda unit: int((unit.position.x - self.start_location.x) ** 2 + (unit.position.y - self.start_location.y) ** 2) // 4
        other_units = sorted(other_units, key=lambda unit: (distance_to_start(unit), unit.name))
        units_text += [await self.unit_to_text(unit) for unit in other_units]

        units_text = "\n".join(units_text)
        return units_text
    
    async def structures_to_text(self, structures: Units):
        if len(structures) == 0:
            return "[Empty]"
        structures = [s for s in structures]
        sorted_structures = []
        current_x, current_y = self.start_location.x, self.start_location.y
        while structures:
            closest_structure = min(structures, key=lambda s: math.sqrt((s.position.x - current_x) ** 2 + (s.position.y - current_y) ** 2))
            sorted_structures.append(closest_structure)
            structures.remove(closest_structure)
            current_x, current_y = closest_structure.position.x, closest_structure.position.y
        return "\n".join([await self.unit_to_text(structure) for structure in sorted_structures])

    async def unit_to_text(self, unit: Unit):
        text = ""

        if unit.build_progress == 1.0:
            text += f"[{self.tag_to_id_new(unit.tag)}]{unit.name}\n"
        else:
            text += f"[{self.tag_to_id_new(unit.tag)}]{unit.name}(building {int(unit.build_progress * 100)}%)\n"
        text += f"Position: ({int(unit.position.x)}, {int(unit.position.y)})\n"

        if unit.build_progress == 1.0:
            if int(unit.health_max) and unit.build_progress == 1.0:
                text += f"Health: {int(unit.health)}/{int(unit.health_max)} ({int(unit.health_percentage * 100)}%)\n"
            if unit.shield_max > 0.0:
                text += f"Shield: {int(unit.shield)}/{int(unit.shield_max)}\n"
            if unit.energy_max > 0.0:
                text += f"Energy: {int(unit.energy)}/{int(unit.energy_max)}\n"
            if unit.name == 'Liberator':
                text += f"You can only attack air forces in this formation. Use MORPH_LIBERATORAGMODE first if you want to attach ground unit.\n"

            if unit.is_mine:
                states = self.unit_state_to_text(unit)
                if states:
                    text += f"State: {states}\n"

                if unit.is_structure:
                    # Supply information
                    assigned = unit.assigned_harvesters
                    ideal = unit.ideal_harvesters
                    surplus = unit.surplus_harvesters
                    if ideal > 0:
                        if surplus > 0:
                            text += f"Harvesters: {assigned}/{ideal} (no more harvesters accepted, surplus {surplus})\n"
                        elif surplus == 0:
                            text += f"Harvesters: {assigned}/{ideal} (no more harvesters accepted)\n"
                        else:
                            text += f"Harvesters: {assigned}/{ideal}\n"

                # Production list
                production_list = []
                unit_orders = unit.orders
                for unit_order in unit_orders:
                    if "Train " in unit_order.ability.friendly_name:
                        production_list.append(unit_order.ability.friendly_name[6:])
                if production_list:
                    text += f"Production list: {', '.join(production_list)}\n"
        return text.strip()

    async def abilities_to_text(self, units: Units):
        units = [unit for unit in units if unit.build_progress == 1.0]
        n_units = len(units)
        units_ability_ids = await self.get_available_abilities(units, ignore_resource_requirements=True)
        units_ability_names = [[ability_id.name for ability_id in units_ability_ids[i]] for i in range(n_units)]
        unit_hash_table = {}
        for i in range(n_units):
            unit = units[i]
            ability_names = units_ability_names[i]
            ability_names = [name for name in ability_names if name != "NULL_NULL"]
            unknown_abilities = [name for name in ability_names if name not in TerranAbility]
            if unknown_abilities:
                print(f"Unit {unit.name} has unknown abilities: {unknown_abilities}")
                import pdb; pdb.set_trace()
            if unit.name in self.miner_units:
                ability_names = [name for name in ability_names if name not in ["MOVE_MOVE", "ATTACK_ATTACK"]]
            ability_names = [name for name in ability_names if TerranAbility[name].get("enabled", False)]
            self._id_to_abilities[self.tag_to_id_new(unit.tag)] = ability_names
            
            unit_hash = unit.name + "|" + ", ".join(ability_names)
            if unit_hash not in unit_hash_table:
                unit_hash_table[unit_hash] = []
            unit_hash_table[unit_hash].append(str(self.tag_to_id_new(unit.tag)))
            
        text = ""
        for unit_hash, ids in unit_hash_table.items():
            unit_name, abilities = unit_hash.split("|")
            ids = ", ".join(ids)
            if abilities:
                text += f"{unit_name}[{ids}]: {abilities}\n"
        
        text = text.strip()
        if not text:
            text = "[Empty]"
        return text

    def unit_state_to_text(self, unit: Unit):
        order_target = unit.order_target or ""
        order_target_name = ""
        if order_target:
            if isinstance(order_target, Point2):
                order_target = f"({int(order_target.x)}, {int(order_target.y)})"
            elif isinstance(order_target, int):
                target_unit = self.get_unit_by_tag(order_target)
                if target_unit:
                    order_target_name = self.get_unit_by_tag(order_target).name
                    order_target = self.tag_to_id_new(order_target)

        states = []
        if unit.is_moving:
            if order_target:
                states.append(f"moving to [{order_target}]{order_target_name}")
            else:
                states.append("moving")
        if unit.is_attacking:
            if order_target:
                states.append(f"attacking [{order_target}]{order_target_name}")
            else:
                states.append("attacking")
        if unit.is_repairing:
            if order_target:
                states.append(f"repairing [{order_target}]{order_target_name}")
            else:
                states.append("repairing")

        if unit.is_idle:
            states.append("idle")
        if unit.is_flying:
            states.append("flying")
        if unit.is_transforming:
            states.append("transforming")
        if unit.is_patrolling:
            states.append("patrolling")
        if unit.tag in self.tag_to_health and unit.health < self.tag_to_health[unit.tag]:
            states.append("under attack")

        if unit.is_constructing_scv:
            if order_target:
                states.append(f"constructing [{order_target}]{order_target_name}")
            else:
                states.append("constructing")

        return "|".join(states)

    def miner_to_text(self):
        center = self.start_location
        miners = []
        num_SCV = len([unit for unit in self.units if unit.name == "SCV"])
        cloest_miners = self.mineral_field.closest_n_units(center, 100)
        cloest_miners = [mineral for mineral in cloest_miners if mineral.mineral_contents > 0]
        cloest_miners = cloest_miners[: 2 * num_SCV]
        for mineral in cloest_miners:
            miners.append(f"[{self.tag_to_id_new(mineral.tag)}]({int(mineral.position.x)}, {int(mineral.position.y)})")
        if len(miners) == 0:
            return "No mineral fields found"
        return "Closest mineral fields: " + ", ".join(miners)

    def gas_to_text(self):
        gases = []
        cloest_gases = self.vespene_geyser.closest_n_units(self.start_location, 100)
        cloest_gases = [gas for gas in cloest_gases if gas.vespene_contents > 0]
        cloest_gases = cloest_gases[:10]
        for gas in cloest_gases:
            # # if there is a structure on the position, ignore it
            # if self.structures.closer_than(0.5, gas.position):
            #     continue
            gases.append(f"[{self.tag_to_id_new(gas.tag)}]({int(gas.position.x)}, {int(gas.position.y)})")
        if len(gases) == 0:
            return "No vespene geysers found"
        return "Closest vespene geysers: " + ", ".join(gases)

    ##### for sc2mm
    # 记录所有我方和敌方的单位/建筑的名称、Tag和ID
    def _record_entity_ids(self):
        """
        将所有(包括已死亡)我方和敌方的单位/建筑，以及地图资源 (矿、气) 的
        名称、Tag、ID、new_id 和存活状态 (is_alive) 记录到 
        self.entity_id_name 变量中。

        同时，将此信息保存到 log_path/entity_id_name/ 文件夹下，
        并以当前帧数 (self.step_count) 命名。
        这个函数中的 is_alive 对于己方单位，指的是单位是否存活，
        对于敌方单位，指的是单位是否可观测到，is_alive 为零表示不可观测到或死亡。
        
        [MODIFIED] new_id 现在存储为 int 类型的序列号。
        [MODIFIED] 新增对 mineral_field 和 vespene_geyser 的追踪。
        """

        # 1. 如果未启用日志记录，则直接返回
        if not self.enable_logging:
            return

        # 2. 初始化核心数据结构 (如果不存在)
        # _entity_tag_map: 核心维护表 {tag: {name, id, new_id, ...}}
        # 这一次，它将存储 *所有* 见过的单位，*不会* 删除已死亡的单位。
        if not hasattr(self, '_entity_tag_map'):
            self._entity_tag_map = {}

        # _entity_new_id_counters: 追踪每种类型的下一个可用序列号
        if not hasattr(self, '_entity_new_id_counters'):
            self._entity_new_id_counters = {}
            
        # 3. 准备本帧的日志目录
        if self.enable_logging:
            # 确保 entity_id_name 子目录存在
            self.entity_log_dir = os.path.join(self.log_path, "entity_id_name")
            try:
                os.makedirs(self.entity_log_dir, exist_ok=True)
            except Exception as e:
                self.logger.error(f"Failed to create entity_id_name directory: {e}")
                self.enable_logging = False # 发生IO错误时停止日志记录
                return

        # 4. 辅助函数：处理一个实体列表
        def process_entity_list(entity_list, alliance, entity_type, live_tags_set):
            """
            遍历一个单位列表 (如 self.units)，
            将 tag 添加到 live_tags_set 中，
            并为新单位在 _entity_tag_map 中创建条目。
            """
            for unit in entity_list:
                try:
                    tag = unit.tag
                    
                    # 将此 tag 标记为本帧存活
                    live_tags_set.add(tag)

                    # 如果是新单位，则创建条目
                    if tag not in self._entity_tag_map:
                        name = unit.name
                        # 1. 获取 LLM simple_id
                        simple_id = self.tag_to_id(tag)
                        
                        # 2. 生成 new_id
                        counter_key = f"{alliance}_{entity_type}_{name}"
                        next_seq_num = self._entity_new_id_counters.get(counter_key, 1)
                        # new_id = f"{name}_{next_seq_num}" # <--- [OLD]
                        new_id = next_seq_num # <--- [NEW]
                        self._entity_new_id_counters[counter_key] = next_seq_num + 1

                        # 3. 存入 _entity_tag_map
                        # (存储 _alliance 和 _type 以便后续分组)
                        self._entity_tag_map[tag] = {
                            "name": name,
                            "tag": tag,
                            "id": simple_id,
                            "new_id": new_id, # <--- [NEW] 存储为 int
                            "_alliance": alliance,
                            "_type": entity_type
                        }
                except Exception as e:
                    self.logger.warning(f"Failed to process new entity {unit.tag} for ID logging: {e}")

        # 5. 遍历所有当前存活的单位，更新 _entity_tag_map
        current_live_tags = set()
        process_entity_list(self.units, "allied", "units", current_live_tags)
        process_entity_list(self.structures, "allied", "structures", current_live_tags)
        process_entity_list(self.enemy_units, "enemy", "units", current_live_tags)
        process_entity_list(self.enemy_structures, "enemy", "structures", current_live_tags)
        
        # --- [NEW] 新增对 mineral 和 gas 的处理 ---
        # self.mineral_field 包含所有可见的晶体矿
        process_entity_list(self.mineral_field, "neutral", "mineral", current_live_tags) 
        # self.vespene_geyser 包含所有可见的气矿点 (无论上面是否有建筑)
        process_entity_list(self.vespene_geyser, "neutral", "gas", current_live_tags)
        # --- [END NEW] ---

        # 6. 基于 _entity_tag_map 重建本帧的 entity_data
        current_entity_data = {
            "allied": {"units": {}, "structures": {}},
            "enemy": {"units": {}, "structures": {}},
            "mineral": {}, # <--- [NEW]
            "gas": {}      # <--- [NEW]
        }

        # 遍历 *所有* 见过的单位，并设置其 is_alive 状态
        for tag, entity_info in self._entity_tag_map.items():
            
            # 判断存活状态
            is_alive = 1 if tag in current_live_tags else 0
            
            # 准备要记录的条目
            log_entry = {
                "name": entity_info["name"],
                "tag": entity_info["tag"],
                "id": entity_info["id"],
                "new_id": entity_info["new_id"],
                "is_alive": is_alive  # <--- 新增的键
            }

            # --- [MODIFIED] 将条目放入 current_entity_data 的正确分组中 ---
            try:
                alliance = entity_info["_alliance"]
                entity_type = entity_info["_type"]
                name = entity_info["name"]

                name_group_container = None
                # 路由到正确的容器
                if alliance in ["allied", "enemy"]:
                    name_group_container = current_entity_data[alliance][entity_type]
                elif entity_type == "mineral": # 处理 'neutral' alliance, 'mineral' type
                    name_group_container = current_entity_data["mineral"]
                elif entity_type == "gas":     # 处理 'neutral' alliance, 'gas' type
                    name_group_container = current_entity_data["gas"]
                else:
                    # 跳过其他不追踪的 'neutral' 类型
                    continue 

                # 将条目添加到其 name 对应的列表中
                if name not in name_group_container:
                    name_group_container[name] = []
                
                name_group_container[name].append(log_entry)
                
            except KeyError as e:
                self.logger.error(f"Failed to group entity {tag}. Data: {entity_info}. Error: {e}")
            # --- [END MODIFIED] ---


        # 7. 排序 (按 new_id 的序列号排序)
        def sort_by_new_id_number(entry):
            try:
                # return int(entry["new_id"].split('_')[-1]) # <--- [OLD]
                return entry["new_id"] # <--- [NEW] 直接按 int 排序
            except:
                return 0
            
        for alliance in ["allied", "enemy"]:
            for entity_type in ["units", "structures"]:
                name_groups = current_entity_data[alliance][entity_type]
                for name in name_groups:
                    name_groups[name].sort(key=sort_by_new_id_number)

        # --- [NEW] 新增对 mineral 和 gas 的排序 ---
        for entity_type in ["mineral", "gas"]:
            name_groups = current_entity_data[entity_type]
            for name in name_groups:
                name_groups[name].sort(key=sort_by_new_id_number)
        # --- [END NEW] ---

        # 8. 将 *当前* 的数据赋给 self.entity_id_name
        self.entity_id_name = current_entity_data

        # 9. 将数据写入 *特定于本帧* 的JSON文件
        if self.enable_logging:
            
            # 格式化文件名，例如：000120_entity_id_name.json
            frame_file_name = f"{self.state.game_loop // 4}_entity_id_name.json"
            file_path = os.path.join(self.entity_log_dir, frame_file_name)
            
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    json.dump(self.entity_id_name, f, indent=4, ensure_ascii=False)
            except Exception as e:
                self.logger.error(f"Failed to write entity log to {file_path}: {e}")

        # 10. 返回更新后的 self.entity_id_name
        return self.entity_id_name
    def tag_to_id_new(self, tag: int):
            """
            通用化版本：
            递归遍历 self.entity_id_name (任意嵌套结构),
            收集所有包含 "tag" 键的字典,
            然后在收集中查找匹配的 "tag" 并返回对应的 "new_id"。
            (根据您的请求修改)
            """
            
            # 1. 用于存储所有找到的包含 "tag" 键的字典
            all_tagged_items = []

            def find_items_recursive(data_structure):
                """
                递归助手函数，用于遍历数据结构并填充 all_tagged_items 列表。
                """
                
                # --- 情况 A: 如果当前数据是字典 (dict) ---
                if isinstance(data_structure, dict):
                    
                    # 检查当前字典是否直接包含 "tag" 键
                    if "tag" in data_structure:
                        # 如果是，将这个字典（元素）添加到收集中
                        all_tagged_items.append(data_structure)
                    
                    # 无论是否包含 "tag"，都必须继续递归遍历它的 *值*
                    for value in data_structure.values():
                        find_items_recursive(value)
                
                # --- 情况 B: 如果当前数据是列表 (list) 或 元组 (tuple) ---
                elif isinstance(data_structure, (list, tuple)):
                    
                    # 递归遍历列表中的 *每个元素*
                    for item in data_structure:
                        find_items_recursive(item)
                
                # --- 情况 C: 其他类型 (int, str, None...) ---
                # 基本类型，停止递归
                else:
                    pass

            # --- 函数主逻辑 ---
            
            # 2. 从 self.entity_id_name 的顶层开始递归查找
            find_items_recursive(self.entity_id_name)
            
            # 3. 遍历收集到的所有包含 "tag" 键的元素
            for item in all_tagged_items:
                
                # 检查 "tag" 的值是否与我们寻找的 tag 相匹配
                if item.get("tag") == tag:
                    
                    # 找到匹配项，返回对应的 "new_id"
                    return item.get("new_id")

            # 4. 如果遍历完整个列表都没有找到匹配项，则返回 None
            # (如果您的代码期望一个整数，您可以返回 -1 或 0)
            return None