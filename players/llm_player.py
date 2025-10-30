from .base_player import BasePlayer, load_knowledge
from agents import PlanAgent, ActionAgent, RagAgent, SingleAgent
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.upgrade_id import UpgradeId
from sc2.ids.buff_id import BuffId
from sc2.ids.ability_id import AbilityId

import random

import json

import re

import copy
import pickle

import math

ignore_list = [
    'HARVEST_GATHER_SCV',
    'COMMANDCENTERTRAIN_SCV', # 生成 SCV
    'MOVE_MOVE',
    'SMART',
    'ATTACK_ATTACK',
    'STOP_STOP',
    'UNSIEGE_UNSIEGE',
    'SIEGEMODE_SIEGEMODE', # 坦克模式
    'MORPH_SUPPLYDEPOT_LOWER', # 降低资源房
    'RALLY_BUILDING',
    'MORPH_LIBERATORAAMODE', # liberator 切换为 AA mode 
    'MORPH_LIBERATORAGMODE',
    'LOAD_MEDIVAC',
    'UNLOADALLAT_MEDIVAC',
    'EFFECT_STIM_MARINE',
    'BURROWDOWN_WIDOWMINE',
    'BURROWUP_WIDOWMINE',
    'EFFECT_MEDIVACIGNITEAFTERBURNERS'
]

start_with_list = [
    'TERRANBUILD',
    'BUILD',
    'UPGRADETOORBITAL',
    'ENGINEERINGBAYRESEARCH',
    'BARRACKSTRAIN',
    'STARPORTTRAIN',
    'FACTORYTRAIN',
    'RESEARCH',
    'BARRACKSTECHLABRESEARCH',
    'UPGRADETOPLANETARYFORTRESS',
    'CALLDOWNMULE'
]

class LLMPlayer(BasePlayer):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        
        agent_config = {
            "model_name": self.model_name,
            "generation_config": self.generation_config,
            "llm_client": self.llm_client,
        }

        if config.enable_rag:
            self.rag_agent = RagAgent(config.own_race, **agent_config)
        if config.enable_plan or config.enable_plan_verifier:
            self.plan_agent = PlanAgent(config.own_race, **agent_config)
            self.action_agent = ActionAgent(config.own_race, **agent_config)
        else:
            self.agent = SingleAgent(config.own_race, **agent_config)

        self.plan_verifier = "llm" if config.enable_plan_verifier else None
        self.action_verifier = self.verify_actions if self.config.enable_action_verifier else None

        self.next_decision_time = -1
        
        # SCV auto-attack settings
        self.scv_auto_attack_distance = 4
        self.scv_auto_attack_time = 240
        
        self.init_bo()

        self.last_decision_minerals = 100

        self.TerranAbility = load_knowledge()

        self.recorded_actions = []

        # with open('logs/history_logs.json', 'r') as f:
        #     self.recorded_actions = json.load(f)

        # with open('logs/history_logs_debug.pk', 'rb') as f:
        #     self.recorded_actions = pickle.load(f)

        # with open('/data2/shy/SC2Arena_BO/logs/default_player/Flat128_Medium_RandomBuild_d5/checkpoint-6000/2025-09-22_19-15-06/history_actions.pk', 'rb') as f:
        #     self.recorded_actions = pickle.load(f)

        self.start_debug = False

        self.recorded_minerals = []
        

    async def distribute_workers(self, resource_ratio: float = 2.0) -> None:
        """
        根据全局矿气比分配工人，优先将工人派往采集gas。
        会将gas_site附近采集mineral的worker调往gas_site以最大化其利用。
        """
        if not self.townhalls.ready or not self.workers:
            return

        # 1. 收集所有基地周围的矿点和气矿
        mineral_patches = {
            m for nexus in self.townhalls.ready
            for m in self.mineral_field.closer_than(12, nexus)
        }
        gas_refineries = {
            g for nexus in self.townhalls.ready
            for g in self.gas_buildings.ready.closer_than(12, nexus)
            if g.has_vespene
        }

        # 2. 派遣MULE（如果是人族）
        if self.config.own_race == "Terran":
            await self._deploy_mules(mineral_patches)

        # 3. 处理gas_site超员问题，将多余的worker释放出来加入可用工人池
        available_idle_workers = list(self.workers.idle)
        
        for gas_site in gas_refineries:
            if gas_site.surplus_harvesters > 0:
                # 找到正在采集这个gas_site的工人
                gas_workers = []
                for worker in self.workers.gathering:
                    # 通过距离判断worker是否在采集这个gas_site
                    if worker.distance_to(gas_site) < 2:
                        gas_workers.append(worker)
                
                # 计算需要释放的工人数量
                excess_count = gas_site.surplus_harvesters
                
                # 将多余的工人加入到可用工人池中（取前excess_count个）
                for i in range(min(excess_count, len(gas_workers))):
                    worker = gas_workers[i]
                    available_idle_workers.append(worker)
                    print(f"Marked excess worker from gas for reassignment: {gas_site}")

        # 4. 统计每个点的缺工数
        gas_tasks = {}
        mineral_tasks = {}

        # 气矿：ideal=3，surplus_harvesters<0 时表示缺工
        for g in gas_refineries:
            missing = max(0, -g.surplus_harvesters)
            if missing:
                gas_tasks[g] = missing
        
        # if self.start_debug:
        # # 移除采尽的矿点
        mineral_patches_ = []
        for m in mineral_patches:
            if m.mineral_contents > 0:
                mineral_patches_.append(m)
        mineral_patches = mineral_patches_

        new_mineral_flag = False
        for mineral in mineral_patches:
            if mineral.tag not in self.recorded_minerals:
                self.recorded_minerals.append(mineral.tag)
                new_mineral_flag = True

        # 矿点：每个矿最多2个工人（不计算MULE和即将被重新分配的工人）
        # if self.start_debug:
        for m in mineral_patches:
            # 统计该矿点的工人数量（不包括MULE和即将被重新分配的工人）
            worker_count = 0
            for worker in self.workers.gathering:
                # if (worker.distance_to(m) < 2 and 
                #     worker not in available_idle_workers):
                #     worker_count += 1
                if (worker.order_target == m.tag and 
                    worker not in available_idle_workers):
                    worker_count += 1
            if worker_count > 2:
                print(m.tag, m.position, worker_count)
            need = max(0, 2 - worker_count)
            if need:
                mineral_tasks[m] = need
        
        # 5. 优先处理gas_sites - 从附近mineral_sites调worker + idle_workers
        for gas_site in list(gas_tasks.keys()):
            needed = gas_tasks[gas_site]
            if needed <= 0:
                continue
            
            # 找到这个gas_site附近正在采集mineral的workers（排除即将被重新分配的工人）
            nearby_mineral_workers = []
        
            for mineral in mineral_patches:
                # 只考虑距离gas_site较近的mineral
                if mineral.distance_to(gas_site) < 10:  # 距离阈值可调整
                    # 找到正在采集这个mineral的workers
                    for worker in self.workers.gathering:
                        # 通过距离判断worker是否在采集这个mineral，且不在重新分配列表中
                        if (worker.distance_to(mineral) < 2 and 
                            worker not in available_idle_workers):
                            nearby_mineral_workers.append((worker, worker.distance_to(gas_site)))
        
            # 按距离gas_site的远近排序，优先调用最近的workers
            nearby_mineral_workers.sort(key=lambda x: x[1])
        
            # 重新分配mineral workers到gas_site
            reassigned = 0
            for worker, _ in nearby_mineral_workers:
                if reassigned >= needed:
                    break
                worker.gather(gas_site)
                print(f"Reassigned worker from mineral to gas: {gas_site}")
                reassigned += 1
        
            # 更新gas_site的需求
            needed -= reassigned
        
            # 如果还有缺工，用idle_workers补充
            if needed > 0 and available_idle_workers:
                # 找到距离gas_site最近的idle_workers
                available_idle_workers.sort(key=lambda w: w.distance_to(gas_site))
            
                assigned = 0
                workers_to_remove = []
                for worker in available_idle_workers:
                    if assigned >= needed:
                        break
                    worker.gather(gas_site)
                    print(f"Assigned idle worker to gas: {gas_site}")
                    workers_to_remove.append(worker)
                    assigned += 1
            
                # 从available_idle_workers中移除已分配的workers
                for worker in workers_to_remove:
                    available_idle_workers.remove(worker)
            
                needed -= assigned
        
            # 更新gas_tasks
            gas_tasks[gas_site] = needed
            if gas_tasks[gas_site] <= 0:
                del gas_tasks[gas_site]
        
        # 只有当出现了新的矿的时候，才会对此前的工人进行重分配。
        if new_mineral_flag:
            for m in mineral_patches:
                worker_count = 0
                for worker in self.workers.gathering:
                    # if (worker.distance_to(m) < 2 and
                    if (worker.order_target == m.tag and 
                        worker not in available_idle_workers):
                        if worker_count == 2:
                            available_idle_workers.append(worker)
                        else:
                            worker_count += 1
        
        # 6. 用剩余的idle_workers填补mineral_sites
        for worker in available_idle_workers:
            if not mineral_tasks:
                break
        
            # 选择距离最近的mineral_site
            target = min(mineral_tasks.keys(), key=lambda s: s.distance_to(worker))
            worker.gather(target)
            print(f"Assigned idle worker to mineral: {target}")
        
            # 更新mineral_site的需求
            mineral_tasks[target] -= 1
            if mineral_tasks[target] <= 0:
                del mineral_tasks[target]

        # 7. 如果还有gas_site缺工且还有mineral workers可调配，进行第二轮调配
        if gas_tasks:
            for gas_site in list(gas_tasks.keys()):
                needed = gas_tasks[gas_site]
                if needed <= 0:
                    continue
                
                # 扩大搜索范围，找到更远的mineral workers
                distant_mineral_workers = []
                for mineral in mineral_patches:
                    if mineral.distance_to(gas_site) < 15:  # 扩大搜索范围
                        for worker in self.workers.gathering:
                            if (worker.distance_to(mineral) < 2 and 
                                worker not in available_idle_workers):
                                distant_mineral_workers.append((worker, worker.distance_to(gas_site)))
            
                # 按距离排序
                distant_mineral_workers.sort(key=lambda x: x[1])
            
                # 重新分配
                reassigned = 0
                for worker, _ in distant_mineral_workers:
                    if reassigned >= needed:
                        break
                    worker.gather(gas_site)
                    print(f"Reassigned distant worker from mineral to gas: {gas_site}")
                    reassigned += 1
        # else:
        #     # 矿点：每个矿最多2个工人（不计算MULE和即将被重新分配的工人）
        #     for m in mineral_patches:
        #         # 统计该矿点的工人数量（不包括MULE和即将被重新分配的工人）
        #         worker_count = 0
        #         for worker in self.workers.gathering:
        #             # if (worker.distance_to(m) < 2 and 
        #             #     worker not in available_idle_workers):
        #             #     worker_count += 1
        #             if (worker.distance_to(m) < 2 and 
        #                 worker not in available_idle_workers):
        #                 worker_count += 1
        #         need = max(0, 2 - worker_count)
        #         if need:
        #             mineral_tasks[m] = need
            
        #     # 5. 优先处理gas_sites - 从附近mineral_sites调worker + idle_workers
        #     for gas_site in list(gas_tasks.keys()):
        #         needed = gas_tasks[gas_site]
        #         if needed <= 0:
        #             continue
                
        #         # 找到这个gas_site附近正在采集mineral的workers（排除即将被重新分配的工人）
        #         nearby_mineral_workers = []
            
        #         for mineral in mineral_patches:
        #             # 只考虑距离gas_site较近的mineral
        #             if mineral.distance_to(gas_site) < 10:  # 距离阈值可调整
        #                 # 找到正在采集这个mineral的workers
        #                 for worker in self.workers.gathering:
        #                     # 通过距离判断worker是否在采集这个mineral，且不在重新分配列表中
        #                     if (worker.distance_to(mineral) < 2 and 
        #                         worker not in available_idle_workers):
        #                         nearby_mineral_workers.append((worker, worker.distance_to(gas_site)))
            
        #         # 按距离gas_site的远近排序，优先调用最近的workers
        #         nearby_mineral_workers.sort(key=lambda x: x[1])
            
        #         # 重新分配mineral workers到gas_site
        #         reassigned = 0
        #         for worker, _ in nearby_mineral_workers:
        #             if reassigned >= needed:
        #                 break
        #             worker.gather(gas_site)
        #             print(f"Reassigned worker from mineral to gas: {gas_site}")
        #             reassigned += 1
            
        #         # 更新gas_site的需求
        #         needed -= reassigned
            
        #         # 如果还有缺工，用idle_workers补充
        #         if needed > 0 and available_idle_workers:
        #             # 找到距离gas_site最近的idle_workers
        #             available_idle_workers.sort(key=lambda w: w.distance_to(gas_site))
                
        #             assigned = 0
        #             workers_to_remove = []
        #             for worker in available_idle_workers:
        #                 if assigned >= needed:
        #                     break
        #                 worker.gather(gas_site)
        #                 print(f"Assigned idle worker to gas: {gas_site}")
        #                 workers_to_remove.append(worker)
        #                 assigned += 1
                
        #             # 从available_idle_workers中移除已分配的workers
        #             for worker in workers_to_remove:
        #                 available_idle_workers.remove(worker)
                
        #             needed -= assigned
            
        #         # 更新gas_tasks
        #         gas_tasks[gas_site] = needed
        #         if gas_tasks[gas_site] <= 0:
        #             del gas_tasks[gas_site]
            
        #     # 6. 用剩余的idle_workers填补mineral_sites
        #     for worker in available_idle_workers:
        #         if not mineral_tasks:
        #             break
            
        #         # 选择距离最近的mineral_site
        #         target = min(mineral_tasks.keys(), key=lambda s: s.distance_to(worker))
        #         worker.gather(target)
        #         print(f"Assigned idle worker to mineral: {target}")
            
        #         # 更新mineral_site的需求
        #         mineral_tasks[target] -= 1
        #         if mineral_tasks[target] <= 0:
        #             del mineral_tasks[target]

        #     # 7. 如果还有gas_site缺工且还有mineral workers可调配，进行第二轮调配
        #     if gas_tasks:
        #         for gas_site in list(gas_tasks.keys()):
        #             needed = gas_tasks[gas_site]
        #             if needed <= 0:
        #                 continue
                    
        #             # 扩大搜索范围，找到更远的mineral workers
        #             distant_mineral_workers = []
        #             for mineral in mineral_patches:
        #                 if mineral.distance_to(gas_site) < 15:  # 扩大搜索范围
        #                     for worker in self.workers.gathering:
        #                         if (worker.distance_to(mineral) < 2 and 
        #                             worker not in available_idle_workers):
        #                             distant_mineral_workers.append((worker, worker.distance_to(gas_site)))
                
        #             # 按距离排序
        #             distant_mineral_workers.sort(key=lambda x: x[1])
                
        #             # 重新分配
        #             reassigned = 0
        #             for worker, _ in distant_mineral_workers:
        #                 if reassigned >= needed:
        #                     break
        #                 worker.gather(gas_site)
        #                 print(f"Reassigned distant worker from mineral to gas: {gas_site}")
        #                 reassigned += 1

    async def _deploy_mules(self, mineral_patches) -> None:
        """
        部署MULE到合适的矿点
        """
        mule_units = self.units(UnitTypeId.MULE).idle
        for mule in mule_units:
            nearby_minerals = [m for m in mineral_patches if m.distance_to(mule) < 12]
            best_mineral = self._select_best_mineral_for_mule(nearby_minerals, mule)
            if best_mineral:
                mule.gather(best_mineral)

    def _select_best_mineral_for_mule(self, mineral_patches, orbital_command):
        """
        为MULE选择最佳的矿点
        优先选择：
        1. 资源量较多的矿点
        2. 距离基地较近的矿点
        3. 当前采集单位较少的矿点
        4. 没有MULE的矿点
        """
        if not mineral_patches:
            return None
        
        best_mineral = None
        best_score = -1
        
        for mineral in mineral_patches:
            # 计算该矿点的评分
            score = 0
            
            # 资源量权重（剩余资源越多越好）
            resource_weight = mineral.mineral_contents / 1800  # 1800是矿点的初始资源量
            score += resource_weight * 40
            
            # 距离权重（距离越近越好）
            distance_weight = 1 - (mineral.distance_to(orbital_command) / 12)
            score += distance_weight * 20
            
            # 采集单位数量权重（采集单位越少越好）
            current_harvesters = 0
            for unit in self.units:
                if (hasattr(unit, 'order_target') and unit.order_target == mineral.tag and 
                    unit.type_id in [UnitTypeId.SCV, UnitTypeId.MULE]):
                    current_harvesters += 1
            
            harvester_weight = max(0, 1 - current_harvesters / 4)  # 最多4个采集单位(2个SCV + 2个MULE)
            score += harvester_weight * 30
            
            # 检查是否已经有MULE在这个矿点
            mule_count = 0
            for unit in self.units.filter(lambda u: u.type_id == UnitTypeId.MULE):
                if hasattr(unit, 'order_target') and unit.order_target == mineral.tag:
                    mule_count += 1
                    
            if mule_count >= 1:  # 每个矿点最多1个MULE
                score -= 50
            
            # 优先选择资源量充足的矿点
            if mineral.mineral_contents < 500:  # 资源量过低的矿点降低优先级
                score -= 20
            
            if score > best_score:
                best_score = score
                best_mineral = mineral
        
        return best_mineral if best_score > 0 else None

    def get_terran_suggestions(self):
        suggestions = []
        # 人口不足时建议建造Supply Depot
        if (
            self.supply_left < 5
            and not self.already_pending(UnitTypeId.SUPPLYDEPOT)
            and self._can_build(UnitTypeId.SUPPLYDEPOT)
        ):
            suggestions.append("Supply is low! Build a Supply Depot immediately.")
        # 没有Supply Depot时建议建造
        if (
            self.get_total_amount(UnitTypeId.SUPPLYDEPOT) < 1
            and self._can_build(UnitTypeId.SUPPLYDEPOT)
            and not self.already_pending(UnitTypeId.SUPPLYDEPOT)
        ):
            suggestions.append("At least one Supply Depot is necessary for development, consider building one.")
        # 没有MULE时建议建造
        if (
            self.get_total_amount(UnitTypeId.MULE) < 5
            and not self.already_pending(UnitTypeId.MULE)
            and self.townhalls(UnitTypeId.ORBITALCOMMAND).ready.exists
        ):
            suggestions.append("MULE can boost your economy, consider calling one from your Command Center.")
        # 没有Refinery时建议建造
        if self.get_total_amount(UnitTypeId.REFINERY) < 1 and self._can_build(UnitTypeId.REFINERY):
            suggestions.append("At least one Refinery is necessary for gas collection, consider building one.")
        # 没有Barracks时建议建造
        if (
            self.structures(UnitTypeId.SUPPLYDEPOT).exists
            and self.get_total_amount(UnitTypeId.BARRACKS) < 1
            and self.structures(UnitTypeId.SUPPLYDEPOT).ready.exists
        ):
            suggestions.append("At least one Barracks is necessary for attacking units, consider building one.")
        # 没有Barracks Tech Lab时建议建造
        barracks = self.structures(UnitTypeId.BARRACKS).ready
        if (
            barracks.exists
            and self.get_total_amount(UnitTypeId.BARRACKSTECHLAB) < 1
            and self._can_build(UnitTypeId.BARRACKSTECHLAB)
        ):
            if barracks.idle.exists:
                suggestions.append("At least one Barracks Tech Lab is necessary for advanced units, consider building one.")
            else:
                suggestions.append(
                    "Consider building a Barracks Tech Lab when one of your Barracks is idle to unlock advanced units."
                )
        # Marine数量少于2时建议建造
        if (
            self.structures(UnitTypeId.BARRACKS).ready.exists
            and self.get_total_amount(UnitTypeId.MARINE) < 2
            and self._can_build(UnitTypeId.MARINE)
        ):
            suggestions.append("At least 2 Marines are necessary for defensing, consider training one.")
        # 没有Marauder时建议建造
        if (
            self.structures(UnitTypeId.BARRACKSTECHLAB).ready.exists
            and self.get_total_amount(UnitTypeId.MARAUDER) < 1
            and self._can_build(UnitTypeId.MARAUDER)
        ):
            suggestions.append("At least one Marauder is necessary for defensing, consider training one.")
        # 只有一座Barracks时建议建造第二座
        if self.get_total_amount(UnitTypeId.BARRACKS) == 1 and self._can_build(UnitTypeId.BARRACKS):
            suggestions.append("Consider building a second Barracks to increase unit production.")
        # 如果有2个兵营且没有Factory时建议建造
        if (
            self.structures(UnitTypeId.BARRACKS).ready.amount >= 2
            and self.structures(UnitTypeId.BARRACKSTECHLAB).ready.exists
            and self.get_total_amount(UnitTypeId.FACTORY) == 0
            and self._can_build(UnitTypeId.FACTORY)
        ):
            suggestions.append("Consider building a Factory to unlock mechanical units.")
        # 有Factory时建议升级TechLab
        if (
            self.structures(UnitTypeId.FACTORY).ready.exists
            and self.get_total_amount(UnitTypeId.FACTORYTECHLAB) == 0
            and self._can_build(UnitTypeId.FACTORYTECHLAB)
        ):
            suggestions.append("Consider upgrade Factory Tech Lab to train powerful units.")
        if self.structures(UnitTypeId.FACTORYTECHLAB).ready.exists and self.get_total_amount(UnitTypeId.SIEGETANK) < 3:
            suggestions.append("Consider train Siege Tank to increase your army's firepower.")
        # 建议升级Command Center到Orbital Command
        cc = self.townhalls(UnitTypeId.COMMANDCENTER).ready
        if cc.exists:
            main_cc = cc.first  # 通常主基地优先升级
            if main_cc.is_idle and self._can_build(UnitTypeId.ORBITALCOMMAND) and self.get_total_amount(UnitTypeId.SCV) >= 16:
                suggestions.append("Upgrade Command Center to Orbital Command for better economy.")
        # 如果只有一座Orbital Command且没有Command Center时，建议建造新的Command Center
        if (
            self.get_total_amount(UnitTypeId.ORBITALCOMMAND) == 1
            and self.get_total_amount(UnitTypeId.COMMANDCENTER) == 0
            and self._can_build(UnitTypeId.COMMANDCENTER)
        ):
            suggestions.append("Consider building another Command Center to expand your base at another resource location.")
        # 维持适当的Marine和Marauder比例
        marine_count = self.get_total_amount(UnitTypeId.MARINE)
        marauder_count = self.get_total_amount(UnitTypeId.MARAUDER)

        if marine_count + marauder_count > 10:
            ratio = marauder_count / max(1, marine_count)
            if ratio < 0.5:
                suggestions.append("Increase Marauder production for better tanking.")
            elif ratio > 2.5:
                suggestions.append("Produce more Marines for DPS against light units.")

        return suggestions

    def get_protoss_suggestions(self):
        suggestions = []

        # 人口不足时建议建造Pylon (水晶塔)
        if (
            self.supply_left < 4
            and not self.already_pending(UnitTypeId.PYLON)
            and self._can_build(UnitTypeId.PYLON)
        ):
            suggestions.append("Supply is low! Build a Pylon immediately.")
        
        # 建筑没有能量时建议建造Pylon
        if self.structures.filter(lambda s: not s.is_powered and s.build_progress > 0.1).exists:
             if self._can_build(UnitTypeId.PYLON) and not self.already_pending(UnitTypeId.PYLON):
                suggestions.append("Some of your structures are unpowered! Build a Pylon nearby.")

        # 没有Pylon时建议建造
        if (
            self.get_total_amount(UnitTypeId.PYLON) < 1
            and self._can_build(UnitTypeId.PYLON)
            and not self.already_pending(UnitTypeId.PYLON)
        ):
            suggestions.append("At least one Pylon is necessary for development and power, consider building one.")

        # 有多余能量时建议使用Chrono Boost (星空加速)
        nexus = self.townhalls(UnitTypeId.NEXUS).ready
        if nexus.exists and nexus.first.energy >= 50:
            suggestions.append("Your Nexus has enough energy for Chrono Boost. Use it on the Nexus for more Probes or on a production building.")

        # 没有Assimilator (吸收厂) 时建议建造
        if self.get_total_amount(UnitTypeId.ASSIMILATOR) < 1 and self._can_build(UnitTypeId.ASSIMILATOR):
            suggestions.append("At least one Assimilator is necessary for gas collection, consider building one.")

        # 没有Gateway (传送门) 时建议建造
        if (
            self.structures(UnitTypeId.PYLON).exists
            and self.get_total_amount(UnitTypeId.GATEWAY) < 1
            and self._can_build(UnitTypeId.GATEWAY)
        ):
            suggestions.append("At least one Gateway is necessary for training ground units, consider building one.")

        # 没有Cybernetics Core (控制核心) 时建议建造
        if (
            self.structures(UnitTypeId.GATEWAY).ready.exists
            and self.get_total_amount(UnitTypeId.CYBERNETICSCORE) < 1
            and self._can_build(UnitTypeId.CYBERNETICSCORE)
        ):
            suggestions.append("A Cybernetics Core is necessary to unlock advanced units like Stalkers, consider building one.")

        # 建议研究Warpgate (折跃门) 科技
        cyber_core = self.structures(UnitTypeId.CYBERNETICSCORE).ready
        if (
            cyber_core.exists
            and self.already_pending_upgrade(UpgradeId.WARPGATERESEARCH) == 0
            and self.can_afford(UpgradeId.WARPGATERESEARCH)
        ):
            if cyber_core.idle.exists:
                suggestions.append("Cybernetics Core is ready. Research Warpgate technology to reinforce your army faster.")
            else:
                suggestions.append("Consider researching Warpgate technology when your Cybernetics Core is idle.")
                
        # Zealot (狂热者) 数量少于2时建议建造
        if (
            self.structures(UnitTypeId.GATEWAY).exists
            and self.get_total_amount(UnitTypeId.ZEALOT) < 2
            and self._can_build(UnitTypeId.ZEALOT)
        ):
            suggestions.append("At least 2 Zealots are necessary for early defense, consider training one.")

        # 没有Stalker (追猎者) 时建议建造
        if (
            self.structures(UnitTypeId.CYBERNETICSCORE).ready.exists
            and self.get_total_amount(UnitTypeId.STALKER) < 1
            and self._can_build(UnitTypeId.STALKER)
        ):
            suggestions.append("At least one Stalker is useful for anti-air and kiting, consider training one.")
            
        # 传送门数量不足时建议建造更多
        gateway_count = self.get_total_amount(UnitTypeId.GATEWAY) + self.get_total_amount(UnitTypeId.WARPGATE)
        if 1 <= gateway_count < 3 and self._can_build(UnitTypeId.GATEWAY):
            suggestions.append("Consider building more Gateways to increase unit production.")

        # 维持适当的Zealot和Stalker比例
        zealot_count = self.get_total_amount(UnitTypeId.ZEALOT)
        stalker_count = self.get_total_amount(UnitTypeId.STALKER)

        if zealot_count + stalker_count > 10:
            # 理想比例：大约1个狂热者对应2个追猎者
            ratio = zealot_count / max(1, stalker_count)
            if ratio > 0.8: # 狂热者过多
                suggestions.append("Your army has many Zealots. Produce more Stalkers for ranged support.")
            elif ratio < 0.3: # 追猎者过多
                suggestions.append("Increase Zealot production to create a stronger frontline for your Stalkers.")

        return suggestions

    def get_zerg_suggestions(self):
        suggestions = []
        
        # 人口不足时建议建造Overlord
        if (
            self.supply_left < 3
            and self.supply_cap < 200 # 避免在200人口时仍然提示
            and not self.already_pending(UnitTypeId.OVERLORD)
            and self._can_build(UnitTypeId.OVERLORD)
        ):
            suggestions.append("Supply is low! Morph an Overlord immediately.")

        # 没有Spawning Pool时建议建造
        if (
            self.get_total_amount(UnitTypeId.SPAWNINGPOOL) < 1
            and not self.already_pending(UnitTypeId.SPAWNINGPOOL)
            and self._can_build(UnitTypeId.SPAWNINGPOOL)
        ):
            suggestions.append("A Spawning Pool is required to create Zerglings, build one.")

        # 没有Queen时建议建造
        # 每个基地至少一个女王用于注卵和防御
        if (
            self.structures(UnitTypeId.SPAWNINGPOOL).ready.exists
            and self.get_total_amount(UnitTypeId.QUEEN) < self.townhalls.amount
            and self._can_build(UnitTypeId.QUEEN)
        ):
            suggestions.append("Build a Queen for each Hatchery to inject larva and defend.")

        # 有女王但基地没有注卵时建议注卵
        queens_with_energy = self.units(UnitTypeId.QUEEN).filter(lambda q: q.energy >= 25)
        hatcheries_needing_inject = self.townhalls.ready.filter(lambda h: not h.has_buff(BuffId.QUEENSPAWNLARVATIMER))
        if queens_with_energy.exists and hatcheries_needing_inject.exists:
            suggestions.append("Your Queen has energy! Use 'Inject Larva' on a Hatchery to boost production.")

        # 没有Extractor时建议建造
        if self.get_total_amount(UnitTypeId.EXTRACTOR) < 1 and self._can_build(UnitTypeId.EXTRACTOR):
            suggestions.append("At least one Extractor is necessary for gas collection, consider building one.")

        # Zergling数量少于6时建议建造
        if (
            self.structures(UnitTypeId.SPAWNINGPOOL).ready.exists
            and self.get_total_amount(UnitTypeId.ZERGLING) < 6
            and self._can_build(UnitTypeId.ZERGLING)
        ):
            suggestions.append("At least 6 Zerglings are necessary for early defense, consider training some.")

        # 建议扩张（建造第二个基地）
        if self.townhalls.amount < 2 and self._can_build(UnitTypeId.HATCHERY):
            suggestions.append("Consider building a second Hatchery to expand your economy and production.")

        # 建议建造Roach Warren
        if (
            self.structures(UnitTypeId.SPAWNINGPOOL).ready.exists
            and self.get_total_amount(UnitTypeId.ROACHWARREN) == 0
            and self._can_build(UnitTypeId.ROACHWARREN)
        ):
            suggestions.append("Consider building a Roach Warren to unlock Roaches, a strong armored unit.")

        # 没有Roach时建议建造
        if (
            self.structures(UnitTypeId.ROACHWARREN).ready.exists
            and self.get_total_amount(UnitTypeId.ROACH) < 5
            and self._can_build(UnitTypeId.ROACH)
        ):
            suggestions.append("Roaches are strong against many early units, consider training some.")

        # 建议升级到Lair (T2科技)
        if (
            self.structures(UnitTypeId.SPAWNINGPOOL).ready.exists
            and self.get_total_amount(UnitTypeId.LAIR) == 0
            and self.townhalls(UnitTypeId.HATCHERY).idle.exists
            and self._can_build(UnitTypeId.LAIR)
        ):
            suggestions.append("Upgrade a Hatchery to a Lair to unlock powerful mid-game units and upgrades.")

        # 有Lair时建议建造Hydralisk Den
        if (
            self.structures(UnitTypeId.LAIR).ready.exists
            and self.get_total_amount(UnitTypeId.HYDRALISKDEN) == 0
            and self._can_build(UnitTypeId.HYDRALISKDEN)
        ):
            suggestions.append("Build a Hydralisk Den to unlock Hydralisks, a versatile ranged unit.")

        # 有Hydralisk Den时建议训练Hydralisk
        if self.structures(UnitTypeId.HYDRALISKDEN).ready.exists and self.get_total_amount(UnitTypeId.HYDRALISK) < 5:
            suggestions.append("Consider training Hydralisks to strengthen your army's anti-air and ranged capabilities.")

        # 维持适当的Zergling和Roach比例
        zergling_count = self.get_total_amount(UnitTypeId.ZERGLING)
        roach_count = self.get_total_amount(UnitTypeId.ROACH)

        if zergling_count + roach_count > 20:
            # 计算蟑螂在(蟑螂+小狗)部队中的价值占比，蟑螂占2人口，小狗占0.5
            roach_supply = roach_count * 2
            zergling_supply = zergling_count * 0.5
            total_supply = roach_supply + zergling_supply
            
            if total_supply > 0:
                roach_ratio = roach_supply / total_supply
                if roach_ratio < 0.3: # 蟑螂占比过低
                    suggestions.append("Your army is Zergling-heavy. Add Roaches for a stronger frontline.")
                elif roach_ratio > 0.8: # 蟑螂占比过高
                    suggestions.append("Your army is Roach-heavy. Add Zerglings for more DPS and to surround enemies.")

        return suggestions
    
    def name_equal(self, built, bo_name):
        if built == bo_name:
            return True
        return False
    
    def normalize_name(self, name: str) -> str:
            """标准化名称：移除非字母数字字符，转为大写"""
            return re.sub(r'[^a-zA-Z0-9]', '', name).upper()

    def get_unit_type_id(self, name: str):
        normalized = self.normalize_name(name)
        for member in UnitTypeId:
            if self.normalize_name(member.name) == normalized:
                return member
        return None
    
    def get_upgrade_id(self, name: str):
        normalized = self.normalize_name(name)
        for member in UpgradeId:
            if self.normalize_name(member.name) == normalized:
                return member
        return None
    
    def init_bo(self):
        bo = []
        # with open('BO/prompts.json', 'r') as f:
        with open('BO/prompts_1007.json', 'r') as f:
            suggestions = json.load(f)
        
        for i in range(len(suggestions)):
            bo = bo + suggestions[i]['bo']
        
        # with open('BO/prompts_d6.json', 'r', encoding='utf-8') as f:
        #     suggestions = json.load(f)
        
        # stages = suggestions['Phase Division']
        # for i in range(len(stages)):
        #     bo = bo + stages[i]['BO Segment']
        
        new_bo = []
        for build in bo:
            if ' x ' in build[2]:
                structure_name = build[2].split(' x ')[0]
                structure_num = int(build[2].split(' x ')[1])
                for i in range(structure_num):
                    new_bo.append([build[0], structure_name])
            else:
                new_bo.append([build[0], build[2]])

        self.bo = new_bo

        self.ability_2_bo_id = {
            'TERRANBUILD_SUPPLYDEPOT': 'supplydepot', 
            'TERRANBUILD_BARRACKS': 'barracks', 
            'TERRANBUILD_COMMANDCENTER': 'commandcenter', 
            'BARRACKSTRAIN_MARINE': 'marine', 
            'UPGRADETOORBITAL_ORBITALCOMMAND': 'orbitalcommand', 
            'TERRANBUILD_REFINERY': 'refinery', 
            'CALLDOWNMULE_CALLDOWNMULE': 'mule', 
            'TERRANBUILD_ENGINEERINGBAY': 'engineeringbay', 
            'UPGRADETOPLANETARYFORTRESS_PLANETARYFORTRESS': 'planetaryfortress', 
            'TERRANBUILD_FACTORY': 'factory', 
            'BUILD_TECHLAB_FACTORY': 'techlab factory', 
            'TERRANBUILD_MISSILETURRET': 'missileturret', 
            'TERRANBUILD_STARPORT': 'starport', 
            'FACTORYTRAIN_SIEGETANK': 'siegetank', 
            'STARPORTTRAIN_LIBERATOR': 'liberator', 
            'BUILD_TECHLAB_BARRACKS': 'techlab barracks', 
            'BUILD_REACTOR_BARRACKS': 'reactor barracks', 
            'STARPORTTRAIN_VIKINGFIGHTER': 'vikingfighter', 
            'ENGINEERINGBAYRESEARCH_TERRANINFANTRYARMORLEVEL1': 'terraninfantryarmorlevel1', 
            'TERRANBUILD_GHOSTACADEMY': 'ghostacademy', 
            'BUILD_TECHLAB_STARPORT': 'techlab starport', 
            'FACTORYTRAIN_WIDOWMINE': 'widowmine', 
            'BARRACKSTECHLABRESEARCH_STIMPACK': 'stimpack', 
            'RESEARCH_COMBATSHIELD': 'combatshield', 
            'STARPORTTRAIN_MEDIVAC': 'medivac', 
            'TERRANBUILD_FUSIONCORE': 'fusioncore', 
            'RESEARCH_TERRANSTRUCTUREARMORUPGRADE': 'terranstructurearmorupgrade', 
            'ENGINEERINGBAYRESEARCH_TERRANINFANTRYWEAPONSLEVEL1': 'terraninfantryweaponslevel1', 
            'TERRANBUILD_ARMORY': 'armory', 'STARPORTTRAIN_BANSHEE': 'banshee', 
            'RESEARCH_BANSHEECLOAKINGFIELD': 'bansheecloakingfield', 
            'BARRACKSTRAIN_MARAUDER': 'marauder', 
            'RESEARCH_CONCUSSIVESHELLS': 'concussiveshells', 
            'ENGINEERINGBAYRESEARCH_TERRANINFANTRYARMORLEVEL2': 'terraninfantryarmorlevel2',
            'FACTORYTRAIN_HELLION': 'hellion',
            'BUILD_REACTOR_STARPORT': 'reactor starport'
        }

    def transform_bo(self):
        new_bo = copy.deepcopy(self.bo)
        
        new_bo_ = []
        for idx, build in enumerate(new_bo):
            if build[1] not in ['mule']:
                new_bo_.append(build)
        new_bo = new_bo_

        history_actions = self.last_action

        for ability in history_actions:
            ability = json.loads(ability)['action']
            action = None
            if ability in ignore_list:
                continue
            
            for s in start_with_list:
                if ability.startswith(s):
                    action = s
                    break
            
            if action == None:
                continue
            
            try:
                history_action_name = self.ability_2_bo_id[ability]
            except:
                print(f"Not in BO: {ability}")

            for idx, build in enumerate(new_bo):
                if self.name_equal(history_action_name, build[1]):
                    del new_bo[idx]
                    break

        # for s in self.structures:
        #     s = s.name.lower()
        #     for idx, build in enumerate(new_bo):
        #         if self.name_equal(s, build[1]):
        #             del new_bo[idx]
        #             break
        
        # for s in self.units:
        #     s = s.name.lower()
        #     for idx, unit in enumerate(new_bo):
        #         if self.name_equal(s, unit[1]):
        #             del new_bo[idx]
        #             break
        
        new_bo_ = []
        for bo in new_bo:
            try:
                unit_id = self.get_unit_type_id(bo[1])
                # upgrade_id = self.get_upgrade_id(bo[1])
                # if unit_id and self.already_pending(self.get_unit_type_id(bo[1])):
                #     pass
                # elif upgrade_id and self.already_pending_upgrade(self.get_upgrade_id(bo[1])):
                #     pass
                # else:
                if unit_id:
                    orbital_data = self.game_data.units[unit_id.value]
                    tech_requirement = orbital_data.tech_requirement
                    prerequisite = []
                    
                    if tech_requirement:
                        prerequisite.append(tech_requirement.name)

                    bo.append(prerequisite)
                else:
                    bo.append([])
                new_bo_.append(bo)
            except:
                if unit_id:
                    orbital_data = self.game_data.units[unit_id.value]
                    tech_requirement = orbital_data.tech_requirement
                    prerequisite = []
                    
                    if tech_requirement:
                        prerequisite.append(tech_requirement.name)

                    bo.append(prerequisite)
                else:
                    bo.append([])
                new_bo_.append(bo)

        return new_bo_
    
    def get_suggestions(self):
        with open('BO/prompts.json', 'r') as f:
            suggestions = json.load(f)
        
        time = self.time
        
        bo = []
        
        for i in range(len(suggestions)):
            bo = bo + suggestions[i]['bo']
        
        new_bo = self.transform_bo()
        
        suggestions_res = []
        if len(new_bo):
            suggestions_res = [
                "Defend your structures if they are attaked by enemys",
                "RULES FOR BUILD ORDER: Consider building the unit/structure/upgrade following the Build Order in given orders with highest priority.",
                # "RULES FOR BUILD ORDER: You should build unit/structure/upgrade following the given orders.",
                "RULES FOR BUILD ORDER: If you are unable to build the first one currently, you should **prepare** for building it instead of building later ones.",
                "RULES FOR BUILD ORDER: You should check the prerequisites first. If the prerequisits are under construction, please be patient and wait; if not, build one."
                "RULES FOR BUILD ORDER: Do not build unit/structure/upgrade that is not in BO, except for SCV, MULE. (SCV production can be estimated by the supply used in BO.)",
                ]

        for i in range(len(new_bo[:4])):
            if len(new_bo[i][2]):
                # suggestion_str = f"Consider building a new {new_bo[i][1]} at {new_bo[i][0]}, with prerequisites: {new_bo[i][2][0]}.\n"
                suggestion_str = f"BUILD ORDER {i+1}: Consider building a new {new_bo[i][1]}, with prerequisites: {new_bo[i][2][0]}.\n"
                suggestions_res.append(suggestion_str)
            else:
                # suggestion_str = f"Consider building a new {new_bo[i][1]} at {new_bo[i][0]}.\n"
                suggestion_str = f"BUILD ORDER {i+1}: Consider building a new {new_bo[i][1]}.\n"
                suggestions_res.append(suggestion_str)
        
        # self.current_first_bo = new_bo[0][1]
        return suggestions_res

        # suggestions = [suggestions[i]['prompt']]
        # suggestions = []

        # # 发现敌人单位时建议攻击
        # if self.enemy_units.exists:
        #     n_enemies = len(
        #         [unit for unit in self.enemy_units if unit.name not in ["Probe", "SCV", "Drone", "MULE", "Overlord"]]
        #     )
        #     if n_enemies > 0:
        #         suggestions.append(
        #             f"Enemy units detected ({n_enemies} units), consider attacking them."
        #         )

        # if self.time < 300 and self.time > 60:
        #     suggestions.append("The enemy will start a fierce attack at 03:00, so you need to start producing a large number of attack units, such as Marauder, at least at 02:30.")
        
        # if self.time > 300 and self.supply_army > 15 and len(self.enemy_units) < 8:
        #     suggestions.append("We can win the game right away! Please find and eliminate all enemies as soon as possible.")
        
        # if self.minerals >= 500:
        #     suggestions.append("Too much minerals! Consider spending them on expanding or developing high technology.")

        # if self.config.own_race == "Terran":
        #     suggestions.extend(self.get_terran_suggestions())
        # elif self.config.own_race == "Protoss":
        #     suggestions.extend(self.get_protoss_suggestions())
        # elif self.config.own_race == "Zerg":
        #     suggestions.extend(self.get_zerg_suggestions())


    def log_current_iteration(self, iteration: int):
        print(f"================ iteration {iteration} ================")
        self.logging("iteration", iteration, save_trace=True)
        self.logging("time_seconds", int(self.time), save_trace=True)
        self.logging("minerals", self.minerals, save_trace=True)
        self.logging("vespene", self.vespene, save_trace=True)
        
        unit_mineral_value, unit_vespene_value = 0, 0
        for unit in self.units:
            unit_value = self.calculate_unit_value(unit.type_id)
            unit_mineral_value += unit_value.minerals
            unit_vespene_value += unit_value.vespene
        self.logging("unit_mineral_value", unit_mineral_value, save_trace=True)
        self.logging("unit_vespene_value", unit_vespene_value, save_trace=True)
        
        structure_mineral_value, structure_vespene_value = 0, 0
        for structure in self.structures:
            structure_value = self.calculate_unit_value(structure.type_id)
            structure_mineral_value += structure_value.minerals
            structure_vespene_value += structure_value.vespene
        self.logging("structure_mineral_value", structure_mineral_value, save_trace=True)
        self.logging("structure_vespene_value", structure_vespene_value, save_trace=True)
        
        self.logging("supply_army", self.supply_army, save_trace=True)
        self.logging("supply_workers", self.supply_workers, save_trace=True)
        self.logging("supply_left", self.supply_left, save_trace=True)
        self.logging("n_structures", len(self.structures), save_trace=True)
        self.logging("n_visible_enemy_units", len(self.enemy_units), save_trace=True)
        self.logging("n_visible_enemy_structures", len(self.enemy_structures), save_trace=True)
        unit_types = set(unit.type_id for unit in self.units)
        structure_types = set(unit.type_id for unit in self.structures)
        self.logging("n_unit_types", len(unit_types), save_trace=True)
        self.logging("n_structure_types", len(structure_types), save_trace=True)

    
    async def run(self, iteration: int):
        # send idle workers to minerals or gas automatically
        await self.distribute_workers()
        for unit in self.units:
            if unit.type_id in [UnitTypeId.MULE] or unit.is_constructing_scv:
                continue
            enemies_in_range = self.enemy_units.in_attack_range_of(unit)
            if enemies_in_range.exists:
                target = self.get_lowest_health_enemy(enemies_in_range)
                if target:
                    unit.attack(target)
            else:
                near_by_enemies = self.enemy_units.closer_than(self.scv_auto_attack_distance, unit.position)
                near_by_enemies = near_by_enemies.closer_than(self.scv_auto_attack_distance, self.start_location)
                target_enemy = self.get_lowest_health_enemy(near_by_enemies)
                if unit.type_id in [UnitTypeId.SCV] and self.time < self.scv_auto_attack_time and target_enemy:
                    unit.attack(target_enemy)
        
        # 10 iteration -> 1.7s
        if self.config.enable_random_decision_interval:
            decision_iteration = random.randint(8, 12)
            decision_minerals = random.randint(130, 200)
        else:
            decision_iteration = 10
            decision_minerals = 100

        if (
            iteration % decision_iteration == 0
            and self.minerals >= decision_minerals
            or iteration == self.next_decision_time
        ):
            self.next_decision_time = iteration + 9 * decision_iteration

            self.log_current_iteration(iteration)

            obs_text = await self.obs_to_text()
            
            flag = False
            # for h in self.recorded_actions[:25]:
            for h in self.recorded_actions[:9]:
                if h['iteration'] == iteration:
                    self.last_action = h['history']
                    self.plan_agent.last_intention = h['intention']
                    self.scouted_locations = h['scouted_locations']
                    self.enemy_locations = h['enemy_locations']
                    flag = True
                    # self._tag_to_id = {int(key): value for key, value in h['tag_to_ids'].items()}
                    # self._id_to_tag = {int(key): value for key, value in h['id_to_tags'].items()}
                    await self.run_actions(h['actions'])
            if not flag:
                self.start_debug = True
                # RAG is not ready yet, so we skip it for now
                # if self.config.enable_rag:
                #     rag_summary, rag_think = self.rag_agent.run(obs_text)
                #     self.logging("rag_summary", rag_summary, save_trace=True)
                #     self.logging("rag_think", rag_think, save_trace=True, print_log=False)
                #     obs_text += "\n\n# Hint\n" + rag_summary

                if self.config.enable_plan or self.config.enable_plan_verifier:
                    suggestions = self.get_suggestions()
                    print(suggestions)
                    self.logging("suggestions", suggestions, save_trace=True, print_log=False)

                    plans, plan_think, plan_chat_history = self.plan_agent.run(obs_text, verifier=self.plan_verifier, suggestions=suggestions)
                    print(plans)

                    self.logging("plans", plans, save_trace=True)
                    self.logging("plan_think", plan_think, save_trace=True, print_log=False)
                    self.logging("plan_chat_history", plan_chat_history, save_trace=True, print_log=False)

                    actions, action_think, action_chat_history = self.action_agent.run(obs_text, plans, verifier=self.action_verifier)
                    self.logging("actions", actions, save_trace=True)
                    self.logging("action_think", action_think, save_trace=True, print_log=False)
                    self.logging("action_chat_history", action_chat_history, save_trace=True, print_log=False)
                else:
                    actions, action_think, action_chat_history = self.agent.run(obs_text, verifier=self.action_verifier)
                    self.logging("actions", actions, save_trace=True)
                    self.logging("action_think", action_think, save_trace=True, print_log=False)
                    self.logging("action_chat_history", action_chat_history, save_trace=True, print_log=False)
                
                self.recorded_actions.append({
                    'actions': actions,
                    'iteration': iteration,
                    'history': copy.deepcopy(self.last_action),
                    'id_to_tags': copy.deepcopy(self._id_to_tag),
                    'tag_to_ids': copy.deepcopy(self._tag_to_id),
                    'intention': copy.deepcopy(self.plan_agent.last_intention),
                    'scouted_locations': copy.deepcopy(self.scouted_locations),
                    'enemy_locations': copy.deepcopy(self.enemy_locations)
                })

                with open(f'{self.log_path}/history_actions.pk', 'wb') as f:
                    # json.dump(self.recorded_actions, f, indent=2)
                    pickle.dump(self.recorded_actions, f)

                await self.run_actions(actions)
                
        elif iteration % 10 == 0:
            self.log_current_iteration(iteration)
