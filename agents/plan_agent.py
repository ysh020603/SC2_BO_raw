from agents.base_agent import BaseAgent
from tools.format import extract_code, json_to_markdown, construct_ordered_list
import json

# strategy_prompt = """
# Our final aim: economic expanding before 08:00; continue building a large army after 08:00 and win by constant attacking.
# Our strategy:
# - Resource collection: produce workers and gather minerals and gas
# - Development: build attacking units and structures
# - Attacking: concentrate forces to search and destroy enemies proactively
# """.strip()

strategy_prompt = """
Our final aim: destroy all enemies as soon as possible.
Our strategy:
- Resource collection: produce workers and gather minerals and gas
- Development: build attacking units and structures
- Attacking: concentrate forces to search and destroy enemies proactively
""".strip()

def construct_plan_example(race: str):
    if race == "Terran":
        return """
Following are some examples:
- Do nothing and just wait;
- Train 1/2/3/... SCV/Marine/Viking/...
- Build a supply depot;
- Upgrade to Orbital Command;
- Attack visible enemies;
- ...
""".strip()
    elif race == "Protoss":
        return """
Following are some examples:
- Do nothing and just wait;
- Train 1/2/3/... Probe/Stalker/Zealot/...
- Build a Pylon;
- Upgrade to Warp Gate;
- Attack visible enemies;
- ...
""".strip()
    elif race == "Zerg":
        return """
Following are some examples:
- Do nothing and just wait;
- Train 1/2/3/... Drone/Zergling/Hydralisk/...
- Build a Hatchery;
- Upgrade to Lair;
- Attack visible enemies;
- ...
""".strip()
    else:
        raise ValueError(f"Unknown race: {race}")

def construct_rules(race: str):
    rules = [
        "Commands should be natural language, instead of code.",
        "When outputting commands, you can perform different types of operations simultaneously. You should consider these types of actions in order, such as whether movement is necessary in the current state. Add these types of commands together as the final commands. Do not overlook other types of actions because you are focused on one category. For example, do not forget to construct facilities while commanding troops. Of course, if these actions are unnecessary, you can still skip them.",
        # "Produce as many units with the strongest attack power as possible.",
        "The total cost of all commands should not exceed the current resources (minerals and gas).",
        "Buildings should be constructed within 12 of the Command Center.",
        # "Commands should not build redundant structures while the existing ones are idle.",
        # "Commands should not build redundant structures(e.g. 2 Refinery while one is not fully utilized).",
        "Commands should not use abilities that are not supported currently.",
        "Commands should not contain 'cancel' or 'stop' commands",
        "Facilities other than the Refinery should be built at a certain distance from the vespene geyser to allow for the construction of a refinery on it.", 
        # "Commands should not build a structure that is not needed now (e.g. build a Missile Turret but there is no enemy air unit).",
        "The unit production list capacity of structures is 5. If the list is full, do not add more units to it.",
        "You can take multiple actions at the same time if possible and needed. (Like train a SCV when building a barracks, if needed. Or attack an enemy when building a new command center.)",
        "When you wish to conserve mineral resources to construct more expensive buildings (such as a Command Center), use the command: 'Do nothing and just wait'. You must not issue no command at all. "
        # "Before 08:00, you should focus on econmic expanding in this stage. You should follow the build order to build 3 command center.", 
        # "Before 08:00, you should send a few Marines if you have any to scout coordinate that are not yet scouted to locate enemy's base if no enemy's structures have been found. You should not send SCV to scout.",
        # "Before 08:00, after you locate any enemy's structures, stop scouting immediately. Move all the Marines back to the base and guard the bases.",
        # "Before 08:00, dispatch a small amount of military units you produce, such as Hellion, to harrass enemy's units and structures instead of doing nothing. However, Marines should be kept at base for defending and act as the main force to participate in the full-scale assault at 08:00.",
        # "After 08:00, start a large-scale assault: Send your Marines in group, destroy all the enemy's structures you find",
        "During (00:00-03:15), you should focus on Early Economy and Tech Setup. The objective in this phase is: The phase prioritizes rapid expansion with a 17-supply Command Center. Three Refineries by 02:37 ensure consistent gas flow for Marauders and upgrades. Early Orbital Commands maximize mineral income via MULEs. The dual Tech Lab Barracks setup enables simultaneous Marauder production and Concussive Shells research. Marines provide anti-reaper defense while Marauders counter early enemy Hellions or Reapers. Supply Depot timing avoids blocks during unit production spikes.",
        "During (03:15-06:53), you should focus on Infantry Massing and Expansion. The objective in this phase is: Focus shifts to overwhelming infantry production: Reactor Barracks mass Marines while Tech Lab Barracks produce Marauders. The third Command Center (04:29) and fourth Command Center (06:53) secure economic superiority. Combat Shield (05:36) enhances Marine survivability. Factory (05:19) and Ghost Academy (05:57) enable late-game tech transitions. Continuous Marine production supports scouting \u2013 send pairs to locate enemy bases then retreat. Supply Depots are timed to match production spikes (4-5 per minute). MULEs prioritize mineral lines for expansion saturation.",
        "During (06:53-10:00), you should focus on Tech Transition and Assault Preparation. The objective in this phase is: Starport with Tech Lab adds Liberators for air control. Infantry Weapons/Armor Level 1 and Stimpack significantly boost DPS. Missile Turrets defend against air harassment. Fifth and sixth Command Centers cement economic dominance. Fusion Core enables advanced air upgrades. Liberators harass enemy siege lines or mineral lines if opportunities arise, but main infantry stays grouped. Army hits 190+ supply by 09:00 through continuous production from 10+ Barracks.",
        "During (10:00 onwards), you should focus on Full-Scale Assault Execution, find and destory all enemy's structures and units. The objective in this phase is: Engage with army: Marines/Marauders form the core, supported by Liberators for area denial and Vikings for air superiority. Banshees harass undefended bases during the main push. Siege Tanks provide siege support. Infantry Armor Level 2 enhances durability. Continuous production from 15+ structures reinforces the assault. Post-engagement, seventh and eighth Command Centers ensure economic recovery. Missile Turrets protect expansions from counter-harassment.",
    ]
    if race == "Terran":
        rules += [
            "Commands should not send SCV or MULE to gather resources because the system will do it automatically.",
            "Commands should not train too many SCVs or MULEs, whose number should not exceed the capacity of CommandCenter and Refinery.",
            # "Commands can construct a new one Supply Depot only when the remaining unused supply is less than 7.",
            "Rules for Building Command Center 1: You should choose given coordinates as expansion locations.",
            "Rules for Building Command Center 2: The coordinates of command centers should follow the order of given possible coordinates.",
        ]
    elif race == "Protoss":
        rules += [
            "Commands should not send Probe to gather resources because the system will do it automatically.",
            "Commands should not train too many Probes, whose number should not exceed the capacity of Nexus and Assimilator.",
            "Commands can construct a new one Pylon only when the remaining unused supply is less than 7.",
        ]
    elif race == "Zerg":
        rules += [
            "Commands should not send Drone to gather resources because the system will do it automatically.",
            "Commands should not train too many Drones, whose number should not exceed the capacity of Hatchery and Extractor.",
            "Commands can construct a new one Overlord only when the remaining unused supply is less than 7.",
            "Commands should not train another Overlord if any [Egg] unit in 'Own units' has 'Production list: Overlord'.",
        ]
    else:
        raise ValueError(f"Unknown race: {race}")
    return rules

############## Plan Role Prompt ###############
# def create_plan_prompt(race: str, rules: list[str], obs_text: str, last_intention: str):
#     plan_example_prompt = construct_plan_example(race)
#     rules_prompt = "Rule checklist:\n" + construct_ordered_list(rules)
#     return f"""
# As a top-tier StarCraft II strategist, your task is to give one or more commands based on the current game state. Only give commands which can be executed immediately, instead of waiting for certain events.

# ### Aim
# {strategy_prompt}

# ### Last Step"s Intention
# What you intented to do in the last time step: {last_intention}.

# ### Current Game State
# {obs_text}

# ### Rules
# {rules_prompt}

# ### Examples
# {plan_example_prompt}

# Think step by step,, and conclude your intention into one sentence (<intention>) to help the executor understand your intention before giving the commands.
# And then give commands in natural language as a dict JSON in the following format wrapped with triple backticks: 
# ```
# [
#     "Intention: <intention>", 
#     "Command: <command_1>",
#     "Command: <command_2>",
#     ...
# ]
# ```
#     """.strip()

def create_plan_prompt(race: str, rules: list[str], obs_text: str, last_intention: str):
    plan_example_prompt = construct_plan_example(race)
    rules_prompt = "Rule checklist:\n" + construct_ordered_list(rules)
    return f"""
As a top-tier StarCraft II strategist, your task is to give one or more commands based on the current game state. Only give commands which can be executed immediately, instead of waiting for certain events.

### Aim
{strategy_prompt}

### Current Game State
{obs_text}

### Rules
{rules_prompt}

### Examples

Here is an example of the desired output format:

<analysis>
Observing the current supply at 13/15 and limited mineral resources, constructing a new Supply Depot at (18, 132) is the highest priority. This action prevents an imminent supply block, which would halt SCV and Marine production, critically slowing down our early economy and military presence. The location at (18, 132) is optimal as it's defensively positioned behind the mineral line and helps form a choke point against early rushes.
</analysis>
<answer>
```
[
    "<command_1>",
    "<command_2>",
    ...
]
```
</answer>

**Your task is to strictly follow these steps:**
1.  **First, you MUST provide your strategic analysis inside `<analysis>` and `</analysis>` tags.** Your analysis should be thorough and justify your final decision.
2.  **Second, you MUST provide the final command list in a JSON format, wrapped with triple backticks, inside `<answer>` and `</answer>` tags.**

**The output format is not optional and must be followed precisely. Do not add any other text outside of these tags.**
    """.strip()

# Think step by step, and then give commands as a list JSON in the following format wrapped with triple backticks:
# ```
# [
#     "<command_1>",
#     "<command_2>",
#     ...
# ]
# ```
#     """.strip()


############## Plan Critic Role Prompt ###############
def create_plan_critic_prompt(rules: list[str], obs_text: str, plans: list[str]):
    rules_text = construct_ordered_list(rules)
    plans_text = construct_ordered_list(plans)
    return """
As a top-tier StarCraft II player, your task is to check if the given commands for current game state violate any rules.

### Current Game State
%s

### Given Commands
%s

### Rules Checklist
%s

Analyze the given rules one by one, and then provide a summary for errors at the end as follows, wrapped with triple backticks:
```
{
    "errors": [
        "Error 1: ...",
        "Error 2: ...",
        ...
    ],
    "error_number": 0/1/2/...
}
```
    """.strip() % (obs_text, plans_text, rules_text)


class PlanAgent(BaseAgent):
    def __init__(self, race, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.race = race
        self.rules = construct_rules(race)
        self.plan_example = construct_plan_example(race)
        
        self.max_refine_times = 3
        self.think = []
        self.chat_history = []
        self.last_intention = "This is your first step."
        self.mineral_needed = 100

    def gene_new_plan(self, obs_text: str, rules: list[str]):
        prompt = create_plan_prompt(self.race, rules, obs_text, self.last_intention)
        response, messages = self.llm_client.call(**self.generation_config, prompt=prompt, need_json=True)
        self.think.append([response])
        self.chat_history.append(messages)
        return json.loads(extract_code(response))

    def critic_plan(self, plan: list[str], obs_text: str, rules: list[str]):
        prompt = create_plan_critic_prompt(rules, obs_text, plan)
        response, messages = self.llm_client.call(**self.generation_config, prompt=prompt, need_json=True)
        self.think[-1].append(response)
        self.chat_history.append(messages)
        return response

    def refine_plan(self, obs_text: str, plan: list[str], critic: str, rules: list[str]):
        gene_prompt = create_plan_prompt(self.race, rules, obs_text, self.last_intention)
        history = [
            {"role": "user", "content": gene_prompt},
            {"role": "assistant", "content": json_to_markdown(plan)},
        ]
        prompt = (
            "Errors:\n"
            + critic
            + "\nRethink with the given rules and errors step by step, and then give a refined plan based on the current game state."
        )
        response, messages = self.llm_client.call(**self.generation_config, prompt=prompt, history=history, need_json=True)
        self.think.append([response])
        self.chat_history.append(messages)
        return json.loads(extract_code(response))

    def refine_plan_until_ready(self, obs_text: str, plan: list[str], rules: list[str]):
        for _ in range(self.max_refine_times):
            critic = self.critic_plan(plan, obs_text, rules)
            critic = json.loads(extract_code(critic))
            if isinstance(critic, list):
                critic = {"error_number": len(critic), "errors": critic}
            if critic.get("error_number", 0) == 0:
                return plan
            critic = construct_ordered_list(critic.get("errors", []))
            plan = self.refine_plan(obs_text, plan, critic, rules)
        return plan

    def run(self, obs_text: str, verifier=None, suggestions: list[str] = []):
        self.think = []
        self.chat_history = []
        rules = self.rules + suggestions
        plan = self.gene_new_plan(obs_text, rules)
        if verifier == "llm":
            plan = self.refine_plan_until_ready(obs_text, plan, rules)
        
        try:
            if len(plan)>0 and plan[0].startswith("Intention"):
                self.last_intention = plan[0]
            else:
                self.last_intention = "Fail to get last intention."
        except:
            self.last_intention = "Fail to get last intention."

        return plan, self.think, self.chat_history
