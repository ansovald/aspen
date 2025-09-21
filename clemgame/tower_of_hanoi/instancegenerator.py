from math import e
from mimetypes import init
import os
import logging
import re

from clemcore.clemgame import GameInstanceGenerator
from resources.game_state import GameState
from string import Template

logger = logging.getLogger(__name__)

N_INSTANCES = 1
DISK_RANGE = range(3, 7) #8)  # Number of disks to generate instances for
# ALIGNMENTS = ["vertical"] #, "horizontal"]
ALIGNMENTS = ["horizontal", "vertical"]
PEG_NAMES = ["alpha"] #, "num"]
DISK_ORDERS = ["ascending", "descending"]
# DISK_ORDERS = ["descending"]
TURN_TYPES = ["single", "multi", "multi_asp"]

class ToHInstanceGenerator(GameInstanceGenerator):

    def __init__(self):
        super().__init__(os.path.dirname(__file__))

    def on_generate(self, seed, turn_type, **kwargs):
        self.turn_type = turn_type.split("_")[0]  # "single" or "multi" (also from "multi_asp")
        if "asp" in turn_type:
            self.asp = True
        else:
            self.asp = False
        for disk_n in DISK_RANGE:
            for align in ALIGNMENTS:
                for peg_names in PEG_NAMES:
                    for disk_order in DISK_ORDERS:
                        prompts = self.load_config_json("resources/prompts.json")
                        experiment = self.add_experiment(f"toh_{disk_n}_disks_{align}_{peg_names}_{disk_order}")
                        default_instance = {}
                        if self.asp:
                            default_instance["asp"] = True
                            default_instance["asp_prompts"] = self.load_json("resources/asp_prompts.json")
                        game_state = GameState(disk_n, align=align, peg_names=peg_names, disk_order=disk_order)
                        peg_names_str = ", ".join(game_state.peg_names)
                        initial_board = str(game_state)
                        default_instance["turn_type"] = self.turn_type
                        default_instance["num_disks"] = disk_n
                        default_instance["align"] = align
                        default_instance["peg_names"] = peg_names
                        default_instance["disk_order"] = disk_order
                        default_instance["num_disks"] = disk_n
                        default_instance["initial_board"] = initial_board
                        initial_prompt = Template(prompts["initial_prompt"]).substitute(
                            peg_names=peg_names_str,
                            initial_board=initial_board,
                            source_peg=game_state.peg_names[0],
                            destination_peg=game_state.peg_names[2]
                        )
                        if self.turn_type == "multi":
                            max_turns = 2 ** disk_n * 2  # Allow twice + 2 the optimal number of moves
                            default_instance["initial_prompt"] = initial_prompt + Template(prompts["start_message"]).substitute(
                                max_turns=max_turns
                            )
                            default_instance["parse_error_message"] = prompts["parse_error_message"]
                            default_instance["rule_violation_message"] = prompts["rule_violation_message"]
                        else:
                            max_turns = 3 # allow three tries for single turn
                            default_instance["initial_prompt"] = initial_prompt + prompts["start_message"]
                            default_instance["not_finished"] = prompts["not_finished"]
                            default_instance["tries"] = prompts["tries"]
                        default_instance["max_turns"] = max_turns
                        default_instance["new_turn_prompt"] = prompts["new_turn_prompt"]
                        default_instance["move_pattern"] = prompts["move_pattern"]
                        default_instance["parse_error_message"] = prompts["parse_error_message"]
                        for i in range(N_INSTANCES):
                            game_instance = self.add_game_instance(experiment, i)
                            for key, value in default_instance.items():
                                game_instance[key] = value

    def load_config_json(self, file_path: str) -> dict:
        """
        Load a JSON file from the game directory.
        If the JSON file contains entries for different languages, load the specified language.
        Then, for all keys that differ on modality, load the specified modality.
        Modalities can also be combined with a comma, like 'text,hybrid'
        """
        data = super().load_json(file_path)
        for key, value in data.items():
            if self.turn_type in value:
                data[key] = value[self.turn_type]
        return data

if __name__ == '__main__':
    for turn_type in TURN_TYPES:
        file_name = f"{turn_type}_instances.json"
        ToHInstanceGenerator().generate(seed=73128361, filename=file_name, turn_type=turn_type)
