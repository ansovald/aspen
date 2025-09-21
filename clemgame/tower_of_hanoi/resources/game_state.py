from ast import Dict, parse
from clingo.control import Control
from clingo.symbol import Function, Number
from clingo.core import MessageCode
import os
import logging
from copy import deepcopy
import re
from typing import TypedDict

logger = logging.getLogger(__name__)

def control_logger(code: MessageCode, message: str) -> None:
    message = message.strip().replace('\n', ' ')
    logger.info(f"Clingo [{code.name}]: {message}")

class GameState:
    def __init__(self, num_disks, align="vertical", peg_names="alpha", disk_order="descending"):
        self.num_disks = num_disks
        self.align = align
        if peg_names == "alpha":
            self.peg_names = ["A", "B", "C"]
            disk_type = "num"
        elif peg_names == "num":
            self.peg_names = ["1", "2", "3"]
            disk_type = "alpha"
        self.disks = {i: None for i in range(1, num_disks + 1)}  # Disk size from 1 (largest) to num_disks (smallest)
        if disk_type == "num":
            self.disk_names = list(range(1, num_disks + 1))
        elif disk_type == "alpha":
            self.disk_names = [chr(i) for i in range(65, 65 + num_disks)]  # A, B, C, ...
        if disk_order == "descending":
            self.disk_names = list(reversed(self.disk_names))
        if align == "vertical":
            for i, disk_name in enumerate(self.disk_names):
                self.disks[i + 1] = disk_name
        elif align == "horizontal":
            for i, disk_name in enumerate(self.disk_names):
                size = num_disks - i
                # center disk name with dashes to represent size
                disk_name = str(disk_name).center(size * 2 + 1, '-')
                # fill with spaces to make all disk names the same length
                disk_name = disk_name.center(num_disks * 2 + 1, ' ')
                self.disks[i + 1] = disk_name
        self.pegs = [list(self.disks.keys()), [], []]  # Start with all disks on peg A

    def __str__(self):
        out_string = ""
        if self.align == "horizontal":
            empty_peg = "|".center(self.num_disks * 2 + 1, ' ')
            max_height = self.num_disks + 1
            for level in range(max_height, 0, -1):
                line = []
                for peg in self.pegs:
                    if len(peg) >= level:
                        line.append(self.disks[peg[level - 1]])
                    else:
                        line.append(empty_peg)
                out_string += " ".join(line) + "\n"
            out_string += " ".join(name.center(self.num_disks * 2 + 1, ' ') for name in self.peg_names) + "\n"
        elif self.align == "vertical":
            for i, peg in enumerate(self.pegs):
                line = [self.peg_names[i] + ": "]
                for disk_size in peg:
                    disk_name = self.disk_names[disk_size - 1]
                    line.append(f"{disk_name} ")
                out_string += "".join(line) + "\n"
        return out_string

    def move_disk(self, from_peg, to_peg):
        if to_peg not in self.peg_names:
            return False, f"peg '{to_peg}' does not exist!"
        
        if from_peg not in self.peg_names:
            return False, f"peg '{from_peg}' does not exist!"
        from_peg_idx = self.peg_names.index(from_peg)
        if len(self.pegs[from_peg_idx]) == 0:
            return False, f"peg '{from_peg}' is empty!"  # No disk to move
        disk_num = self.pegs[from_peg_idx][-1]
        
        to_peg_idx = self.peg_names.index(to_peg)
        if from_peg_idx == to_peg_idx:
            return False, "source and target pegs are the same!"
        if len(self.pegs[to_peg_idx]) > 0 and self.pegs[to_peg_idx][-1] > disk_num: # smaller disk_num means larger disk
            return False, f"you cannot place disk '{str(self.disks[disk_num]).strip()}' on top of smaller disk '{str(self.disks[self.pegs[to_peg_idx][-1]]).strip()}'!"
        # Move is valid, perform the move
        self.pegs[from_peg_idx].pop()
        self.pegs[to_peg_idx].append(disk_num)
        return True, f"Moved disk '{str(self.disks[disk_num]).strip()}' from '{self.peg_names[from_peg_idx]}' to '{to_peg}'."

    def check_solvable(self, max_steps):
        instance_solver = InstanceSolver(self.to_asp())
        return instance_solver.solve(max_steps=max_steps)
    
    def check_win(self):
        # Win if all disks are on peg C
        return len(self.pegs[2]) == self.num_disks
    
    def to_asp(self):
        asp_facts = []
        for peg_name in self.peg_names:
            asp_facts.append(f"peg(\"{peg_name}\").")
        for i, disk_size in enumerate(reversed(self.disks.keys())):
            asp_facts.append(f"disk({disk_size}).")
            asp_facts.append(f"goal_on({disk_size}, \"{self.peg_names[-1]}\").")
        for i, peg in enumerate(self.pegs):
            for disk_size in peg:
                asp_facts.append(f"init_on({disk_size}, \"{self.peg_names[i]}\").")
        return "\n".join(asp_facts)
    
def parse_standardized_state(state_str: str):
    """
    Hopefully, the player has provided a standardized state in the format:
    ```
    a: [1, 2]
    b: []
    ...
    ```
    Args:
        state_str (str): the standardized state string
    Returns:
        an ASP instance string
    """
    asp_instance = []
    peg_pattern = re.compile(r'^(?P<peg>\w): *\[(?P<disks>[^\]]*)\] ?$', re.MULTILINE)
    disk_list = []
    pegs = []
    for match in peg_pattern.finditer(state_str):
        peg = match.group("peg")
        pegs.append(peg)
        asp_instance.append(f'peg("{peg}").')
        disks = match.group("disks").split(",") if match.group("disks").strip() != "" else []
        for disk in disks:
            disk_id = disk.strip()
            if not disk_id.isdigit():
                return False, f"<disk_id> '{disk_id}' is not a valid integer"
            if disk_id in disk_list:
                return False, f"you used <disk_id> '{disk_id}' more than once"
            asp_instance.append(f"disk({disk_id}).")
            asp_instance.append(f"init_on({disk_id}, \"{peg}\").")
    if len(pegs) == 0:
        return False, f"I couldn't find any pegs in your message"
    asp_instance.append(f"goal_on(D,\"{pegs[-1]}\") :- disk(D).")
    return True, "\n".join(asp_instance)

class Move(TypedDict):
    disk: int
    from_peg: str = None
    to_peg: str

class InstanceSolver:
    def __init__(self, instance):
        self.ctl = Control(logger=control_logger)
        encoding_path = os.path.join(os.path.dirname(__file__), "tohE.lp")
        self.ctl.load(encoding_path)
        self.ctl.add("check", ["t"], "#external query(t).")
        self.ctl.add("instance", [], instance)
        self.ctl.ground([("instance", []), ("base", [])])
        self.result = None
        self.step = 1
        self.next_move = None
        self.moves: Dict[int, Move] = {}
        self.states = {}

    def solve(self, parse_model=False, max_steps=1000):
        if parse_model: 
            on_model = self.parse_model
        else:
            on_model = None
        while (self.result == None or self.result.unsatisfiable) and self.step < max_steps:
            self.multi_shot_step(self.step)
            self.result = self.ctl.solve(on_model=on_model)
            if self.result.satisfiable:
                return True, self.step
            self.step += 1
        return False, 0

    def multi_shot_step(self, step):
        parts = []
        parts.append(("check", [Number(step)]))
        query = Function("query", [Number(step - 1)])
        self.ctl.release_external(query)
        parts.append(("step", [Number(step)]))
        self.ctl.ground(parts)
        query = Function("query", [Number(step)])
        self.ctl.assign_external(query, True)
    
    def parse_model(self, model):
        symbols = model.symbols(shown=True)
        for sym in symbols:
            if sym.name == "move" and len(sym.arguments) == 3:
                disk = sym.arguments[0].number
                to_peg = sym.arguments[1].string
                turn = sym.arguments[2].number
                self.moves[turn] = {"disk": disk, "to_peg": to_peg}
            elif sym.name == "on" and len(sym.arguments) == 3:
                disk = sym.arguments[0].number
                peg = sym.arguments[1].string
                turn = sym.arguments[2].number
                if turn not in self.states:
                    self.states[turn] = {}
                self.states[turn][disk] = peg
        self.moves = dict(sorted(self.moves.items()))
        for turn in self.moves:
            if turn-1 in self.states:
                self.moves[turn]["from_peg"] = self.states[turn-1][self.moves[turn]["disk"]]

if __name__ == "__main__":
    configs = [
        {"num_disks": 6, "align": "vertical", "peg_names": "alpha", "disk_order": "ascending"},
        {"num_disks": 6, "align": "vertical", "peg_names": "alpha", "disk_order": "descending"},
        {"num_disks": 6, "align": "vertical", "peg_names": "num", "disk_order": "descending"},
        {"num_disks": 6, "align": "vertical", "peg_names": "num", "disk_order": "ascending"},
        # {"num_disks": 5, "align": "horizontal", "peg_names": "alpha", "disk_order": "ascending"},
        # {"num_disks": 5, "align": "horizontal", "peg_names": "alpha", "disk_order": "descending"},
        # {"num_disks": 5, "align": "horizontal", "peg_names": "num", "disk_order": "ascending"},
        # {"num_disks": 5, "align": "horizontal", "peg_names": "num", "disk_order": "descending"},
    ]
    # for config in configs:
        # print(config)
        # game_state = GameState(**config)
        # print(game_state)
        # _, _ = game_state.move_disk("A", "B")
        # print(game_state)
        # success, message = game_state.move_disk("A", "B")
        # print(f"Move A to B: {success}, {message}")
        # # print(game_state.to_asp())
        # solvable, steps = game_state.check_solvable(200)
        # print(f"Solvable: {solvable} in {steps} steps")

    message = "A: [5, 4, 3, 2, 1]\nB: []\nC: []\n\nMove: [A C]"
    success, asp_instance = parse_standardized_state(message)
    print(asp_instance)
    instance_solver = InstanceSolver(asp_instance)
    solvable, steps = instance_solver.solve(parse_model=True, max_steps=200)
    print(f"Solvable: {solvable} in {steps} steps")
