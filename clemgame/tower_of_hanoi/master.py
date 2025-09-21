import re
import os
import logging
import random
from string import Template
import trace
from typing import List, Dict, Optional
import numpy as np
from copy import deepcopy

from regex import P

from clemcore.backends import Model, HumanModel, CustomResponseModel
from clemcore.clemgame import GameSpec, GameMaster, GameBenchmark, Player, DialogueGameMaster, GameScorer, ParseError, ProtocolError
from clemcore.clemgame.metrics import METRIC_ABORTED, METRIC_SUCCESS, METRIC_LOSE, BENCH_SCORE
from clemcore.clemgame.events import GameEventSource
from resources.game_state import GameState, InstanceSolver, parse_standardized_state

logger = logging.getLogger(__name__)

class Solver(Player):
    def __init__(self, model: Model, peg_names: list[str], turn_type: str):
        super().__init__(model)
        self.peg_names = peg_names
        self.turn_type = turn_type
    
    def random_move(self) -> str:
        from_peg, to_peg = random.sample(self.peg_names, 2)
        return f"[{from_peg} {to_peg}]"
    
    def _custom_response(self, prompt: str) -> str:
        if self.turn_type == "single":
            reply = ""
            for i in range(random.randint(1, 10)):
                move = self.random_move()
                reply += f"{i+1}. {move}\n"
            return reply
        else:
            return self.random_move()

class ToHMaster(DialogueGameMaster):
    def __init__(self, game_spec, experiment: Dict, player_models: List[Model]):
        super().__init__(game_spec, experiment, player_models)
    
    def _on_setup(self, **game_instance):
        self.asp = game_instance.get("asp", False)
        self.log_key("markdown", True)
        self._game_state = GameState(
            num_disks=self.game_instance["num_disks"],
            align=self.game_instance["align"],
            peg_names=self.game_instance["peg_names"],
            disk_order=self.game_instance["disk_order"]
        )
        self.game_instance = game_instance
        self.turn_type = self.game_instance["turn_type"]
        self.move_pattern = re.compile(self.game_instance["move_pattern"])
        self.new_turn_prompt = self.game_instance["new_turn_prompt"]
        self.max_turns = self.game_instance["max_turns"]
        self.parse_error_message = self.game_instance["parse_error_message"]
        self.solver = Solver(self.player_models[0], self._game_state.peg_names, self.turn_type)
        if self.asp:
            self.add_player(self.solver)
            self.asp_parse_success = False
            self.asp_parse_errors = 0
            self.asp_prompts = self.game_instance["asp_prompts"]
        else:
            self.add_player(self.solver)
        self.current_errors = 0
        self.total_errors = 0
        self.rule_violations = 0
        self.valid_moves = 0
        self.invalid_moves = 0
        # self.aborted = True
        self.aborted = False
        self.success = False
        self.optimal_steps = 2 ** self.game_instance["num_disks"] - 1
        self.opt_steps_to_finish = (self.optimal_steps, 0)
        self.steps_to_finish = {}

    def _on_before_game(self):
        if not self.asp:
            self.set_context_for(self.solver, self.game_instance["initial_prompt"])
        else:
            initial_context = {
                "role": "user",
                "content": self.game_instance["initial_prompt"]
            }
            self.solver.perceive_context(context=initial_context)
            self.preliminary_questions()

    def preliminary_questions(self):
        for _, prompt in self.asp_prompts["preliminary_prompts"].items():
            context = { 
                "role": "user", 
                "content": prompt }
            response = self.solver(context=context)
        # last response should contain the standardized game state
        asp_instance = parse_standardized_state(response)
        while self.asp_parse_errors <= 3 and not self.asp_parse_success:
            error_prompt = None
            try:
                instance_solver = InstanceSolver(asp_instance)
                solvable, steps = instance_solver.solve(parse_model=True, max_steps=self.max_turns)
                if solvable:
                    self.log_to_self("ASP Initialization", f"Game solvable in {steps} steps.")
                    self.asp_parse_success = True
                    move_template = self.asp_prompts["move_template"]
                    move_list = []
                    for turn in range(1, len(instance_solver.moves.keys())+1):
                        move = instance_solver.moves[turn]
                        move_list.append(
                            Template(move_template).substitute(
                                turn=turn, disk=move["disk"], from_peg=move["from_peg"], to_peg=move["to_peg"]))
                    self.set_context_for(self.solver, self.asp_prompts["prelim_end"]["success"].replace("$move_list", "\n".join(move_list)))
                else:
                    self.asp_parse_errors += 1
                    error_prompt = self.asp_prompts["ground_error"]
            except AssertionError as e:
                self.asp_parse_errors += 1
                logger.error(e.args)
                error_prompt = Template(self.asp_prompts["parse_error"]).substitute(reason=e)
                logger.error(f"could not ground following instance:\n{asp_instance}:\n{e}")
                self.log_to_self("ASP Initialization", "ASP parse error during initialization.")
            if error_prompt:
                response = self.solver(
                    context = { "role": "user", "content": error_prompt }
                    )
                logger.info(f"response:\n{response}")
                asp_instance = parse_standardized_state(response)
        if not self.asp_parse_success:
            self.set_context_for(self.solver, self.asp_prompts["prelim_end"]["failure"])


    def compute_turn_score(self):
        """Score response based on last context (for playpen RL)
        :return: the performance score for a player's response given its last context
        """
        return 0

    def compute_episode_score(self):
        """
        :return: the performance of the agent over the whole episode
        """
        return 0

    def _parse_response(self, player: Player, response: str):
        # see if response matches move pattern
        match = self.move_pattern.search(response)
        if match:
            return response
        else:
            self.log_to_self("Invalid Response", f"Response does not match move pattern: {response}")
            raise ParseError()
    
    def _count_error(self):
        self.current_errors += 1
        self.log_to_self("Error Count", f"Current consecutive errors: {self.current_errors}")
        if self.current_errors > 2:
            self.log_to_self("Aborting", "Too many errors, aborting game.")
            self.aborted = True

    def _on_parse_error(self, error: ParseError):
        self._count_error()
        self.set_context_for(self.solver, self.parse_error_message)
    
    def _advance_game(self, player: Player, parsed_response: str):
        matches = list(self.move_pattern.finditer(parsed_response))
        if self.turn_type == "multi":
            move = matches[-1]  # Assume the last move is the intended one
            success, message = self._game_state.move_disk(from_peg=move.group("source"), to_peg=move.group("target"))
            if success:
                self.current_errors = 0
                self.success = self._game_state.check_win()
                asp_message = ""
                if not self.success:
                    solvable, steps = self._game_state.check_solvable(max_steps=self.max_turns - self.current_round - 1)
                    if solvable:
                        asp_message = f"Game solvable in {steps} steps."
                        self.steps_to_finish[self.current_round] = steps
                        if steps < self.opt_steps_to_finish[0]:
                            self.opt_steps_to_finish = (steps, self.current_round)
                        if steps > self.max_turns - self.current_round - 1:
                            self.log_to_self("Aborting", "Game not solvable within remaining turns, aborting game.")
                            self.aborted = True
                    else:
                        asp_message = "ASP check", "Game no longer solvable, aborting."
                        self.aborted = True
                self.valid_moves += 1
                self.log_to_self("Success", f"Valid move. Game finished: {self.success}\n\n{asp_message}")
                message += "\n" + Template(self.new_turn_prompt).substitute(
                    current_board=str(self._game_state),
                    remaining_turns=self.max_turns - self.current_round - 1
                )
                self.set_context_for(self.solver, message)
            else:
                self.rule_violations += 1
                self.invalid_moves += 1
                self._count_error()
                self.log_to_self("Invalid Move", f"Invalid move: {message}")
                message = Template(self.game_instance["rule_violation_message"]).substitute(reason=message)
                self.set_context_for(self.solver, message)
        elif self.turn_type == "single":
            for move in matches:
                success, message = self._game_state.move_disk(from_peg=move.group("source"), to_peg=move.group("target"))
                if not success:
                    self.invalid_moves += 1
                    move_n = move.group('move_n')
                    solvable, steps = self._game_state.check_solvable(max_steps=self.max_turns - self.current_round - 1)
                    asp_message = ""
                    if solvable:
                        asp_message = f"Game would have been solvable in {steps} more steps."
                        if steps < self.opt_steps_to_finish[0]:
                            self.opt_steps_to_finish = (steps, self.current_round)
                    else:
                        asp_message = "Game wouldn't have been solvable at this point."
                    self.log_to_self("Failure", f"Failed at move {move_n}: {message}\n{asp_message}")
                    self.rule_violations += 1
                    message = Template(self.new_turn_prompt).substitute(
                        move=move.group('move_n'),
                        reason=message,
                        remaining_turns=self.max_turns - self.current_round - 1,
                        tries=self._get_tries_form()
                    )
                    self.set_context_for(self.solver, message)
                    return
                else:
                    self.valid_moves += 1
                    self.current_errors = 0
                    self.success = self._game_state.check_win()
            if self.success:
                self.log_to_self("Success", f"Game finished with {self.valid_moves} moves.")
            else:
                self.log_to_self("Valid Moves", f"All moves valid, but goal not reached:\n```\n{self._game_state}\n```")
                message = Template(self.game_instance["not_finished"]).substitute(
                    tries=self._get_tries_form(),
                    remaining_turns=self.max_turns - self.current_round - 1
                    )
                self.reset_game_state()
                self.set_context_for(self.solver, message)
    
    def _get_tries_form(self) -> str:
        tries = self.max_turns - self.current_round - 1
        if tries == 1:
            return self.game_instance["tries"]["sg"]
        else:
            return self.game_instance["tries"]["pl"]

    def _does_game_proceed(self) -> bool:
        if self.aborted:
            return False
        if self.current_round >= self.max_turns:
            self.aborted = True
            self.log_to_self("Aborting", "Maximum number of turns reached, aborting game.")
            return False
        return not self.success
    
    def _on_after_game(self):
        self.log_to_self("Final Board", "Final Board:\n```\n" + str(self._game_state) + "\n```")
        lose = not self.success
        self.log_key("Turns", self.current_round)
        self.log_key(METRIC_ABORTED, int(self.aborted))
        self.log_key(METRIC_LOSE, int(lose))
        self.log_key(METRIC_SUCCESS, int(self.success))
        if self.success:
            self.log_key(BENCH_SCORE, 1.0)
        else:
            self.log_key(BENCH_SCORE, 0.0)
        self.log_key("Optimal Steps", self.optimal_steps)
        self.log_key("Closest to Solution", self.opt_steps_to_finish[0])
        self.log_key("Steps to Finish", str(self.steps_to_finish))
        self.log_key("Turn at Closest", self.opt_steps_to_finish[1])
        self.log_key("Valid Moves", self.valid_moves)
        self.log_key("Invalid Moves", self.invalid_moves)
        if self.asp:
            self.log_key("ASP Parse Success", self.asp_parse_success)
            self.log_key("ASP Parse Errors", self.asp_parse_errors)

class ToHScorer(GameScorer):
    def __init__(self, game_name: str, experiment: Dict, game_instance: Dict):
        super().__init__(game_name, experiment, game_instance)

    def compute_episode_scores(self, episode_interactions: Dict) -> float:
        if episode_interactions[METRIC_SUCCESS]:
            bench_score =  (int(episode_interactions["Optimal Steps"])) / int(episode_interactions["Valid Moves"]) * 100
        else:
            bench_score = 0
        self.log_episode_score(BENCH_SCORE, bench_score)
        self.log_episode_score("Optimal Steps", episode_interactions["Optimal Steps"])
        self.log_episode_score("Valid Moves", episode_interactions["Valid Moves"])
        self.log_episode_score("Invalid Moves", episode_interactions["Invalid Moves"])
        for key in ["ASP Parse Success", "ASP Parse Errors"]:
            if key in episode_interactions:
                self.log_episode_score(key, episode_interactions[key])

class ToHBenchmark(GameBenchmark):

    def __init__(self, game_spec: GameSpec):
        super().__init__(game_spec)

    def create_game_master(self, experiment: Dict, player_models: List[Model]) -> GameMaster:
        return ToHMaster(self.game_spec, experiment, player_models)

    def create_game_scorer(self, experiment: Dict, game_instance: Dict) -> GameScorer:
        return ToHScorer(self.game_name, experiment, game_instance)