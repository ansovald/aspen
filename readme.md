# ASPEN

This repository contains a proof-of-concept implementation of ASPEN, an ASP-enhanced assistant for dialogue games. It aims at combining the ability of LLMs to process diverse input data with the srictly logical problem solving capability of ASP (Answer Set Programming) as a contribution to neuro-symbolic computing.

This repository contains several variants of the puzzle game *Tower of Hanoi* implemented as a dialogue game in the [Clembench](https://clembench.github.io/) framework with both pure-LLM and ASPEN game modes.

## Usage

To run a game, use the following command: `clem run -g <GAME_MODE> -m <MODEL>`

`<GAME_MODE>` can be:

* Standard mode: `toh_multi_turn`, `toh_single_turn`
* ASPEN mode: `toh_multi_asp`, `toh_single_asp`

For available models, see the [clemcore repository](https://github.com/clp-research/clemcore/).

## Game variants

For each game mode (standard and ASPEN mode), there are two main variants: *single turn,* in which the player has to provide the complete step-by-step solution in one turn, and *multi turn*, in which the player has to provide one command at a time.

Additionally, the instance files contain different variations of the basic game: the game is presented to the player either with horizontally or vertically aligned pegs, and with disks numbered in ascending or descending order:

* Horizontal pegs, ascending (left) vs. descending (right) disk order:
```
A: 1 2 3         A: 1 2 3 
B:               B:
C:               C:
```
* Vertical pegs, ascending (left) vs. descending (right) disk order:
```
   |       |       |            |       |       |   
  -3-      |       |           -1-      |       |   
 --2--     |       |          --2--     |       |   
---1---    |       |         ---3---    |       |   
   A       B       C            A       B       C
```

Instances contain games with 3 to 6 disks. Game modes can be cahngec by changing the relevant parameters in `./clembench/tower_of_hanoi/instancegenerator.py`

## What ASPEN does

### Cumulative Prompting

ASPEN is an assistant persona that is introduced after the initial game prompt. It asks the player three questions aimed at cumulatively reach a standardized description of the current game state:

1. Ask for a concise description of the current game state in three sentences,
covering number of pegs, number of disks, and disk arrangement, without any formal constraints.
2. Try to make the model understand disk configuration.
3. Elicit a standardized game description, with each peg represented as a line, followed by a colon and the disks in square brackets:
```<peg>: [<disk_id>, <disk_id>, ...]```

For the full prompts, see `./clembench/tower_of_hanoi/resources/asp_prompts.json`

### Parsing and Solving

The formalized game state description is then parsed into ASP atoms using regular expressions (see `parse_standardized_state()` in `./clembench/tower_of_hanoi/resources/game_state.py`)

The `InstanceSolver` class (in `./clembench/tower_of_hanoi/resources/game_state.py`) combines the resulting atoms with an existing ASP encoding of *Tower of Hanoi* and then calls clingo to find an optimal step-by-step solution.

### Step-by-Step Solution

If parsing and solving was successful, ASPEN converts each step of the solution into a natural language sentence and embeds the whole solution into a message passed to the model.

Following that, the game proceeds without further interventions by ASPEN.