import re

on_pattern = r"on\((?P<turn>[0-9]+),\"(?P<peg>[A-Z])\",(?P<disk>[0-9]+)\)"
on_pattern = re.compile(on_pattern)
move_pattern = r"move\((?P<turn>[0-9]+),\"(?P<peg>[A-Z]+)\",(?P<disk>[0-9]+)\)"
move_pattern = re.compile(move_pattern)
moves_pattern = r"moves\((?P<moves>[0-9]+)\)"
moves_pattern = re.compile(moves_pattern)

def parse_model(model):
    model = str(model)
    max_move_match = moves_pattern.match(model)
    max_turns = int(max_move_match.group('moves')) + 1
    parsed_model = { 'moves': max_turns-1, 'turns': {t : {'move': None, 'state': {'A': [], 'B': [], 'C': []}} for t in range(max_turns) }}
    for match in list(move_pattern.finditer(model)):
        turn = int(match.group('turn'))
        peg = match.group('peg')
        disk = match.group('disk')
        parsed_model['turns'][turn]['move'] = f"{disk}{peg}"
    for match in list(on_pattern.finditer(model)):
        turn = int(match.group('turn'))
        peg = match.group('peg')
        disk = int(match.group('disk'))
        parsed_model['turns'][turn]['state'][peg].append(disk)
    return parsed_model

def print_moves(model):
    for turn in model['turns']:
        print(turn)
        for peg in model['turns'][turn]['state']:
            disks = model['turns'][turn]['state'][peg]
            disks.sort(reverse=True)
            print(f"{peg}: {disks}")
