from clingo.control import Control
from model_handling import parse_model, print_moves
import time

def generate_models(disks=3, encoding='tower_encoding.lp', verbose=False):
    with open(encoding, 'r', encoding='utf-8') as lp_file:
        tower_encoding = lp_file.read()
    
    disk_string = f"disks({disks}).\n"
    tower_encoding = disk_string + tower_encoding
    ctl = Control()
    ctl.add(tower_encoding)
    total_time = time.time()
    ground_time = time.time()
    ctl.ground()
    ground_time = time.time() - ground_time
    if verbose:
        print(f"Grounding for {disks} disks completed in {ground_time:.4f} seconds.")
    solve_time = time.time()
    models = []
    with ctl.solve(yield_=True) as solve:
        for model in solve:
            models.append(parse_model(model))
    solve_time = time.time() - solve_time
    if verbose:
        print(f"Solving for {disks} disks completed in {solve_time:.4f} seconds.")
    total_time = time.time() - total_time
    if verbose:
        print(f"Total time for {disks} disks: {total_time:.4f} seconds.")
    times = {
        'ground_time': ground_time,
        'solve_time': solve_time,
        'total_time': total_time
    }
    return models[0], times

if __name__ == "__main__":
    model, _ = generate_models(disks=3)
    print_moves(model)
