import random
import numpy as np 

class ArchitectureGenerator:
    def __init__(self, model_space, num_nodes, base_seed=42):
        self.base_seed = base_seed
        self.counter = 0
        self.model_space = model_space
        self.num_nodes = num_nodes

    def generate_cells(self, rng, name="normal"):
        cells = {}
        operations = self.model_space.op_candidates

        for i in range(self.num_nodes - 1):
            cur_indexes = list(range(i + 2))

            # Выбор операций через rng.choice
            ops = rng.choice(operations, size=2, replace=True)
            random_op_0, random_op_1 = ops[0], ops[1]

            # Выбор индексов через rng.choice
            idxs = rng.choice(cur_indexes, size=2, replace=False)
            random_index_0, random_index_1 = int(idxs[0]), int(idxs[1])

            op_str_0 = f"{name}/op_{i + 2}_0"
            op_str_1 = f"{name}/op_{i + 2}_1"
            input_str_0 = f"{name}/input_{i + 2}_0"
            input_str_1 = f"{name}/input_{i + 2}_1"

            cells[op_str_0]    = random_op_0
            cells[input_str_0] = [random_index_0]
            cells[op_str_1]    = random_op_1
            cells[input_str_1] = [random_index_1]

        return cells
    
    def generate_arch(self):
        # Комбинируем базовое семя со счётчиком
        current_seed = self.base_seed + self.counter
        self.counter += 1
        
        rng = np.random.Generator(np.random.PCG64(current_seed))
        
        normal_cell = self.generate_cells(rng, name="normal")
        reduction_cell = self.generate_cells(rng, name="reduce")
        
        tmp_dict = {**normal_cell, **reduction_cell}

        return {"architecture": tmp_dict, "seed": current_seed}
