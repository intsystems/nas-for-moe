import random
import numpy as np
from typing import Optional
from nas_moe.nni_utils import OPS_PARAMS_NUM


class ArchitectureGenerator:
    def __init__(self, model_space, num_nodes, base_seed=42, remove_permutation=False):
        self.base_seed = base_seed
        self.counter = 0
        self.model_space = model_space
        self.num_nodes = num_nodes
        self.remove_permutation = remove_permutation

    def generate_cells(self, rng, name="normal"):
        cells = {}
        operations = self.model_space.op_candidates.copy()
        if self.remove_permutation:
            operations.remove('pixel_permutation')

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
    
class ParamFilteredArchitectureGenerator(ArchitectureGenerator):
    """
    Генератор архитектур, который выдаёт только архитектуры с числом параметров
    меньше заданного порога. Порог можно вычислить как среднее по случайной выборке.
    Если заморозка модели дорогая/ломается, используется быстая оценка по OPS_PARAMS_NUM.
    """

    def __init__(self, model_space, num_nodes, base_seed=42, remove_permutation=False,
                 threshold: Optional[float] = 1.,
                 mean_sample_size: int = 1000):
        super().__init__(model_space, num_nodes, base_seed, remove_permutation)
        self.model_space = model_space
        self._cache = {}
        self.threshold = self._compute_mean_params(mean_sample_size) * threshold

    def _count_params_estimate(self, arch: dict) -> int:
        """Быстрая оценка числа параметров по OPS_PARAMS_NUM (суммирование по операциям)."""
        total = 0
        for k, v in arch.items():
            # интересуют только строки с op_, значения - строки вида 'sep_conv_3x3'
            if 'op' in k:
                op = v
                total += OPS_PARAMS_NUM[op]
        return total

    def count_params(self, arch: dict) -> int:
        key = str(arch)
        if key in self._cache:
            return self._cache[key]
            
        cnt = self._count_params_estimate(arch)
        self._cache[key] = cnt
        return cnt

    def _compute_mean_params(self, sample_size: int) -> float:
        s = 0
        n = 0
        for _ in range(sample_size):
            arch_obj = self.generate_arch()
            arch = arch_obj['architecture']
            try:
                cnt = self.count_params(arch)
            except RuntimeError:
                cnt = self._count_params_estimate(arch)
            s += cnt
            n += 1
        return s / max(1, n)

    def generate_arch_filtered(self, max_tries: int = 1000):
        """Генерирует одну архитектуру с params < threshold.
           Если не удалось за max_tries попыток — выбрасывает исключение."""
        for _ in range(max_tries):
            arch_obj = self.generate_arch()
            arch = arch_obj['architecture']
            cnt = self.count_params(arch)
            if cnt < self.threshold:
                # возвращаем вместе с подсчитанными параметрами и seed
                return {"architecture": arch, "params": cnt, "seed": arch_obj["seed"]}
        raise RuntimeError(f"Не удалось сгенерировать архитектуру с params < {self.threshold} за {max_tries} попыток")



