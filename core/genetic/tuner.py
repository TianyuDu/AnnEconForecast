"""
Created: Dec. 26 2018
The genetic hyper parameter tuner for neural networks.
"""
import sys
from typing import Dict, List, Tuple, Union, Callable, Iterable

import numpy as np

sys.path.extend("./")
from core.genetic.optimizer import GeneticOptimizer


class GeneticTuner(GeneticOptimizer):
    """
    Genetic Hyper-Parameter Tuner for neural networks.
    """

    def __init__(
        self,
        gene_pool: Dict[str, Union[object, List[object]]],
        pop_size: int,
        eval_func: Callable[[Dict[str, object]], Union[float, int]],
        mode: Union["min", "max"],
        retain: float=0.3,
        shot_prob: float=0.05,
        mutate_prob: float=0.05,
        verbose: bool=False,
        ignore: Iterable[str]=()
    ) -> None:
        """
        Docstring refer to the parent class GeneticOptimizer.
        """
        # ======== Spec ========
        # Ignore the following feature (not considered as hyper parameters) in evolution.
        # TODO: consider add reg-ex here.
        # nn_ignore = (
        #     "ORDER",  # TODO: order can be supported after fix the recursive differencing method.
        #     "TRAIN_RATIO",
        #     "VAL_RATIO",
        #     "tensorboard_path",
        #     "model_path",
        #     "fig_path"
        # )
        # # ======== End ========
        super().__init__(
            gene_pool,
            pop_size,
            eval_func,
            mode,
            retain,
            shot_prob,
            mutate_prob,
            verbose,
            ignore
        )

    def evaluate(
        self,
        verbose: bool = False
    ) -> None:
        """
        Assign the evaluated score to each entity.
        Sort the current population so that the more competitive
        entities will be placed at the beginning of the sorted
        list.
        NOTE: the only difference between this method and the one in generic optimizer
        is the progress bar visualization.
        """
        def progbar(curr, total, full_progbar, net_size, max_ep, lr):
            """
            Progress bar used in training process.
            Modified version, the original one is located in core.tools.visualize
            """
            frac = curr/total
            filled_progbar = round(frac*full_progbar)
        #     print('\r', '#'*filled_progbar + '-'*(
        #         full_progbar-filled_progbar), '[{:>7.2%}]'.format(frac), end='')
            print('\r', '#'*filled_progbar + '-'*(
                full_progbar-filled_progbar),\
                f"Evaluating... [{curr}/{total}, {frac:>7.2%}] Current Net: size={net_size}, ep={max_ep}, lr={lr: 0.4f}", end='')
        # Evaluation Phase.
        for (idx, entity) in enumerate(self.population):
            # NOTE: each entity in format (dictionary, score).
            self.population[idx] = (entity[0], self.eval_func(entity[0]))

            if verbose:
                progbar(idx+1,
                        len(self.population),
                        min(100, len(self.population)),
                        net_size=entity[0]["num_neurons"],
                        max_ep=entity[0]["epochs"],
                        lr=entity[0]["learning_rate"]
                       )

        # Rank Phase.
        if self.mode == "min":
            # If this is a minimization problem, the entity with
            # lowest SCORE will be placed at the beginning of the
            # sorted population.
            self.population.sort(key=lambda x: x[1])
        elif self.mode == "max":
            # If this is a maximization probblem, the entity with
            # highest SCORE will be placed at the beginning.
            self.population.sort(key=lambda x: x[1], reverse=True)
        else:
            raise ValueError("Unsupported optimization task type, must be either MINimization or MAXimization.")

    def cross_over(
        self,
        p1: Dict[str, object],
        p2: Dict[str, object]
    ) -> Tuple[dict]:
        """
        Individual cross over method. This method should only be called in
        evolve phase.
        If the feature is a string, randomly choose one from the chromosome of 
        parents.
        If the method is a float or integer, cross over methods take a weighted
        average with random weight. For integers, a round with int operation will 
        be added.
        """
        child1 = {key: None for key in p1.keys()}
        child2 = {key: None for key in p1.keys()}

        def mixup_float(f1: float, f2: float) -> (float, float):
            """
            Helper function, crossover float objects.
            """
            # Take the random convex combination.
            z = np.random.random()
            new_f1 = z * f1 + (1 - z) * f2
            new_f2 = (1 - z) * f1 + z * f2
            assert all(isinstance(x, float) for x in [new_f1, new_f2]), "Wrong return type(s), both should be float."
            return float(new_f1), float(new_f2)

        def mixup_integer(
            i1: int,
            i2: int,
            choose: bool=True
        ) -> (int, int):
            """
            Helper function, crossover integer objects.
            Args:
                choose: randomly choose an integer or taking the convex combination.
            """
            if choose:
                # TODO: Consider which method to use.
                # Randomly distribute.
                new_i1, new_i2 = np.random.choice([i1, i2], size=2, replace=False)
                new_i1, new_i2 = int(new_i1), int(new_i2)
            else:
                f1, f2 = mixup_float(float(i1), float(i2))
                rd = lambda x: int(round(x))
                new_i1, new_i2 = rd(f1), rd(f2)
            assert isinstance(new_i1, int) and isinstance(new_i2, int),\
            f"Wrong return type(s), both should be int. Found: {new_i1} and {new_i2}."
            return new_i1, new_i2

        def mixup_numerical(
            x1: Union[float, int],
            x2: Union[float, int]
        ) -> Tuple[Union[float, int]]:
            """
            Helper function, crossover numerical objects.
            """
            assert type(x1) == type(x2), "x1 and x2 should have the same numerical type."
            if isinstance(x1, int):
                return mixup_integer(x1, x2)
            elif isinstance(x1, float):
                return mixup_float(x1, x2)
            else:
                raise TypeError("Unsupported data type.")

        for k in p1.keys():
            if type(p1[k]) != type(p2[k]) or isinstance(p1[k], str) or isinstance(p1[k], int):
                # For different types to cross over,
                # E.g. p1 has grad clipping activated (10.0) and p2 has no gradient clipping (None)
                # Randomly distribute.
                if np.random.random() >= 0.5:
                    new_gene1, new_gene2 = p1[k], p2[k]
                else:
                    new_gene1, new_gene2 = p2[k], p1[k]

            # NOTE: Dec. 29 2018, merged to above case.     
            # elif isinstance(p1[k], str) or isinstance(p1[k], int):
            #     if np.random.random() >= 0.5:
            #         new_gene1, new_gene2 = p1[k], p2[k]
            #     else:
            #         new_gene1, new_gene2 = p2[k], p1[k]

            elif isinstance(p1[k], float):
                new_gene1, new_gene2 = mixup_float(p1[k], p2[k])
            elif isinstance(p1[k], list) or isinstance(p1[k], tuple):
                tp = type(p1[k])
                if len(p1[k]) == len(p2[k]):
                    # Case 1: same length.
                    # e.g. two parameter sets give two multi-layer LSTM with neurons [16, 32, 64] and [32, 64, 128]
                    # The data type will be preserved after cross over.
                    mixed_gene = [
                        mixup_numerical(x1, x2)
                        for x1, x2 in zip(p1[k], p2[k])
                    ]
                    new_gene1 = tp(x[0] for x in mixed_gene)
                    new_gene2 = tp(x[1] for x in mixed_gene)
                else:
                    # Case 2: with different length.
                    # TODO: do we need to consider the mixed case.
                    # E.g. different length and different data type.
                    typical_type_1, typical_type_2 = (type(x[0]) for x in [p1[k], p2[k]])
                    assert all(isinstance(x, typical_type_1) for x in p1[k]),\
                    "All elements in p1 should have the same type."
                    assert all(isinstance(x, typical_type_2) for x in p2[k]),\
                    "All elements in p2 should have the same type."
                    assert typical_type_1 == typical_type_2, "Cannot cross over between different data type."

                    # TODO: think about this, how to cross over two list with different lengths. 
                    # e.g. two LSTM spec with different numbers of layers.
                    # (1): [16, 32, 64], (2): [128, 256]
                    # Current solution: randomly distribute.
                    new_gene1, new_gene2 = np.random.choice(
                        [p1[k], p2[k]],
                        size=2,
                        replace=False
                    )
            else:
                # raise TypeError("Unsupported data type to cross over.")
                new_gene1, new_gene2 = np.random.choice(
                    [p1[k], p2[k]],
                    size=2,
                    replace=False
                )

            if k in self.ignore:
                new_gene1, new_gene2 = np.random.choice(
                    [p1[k], p2[k]],
                    size=2,
                    replace=False
                )
                
            child1[k], child2[k] = new_gene1, new_gene2
        return (child1, child2)
