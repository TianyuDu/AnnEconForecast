"""
Created: Dec. 24 2018
The base package containing the genetic algorithm packages.
"""
import numpy as np
from typing import Dict, List, Union, Callable, Tuple


class GeneticOptimizer:
    """
    The baseline optimizer using genetic algorithm.
    """

    def __init__(
        self,
        gene_pool: Dict[str, Union[object, List[object]]],
        pop_size: int,
        eval_func: Callable[[object], Union[float,int]],
        mode: Union["min", "max"],
        retain: float,
        shot_prob: float=0.2,
        mutate_prob: float=0.2,
        verbose: bool=False
    ) -> None:
        """
        #TODO: Docstring
        """
        assert isinstance(
            pop_size, int) and pop_size > 0, "Population size should be a positive integer."

        if verbose:
            print(f"Creating initial population: {pop_size} entities.")

        # Create initial population.
        self.population = list()
        for _ in range(pop_size):
            new_entity = dict()
            # Construct a random new entity from the gene pool.
            for (key, val) in gene_pool.items():
                # Suppose each value faces the same probability of being selected.
                val_lst = val if isinstance(val, list) else [val]
                new_entity[key] = np.random.choice(val_lst)
            self.population.append(new_entity)
        if verbose:
            print("Done.")
            unique_chromosome = self.count_unique()
            print(f"Unique entity chromosome created: {unique_chromosome}")

        # Admit fitness/evaluating function.
        self.eval_func = eval_func

        # The mode determines the correlation between fitness
        # and probability of being admitted the next generation
        # during the selection phase.
        assert mode in ["max", "min"], "Invalid optimization mode."
        self.mode = mode

        # The ratio of population to be retained to the next generation
        # after the selection/elimination phase
        assert isinstance(retain, float) and 0 <= retain <= 1, "Invalid retain ratio."
        self.retain = retain

        assert isinstance(shot_prob, float) and 0 <= shot_prob <= 1, "Invalid shot probability."
        self.shot_prob = shot_prob

        assert isinstance(mutate_prob, float) and 0 <= mutate_prob <= 1, "Invalid mutation probability."
        self.mutate_prob = mutate_prob

        if verbose:
            print("Initial population created.")
    

    def count_population(self) -> int:
        """
        Return the current population size.
        """
        return len(self.population)


    def count_unique(
        self
    ) -> int:
        """
        Count the unique chromosome in current poplation.
        This measures the varity of population gene pool.
        """
        count = np.unique(
            np.array(
                [np.array(list(entity.values())) for entity in self.population]
            ),
            axis=0
        )
        return len(count)
        
    def evaluation(
        self
    ) -> None:
        """
        Sort the current population so that the more competitive
        entities will be placed at the beginning of the sorted
        list.
        See comments below.
        """
        if self.mode == "min":
            # If this is a minimization problem, entities with 
            # lowest SCORE will be placed at the beginning of the
            # sorted population.
            self.population.sort(key=self.eval_func)
        elif self.mode == "max":
            # If this is a maximization probblem, entities with
            # highest SCORE will be placed at the beginning.
            self.population.sort(key=self.eval_func, reverse=True)
    
    def select(
        self,
        verbose: bool=False
    ) -> None:
        retain_length = int(self.retain * self.count_population())
        retained = self.population[:retain_length]
        dropped = self.population[retain_length:]

        # Retain some entity chromosome from the dropped list
        # with small probability.
        for entity in dropped:
            if np.random.random >= self.shot_prob:
                retained.append(entity)
        
        percent_retained = len(retained) / len(self.population)
        if verbose:
            print(f"Actual proportion retained: {percent_retained}")
            print("Assigning retained entities to replace the population.")
        self.population = retained

    def mutate(
        self,
        chromosome: Dict[str, object],
        mutate_rate: float=0.1
    ) -> None:
        """
        Randomly mutate genetic information encoded in dictionary.
        """
        def mutate_float(src: float) -> float:
            # NOTE: change factor formula here to tune the mutation process.
            factor = np.exp(np.random.randn())
            assert factor >= 0
            result = factor * src
            assert np.sign(src) == np.sign(result)  # we wish to preserve the sign of feature.
            return result
        
        def mutate_int(src: int) -> int:
            f = mutate_float(src)
            result = int(np.round(f))
            assert np.sign(src) == np.sign(result)
            return result
        
        def mutate_numerical(src: Union[float, int]) -> Union[float, int]:
            if isinstance(src, float):
                return mutate_float(src)
            elif isinstance(src, int):
                return mutate_int(src)
            else:
                raise TypeError("Unsupported data type for mutation.")

        mutated = {key: None for key in chromosome.keys()}

        for key in chromosome.keys():
            if np.random.rand() <= mutate_rate:
                if isinstance(chromosome[key], int) or isinstance(chromosome[key], float):
                    new = mutate_numerical(chromosome[key])
                elif isinstance(chromosome[key], list):
                    assert all(
                        type(x) in [float, int]
                        for x in chromosome[key]
                    )
                    new = [
                        mutate_numerical(x)
                        for x in chromosome[key]
                    ]
                else:
                    # NOTE: we can either raise a type error here or leave unsupported type unchanged.
                    new = chromosome[key]
                # Assign back.
                mutated[key] = new
            else:
                mutated[key] = chromosome[key]
        return mutated

    def evolve(
        self
    ) -> None:
        """
        The evolving step is a wrapper for cross over and mutation process.
        This method updates population.
        """
        

    def _cross_over(
        p1: Dict[str, Union[str, float]],
        p2: Dict[str, Union[str, float]],
        self
    ) -> Tuple[dict]:
        """
        The basic cross over method, used for string and float data type only.
        """
        child1 = {key: None for key in p1.keys()}
        child2 = {key: None for key in p1.keys()}
        
        for k in p1.keys():
            if isinstance(p1[k], str):
                new_gene1 = np.random.choice([p1[k], p2[k]])
                new_gene2 = np.random.choice([p1[k], p2[k]])
            elif isinstance(p1[k], float):
                z = np.random.random()
                # Take the weighted average with random weight.
                new_gene1 = z * p1[k] + (1 - z) * p2[k]
                new_gene2 = (1 - z) * p1[k] + z * p2[k]
            else:
                raise TypeError("Unsupported data type to cross over.")
            child1[k], child2[k] = new_gene1, new_gene2
        return (child1, child2)


class GeneticHPT(GeneticOptimizer):
    """
    Genetic Hyper-Parameter Tuner
    """
    def __init__(self):
        raise NotImplementedError()

    def _cross_over(
        p1: Dict[str, object],
        p2: Dict[str, object],
        self
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

        for k in p1.keys():
            if isinstance(p1[k], str):
                new_gene1 = np.random.choice([p1[k], p2[k]])
                new_gene2 = np.random.choice([p1[k], p2[k]])
            elif isinstance(p1[k], float):
                z = np.random.random()
                # Take the weighted average with random weight.
                new_gene1 = z * p1[k] + (1 - z) * p2[k]
                new_gene2 = (1 - z) * p1[k] + z * p2[k]
            elif isinstance(p1[k], int):
                z = np.random.random()
                # Take the weighted average with random weight.
                # And then round to the nearest integer.
                new_gene1 = int(np.round(
                    z * p1[k] + (1 - z) * p2[k]
                ))
                new_gene2 = int(np.round(
                    (1 - z) * p1[k] + z * p2[k]
                ))
            elif isinstance(p1[k], list):
                # TODO: add cross over rules for list with possibly different length.
                raise NotImplementedError()
            else:
                raise TypeError("Unsupported data type to cross over.")
            child1[k], child2[k] = new_gene1, new_gene2

        return (child1, child2)
