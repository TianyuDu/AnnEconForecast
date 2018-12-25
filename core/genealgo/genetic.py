"""
Created: Dec. 24 2018
The base package containing the genetic algorithm packages.
"""
import numpy as np
from typing import Dict, List, Union, Callable, Tuple


class GenericGeneticOptimizer:
    """
    The baseline optimizer using genetic algorithm.
    """

    def __init__(
        self,
        gene_pool: Dict[str, Union[object, List[object]]],
        pop_size: int,
        eval_func: Callable[Tuple[object], float],
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


    def evolve(
        self
    ) -> None:
        raise NotImplementedError()

    def _cross_over(
        p1: Dict[str, object],
        p2: Dict[str, object],
        self
    ) -> None:
        """
        Individual cross over method. This method should only be called in
        evolve phase.
        If the feature is a string, randomly choose one from the chromosome of 
        parents.
        If the method is a float or integer, cross over methods take a weighted
        average with random weight. For integers, a round with int operation will 
        be added.
        """
        child = {key: None for key in p1.keys()}
        
        for k in p1.keys():
            if isinstance(p1[k], str):
                new_gene = np.random.choice([p1[k], p2[k]])
            elif isinstance(p1[k], float):
                z = np.random.random()
                # Take the weighted average with random weight.
                new_gene = z * p1[k] + (1 - z) * p2[k]
            elif isinstance(p1[k], int):
                z = np.random.random()
                # Take the weighted average with random weight.
                # And then round to the nearest integer.
                new_gene = z * p1[k] + (1 - z) * p2[k]
                new_gene = int(np.round(new_gene))
            else:
                raise ValueError()
            child[k] = new_gene
            
        return child
                
        

    def mutate(self, target) -> Dict[str, object]:
        raise NotImplementedError()


class GeneticHyperParameterTuner(GenericGeneticOptimizer):
    raise NotImplementedError
