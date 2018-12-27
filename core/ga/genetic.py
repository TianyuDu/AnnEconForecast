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
        eval_func: Callable[[Dict[str, object], Union[float, int]]],
        mode: Union["min", "max"],
        retain: float = 0.3,
        shot_prob: float = 0.05,
        mutate_prob: float = 0.05,
        verbose: bool = False
    ) -> None:
        """
        Args:
            gene_pool:
                A dictionary contains features as keys and possible chromosome as values.
                    - For flexiable chromosome, use list object as value.
                    - For fixed chromosome, use object as value.
                Each entity of the initial population randomly choose one value from the gene pool for each feature chromosome,
                    if the value at this position in gene pool is a list.
                    Otherwise, if the value in gene pool is not a list, the entity takes the value for sure.
            pop_size:
                The size of population, this size will be maintained via selection and breeding operations in each generation.
            eval_func:
                A real-valued function takes an entity/individual (a dictionary) in population and use it as the parameter.
            mode:
                Mode specifies the type of optimization task to be solved, either maximizing the eval_func or minimizing it.
            retain:
                A float specifying the percentage of (best fitted) entities in population to be retained after selection phase.
            shot_prob:
                A float specifying the chance of an entity not in the best-fitted group to be selected as a parent to the next generation.
                This operation reduces the chance that our optimizer stucks in  local extrema.
            mutate_prob:
                A float specifying the chance of chromosome (value) to be randomly mutated.
                Different data type would be mutated in different ways.
                Mutation process preserves the sign of numerical values.
            verbose:
                A bool specifying if the optimizer prints out logs during training session.
        """
        # ======== Argument Checking Phase ========
        assert isinstance(
            pop_size, int) and pop_size > 0, "Population size should be a positive integer."

        assert all(
            isinstance(key, str) for key in gene_pool.keys()
        ), "All keys in gene pool dictionary must be strings."

        assert mode in [
            "min", "max"], "Optimization mode must be either MINimization or MAXimization."

        assert isinstance(
            retian, float) and 0 < retain < 1, "Retain must be a float in range (0, 1)."

        assert isinstance(
            shot_prob, float) and 0 <= shot_prob <= 1, "Shot probability must be a float in range [0, 1]."

        assert isinstance(
            mutate_prob, float) and 0 <= mutate_prob <= 1, "Mutation probability must be a float in range [0, 1]."

        assert isinstance(verbose, bool), "Verbose must be a bool."

        # ======== End ========

        # Admit argument.
        self.verbose = verbose
        if self.verbose:
            print(f"Creating initial population: {pop_size} entities.")

        # Create initial population.
        self.population = self.create_population(gene_pool, pop_size)

        if self.verbose:
            unique_chromosome = self.count_unique()
            print(f"Unique entity chromosome created: {unique_chromosome}")

        # Admit fitness/evaluating function.
        self.eval_func = eval_func
        self.mode = mode
        self.retain = retain
        self.shot_prob = shot_prob
        self.mutate_prob = mutate_prob

        if self.verbose:
            print("Population initialized.")

    def create_population(
        self,
        gene_pool: dict,
        pop_size: int
    ) -> List[dict]:
        """
        Args:
            Refer to docstring in __init__ method.
        Returns:
            A list of dictionaries.

        This method create a list of entities (dictionaries), in which each entity has the same keys as gene_pool,
        but it randomly choose a value avaiable in the gene pool for flexiable chromosome(list of objects). For fixed
        chromosome (single object), the entity takes it for sure.

        Example:
            gene_pool = {"c1": [1, 2, 3], "c2": 10}
            then a typical entity generated from above gene_pool would be {"c1": 2, "c2": 10}.
        """
        population = ()
        for _ in range(pop_size):
            entity = dict()
            # Construct a random new entity from the gene pool.
            for (key, val) in gene_pool.items():
                # Suppose each value faces the same probability of being selected.
                # If single value (certainity) found as chromosome, convert it into a singleton.
                val_lst = val if isinstance(val, list) else [val]
                entity[key] = np.random.choice(val_lst)
            # The second term is the 'score' or 'fittness' for entity,
            # New entity not evaluated yet would be marked with None as score.
            population.append((entity, None))

        if self.verbose:
            print(f"Population created, with size = {pop_size}")
        return population

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
                [np.array(list(entity[0].values()))
                 for entity in self.population]
            ),
            axis=0
        )
        return len(count)

    def evaluation(
        self
    ) -> None:
        """
        # TODO: associate score with population.
        Sort the current population so that the more competitive
        entities will be placed at the beginning of the sorted
        list.
        See comments below.
        """
        for entity in self.population:
            # NOTE: each entity in format (dictionary, score).
            entity[1] = self.eval_func(entity[0])

        if self.mode == "min":
            # If this is a minimization problem, entities with
            # lowest SCORE will be placed at the beginning of the
            # sorted population.
            self.population.sort(key=lambda x: x[1])
        elif self.mode == "max":
            # If this is a maximization probblem, entities with
            # highest SCORE will be placed at the beginning.
            self.population.sort(key=lambda x: x[1], reverse=True)
        else:
            raise ValueError("Unsupported optimization task type, must be either MINimization or MAXimization.")

    def select(
        self,
        verbose: bool = False
    ) -> None:
        retain_length = int(self.retain * self.count_population())
        retained = self.population[:retain_length]
        dropped = self.population[retain_length:]

        # Retain some entity chromosome from the dropped list
        # with small probability.
        for entity in dropped:
            if np.random.random() <= self.shot_prob:
                retained.append(entity)

        percent_retained = len(retained) / len(self.population)
        if verbose:
            print(f"Actual proportion retained: {percent_retained}")
            print("Assigning retained entities to replace the population.")
        self.population = retained

    def mutate(
        self,
        chromosome: Dict[str, object],
        mutate_rate: float = 0.1
    ) -> None:
        """
        Randomly mutate genetic information encoded in dictionary.
        """
        def mutate_float(src: float) -> float:
            # NOTE: change factor formula here to tune the mutation process.
            factor = np.exp(np.random.randn())
            assert factor >= 0
            result = factor * src
            # we wish to preserve the sign of feature.
            assert np.sign(src) == np.sign(result)
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
        assert len(self.population) >= 2, "Insufficient population."
        while len(self.population) < self.init_pop_size:
            [p1, p2] = np.random.choice(self.population, size=2, replace=False)
            off_springs = self._cross_over(p1=p1, p2=p2)
            self.population.extend(off_springs)

    def _cross_over(
        self,
        p1: Dict[str, Union[str, float]],
        p2: Dict[str, Union[str, float]]
    ) -> List[dict]:
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
        return [child1, child2]


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
