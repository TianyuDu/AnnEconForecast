"""
Created: Dec. 24 2018
The baseline genetic optimizer.
"""
from typing import Callable, Dict, List, Tuple, Union, Iterable

import numpy as np


class GeneticOptimizer:
    """
    The baseline optimizer using genetic algorithm.
    The generic genetic optimizer runs on string, numerical, and iterable of numerical data types.
    # TODO: optimize code where ignore is encountered.
    """

    def __init__(
        self,
        gene_pool: Dict[str, Union[object, List[object]]],
        pop_size: int,
        eval_func: Callable[[Dict[str, object]], Union[float, int]],
        mode: Union["min", "max"],
        retain: float = 0.3,
        shot_prob: float = 0.05,
        mutate_prob: float = 0.05,
        verbose: bool = False,
        ignore: Iterable[str] = ()
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
            ignore:
                A tuple of strings, which are keys for parameters. Those keys will be skipped
                in the evolution. (They will be preserved during cross-over and mutate phase)
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
            retain, float) and 0 < retain < 1, "Retain must be a float in range (0, 1)."

        assert isinstance(
            shot_prob, float) and 0 <= shot_prob <= 1, "Shot probability must be a float in range [0, 1]."

        assert isinstance(
            mutate_prob, float) and 0 <= mutate_prob <= 1, "Mutation probability must be a float in range [0, 1]."

        assert isinstance(verbose, bool), "Verbose must be a bool."

        assert all(key in gene_pool.keys() for key in ignore),\
        "Some key(s) in ignore are not valid key for the gene pool."

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
        self.pop_size = pop_size
        self.eval_func = eval_func
        self.mode = mode
        self.retain = retain
        self.shot_prob = shot_prob
        self.mutate_prob = mutate_prob
        self.ignore = ignore

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
        population = []
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

    def evaluate(
        self,
        verbose: bool = False
    ) -> None:
        """
        Assign the evaluated score to each entity.
        Sort the current population so that the more competitive
        entities will be placed at the beginning of the sorted
        list.

        See comments below.
        """
        # Evaluation Phase.
        for (idx, entity) in enumerate(self.population):
            # NOTE: each entity in format (dictionary, score).
            self.population[idx] = (entity[0], self.eval_func(entity[0]))

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

    def select(
        self
    ) -> None:
        """
        Select the top(elite) entity in terms of fitness to be the parents for the next generation.
        As well, entities NOT in the elite group could also be selected with a minor chance.
        """
        retain_length = int(self.retain * self.count_population())

        retained = self.population[:retain_length]
        dropped = self.population[retain_length:]

        # Retain some entity chromosome from the dropped list
        # with minor probability.
        for entity in dropped:
            if np.random.random() <= self.shot_prob:
                retained.append(entity)

        percent_retained = len(retained) / len(self.population)
        if self.verbose:
            print(f"Actual percentage retained: {percent_retained}")
            print("Assigning retained entities to replace the population...")
        self.population = retained

    def mutate(
        self,
        chromosome: Dict[str, object],
        mutate_prob: float = 0.05
    ) -> Dict[str, object]:
        """
        Args:
            chromosome:
                A typical entity in the population to be mutated.
        Returns:
            An entity gene that is mutated.
        
        Randomly mutate genetic information encoded in dictionary with a minor probability.
        NOTE: in most cases, the target chromosome will not be altered.

        For the baseline optimizer, only numerical data type (int or float) 
        are supported for mutation.
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
            if np.random.rand() <= mutate_prob and key not in self.ignore:
                if isinstance(chromosome[key], int) or isinstance(chromosome[key], float):
                    new = mutate_numerical(chromosome[key])
                elif type(chromosome[key]) in [list, tuple]:
                    assert all(
                        type(x) in [float, int]
                        for x in chromosome[key]
                    )
                    iterable_type = type(chromosome[key])
                    new = iterable_type(
                        mutate_numerical(x)
                        for x in chromosome[key]
                    )
                    assert type(new) == type(chromosome[key]),\
                    f"Wrong return type, expect: {type(chromosome[key])}, received{type(new)}."
                else:
                    # NOTE: we can either raise a type error here or leave unsupported type unchanged.
                    new = chromosome[key]
                # Assign back.
                mutated[key] = new
            else:
                # Leave the genetic value unchanged.
                mutated[key] = chromosome[key]

        assert isinstance(mutated, dict)
        assert all(isinstance(key, str) for key in mutated.keys())
        return mutated

    def evolve(
        self
    ) -> None:
        """
        The evolving step is a wrapper for cross over and mutation process.
        This method updates population.
        """
        # Breeding Phase.
        assert len(self.population) >= 2, "Insufficient population."
        while len(self.population) < self.pop_size:
            (p1_idx, p2_idx) = np.random.randint(
                len(self.population),
                size=2
            )
            if p1_idx != p2_idx:
                (p1, p2) = self.population[p1_idx], self.population[p2_idx]
                assert len(p1) == 2 \
                and isinstance(p1, tuple), f"Invalid p1: {p1}"
                assert len(p2) == 2 \
                and isinstance(p2, tuple), f"Invalid p2: {p2}"

                [child1, child2] = self.cross_over(p1[0], p2[0])
                self.population.extend(
                    [(child1, None), (child2, None)]
                )
            
        # Mutation Phase.
        # For checking purpose only
        init_len = self.count_population()
        for idx, entity in enumerate(self.population):
            # Execute mutation process.
            gene, score = entity
            mutated_gene = self.mutate(gene, mutate_prob=self.mutate_prob)
            # Replace the original entity.
            self.population[idx] = (mutated_gene, score)
        
        # For checking purpose only,
        # assert the population size is unchanged before and after mutation
        # to ensure there's no duplicate of chromosome.
        assert init_len == self.count_population()
            
    def cross_over(
        self,
        p1: Dict[str, Union[str, float, int]],
        p2: Dict[str, Union[str, float, int]]
    ) -> [dict, dict]:
        """
        Args:
            p1, p2:
            Two typical entity dictionaries as parents.
        Returns:
            Two typical entity dictionaries as children.
        The basic cross over method, used for string and numerical (float and int) data types only.
        """
        child1 = {key: None for key in p1.keys()}
        child2 = {key: None for key in p1.keys()}

        for key in p1.keys():
            if key in self.ignore:
                new_gene1, new_gene2 = p1[key], p2[key]
            else:
                if (isinstance(p1[key], str) and isinstance(p2[key], str))\
                    or (isinstance(p1[key], int) and isinstance(p2[key], int)):
                    new_gene1 = np.random.choice([p1[key], p2[key]])
                    new_gene2 = np.random.choice([p1[key], p2[key]])
                elif isinstance(p1[key], float) and isinstance(p2[key], float):
                    z = np.random.random()
                    # Take the weighted average with random weight.
                    new_gene1 = z * p1[key] + (1 - z) * p2[key]
                    new_gene2 = (1 - z) * p1[key] + z * p2[key]
                else:
                    raise TypeError("Unsupported data type to cross over.")
            child1[key], child2[key] = new_gene1, new_gene2
        return [child1, child2]
