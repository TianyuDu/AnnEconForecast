"""
Created: Dec. 29 2018
This script contains methods used in genetic algorithm 
powered hyper parameter searching.
"""
import sys
import os
from typing import Dict, Union, List
import numpy as np
import datetime

sys.path.append("../")
import core.tools.rnn_prepare as rnn_prepare
import core.tools.json_rec as json_rec
import core.tools.dtype_cleaner as dtype_cleaner 


def eval_net(
    model: object,
    param: Dict[str, object],
    file_dir: str,
    metric: str = "mse_val",
    smooth_metric: Union[float, None] = 0.05
) -> float:
    """
    NOTE: You still need a wrapper! A valid evaluation function should only taken
    This methods execute a single training session on the neural network
    built based on the spec. provided in param set and evaluate the 
    performance of the network using provided metric.

    It will be passed as an argument to the genetic optimizer
    directly, so it returns a single float number, as required by the
    optimizer.
    
    Args:
        model:
            The model object to be built.

        param:
            A dictionary specifying the following information.
            NOTE: For more detailed info. on the parameter set, refer to
            the template configuration file.
                - Data pre-processing spec.
                - Neural network spec.
                - Model saving spec.

        file_dir:
            The directory of dataset.

        metric:
            The metric used to evaluate the performance.

        smooth_metric:
            None or a float in range (0, 1)
            If None is specified, the metric on the very last epoch will be 
            returned.
            Otherwise, the average of metric evaluated in the last few epochs
            will be returned.
            - Example: 
            if smooth_metric = 0.05, then the average on metric evaluated on the 
            very last 5% epochs will be returned.

    Returns:
        A float specified by 'metric'.
    """
    # ======== Argument Checking ========
    # ======== End ========
    # Prepare the dataset.
    df_ready = rnn_prepare.prepare_dataset(
        file_dir=file_dir,
        periods=int(param["PERIODS"]),
        order=int(param["ORDER"]),
        remove=None,
        verbose=False
    )

    # Split dataset.
    (X_train, X_val, X_test,
     y_train, y_val, y_test)\
    = rnn_prepare.split_dataset(
        raw=df_ready,
        train_ratio=param["TRAIN_RATIO"],
        val_ratio=param["VAL_RATIO"],
        lags=param["LAGS"]
    )

    # The gross dataset excluding the test set.
    # Excluding the test set for isolation purpose.
    data_feed = {
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
    }
    ep = param["epochs"]
    ckpts = range(int(ep * 0.95), ep)  # Take the final 5% epochs.
    net = model(
        param=param,
        prediction_checkpoints=ckpts,
        verbose=False
    )
    ret_pack = net.fit(
        data=data_feed,
        ret=[metric]
    )
    return float(np.mean(list(ret_pack["mse_val"].values())))


def save_generation(
    population: List[dict],
    generation: int,
    file_dir: str,
    verbose: bool = True
) -> None:
    """
    Save the population genetic information.
    Args:
        population:
            A list of chromosome/gene to be stored.
        generation:
            The index of current generation.
        file_dir:
            folder to save.
    """
    assert os.path.isdir(file_dir), f"{file_dir} is not a valid path."
    
    cur_gen_dir = file_dir + "gen" + str(generation) + "/"

    try:
        os.mkdir(cur_gen_dir)
    except FileExistsError:
        print(f"Folder {cur_gen_dir}occupied, directly write to it.")

    if verbose:
        print(f"Gene container folder created: {cur_gen_dir}")

    for (rank, chromosome) in enumerate(population):
        js_file = f"{cur_gen_dir}rank{str(rank)}.json"
        if verbose:
            print(f"Save to {js_file}")
        writer = json_rec.ParamWriter(
            file_dir=js_file
        )
        cleaned = dtype_cleaner.clean(chromosome)
        writer.write(cleaned)
    if verbose:
        print(f"Generation {str(generation)} saved.")

def train_op(
    optimizer,
    total_gen: int,
    elite: Union[int, float] = 1,
    write_to_disk: Union[None, str] = None
) -> Dict[int, List[object]]:
    """
    Run the genetic optimizer and returns chromosomes
    of best performing entities (the elite group)
    Args:
        optimizer:
            A genetic optimizer.
        
        total_gen:
            An integer denoting the total number of
            generation to evolve.

        elite:
            i) If an integer is given:
            An integer defining the elite class.
            All entities in the top ${elite} will be 
            considered as the elite group in their generation
            and their chromosomes will be stored.

            ii) If a float between (0, 1] is given, it will be 
            interpreted as:
            'The top ${elite}*100 PERCENT is defined as the elite group'
        
        write_to_disk:
            If save the chromosome of the elite group in each generation to
            json files.
            If wish to save chromosome, pass in a file directory. 
            NOTE: This should be a folder/dir, not a json files.
        
    Returns:
        A dictionary in which keys are the generation index
        and the corresponding value is a list of the elite 
        group in that generation.
        Each element is a tuple where the first element is the chromosome and
        the second element is the fittness.
    """
    # ======== Argument Checking ========
    assert isinstance(total_gen, int),\
    f"Total generation should be an integer, received: {total_gen} with type {type(total_gen)}"

    assert (isinstance(elite, int) and elite >= 1)\
    or (isinstance(elite, float) and 0.0 < elite <= 1.0),\
    f"Elite class should be an integer >= 1 or a float in (0, 1], received: {elite} with type {type(elite)}."

    if write_to_disk is not None:
        assert os.path.isdir(write_to_disk),\
        "write_to_disk arg should either be None or a valid directory."
        if not write_to_disk.endswith("/"):
            write_to_disk += "/"
    # ======== End ========

    def report(optimizer) -> None:
        print(f"\nBest fitted entity validatiton MSE: {optimizer.population[0][1]: 0.7f}\
        \nWorst fitted entity validation MSE: {optimizer.population[-1][1]: 0.7f}")

    best_rec = list()
    elite_chromosome = dict()
    
    print(f"Generation: [0/{total_gen}]\nEvaluating the initial population.")
    optimizer.evaluate(verbose=True)
    report(optimizer)

    for gen in range(total_gen):
        start_time = datetime.datetime.now()
        print(f"Generation: [{gen + 1}/{total_gen}]")
        optimizer.select()
        optimizer.evolve()
        optimizer.evaluate(verbose=True)
        report(optimizer)
        best_rec.append(optimizer.population[0][1])
        
        # Store the elite chromosome.
        if isinstance(elite, int):
            # If elite group cutoff is defined by group SIZE.
            cutoff = elite
        elif isinstance(elite, float):
            # If elite group cutoff is defined by population PERCENTILE.
            cutoff = int(elite * len(optimizer.population))
        
        elite_group = optimizer.population[:cutoff]
        elite_chromosome[gen] = elite_group
        
        # NOTE: entity format: (gene, fittness_Score)
        # Write current generation elite 
        if write_to_disk is not None:
            save_generation(
                population=[p[0] for p in elite_group],
                generation=gen,
                file_dir=write_to_disk
            )
        
        end_time = datetime.datetime.now()
        print(f"Time taken: {str(end_time - start_time)}")
    
    print("Final:")
    report(optimizer)
    return elite_chromosome

