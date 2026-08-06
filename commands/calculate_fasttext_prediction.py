from typing import Optional

import click
from fsspec import get_fs_token_paths

# from text_classification.calculate_metric import calculate_predictions
from tools.calculate_metric_v2 import calculate_predictions


@click.command()
# @click.argument('config', type=click.Path(exists=True))

#
# click.option(
#     '--collect-type',
#     '-ct',
#     'collect_type',
#     default=None,
#     type=str,
#     help='The type of the collect data.')
@click.option(
    '--gt-path',
    '-gt',
    'gt_path',
    default=None,
    type=str,
    help='Path to load gt_path.')

@click.option(
    '--pred-path',
    '-pred',
    'pred_path',
    default=False,
    type=str,
    help='pred_file to load.')

@click.option(
    '--save-results-path',
    '-srp',
    'save_results_path',
    default=None,
    type=str,
    help='Path to save the results.')


def calculate_evaluation(
    pred_path : str, gt_path : str, save_results_path:str):
    """Evaluate a FastText model.

    \b
    Args:
        config (str): Path to the config file.
        model_path (str, optional): Path to the model. Defaults to None.
        save_results_path (str, optional): Path to save the results.
            Defaults to None.
        keep_results (bool): Whether to keep the results.
            Defaults to True.
        num_workers (int): The number of workers to process the data.
            May slow down if the dataset is small, but it will speed up
            the process if the dataset is large.
        process_workers (int): The number of workers to process the `run_dataset_processor`.
            Defaults to 1.
        seed (int, optional): The seed for the random. Defaults to None.
        inplace (bool): Whether to overwrite the processed data.
    """

    calculate_predictions(
        pred_file=pred_path,
        ground_truth_file=gt_path,
        save_results_path=save_results_path
    )


if __name__ == '__main__':
    calculate_evaluation()
