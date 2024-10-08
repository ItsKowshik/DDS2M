import argparse
import traceback
import logging
import yaml
import sys
import os
import torch
from utils import utils_logger
import numpy as np
import os
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
from runners.diffusion import Diffusion
torch.set_printoptions(sci_mode=False)
torch.cuda.set_device(7)

def parse_args_and_config():
    """
    Parse arguments and configuration file

    Returns:
        Namespace: Command line arguments
        Namespace: Configuration parameters
    """

    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "--config", type=str, default='demo_denoising.yml', help="Path to the config file"
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "--exp", type=str, default="exp", help="Path for saving running related data."
    )
    parser.add_argument(
        "--timesteps", type=int, default=2000, help="number of steps involved"
    )
    parser.add_argument(
        "--start_point", type=float, default=1000
    )
    parser.add_argument(
        "--deg", type=str, default='denoising0.1', help="Degradation"
    )
    parser.add_argument(
        "--sigma_0", type=float, default=0, help="Sigma_0"
    )
    parser.add_argument(
        "--eta", type=float, default=1, help="Eta"
    )
    parser.add_argument(
        "--etaB", type=float, default=1, help="Eta_b (before)"
    )
    parser.add_argument(
        '--beta', type=float, default=0)
    parser.add_argument(
        '--rank', type=int, default=10)
    args = parser.parse_args()

    # parse config file
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    if new_config.model.iter_number is not list:
        new_config.model.iter_number = [new_config.model.iter_number] * args.timesteps

    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    """
    Convert a dictionary to a Namespace object

    Args:
        config (dict): Dictionary containing configuration parameters

    Returns:
        Namespace: Namespace object containing configuration parameters
    """
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def main():
    args, config = parse_args_and_config()

    # create output file
    args.logger_name = "test"
    args.image_folder = os.path.join('./results', args.logger_name)
    if not os.path.exists(args.image_folder):
        os.makedirs(args.image_folder)
    
    # setup logger
    utils_logger.logger_info(args.logger_name, os.path.join(args.image_folder, args.logger_name+'.log'))
    logger = logging.getLogger(args.logger_name)
    logger.info(f'Writing to {args.image_folder}')
    logger.info("Writing log file to {}".format(args.logger_name))

    try:
        # Creates a runner for the diffusion model
        runner = Diffusion(args, config)
        runner.sample(logger, config, args.image_folder)
    except Exception:
        logging.error(traceback.format_exc())

    return 0


if __name__ == "__main__":
    sys.exit(main())
