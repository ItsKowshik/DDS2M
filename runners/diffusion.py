import os
import logging
import time
import glob
import scipy
import numpy as np
import tqdm
import torch
import torch.utils.data as data
from models.skip import skip
from models.diffusion import Model
from runners.unet import UNet
from utils.data_utils import data_transform, inverse_data_transform
from functions.denoising import efficient_generalized_steps
from functions.svd_replacement import Denoising
import torchvision.utils as tvu
from runners.VS2M import VS2M
import random
import pickle
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    """
    Get beta schedule for diffusion

    Args:
        beta_schedule (str): The type of beta schedule (quad, linear, const, jsd, sigmoid)
        beta_start (float): The starting value of beta
        beta_end (float): The ending value of beta
        num_diffusion_timesteps (int): The number of diffusion timesteps

    Returns:
        np.ndarray: An array of betas for each timestep
    """
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config, device=None):
        """
        Class to run the diffusion process

        Args:
            args (Namespace): Command line arguments
            config (Namespace): Configuration parameters
            device (torch.device): Device to run the model on
        """
        self.args = args
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def sample(self, logger, config, image_folder):
        """
        Sample images using diffusion

        Args:
            logger (logging.Logger): Logger for logging information
            config (Namespace): Configuration parameters
            image_folder (str): Folder to save the sampled images
        """
        path = os.path.join(config.data.root, config.data.filename)
        with open(path, 'rb') as f:
            x_test, y_test, observed_mask, missing_mask, ground_truth = pickle.load(f, encoding='latin1')

        # x_test = (x_test-np.min(x_test))/(np.max(x_test)-np.min(x_test))
        # y_test = (y_test-np.min(y_test))/(np.max(y_test)-np.min(y_test))
        mask_1 = np.float32(np.tile(np.expand_dims(observed_mask, -1), (x_test.shape[0], 1, 1, 1, 1)))
        mask_2 = np.float32(np.tile(np.expand_dims(missing_mask, -1), (x_test.shape[0], 1, 1, 1, 1)))

        # dataset = TensorDataset(torch.tensor(x_test, dtype=torch.float32), 
        #                         torch.tensor(y_test, dtype=torch.float32), 
        #                         torch.tensor(mask_1, dtype=torch.float32), 
        #                         torch.tensor(mask_2, dtype=torch.float32))

        # dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        x_test_recon = np.empty(x_test.shape)
        all_results = []
        for i, image in enumerate(x_test):
            img_range = np.max(image) - np.min(image)
            img_min = np.min(image)
            image = (image - img_min)/img_range
            model = VS2M(
                self.args.rank, np.ones((self.config.data.image_size, self.config.data.image_size, self.config.data.image_size, self.config.data.channels)),
                np.ones((self.config.data.image_size, self.config.data.image_size, self.config.data.image_size, self.config.data.channels)), 
                self.args.beta, self.config.model.iter_number, self.config.model.lr
            )
            result = self.sample_sequence(model, image, config, logger, image_folder=image_folder,mask=mask_1[i])
            x_test_recon[i] = (result['x_recon'] * img_range) + img_min
            all_results.append(result)
        
            with open(os.path.join(image_folder, f"demo_dataset_recon.pickle"), 'wb') as f:
                new_data = (x_test_recon, y_test, observed_mask, missing_mask, ground_truth)
                pickle.dump(new_data, f)
                f.close()

            with open(os.path.join(image_folder, f"all_stats.pickle"), 'wb') as f:
                pickle.dump(all_results, f)
                f.close()

    def sample_sequence(self, model, image, config=None, logger=None, image_folder=None,mask=None):
        """
        Start sampling a single image

        Args:
            model (VS2M): The model used for sampling
            config (Namespace): Configuration for sampling
            logger (logging.Logger): Logger to record the process
            image_folder (str): Path to save sampled images
        """
        args, config = self.args, self.config
        deg = args.deg
        
        mask = None

        # get degradation matrix
        args.sigma_0 = float(deg[9:])
        H_funcs = Denoising(config.data.channels, config.data.image_size, self.device)
        img_clean = torch.from_numpy(np.float32(image)).permute(3, 0, 1, 2).unsqueeze(0)
        ## to account for scaling to [-1,1]
        args.sigma_0 = 2 * args.sigma_0 
        sigma_0 = args.sigma_0
        
        x_orig = img_clean
        x_orig = x_orig.to(self.device)

        x_orig = data_transform(self.config, x_orig)          # x = 2x - 1

        y_0 = H_funcs.H(x_orig) # (1, 629930) only include the konwn pixel for completion
        y_0 = y_0 + sigma_0 * torch.randn_like(y_0)  #add noise on the konwn pixel for completion

        ## in this operation, the known pixels remain unchanged, and the unknown pixels are filled with 0, which is essentially a rearrangement process for completion
        pinv_y_0 = H_funcs.H_pinv(y_0).view(y_0.shape[0], config.data.channels, self.config.data.image_size, self.config.data.image_size, self.config.data.image_size) 
        ## processing the unknown pixel value for completion
        if mask is not None:
            pinv_y_0 += H_funcs.H_pinv(H_funcs.H(torch.ones_like(pinv_y_0))).reshape(*pinv_y_0.shape) - 1

        pinv_y_0 = inverse_data_transform(config, pinv_y_0[0,:,:,:,:]).detach().permute(1,2,3,0).cpu().numpy()

        x = torch.randn(
            y_0.shape[0],
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )

        return self.sample_image(pinv_y_0, x, model, H_funcs, y_0, sigma_0, mask=mask, img_clean=img_clean, logger=logger, image_folder=image_folder)


    def sample_image(self, pinv_y_0, x, model, H_funcs, y_0, sigma_0, mask=None, img_clean=None, logger=None, image_folder=None):
        """
        Sample an image and denoise it

        Args:
            pinv_y_0 (np.ndarray): Pseudo-inverse of y_0
            x (torch.Tensor): The initial noise tensor
            model (VS2M): The model used for sampling
            H_funcs: The degradation functions
            y_0 (torch.Tensor): The degraded image
            sigma_0 (float): Noise standard deviation
            mask (torch.Tensor, optional): Mask for inpainting
            img_clean (torch.Tensor, optional): Clean image
            logger (logging.Logger, optional): Logger to record the process
            image_folder (str, optional): Path to save sampled images
        """
        skip = self.num_timesteps // self.args.timesteps
        seq = range(0, self.num_timesteps, skip)
        
        # runs the diffusion model for denoising
        return efficient_generalized_steps(pinv_y_0, x, seq, model, self.betas, H_funcs, y_0, sigma_0, \
            etaB=self.args.etaB, etaA=self.args.eta, etaC=self.args.eta, mask=mask, img_clean = img_clean, logger=logger, args=self.args, config=self.config, image_folder=image_folder)

        