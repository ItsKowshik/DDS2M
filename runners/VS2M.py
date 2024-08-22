#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: yuchun   time: 2020/7/10
from collections import namedtuple
from runners.com_psnr import quality
from models import *
from models.fcn import fcn
from models.skip3D import skip
from models.losses import *
from models.noise import *
from utils.image_io import *
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
Result = namedtuple("Result", ['recon', 'psnr', 'step'])

class VS2M(object):
    def __init__(self, rank, image_noisy, image_clean, beta, num_iter, lr=0.001):
        """
        Initialize the VS2M class.

        Args:
            rank (int): Rank of the matrices
            image_noisy (np.ndarray): The noisy input image
            image_clean (np.ndarray): The clean target image
            beta (float): Regularization parameter for total variation loss
            num_iter (int): Number of iterations for optimization
            lr (float): Learning rate for the optimizer
        """
        self.beta = beta
        self.rank = rank
        self.channel = image_noisy.shape[3]
        self.image_size = image_noisy.shape[0]
        self.now_rank = self.rank
        self.image = np.reshape(image_noisy, (-1, image_noisy.shape[3]), order="F")
        self.image_clean = image_clean
        self.num_iter = num_iter
        self.image_net = None
        self.mask_net = None
        self.mse_loss = None
        self.learning_rate = lr
        self.parameters = None
        self.current_result = None
        self.input_depth = 1
        self.output_depth = 1
        self.exp_weight = 0.98
        self.best_result = None
        self.best_result_av = None
        self.image_net_inputs = None
        self.mask_net_inputs = None
        self.image_out = None
        self.mask_out = None
        self.done = False
        self.ambient_out = None
        self.total_loss = None
        self.post = None
        self._init_all()
        self.out_avg = 0
        self.save_every = 1000
        self.o = torch.zeros((self.image_clean.shape[0] * self.image_clean.shape[1] * self.image_clean.shape[2], self.image_clean.shape[3])).type(
            torch.cuda.FloatTensor)
        self.previous = np.zeros(self.image_clean.shape)


    def _init_images(self):
        """
        Initialize noisy input image
        """
        self.original_image = self.image.copy()
        image = self.image
        self.image_torch = np_to_torch(image).type(torch.cuda.FloatTensor)
        self.image_torch = self.image_torch.squeeze(0)

    def _init_nets(self):
        """
        Initialize neural networks for the optimization process
        """
        pad = 'reflection'
        data_type = torch.cuda.FloatTensor
        self.image_net = []
        self.parameters = []
        for i in range(self.rank):
            net = skip(self.input_depth, self.output_depth,  num_channels_down = [4, 8, 16, 32],
                           num_channels_up = [4, 8, 16, 32],
                           num_channels_skip = [0, 0, 2, 2],
                           filter_size_down = [5, 5, 3, 3], filter_size_up = [5, 5, 3, 3],
                           upsample_mode='trilinear', downsample_mode='avg',
                           need_sigmoid=False, pad=pad, act_fun='LeakyReLU').type(data_type)
            self.parameters = [p for p in net.parameters()] + self.parameters
            self.image_net.append(net)

        self.mask_net = []
        for i in range(self.rank):
            net = fcn(self.image_clean.shape[3], self.image_clean.shape[3], num_hidden=[128, 256, 256, 128]).type(data_type)
            self.parameters = self.parameters + [p for p in net.parameters()]
            self.mask_net.append(net)

    def _init_loss(self):
        """
        Initialize loss functions
        """
        data_type = torch.cuda.FloatTensor
        self.mse_loss = torch.nn.MSELoss().type(data_type)
        self.sp_loss = SPLoss().type(data_type)
        self.kl_loss = KLLoss().type(data_type)
        self.tv_loss = TVLoss3d().type(data_type)

    def _init_inputs(self):
        """
        Initialize inputs to neural net
        """
        original_noise = torch_to_np(get_noise1(1, 'noise', (self.input_depth, *self.image_clean.shape[:3]), noise_type='u',
                                                                     var=10/10.).type(torch.cuda.FloatTensor).detach())
        self.image_net_inputs = np_to_torch(original_noise).type(torch.cuda.FloatTensor).detach()[0, :, :, :, :]
        original_noise = torch_to_np(get_noise2(1, 'noise', self.image.shape[1], noise_type='u', var=10/ 10.).type(torch.cuda.FloatTensor).detach())
        self.mask_net_inputs = np_to_torch(original_noise).type(torch.cuda.FloatTensor).detach()[0, :, :, :]
        self.mask_net_inputs = self.mask_net_inputs

    def _init_optimizer(self):
        """
        Initialize optimizer for neural net
        """
        self.optimizer = torch.optim.Adam(self.parameters, lr=self.learning_rate)

    def _init_all(self):
        """
        Wrapper function for initializing all variables
        """
        self._init_images()
        self._init_nets()
        self._init_inputs()
        self._init_loss()
        self._init_optimizer()

    def reinit(self):
        """
        Reinitialize the neural networks and optimizer
        """
        self._init_nets()
        self._init_optimizer()

    def optimize(self, image_noisy, image_clean, at, mask, iteration, logger, avg, update):
        """
        Train the neural networks to reconstruct the clean image from the noisy input

        Args:
            image_noisy (np.ndarray): The noisy input image
            image_clean (np.ndarray): The clean target image
            at (torch.Tensor): Alpha values for diffusion process
            mask (np.ndarray): Mask
            iteration (int): Number of iterations for optimization
            logger (logging.Logger): Logger to record the process
            avg (np.ndarray): Averaged image from previous iterations
            update (bool): Flag to indicate whether to update the parameters

        Returns:
            (np.ndarray, ): The reconstructed image, the best step, the best reconstruction, and the best PSNR
        """
        self.num_iter = iteration
        # self.mask = torch.from_numpy(mask).cuda()
        self.image = np.reshape(image_noisy, (-1, image_noisy.shape[3]), order="F")
        self.out_avg = avg
        self.image_clean = image_clean
        self._init_images()

        # update parameters for a set number of iterations and save predicted denoised image
        for j in range(self.num_iter + 1):
            self.optimizer.zero_grad()
            self._optimization_closure(at, j)
            self._obtain_current_result(j)
            self._plot_closure(j, logger)
            if update:
                self.optimizer.step()
            else:
                break
        return self.current_result_av.recon, self.best_result_av.step, self.best_result_av.recon, self.best_result_av.psnr

    def _optimization_closure(self, at, j):
        """
        Closure function for the optimization step

        Args:
            at (torch.Tensor): Alpha values for diffusion process
            j (int): Current iteration step
        """
        at = at[0,0,0,0,0]
        m = 0

        # pass image input through all networks
        M = self.image_net_inputs
        out = self.image_net[0](M)
        for i in range(1, self.now_rank):
            out = torch.cat((out, self.image_net[i](M)), 0)
        out = out[:, :, :self.image_clean.shape[0], :self.image_clean.shape[1], :self.image_clean.shape[2]]

        # reshape and combine outputs for all ranks
        self.image_out = out[m, :, :, :, :].squeeze(0).reshape((-1, 1))
        for m in range(1, self.now_rank):
            self.image_out = torch.cat((self.image_out, out[m, :, :, :, :].squeeze(0).reshape((-1, 1))), 1)
        self.image_out_np = torch_to_np(self.image_out)

        # run mask networks
        M = self.mask_net_inputs
        out = self.mask_net[0](M)
        for i in range(1, self.now_rank):
            out = torch.cat((out, self.mask_net[i](M)), 0)

        self.mask_out = out.squeeze(1)
        self.mask_out_np = torch_to_np(self.mask_out)

        # combine image and mask network outputs
        self.image_com = self.image_out.mm(self.mask_out)
        self.image_com_np = np.matmul(self.image_out_np, self.mask_out_np)
        self.image_com_np = np.reshape(self.image_com_np, self.image_clean.shape, order='F')
        self.out_avg = self.out_avg * self.exp_weight + self.image_com_np * (1 - self.exp_weight)

        self.image_com_rescale = self.image_com * 2 - 1.0

        self.out_avg_rescale = self.out_avg * 2 - 1.0
        
        # Compute the residual tensor for the KL divergence loss
        self.et = (self.image_torch - at.sqrt() * self.image_com_rescale) / (1 - at).sqrt()
        self.mean = torch.mean(self.et)
        self.var = torch.var(self.et)

        # compute losses
        self.loss1 = self.mse_loss(self.image_com_rescale * at.sqrt(), self.image_torch)
        self.loss2 = self.kl_loss(self.et)
        self.image_com_rescale = torch.reshape(self.image_com_rescale, (self.image_size,self.image_size, self.image_size, self.channel)).permute(3,0,1,2).unsqueeze(0)
        self.loss3 = self.tv_loss(self.image_com_rescale)
        self.total_loss = self.loss1 + self.beta * self.loss3
        self.total_loss.backward(retain_graph=True)
        self.res = np.sqrt(np.sum(np.square(self.out_avg - self.previous)) / np.sum(np.square(self.previous)))
        self.previous = self.out_avg

    def _obtain_current_result(self, step):
        """
        Obtain current results and update the best result

        Args:
            step (int): Current iteration step
        """
        self.psnr = quality(self.image_clean, np.clip(self.image_com_np.astype(np.float64),0,1))
        self.psnr_av = quality(self.image_clean, np.clip(self.out_avg.astype(np.float64),0,1))
        self.current_result = Result(recon=np.clip(self.image_com_np,0,1),  psnr=self.psnr, step=step)
        self.current_result_av = Result(recon=np.clip(self.out_avg,0,1),  psnr=self.psnr_av, step=step)
        if self.best_result is None or self.best_result.psnr < self.current_result.psnr:
            self.best_result = self.current_result
        if self.best_result_av is None or self.best_result_av.psnr < self.current_result_av.psnr:
            self.best_result_av = self.current_result_av

    def _plot_closure(self, step, logger):
        """
        Log status of the optimization process

        Args:
            step (int): Current iteration step.
            logger (logging.Logger): Logger to record the process.
        """
        logger.info('--------->Iteration %05d  kl_loss %f tol_loss %f   current_psnr: %f  max_psnr %f  current_psnr_av: %f max_psnr_av: %f mean: %f var: %f res: %f  step: %f'   % (step, self.loss2.item(), self.total_loss.item(),
                                                                                self.current_result.psnr, self.best_result.psnr,
                                                                                self.current_result_av.psnr, self.best_result_av.psnr, self.mean, self.var, self.res, self.current_result_av.step ))



