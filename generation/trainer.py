import math
import logging
import models
import os
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from utils.core import imresize, timeseries_resize
from torch.optim.lr_scheduler import StepLR
from torch.autograd import grad as torch_grad, Variable
from data import get_loader
from utils.recorderx import RecoderX
from utils.misc import mkdir

class Trainer():
    def __init__(self, args):
        # parameters
        self.args = args
        self.print_model = True
        self.invalidity_margins = None
        self.init_generator = True #initially True and later set to false in the _init_models function once generator is initialised
        self.parallel = False
        
        if self.args.use_tb:
            self.tb = RecoderX(log_dir=args.save_path)

    def _init_models(self, loader):
        # number of features

        # initialize discriminator model
        
        d_model_config = {'input_size': self.args.input_size,
                          'hidden_size': self.args.hidden_size, 
                          'num_layers': self.args.num_layers, 
                          'kernel_size': self.args.kernel_size,
                          'dropout': self.args.dropout,
                          'normalization': self.args.normalization}
        
        g_model_config  = {'input_size': self.args.input_size,
                          'hidden_size': self.args.hidden_size, 
                          'num_layers': self.args.num_layers,
                          'kernel_size': self.args.kernel_size,
                          'dropout': self.args.dropout}
        
        d_model = models.__dict__[self.args.dis_model]
        self.d_model = d_model(**d_model_config)
        self.d_model = self.d_model.to(self.args.device)

        # parallel
        if self.args.device_ids and len(self.args.device_ids) > 1:
            self.d_model = torch.nn.DataParallel(self.d_model, self.args.device_ids)
            self.d_model = self.d_model.module if isinstance(self.d_model, torch.nn.DataParallel) else self.d_model
            self.parallel = True
            print("Using multiple GPUs for training...", flush=True)

        # init generator
        # this is only for s0- the coarsest scale
        if self.init_generator:
            g_model = models.__dict__[self.args.gen_model]
            self.g_model = g_model(**g_model_config)
            self.g_model = self.g_model.to(self.args.device)
            if self.args.device_ids and len(self.args.device_ids) > 1:
                self.g_model = torch.nn.DataParallel(self.g_model, self.args.device_ids)

            self.g_model = self.g_model.module if isinstance(self.g_model, torch.nn.DataParallel) else self.g_model

            self.g_model.scale_factor = self.args.scale_factor #why does generator have a scale factor?
            self.init_generator = False
            loader.dataset.amps = {'s0': torch.tensor(1.).to(self.args.device)}
        else:
            # add amp
            """
            amps correspond to: Specifically, we take sigma_n to be proportional to the root mean squared error (RMSE) be-
            tween x_recon_n+1 ↑r and xn, which gives an indication of the amount of details that need to be added at that scale.
            amps are the amplitude values that help in scaling the noise inputs for different scales
            """
            data = next(iter(loader))
            amps = data['amps']
            reals = data['reals']
            noises = data['noises']
            keys = list(reals.keys())
            next_key = keys[keys.index(self.key) + 1]
            z = self.g_model(reals, amps, noises)
            #z is the generated image (b,c,h,w)
            z = timeseries_resize(z.detach(), 1. / self.g_model.scale_factor)
            z = z[:, 0:reals[next_key].size(1),:] #ensures that the size of z is the same as the size of the real timeseries at the next scale

            a = self.args.noise_weight * torch.sqrt(F.mse_loss(z, reals[next_key]))
            loader.dataset.amps.update({next_key: a.to(self.args.device)})

            # add scale
            self.g_model.add_scale(self.args.device)

        # print model
        if self.print_model:
            logging.info(self.g_model)
            logging.info(self.d_model)
            logging.info('Number of parameters in generator: {}'.format(sum([l.nelement() for l in self.g_model.parameters()])))
            logging.info('Number of parameters in discriminator: {}'.format(sum([l.nelement() for l in self.d_model.parameters()])))
            self.print_model = False

        # training mode
        self.g_model.train()
        self.d_model.train()
    
    def _init_eval(self, loader):
        # paramaters 
        self.scale = 0

        # config
        g_model_config  = {'input_size': self.args.input_size,
                          'hidden_size': self.args.hidden_size, 
                          'num_layers': self.args.num_layers, 
                          'dropout': self.args.dropout}
        # init first scale
        g_model = models.__dict__[self.args.gen_model]
        self.g_model = g_model(**g_model_config)
        self.g_model.scale_factor = self.args.scale_factor

        # add scales
        for self.scale in range(1, self.args.stop_scale + 1):
            self.g_model.add_scale('cpu')

        # #printing the parameters expected by the model
        # print("Parameters expected by the model:")
        # for name, param in self.g_model.named_parameters():
        #     print(f"{name}: {param.shape}",flush=True)

        # #printing the parameters in the loaded state_dict
        # loaded_state_dict = torch.load(self.args.model_to_load, map_location='cpu')
        # print("\nParameters in loaded state_dict:")
        # for name, param in loaded_state_dict.items():
        #     print(f"{name}: {param.shape}", flush=True)

        
        # load model
        logging.info('Loading model...')
        self.g_model.load_state_dict(torch.load(self.args.model_to_load, map_location='cpu'))
        loader.dataset.amps = torch.load(self.args.amps_to_load, map_location='cpu')

        # cuda
        self.g_model = self.g_model.to(self.args.device)
        for key in loader.dataset.amps.keys():
            loader.dataset.amps[key] = loader.dataset.amps[key].to(self.args.device)

        # print 
        logging.info(self.g_model)
        logging.info(f'Number of parameters in generator: {sum([l.nelement() for l in self.g_model.parameters()])}')

        # key
        self.key = f's{self.args.stop_scale + 1}'

        return loader

    def _init_optim(self):
        # initialize optimizer
        self.g_optimizer = torch.optim.Adam(self.g_model.curr.parameters(), lr=self.args.lr, betas=self.args.gen_betas)
        self.d_optimizer = torch.optim.Adam(self.d_model.parameters(), lr=self.args.lr, betas=self.args.dis_betas)

        # initialize scheduler
        self.g_scheduler = StepLR(self.g_optimizer, step_size=self.args.step_size, gamma=self.args.gamma)
        self.d_scheduler = StepLR(self.d_optimizer, step_size=self.args.step_size, gamma=self.args.gamma)

        # criterion
        self.reconstruction = torch.nn.MSELoss()

    def _init_global(self, loader):
        # adjust scales
        real = loader.dataset.data.clone().to(self.args.device)
        #print("shape of real input is : ", real.shape)
        self._adjust_scales(real)

        # set reals
        real = timeseries_resize(real, self.args.scale_one)
        #after resizing
        #print("shape of real input after resizing is : ", real.shape)

        loader.dataset.reals = self._set_reals(real)
        loader.dataset.noises = self._set_noises(loader.dataset.reals)

    def _init_local(self, loader):
        """
        This initializes the models, optimizers for each scale
        It stores the current scale in self.key
        Since the scales are trained one at a time, the losses are stored in a dictionary with the key as the scale
        """    
        # initialize models
        self._init_models(loader)

        # initialize optimization
        self._init_optim()

        # parameters
        self.losses = {'D': [], 'D_r': [], 'D_gp': [], 'D_f': [], 'G': [], 'G_recon': [], 'G_adv': [], 'G_euclidean': []}
        self.key = f's{self.scale}'
    
    
    #updated
    def _adjust_scales(self, time_series):
        """
        scale_one is the initial downscaling(if needed) to bring the input image to the max size of the pyramid
        num_scales is the total number of scales. Including the coarsest to the finest

        stop scale is the index of the finest scale where the training goes
        """
        T = time_series.size(1)
        self.args.num_scales = math.ceil(math.log(self.args.min_size / T, self.args.scale_factor_init)) + 1
        self.args.scale_to_stop = math.ceil(math.log(min([self.args.max_size, T]) / T, self.args.scale_factor_init))
        self.args.stop_scale = self.args.num_scales - self.args.scale_to_stop
        self.args.scale_one = min(self.args.max_size / T, 1)
        time_series_resized = timeseries_resize(time_series, self.args.scale_one)
        #now, our time_series is resized
        T_resized = time_series_resized.size(1)
        self.args.scale_factor = math.pow(self.args.min_size / T_resized, 1 / (self.args.stop_scale))
        self.args.scale_to_stop = math.ceil(math.log(min([self.args.max_size, T_resized]) / T_resized, self.args.scale_factor_init))
        self.args.stop_scale = self.args.num_scales - self.args.scale_to_stop

        # self.args.num_scales = math.ceil(math.log(math.pow(self.args.min_size / (min(image.size(2), image.size(3))), 1), self.args.scale_factor_init)) + 1
        # self.args.scale_to_stop = math.ceil(math.log(min([self.args.max_size, max([image.size(2), image.size(3)])]) / max([image.size(2), image.size(3)]), self.args.scale_factor_init))
        # self.args.stop_scale = self.args.num_scales - self.args.scale_to_stop

        # self.args.scale_one = min(self.args.max_size / max([image.size(2), image.size(3)]), 1)
        # image_resized = imresize(image, self.args.scale_one)

        # #now the image is resized
        # self.args.scale_factor = math.pow(self.args.min_size/(min(image_resized.size(2), image_resized.size(3))), 1 / (self.args.stop_scale))
        # self.args.scale_to_stop = math.ceil(math.log(min([self.args.max_size, max([image_resized.size(2), image_resized.size(3)])]) / max([image_resized.size(2), image_resized.size(3)]), self.args.scale_factor_init))
        # self.args.stop_scale = self.args.num_scales - self.args.scale_to_stop

    #updated
    def _set_reals(self, real):
        reals = {}

        # loop over scales
        for i in range(self.args.stop_scale + 1):
            s = math.pow(self.args.scale_factor, self.args.stop_scale - i)
            reals.update({'s{}'.format(i): timeseries_resize(real.clone().detach(), s).squeeze(dim=0)}) #why are they squeezing the tensor?

        return reals

    def _set_noises(self, reals):
        noises = {}

        # loop over scales
        for key in reals.keys():
            #s0 is the coarsest scale
            noises.update({key: self._generate_noise(reals[key].unsqueeze(dim=0), repeat=(key == 's0')).squeeze(dim=0)})
        
        return noises

    def _generate_noise(self, tensor_like: torch.Tensor, repeat: bool = False) -> torch.Tensor:
       # input tensor is [batch, timesteps, features]
        if not repeat:
            noise = torch.randn(tensor_like.size(), device=tensor_like.device)
        else:
            # For the coarsest scale, create noise with a single (timestep) and repeat it across features.
            batch, timesteps, features = tensor_like.size()
            noise = torch.randn((batch, 1, features), device=tensor_like.device)
            noise = noise.repeat(1, timesteps, 1)
        return noise

    def _save_models(self):
        # save models
        save_dir = os.path.join(self.args.save_path, self.key)
        g_model_path = os.path.join(save_dir, f"{self.args.gen_model}_s{self.step}.pth")
        d_model_path = os.path.join(save_dir, f"{self.args.dis_model}_s{self.step}.pth")
        
        torch.save(self.g_model.state_dict(), g_model_path)
        torch.save(self.d_model.state_dict(), d_model_path)

    def _save_last(self, amps):
        # print parameters stored in the saved state_dict
        state_dict = self.g_model.state_dict()
        print("Parameters in state_dict being saved:")
        for name, param in state_dict.items():
            print(f"{name}: {param.shape}", flush=True)

        # save models
        torch.save(self.g_model.state_dict(), os.path.join(self.args.save_path, f'{self.args.gen_model}.pth'))
        torch.save(self.d_model.state_dict(), os.path.join(self.args.save_path, f'{self.args.dis_model}.pth'))

        # save amps
        torch.save(amps, os.path.join(self.args.save_path, 'amps.pth'))

    def _set_require_grads(self, model, require_grad):
        for p in model.parameters():
            p.requires_grad_(require_grad)

    def _critic_wgan_iteration(self, reals, amps):
        """
        updates discriminator using WGAN loss
        also uses the gradient penalty as used in WGAN
        """
        # require grads
        self._set_require_grads(self.d_model, True)

        # get generated data
        generated_data = self.g_model(reals, amps)

        # zero grads
        self.d_optimizer.zero_grad()

        # calculate probabilities on real and generated data
        d_real = self.d_model(reals[self.key])
        d_generated = self.d_model(generated_data.detach()) #detached so that the gradients are not backpropagated to generator

        # create total loss and optimize
        loss_r = -d_real.mean()
        loss_f = d_generated.mean()
        loss = loss_f + loss_r

        # get gradient penalty
        if self.args.penalty_weight:
            gradient_penalty = self._gradient_penalty(reals[self.key], generated_data)
            loss += gradient_penalty * self.args.penalty_weight

        loss.backward()

        self.d_optimizer.step()

        # record loss
        self.losses['D'].append(loss.data.item())
        self.losses['D_r'].append(loss_r.data.item())
        self.losses['D_f'].append(loss_f.data.item())
        if self.args.penalty_weight:
            self.losses['D_gp'].append(gradient_penalty.data.item())

        # require grads
        self._set_require_grads(self.d_model, False)

        return generated_data

    def _gradient_penalty(self, real_data, generated_data):
        # calculate interpolation
        alpha = torch.rand(real_data.size(0), 1, 1)
        alpha = alpha.expand_as(real_data).to(self.args.device)
        interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
        
        #interpolated = Variable(interpolated, requires_grad=True)
        interpolated.requires_grad = True
        interpolated = interpolated.to(self.args.device)

        # Disable cuDNN for the forward pass that computes the critic's output.
        with torch.backends.cudnn.flags(enabled=False):
            prob_interpolated = self.d_model(interpolated)
        
        # # calculate probability of interpolated examples
        # prob_interpolated = self.d_model(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).to(self.args.device),
                               create_graph=True, retain_graph=True, only_inputs=True)[0]
        
        gradients = gradients.reshape(gradients.size(0), -1)
        gradient_norm = gradients.norm(2, dim=1)
        return ((gradient_norm - 1) ** 2).mean()
    
    def _penalty_distance_loss(self, generated_data, real_data, penalty_distance):
        # Extracting coordinates for the two balls.
        ball1 = generated_data[:,:,:3]  # Shape: [batch, timestamps, 3]
        ball2 = generated_data[:,:,3:6]   # Shape: [batch, timestamps, 3]
        distance = torch.norm(ball1 - ball2, dim=-1)  # Shape: [batch, timestamps]
        loss = F.mse_loss(distance, torch.full_like(distance, penalty_distance))
        return loss

    def _generator_iteration(self, noises, reals, amps, generated_data_adv):
        # zero grads
        self.g_optimizer.zero_grad()

        # get generated data
        generated_data_rec = self.g_model(reals, amps, noises) # reals, amps, noises
        loss = 0.

        # reconstruction loss
        if self.args.reconstruction_weight:
            loss_recon = self.reconstruction(generated_data_rec, reals[self.key])
            loss += loss_recon * self.args.reconstruction_weight
            self.losses['G_recon'].append(loss_recon.data.item())
        
        if self.args.euclidean_weight:
            loss_euclidean = self._penalty_distance_loss(generated_data_rec, reals[self.key], self.args.penalty_distance)
            loss += loss_euclidean * self.args.euclidean_weight
            self.losses['G_euclidean'].append(loss_euclidean.data.item())
        
        # adversarial loss
        if self.args.adversarial_weight:
            d_generated = self.d_model(generated_data_adv)
            loss_adv = -d_generated.mean()
            loss += loss_adv * self.args.adversarial_weight
            self.losses['G_adv'].append(loss_adv.data.item())

        # backward loss
        loss.backward()
        self.g_optimizer.step()

        # record loss
        self.losses['G'].append(loss.data.item())

    def _train_iteration(self, loader):
        # set inputs
        data = next(iter(loader))
        noises = data['noises']
        reals = data['reals']
        amps = data['amps']
        
        # critic iteration
        fakes = self._critic_wgan_iteration(reals, amps)

        # only update generator every |critic_iterations| iterations
        if self.step % self.args.num_critic == 0:
            self._generator_iteration(noises, reals, amps, fakes)

        # logging
        if self.step % self.args.print_every == 0:
            line2print = 'Iteration {}'.format(self.step)
            line2print += ', D: {:.6f}, D_r: {:.6f}, D_f: {:.6f}'.format(self.losses['D'][-1], self.losses['D_r'][-1], self.losses['D_f'][-1])
            line2print += ', D_gp: {:.6f}'.format(self.losses['D_gp'][-1])
            line2print += ', G: {:.5f}, G_recon: {:.5f}, G_adv: {:.5f}'.format(self.losses['G'][-1], self.losses['G_recon'][-1], self.losses['G_adv'][-1])
            line2print += ', G_euclidean: {:.5f}'.format(self.losses['G_euclidean'][-1])
            logging.info(line2print)

        # plots for tensorboard
        if self.args.use_tb:
            if self.args.adversarial_weight:
                self.tb.add_scalar(f'data/s{self.scale}/loss_d', self.losses['D'][-1], self.step)
            if self.step > self.args.num_critic:
                self.tb.add_scalar(f'data/s{self.scale}/loss_g', self.losses['G'][-1], self.step)

    def _eval_iteration(self, loader):
        # set inputs
        data = next(iter(loader))
        noises = data['noises']
        reals = data['reals']
        amps = data['amps']

        # evaluation
        with torch.no_grad():
            generated_fixed = self.g_model(reals, amps, noises)
            generated_sampled = self.g_model(reals, amps)

        # save image
        self._save_timeseries(generated_fixed, f's{self.step}_fixed.csv')
        self._save_timeseries(generated_sampled, f's{self.step}_sampled.csv')

    def _sample_iteration(self, loader):
        # set inputs
        data_reals = loader.dataset.reals
        reals = {}
        amps = loader.dataset.amps

        # set reals
        for key in data_reals.keys():
           reals.update({key: data_reals[key].clone().unsqueeze(dim=0).repeat(self.args.batch_size, 1, 1)}) 

        # evaluation
        with torch.no_grad():
            generated_sampled = self.g_model(reals, amps)

        # save image
        if(self.args.save_name is not None):
            self._save_timeseries(generated_sampled, f'{self.args.save_name}')
        else:
            self._save_timeseries(generated_sampled, f's{self.step}_sampled')

    
    def _save_timeseries(self, timeseries, file_name):
        """
        Save a timeseries tensor to CSV file(s).
        timeseries : A tensor of shape [batch, timesteps, features].
        file_name : Base name for the CSV file(s).

        Behavior:
            - If the batch size is 1, save a single CSV file with timesteps rows and features columns.
            - If batch size > 1, create a subfolder (named using file_name) and save each batch sample 
            as a separate CSV file named: file_name_batch_0.csv, file_name_batch_1.csv, etc.
        """
        directory = os.path.join(self.args.save_path, self.key)
        mkdir(directory)
        ts_np = timeseries.data.cpu().numpy()
        if ts_np.shape[0] == 1:
            save_path = os.path.join(directory, file_name)
            np.savetxt(save_path, ts_np[0], delimiter=",")
        else:
            # More than one sample; create a subfolder.
            subfolder = os.path.join(directory, file_name)
            mkdir(subfolder)
            # Save each batch sample as its own CSV file.
            for i in range(ts_np.shape[0]):
                batch_file = f"{file_name}_batch_{i}.csv"
                batch_save_path = os.path.join(subfolder, batch_file)
                np.savetxt(batch_save_path, ts_np[i], delimiter=",")


    def check_model_device_consistency(self, model):
        devices = set()
        
        # Iterate through all parameters
        for name, param in model.named_parameters():
            device = param.device
            devices.add(device)
            #print(f"Parameter: {name}, Device: {device}")
        
        # check for mismatches
        if len(devices) > 1:
            print("Device mismatch detected!", flush=True)
            for name, param in model.named_parameters():
                print(f"Parameter: {name}, Device: {param.device}", flush=True)
        else:
            print(f"All parameters are on the same device: {devices.pop()}", flush=True)

    def _train_single_scale(self, loader):
        # run step iterations
        logging.info('\nScale #{}'.format(self.scale + 1))
        self.check_model_device_consistency(self.g_model)
        self.check_model_device_consistency(self.d_model)
        
        for self.step in range(self.args.num_steps + 1):
            # train
            self._train_iteration(loader)
            # scheduler
            self.g_scheduler.step()
            self.d_scheduler.step()

            # evaluation
            if (self.step % self.args.eval_every == 0) or (self.step == self.args.num_steps):
                # eval
                self.g_model.eval()
                self._eval_iteration(loader)
                self.g_model.train()

        # sample last
        self.step += 1
        self._sample_iteration(loader)

    def _print_stats(self, loader):
        reals = loader.dataset.reals
        amps = loader.dataset.amps

        logging.info('\nScales:')
        for key in reals.keys():
            logging.info('{}, size: {}x{}, amp: {:.3f}'.format(key, reals[key].size(-2), reals[key].size(-1), amps[key]))

    def train(self):
        # get loader
        loader = get_loader(self.args)
        print("done with loader. data has shape : ", loader.dataset.data.shape, flush=True)
        print("about to initialize global", flush=True)
        # initialize global
        self._init_global(loader)
        print("done with initialising global", flush=True)
        #printing the shapes of the real and noisy outputs.
        for key in loader.dataset.reals.keys():
            print(f"key is {key} with real {loader.dataset.reals[key].shape} on device {loader.dataset.reals[key].device} and with noise {loader.dataset.noises[key].shape} on device {loader.dataset.noises[key].device}", flush=True)
        #     real = loader.dataset.reals[key].cpu().numpy()
        #     noise = loader.dataset.noises[key].cpu().numpy()

        #     real_path = os.path.join("realAirSignsLayers", f"{key}_real.csv")
        #     noise_path = os.path.join("realAirSignsLayers", f"{key}_noise.csv")

        #     pd.DataFrame(real).to_csv(real_path, index=False, float_format="%.18e")
        #     pd.DataFrame(noise).to_csv(noise_path, index=False, float_format="%.18e")

        #     print(f"Saved {key}_real.csv and {key}_noise.csv with shapes {real.shape} and {noise.shape}", flush=True)
        
        # exit(0)
        # iterate scales
        for self.scale in range(self.args.stop_scale + 1):
            # initialize local
            self._init_local(loader)
            self._train_single_scale(loader)
            self._save_models()

        # save last
        self._save_last(loader.dataset.amps)

        # print stats
        self._print_stats(loader)

        print("Training Done!", flush=True)

        # close tensorboard
        if self.args.use_tb:
            self.tb.close()

    def eval(self):
        # get loader
        loader = get_loader(self.args)

        # init
        self._init_global(loader)
        loader = self._init_eval(loader)

        # evaluate
        logging.info('Evaluating...')
        #self.step = 0
        for self.step in range(self.args.num_steps):
            self._sample_iteration(loader)
        self._sample_iteration(loader)

        print("Evaluation Done!", flush=True)

        # close tensorboard
        if self.args.use_tb:
            self.tb.close()

    # def evalCreateSynthetic(self):
    #     # if this function is called, then root isnt actually a single file. 
    #     #but due to the design of the code structure, we first have to create a pseudo loader with self.args.root equal to the path of the first file in the folder
    #     # self.args.root = first file
    #     pseudo_loader = get_loader(self.args)
    #     self._init_global(pseudo_loader)
    #     pseudo_loader = self._init_eval(pseudo_loader)

    #     for(# loop through every file in the folder(whose path is in root))
    #         # set the root to the current file
    #         self.args.root = current_file_path
    #         self.args.save_name = current_file_name
    #         # get loader
    #         loader = get_loader(self.args)
    #         self._init_global(loader) # because we have to set unique noises every time
    #         loader.dataset.amps = pseudo_loader.dataset.amps


    #         for(# i in range 1 to num_synthetic_users+1)
    #             args.seed = i
    #             torch.manual_seed(args.seed)
    #             if 'cuda' in args.device and torch.cuda.is_available():
    #                 torch.cuda.manual_seed_all(args.seed)
    #             for self.step in range(self.args.num_steps):
    #                 self._sample_iteration(loader)
    #             self._sample_iteration(loader)

    #     print("Evaluation Done!", flush=True)

    #     # close tensorboard
    #     if self.args.use_tb:
    #         self.tb.close()


    def evalCreateSynthetic(self):
        # the original folder path is in self.args.root.
        original_root = self.args.root
        
        file_list = sorted(
            [fname for fname in os.listdir(original_root) if os.path.isfile(os.path.join(original_root, fname))]
        )
        if not file_list:
            print(f"No files found in directory: {original_root}")
            return

        # using the first file in the directory as a pseudo-loader.
        # I am doing this so that we dont have to keep loading the model for every file in the folder and for every synthetic user per file
        first_file_path = os.path.join(original_root, file_list[0])
        self.args.root = first_file_path
        pseudo_loader = get_loader(self.args)
        self._init_global(pseudo_loader)
        pseudo_loader = self._init_eval(pseudo_loader)
        # for key in pseudo_loader.dataset.amps:
        #     pseudo_loader.dataset.amps[key] *= 5
            
        # looping through every file in the original folder.
        for filename in file_list:
            current_file_path = os.path.join(original_root, filename)
            self.args.root = current_file_path  # update the root to the current file
            
            # creating a loader for the current file.
            loader = get_loader(self.args)
            self._init_global(loader) 
            # ensuring amplitude settings from the pseudo-loader are used. since it is the same for a base model
            loader.dataset.amps = pseudo_loader.dataset.amps
            
            # iterating for each synthetic user to be generated
            for i in range(1, self.args.num_synthetic_users + 1):
                self.args.seed = i
                self.args.save_name = f"N{i}_{filename}"      # update the save name to the current file name
            
                torch.manual_seed(self.args.seed)
                if 'cuda' in self.args.device and torch.cuda.is_available():
                    torch.cuda.manual_seed_all(self.args.seed)
                
                for step in range(self.args.num_steps):
                    self.step = step
                    self._sample_iteration(loader)
                self._sample_iteration(loader)
        
        print("Evaluation Done!", flush=True)
        
        if self.args.use_tb:
            self.tb.close()