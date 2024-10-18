import torch
import yaml
from dataset.mnist_dataset import MnistDataset
from models.unet_base import Unet
from scheduler.linear_noise_scheduler import LinearNoiseScheduler
from torch.utils.data import DataLoader
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer():
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            try:
                config = yaml.safe_load(file)
            except:
                print("Cannot open .yaml file !")

        self.diffusion_config = config['diffusion_config']
        self.dataset_config = config['dataset_config']
        self.model_config = config['model_config']
        self.train_config = config['train_config']

        self.data = MnistDataset('train', im_path=self.dataset_config['im_path'])
        self.mnist_loader = DataLoader(self.data, batch_size=self.train_config['batch_size'],
                                       shuffle=True, num_workers=4)
        self.num_train_batches = len(self.mnist_loader)

        self.model = Unet(self.model_config).to(device)
        self.model.trainer = self            #


        self.scheduler = LinearNoiseScheduler(num_timesteps=self.diffusion_config['num_timesteps'],
                                     beta_start=self.diffusion_config['beta_start'],
                                     beta_end=self.diffusion_config['beta_end'])

        self.losses = []
    
    def fit(self):

        # Create output directories
        if not os.path.exists(self.train_config['task_name']):
            os.mkdir(self.train_config['task_name'])

        # Load checkpoint if found
        if os.path.exists(os.path.join(self.train_config['task_name'], self.train_config['ckpt_name'])):
            print("Loading checkpoint as found one")
            self.model.load_state_dict(torch.load(os.path.join(self.train_config['task_name'], self.train_config['ckpt_name']),
                                             map_location=device))
        

        self.optim = self.model.configure_optimizers()
        self.epoch = 0
        self.train_batch_idx = 0
        self.val_batch_idx = 0
        for self.epoch in range(self.train_config['num_epoch']):
            self.fit_epoch()
            print('Finished epoch: {} | loss: {:4f}'.format(
                self.epoch + 1,
                np.mean(self.losses)
            ))
            torch.save(self.model.state_dict(), os.path.join(self.train_config['task_name'],
                                                    self.train_config['ckpt_name']))
        
        print('Done training ...')


    def fit_epoch(self):
        self.model.train()
        for batch in self.tqdm(self.mnist_loader):

            self.optim.zero_grad()
            batch = batch.float().to(device) 

            noise = torch.randn_like(batch).to(device)
            t = torch.randint(0, self.diffusion_config['num_timesteps'], (batch.shape[0], )).to(device)

            noisy_im = self.scheduler.add_noise(batch, noise, t)
            noise_pred = self.model(noisy_im, t)
            loss = self.model.training_step(noise_pred, noise )
            self.losses.append(loss.item())
            loss.backward()
            self.otim.step()



if __name__ == "__main__":
    trainer = Trainer(config_path="config/default.yaml")
    trainer.fit()