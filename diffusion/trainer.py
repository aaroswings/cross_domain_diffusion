import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_value_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import os
from pathlib import Path
from dataclasses import dataclass, asdict

import json

from diffusion.vdiffusion import VDiffusionConfig
from diffusion.learning_rate import LinearLR
from diffusion.util import save_samples
from networks.unet import UNetConfig
from networks.ema import EMA
from typing import Tuple
from tqdm import tqdm

@dataclass
class TrainerConfig:
    name: str
    net_config: dict
    vdiffusion_config: dict
    batch_size: int
    val_batch_size: int
    total_steps: int = 1000000
    checkpoint_validate_every: int = 10000
    adam_betas: Tuple[float, float] = (0.9, 0.99)
    lr_peak: float = 5e-4
    lr_end: float = 1e-5
    lr_warmup_steps: int = 10000
    ema_beta: float = 0.999
    ema_update_after_step: int = 999
    ema_update_every: int = 10

class Trainer(nn.Module):
    def __init__(
        self,
        config: dict,
        version: int = None
    ):
        super().__init__()
        self._init_config = TrainerConfig(**config)
        for k, v in asdict(self._init_config).items():
            self.__setattr__(k, v)
        self._net_config = UNetConfig(**self.net_config)
        self._vdiffusion_config = VDiffusionConfig(**self.vdiffusion_config)

        self.version = version
        self.construct_state_objs() # sets self.net, self.vdiffusion, self.optimizer, self.lr_scheduler, self.ema
        self.restore_or_create_root() # sets self.out_dir, self.global_step
        self.writer = SummaryWriter(self.out_dir)
        self.scaler = GradScaler()

    def construct_state_objs(self):
        if not hasattr(self, 'net') or self.net is None:
            self.net = self._net_config.build().cuda()

        if not hasattr(self, 'vdiffusion') or self.vdiffusion is None:
            self.vdiffusion = self._vdiffusion_config.build()

        if not hasattr(self, 'optimizer') or self.optimizer is None:
            self.optimizer = optim.Adam([*self.net.parameters()], lr=0.0, betas=self.adam_betas, eps=1e-6)

        if not hasattr(self, 'lr_scheduler') or self.lr_scheduler is None:
            self.lr_scheduler = LinearLR(
                self.optimizer,
                0.0, 
                self.lr_peak, 
                self.lr_end, 
                self.lr_warmup_steps, 
                self.total_steps)
            
        if not hasattr(self, 'ema') or self.ema is None:
            self.ema = EMA(self.net, None, self.ema_beta, self.ema_update_after_step, self.ema_update_every)

    def destruct_state_objs(self):
        del self.net
        del self.vdiffusion
        del self.optimizer
        del self.lr_scheduler
        del self.ema

        self.net = None
        self.vdiffusion = None
        self.optimizer = None
        self.lr_scheduler = None
        self.ema = None

    def restore_or_create_root(self) -> Path:
        """
        tries to create root if self.version is None, or load it otherwise
        out_dir is ./train_run/{name}/version_{self.version}
        the checkpoint at global_step i is saved at /train_run/version_{self.version}/checkpoint_{i}.ckpt
        """
        train_run_dir = Path('train_run') / self.name
        os.makedirs(train_run_dir, exist_ok=True)
        if self.version is None:
            self.global_step = 0
            highest_version = 1
            for p in train_run_dir.iterdir():
                if p.is_dir():
                    try:
                        ns = p.name.split('version_')
                        if len(ns) == 2:
                            ver = int(ns[1])
                        if ver >= highest_version:
                            highest_version = ver + 1
                    except ValueError:
                        pass
            self.out_dir = train_run_dir / f'version_{highest_version}'
            self.version = highest_version
            os.makedirs(self.out_dir)
            config_dict = {
                **asdict(self._init_config),
                **{'net_config': asdict(self._net_config),
                 'vdiffusion_config': asdict(self._vdiffusion_config)
                }
            }
            with open(self.out_dir / 'config.json', 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, ensure_ascii=False, indent=4)
        else:
            self.out_dir = train_run_dir / f'version_{self.version}'
            if self.out_dir.is_dir():
                step = self.load_latest()
                self.global_step = step
            else:
                raise ValueError("A training version was given to load from, but the directory ")

    def save(self):
        torch.save({
            'net_state': self.net.state_dict(),
            'opt_state': self.optimizer.state_dict(),
            'ema_state': self.ema.state_dict()
        },
        self.out_dir / f'checkpoint_{self.global_step}.ckpt'
        )

    def load_latest(self):
        max_step = 0
        for p in self.out_dir.iterdir():
            if p.is_file() and p.stem.startswith('checkpoint_'):
                step = int(p.stem.split('checkpoint_')[1])
                max_step = step if max_step is None or step > max_step else max_step
        checkpoint = torch.load(self.out_dir / f'checkpoint_{max_step}.ckpt')
        self.global_step = max_step
        self.net.load_state_dict(checkpoint['net_state'])
        self.optimizer.load_state_dict(checkpoint['opt_state'])
        self.ema.load_state_dict(checkpoint['ema_state'])
        return max_step

    def fit(self, train_dataset, val_dataset):
        train_dataloader = DataLoader(train_dataset, self.batch_size, shuffle=True, num_workers=8)
        val_dataloader = DataLoader(val_dataset, self.batch_size)

        # do training
        for _ in range(self.global_step, self.total_steps, self.checkpoint_validate_every):
            self.train_steps(self.checkpoint_validate_every, train_dataloader)
            self.validation_step(val_dataloader)
            self.save()

    def __interruptible(func):
        def wrap(self, *args, **kwargs):
            while True:
                try:
                    return func(self, *args, **kwargs)
                except KeyboardInterrupt:
                    print('Function interrupted, clearning memory.')
                    self.destruct_state_objs()
                    self.save()
                    entry = input('Reload and continue? y=continue, n=quit:\t')
                    if entry == 'y':
                        step = self.load_latest()
                        print(f'Reloaded from step {step}')
                    else:
                        exit(1)
        return wrap

    @__interruptible
    def train_steps(self, steps, train_dataloader):
        print(f'Training for {steps} steps.')
        data_iter = iter(train_dataloader)
        for i in tqdm(range(steps)):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_dataloader)
                batch = next(data_iter)

            self.forward_backward(batch)

    @__interruptible
    @torch.no_grad() 
    def sample(self, batch, guided = False, subfolder = None):
        save_dir = self.out_dir / (f'samples_step_{self.global_step}_' + ('guided' if guided else 'unguided'))
        save_dir = save_dir / subfolder if subfolder else save_dir
        samples = self.vdiffusion.sample(self.ema.ema_model, batch, guided)
        save_samples(samples, save_dir, self.vdiffusion.channel_split)
        

    @__interruptible
    @torch.no_grad()        
    def validation_step(self, val_dataloader):
        batch = next(iter(val_dataloader))
        self.sample(batch, guided = True)
        self.sample(batch, guided = False)


    def forward_backward(self, batch):
        self.lr_scheduler.update(self.global_step)
        self.optimizer.zero_grad()
        loss = self.vdiffusion.loss(self.net, batch)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        clip_grad_value_(self.net.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.writer.add_scalar('Loss/train', loss, self.global_step)
        self.ema.update()
        self.global_step += 1

        if torch.isnan(loss):
            raise ValueError('Encountered NaN loss, stopping training.')