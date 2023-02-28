from train.trainer import SimpleTrainer
from data.dataset import PairedDataset


args = {
    'train_dataset': PairedDataset(
        ("./data/anime_sketch_color/train/color", "./data/anime_sketch_color/train/sketch"), 
        channels=(3, 1)),
    'valid_dataset': PairedDataset(
    ("./data/anime_sketch_color/val/color", "./data/anime_sketch_color/val/sketch"), 
    channels=(3, 1)),
    'net_config': {
        'in_channels': 4,
        'out_channels': 4,
        'emb_dim': 512
    },
    'optimizer_config': {
        'lr': 0,
        'betas': (0.9, 0.99)
    },
    'lr_scheduler_config': {
        'lr_start': 0.0, 
        'lr_peak': 1e-4, 
        'lr_end': 1e-6,
        'warmup_steps': 5000,
        'total_steps': 1000000
    },
    'batch_size': 8,
    'val_batch_size': 4,
    'max_train_steps': 1000000,
    'sample_eta': 0.1,
    'loss_scales': 1,
    'checkpoint_every': 20,
    'sample_steps': 1000,
    'normalize_x': True,
    'x0_scale': 0.5,
    'out_dir_name': 'train_sketch_color1_history',
    'dynamic_thresholding_quantile': 0.995,
    'sample_freeze_dims': (3)
}


trainer = SimpleTrainer(**args).train()