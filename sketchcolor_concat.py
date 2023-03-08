from diffusion.trainer import Trainer
from data.dataset import PairedDataset
import os
from pathlib import Path

def main():
    root = Path(os.getcwd())

    train_data = PairedDataset(
            (root / "data/anime_sketch_color/train/color_256", root / "data/anime_sketch_color/train/sketch_256"), 
            channels=(3, 1)
        )
    valid_data = PairedDataset(
            (root / "data/anime_sketch_color/val/color_256", root / "data/anime_sketch_color/val/sketch_256"), 
            channels=(3, 1)
        )

    trainer_config = {
        'name': 'anime_sketch_lines_guided_by_concat',
        'batch_size': 7, 
        'val_batch_size': 8,
        'total_steps': 1000000,
        'checkpoint_validate_every': 10000,
        'net_config': {
            'in_channels': 4,
            'out_channels': 4,
            'channels': (128, 256, 512, 1024, 1024),
            'block_depth': (1, 2, 3, 6, 1),
            'conv2d_name': 'conv2d'
        },
        'vdiffusion_config': {
            'sample_steps': 1000,
            'sample_intermediates_every': 10,
            'xpred_to_x0_channels': (3),
            'clip_x0_pred': True
        },
        'lr_peak': 1e-3,
        'lr_warmup_steps': 10000,
        'ema_update_after_step': 5000
    }

    trainer = Trainer(trainer_config, version=47)
    trainer.fit(train_data, valid_data)

if __name__ == '__main__':
    main()
