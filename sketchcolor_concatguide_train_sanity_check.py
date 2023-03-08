from diffusion.trainer import Trainer
from data.dataset import PairedDataset
import os
from pathlib import Path

def main():
    root = Path(os.getcwd())

    train_data = PairedDataset(
            (root / "data/anime_sketch_color/train/color", root / "data/anime_sketch_color/train/sketch"), 
            channels=(3, 1)
        )
    valid_data = PairedDataset(
            (root / "data/anime_sketch_color/val/color", root / "data/anime_sketch_color/val/sketch"), 
            channels=(3, 1)
        )

    trainer_config = {
        'name': 'anime_sketch_lines_guided_by_concat',
        'batch_size': 8, 
        'val_batch_size': 8,
        'total_steps': 40,
        'checkpoint_validate_every': 20,
        'net_config': {
            'in_channels': 4,
            'out_channels': 4,
            'channels': (128, 256, 512, 1024, 1024),
            'num_res_blocks': (1, 2, 3, 4, 2)
        },
        'vdiffusion_config': {
            'sample_steps': 50,
            'sample_intermediates_every': 10,
            'xpred_to_x0_channels': (3),
            'clip_x0_pred': True
        },
        'lr_warmup_steps': 3
    }

    trainer = Trainer(trainer_config)
    trainer.fit(train_data, valid_data)

if __name__ == '__main__':
    main()
