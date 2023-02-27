from diffusion.trainer import TrainPairedConcat

trainer = TrainPairedConcat(
    ("./data/anime_sketch_color/train/color", "./data/anime_sketch_color/train/sketch"),
    ("./data/anime_sketch_color/val/color", "./data/anime_sketch_color/val/sketch"),
    {
        'in_channels': 6,
        'out_channels': 6
    },
    batch_size=8,
    val_batch_size=4,
    image_size=256,
    lr=1e-4,
    repeat_noise = 2,
    sample_eta=0.1,
    loss_scales=1,
    validate_every=200
).train()