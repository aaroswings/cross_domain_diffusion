from data.dataset import resize_reformat_files

resize_reformat_files(
    "data/anime_sketch_color/train/color",
    "data/anime_sketch_color/train/color_256", 
    256)

resize_reformat_files(
    "data/anime_sketch_color/train/sketch",
    "data/anime_sketch_color/train/sketch_256",
    256
)

resize_reformat_files(
    "data/anime_sketch_color/val/color",
    "data/anime_sketch_color/val/color_256",
    256
)

resize_reformat_files(
    "data/anime_sketch_color/val/sketch",
     "data/anime_sketch_color/val/sketch_256",
     256
)