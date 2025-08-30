#!/bin/bash

python extract_img_feature_encoder.py --device cuda:0 --dataset rad  --image_dir /home/mixlab/tabular/medthink/Medthink/Medthink_Dataset/R-RAD/images/ --output_dir  /home/mixlab/tabular/medthink/data/R-RAD/
# python extract_img_feature.py --device cuda:0 --image_dir data/R-SLAKE/img/ --output_dir  /home/mixlab/tabular/medthink/data/R-SLAKE/

