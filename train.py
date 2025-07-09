#!/usr/bin/env python3
"""
Scripts to train a keras model using tensorflow.
Basic usage should feel familiar: train.py --tubs data/ --model models/mypilot.h5

Usage:
    train.py [--tubs=tubs] (--model=<model>)
    [--type=(linear|inferred|tensorrt_linear|tflite_linear)]
    [--comment=<comment>]

Options:
    -h --help              Show this screen.
"""

from docopt import docopt
import donkeycar as dk
from donkeycar.pipeline.training import train
# from donkeycar.pipeline.training import purge_bad_data


def main():
    args = docopt(__doc__)
    cfg = dk.load_config()
    tubs = args['--tubs']
    model = args['--model']
    model_type = args['--type']
    comment = args['--comment']
    
    # # 在训练前清理数据
    # if tubs:
    #     purge_bad_data(tub_path=tubs,
    #                    max_brightness=250,
    #                    min_brightness=20,
    #                    max_zero_angle_count=10)
    
    train(cfg, tubs, model, model_type, comment)


if __name__ == "__main__":
    main()


