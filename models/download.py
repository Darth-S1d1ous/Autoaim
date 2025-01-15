#!/usr/bin/python3

import argparse
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description='Download YOLO model')
    parser.add_argument('--model', type=str, required=True, help='the name of model to be downloaded')
    parser.add_argument('--output', type=str, default='.', help='Output directory for the downloaded model')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    model_name = args.model
    output_dir = args.output

    model = YOLO(model_name)