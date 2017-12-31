from PIL import Image

import os
import argparse


parser = argparse.ArgumentParser(description='Feature extractor for image literals')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')

args = parser.parse_args()

directory = os.fsencode(args.data)

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    filename = '{}/{}'.format(args.data, filename)

    try:
        im = Image.open(filename)
        im.verify()
    except Exception as e:
        print(e)
        os.remove(filename)
