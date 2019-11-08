"""Script for running inference on a collection of new images.

NOTE: Server must be launched before running this script.
"""

import argparse
import csv
import glob
import os
import requests
import time


def main(args):
    t0 = time.time()

    # Get list of files to process
    assert os.path.exists(args.path_dir_jpeg)
    fns_jpg = [
        fn for fn in sorted( glob.glob(os.path.join(args.path_dir_jpeg, '*.jpg')) )
    ]

    # Initialize result
    queries = []
    for fn in fns_jpg:
        resp = requests.post(
            args.server_address,
            files={'file': open(fn, 'rb')}
        )
        resp_json = resp.json()
        queries.append([fn.split('/')[-1],
                        resp_json['class_name'],
                        resp_json['class_prob']])

    # Print performance information
    elapsed = time.time() - t0
    rate = (elapsed / len(fns_jpg))**-1
    print('Inference rate: {:d} images per second.'.format(round(rate)))
    print('{:d} files were processed.'.format(len(fns_jpg)))

    # Save CSV file containing the results
    with open(args.csv_out, 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(queries)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run inference on a collection of MNIST jpeg files.'
    )
    parser.add_argument('path_dir_jpeg',
                        type=str,
                        help='Path to directory in which MNIST jpeg files live.')
    parser.add_argument('server_address',
                        type=str,
                        help='Address for the inference server.')
    parser.add_argument('csv_out',
                        type=str,
                        help='Full path to csv file that will contain the results.')
    args = parser.parse_args()
    main(args)
