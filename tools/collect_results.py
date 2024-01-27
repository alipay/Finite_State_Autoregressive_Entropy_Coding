import os
import csv
import argparse


def main(args):
    root = args.root
    output_file = args.output_file
    collect_file = args.collect_file

    result_dicts = []
    fieldnames = set()
    for dirpath, dirnames, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(collect_file):
                filepath = os.path.join(dirpath, filename)
                with open(filepath) as f:
                    reader = csv.DictReader(f)
                    for result_dict in reader:
                        result_dict.update(name=dirpath)
                        result_dicts.append(result_dict)
                    fieldnames.update(reader.fieldnames)

    with open(output_file, 'w') as f:
        fieldnames = ['name'] + sorted(list(fieldnames)) # make sure name goes first
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result_dict in result_dicts:
            writer.writerow(result_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('--root', type=str, default="experiments/", help='root path')
    parser.add_argument('--collect-file', type=str, default="metrics.csv", help='collect file')
    parser.add_argument('--output-file', type=str, default="all_metrics.csv", help='output file')
    args = parser.parse_args()
    main(args)
