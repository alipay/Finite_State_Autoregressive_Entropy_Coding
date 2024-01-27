import os
import csv
import argparse
import numpy as np
from collections import OrderedDict

def read_results(file_path):
    all_results = OrderedDict()
    with open(file_path) as f:
        reader = csv.DictReader(f)
        for result_dict in reader:
            if not 'name' in result_dict:
                raise ValueError()
            else:
                name = result_dict.pop('name')
            all_results[name] = result_dict
    return all_results

def main(args):
    root = args.root
    output_file = os.path.join(root, args.output_file)
    collect_file = args.collect_file

    result_dict = OrderedDict()
    # for dirpath, dirnames, filenames in os.walk(root):
    for dir in os.listdir(root):
        if dir.startswith("repeat_") or dir == collect_file:
            if dir == collect_file:
                metric_file = os.path.join(root, dir)
            else:
                metric_file = os.path.join(root, dir, collect_file)
            if os.path.exists(metric_file):
                results = read_results(metric_file)
                for name, metrics in results.items():
                    if not name in result_dict:
                        result_dict[name] = OrderedDict()
                    for key, value in metrics.items():
                        # filter non-float values
                        try:
                            value = float(value)
                        except:
                            continue
                        if not key in result_dict[name]:
                            result_dict[name][key] = []
                        result_dict[name][key].append(value)

    fieldnames = OrderedDict()
    output_results = []
    for name, metrics in result_dict.items():
        output_result = OrderedDict()
        output_result['name'] = name
        for key, values in metrics.items():
            if args.agg_metric == "range":
                val_mid = (max(values)+min(values)) / 2
                val_var = (max(values)-min(values)) / 2
                output_result[key+"_mid"] = val_mid
                output_result[key+"_var"] = val_var
            elif args.agg_metric == "gaussian":
                values = np.array(values)
                val_mid = (max(values)+min(values)) / 2
                val_var = (max(values)-min(values)) / 2
                output_result[key+"_mean"] = np.mean(values)
                output_result[key+"_std"] = np.std(values)

        fieldnames.update(**output_result)
        output_results.append(output_result) 
    
    fieldnames = list(fieldnames.keys())
    
    with open(output_file, 'w') as f:
        # fieldnames = ['name'] + sorted(list(fieldnames)) # make sure name goes first
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result_dict in output_results:
            writer.writerow(result_dict)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument('--root', '-r', type=str, default="experiments/", help='root path')
    parser.add_argument('--agg-metric', '-m', type=str, default="gaussian", choices=["range", "gaussian"], help='aggregation metric')
    parser.add_argument('--collect-file', type=str, default="metrics.csv", help='collect file')
    parser.add_argument('--output-file', type=str, default="metrics_repeat.csv", help='output file')
    args = parser.parse_args()
    main(args)
