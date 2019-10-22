#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""usage: ./plrl.py [-d <DIR> -m <INT> -f <INT> -h]
    
    -d, --dir <DIR>                             Directory of *.read_length.txt
    -f, --field <INT                            Field(s) to use for prefix based on 
                                                splitted (".") filename [default: 2]
    -m, --max <INT>                             Maximum readlength [default: 40000]
    -h, --help                                  Help

"""

from timeit import default_timer as timer
from docopt import docopt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
sns.set_style('ticks')
sns.set_context('notebook')
import os
from tqdm import tqdm
import collections
import sys 

def parse_readlengths(directory, field):
    read_length_array_by_dataset = {}
    readlength_fs = []
    for f in os.listdir(directory):
        if ".".join(f.split(".")[-2:]) == 'read_length.txt':
            readlength_fs.append(os.path.join(directory, f))
    if len(readlength_fs) == 0:
        sys.exit("[X] No files ending in '*.read_length.txt' in folder %s" % os.path.abspath(directory))
    print("[+] Parsing files ...")
    for readlength_f in tqdm(sorted(readlength_fs), total=len(readlength_fs), desc="[%] ", ncols=80):
        dataset_id = ".".join(os.path.basename(readlength_f).split(".")[:field])
        with open(readlength_f) as readlength_fh:
            readlengths = readlength_fh.read().rstrip("\n").split("\n")
            read_length_array_by_dataset[dataset_id] = np.array(readlengths).astype(int)
    return read_length_array_by_dataset

def get_n50(array):
    all_len = sorted(array, reverse=True)
    csum = np.cumsum(all_len)
    n2 = int(sum(array)/2)
    csumn2 = min(csum[csum >= n2])
    ind = np.where(csum == csumn2)
    return all_len[ind[0][0]]

def plot_readlength_distributions(read_length_array_by_dataset, maxlength):
    steps = 1000
    bins = np.arange(0, maxlength+2*steps, steps)
    xlabels = ["0"] + ["%s" % int(i/1000) for i in bins[1:-1]] + [">%s" % int(bins[-2]/1000)]
    N_labels = len(xlabels)
    xticks = steps * np.arange(N_labels) + steps/2
    data = collections.defaultdict(dict)
    print("[+] Computing histograms ...")
    for idx, (dataset, read_length_array) in tqdm(enumerate(read_length_array_by_dataset.items()), total=len(read_length_array_by_dataset), desc="[%] ", ncols=80):
        data[dataset]['count_weights'] = np.ones_like(read_length_array)/float(len(read_length_array))
        data[dataset]['count_y'], data[dataset]['count_bins'] = np.histogram(np.clip(read_length_array, bins[0], maxlength), bins=bins, weights=data[dataset]['count_weights'])
        data[dataset]['span_weights'] = read_length_array / np.sum(read_length_array)
        data[dataset]['span_y'], data[dataset]['span_bins'] = np.histogram(np.clip(read_length_array, bins[0], maxlength), bins=bins, weights=data[dataset]['span_weights'])
        data[dataset]['n50'] = get_n50(read_length_array)
        data[dataset]['bases'] = '{:,}'.format(np.sum(read_length_array))
        data[dataset]['reads'] = '{:,}'.format(len(read_length_array))
    max_y = max([max([max(data[dataset]['span_y']), max(data[dataset]['count_y'])]) for dataset in read_length_array_by_dataset])
    table = []
    for idx, (dataset, read_length_array) in enumerate(read_length_array_by_dataset.items()):
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.axvline(x=data[dataset]['n50'], ymin=0, lw=3, linestyle='--', color='darkgrey')
        plt.text(data[dataset]['n50']+steps/10, max_y, 'N50', color='darkgrey')
        # proportional count (works !)
        ax.plot(data[dataset]['count_bins'][1:], data[dataset]['count_y'], color='gold', label="Read count (Total = %s)" % data[dataset]['reads'], lw=2, marker='o')
        # proportional span (works !)
        ax.plot(data[dataset]['span_bins'][1:], data[dataset]['span_y'], color='orange', label="Base count (Total = %s b)" % data[dataset]['bases'], lw=2, marker='o')
        
        ax.set_ylim(0, max_y)
        plt.xticks(xticks, xlabels)
        ax.set_xticklabels(xlabels)
        sns.despine(ax=ax)
        ax.set_xlabel('Read length (kb)')
        ax.set_ylabel('Propotion of total')
        plt.legend(frameon=False, title=r'$\bf{%s}$' % dataset.replace("_", "\\_"))
        outfile = "%s.readlengths.png" % dataset
        print("[+] Generating %r" % outfile)
        fig.savefig(outfile, format="png")
        table.append([dataset, data[dataset]['reads'], data[dataset]['bases'], '{:,}'.format(get_n50(read_length_array))])
    col_width = max(len(word) for row in table for word in row) + 2  # padding
    print()
    print(format_table(col_width, ['dataset', 'reads', 'bases', 'N50']))
    print("|%s|" % "|".join([(":%s" % ("-" * (col_width - 1)) if col == 0 else "%s:" % ("-" * (col_width - 1))) for col in range(4)]))
    for row in table:
        print(format_table(col_width, row))
    print()

def format_table(col_width, row):
    return "|%s|" % "|".join(("%s" % word.ljust(col_width) if idx == 0 else "%s" % word.rjust(col_width)) for idx, word in enumerate(row))

if __name__ == "__main__":
    __version__ = '0.2'
    start_time = timer()
    args = docopt(__doc__)
    read_length_array_by_dataset = parse_readlengths(args['--dir'], int(args['--field']))
    plot_readlength_distributions(read_length_array_by_dataset, int(args['--max']))
    print("[*] Total runtime: %.3fs" % (timer() - start_time))