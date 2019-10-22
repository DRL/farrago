#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""usage: ./seqstats.py <FILE>... -o <STR> [-n <INT> -g <FILE> -c -t -m -h]
    
    <FILE>...                          FASTA file(s) of proteins
    -n, --ns <INT>                     Split sequences based on INT consecutive N's
    -g, --config <FILE>                Config file with format "$1,$2,$3", with 
                                            $1 = fasta_f
                                            $2 = name
                                            $3 = genome size in Mb
    -o, --outprefix <STR>              Output prefix
    -t, --tsv                          Output TSV        
    -c, --csv                          Output CSV
    -m, --md                           Output Markdown
    -h, --help                         Help

"""

from timeit import default_timer as timer
from docopt import docopt

import numpy as np
import matplotlib.pyplot as plt
import matplotlib 
import seaborn as sns
import pandas as pd
sns.set_style('white')
sns.set_style('ticks')
sns.set_context('notebook')
import sys
import os
from tqdm import tqdm
import collections

def get_seqs(fasta):
    seqs = []
    with open(fasta) as fasta_fh:
        seq = []
        for l in fasta_fh:
            if l[0] == '>':
                if seq:
                    seqs.append(''.join(seq).upper())
                seq = []
            else:
                seq.append(l[:-1])
        seqs.append(''.join(seq).upper())
    return seqs

def get_stats(seqs):
    lengths = np.array([len(seq) for seq in seqs])
    sorted_lengths = np.sort(lengths)[::-1]
    cum_sum = np.cumsum(sorted_lengths)
    n_50 = int(np.sum(lengths) * 0.5)
    n_90 = int(np.sum(lengths) * 0.9)
    csum_n_50 = np.min(cum_sum[cum_sum >= n_50])
    csum_n_90 = np.min(cum_sum[cum_sum >= n_90])
    ind_n_50 = np.where(cum_sum == csum_n_50)
    ind_n_90 = np.where(cum_sum == csum_n_90)
    n50 = sorted_lengths[ind_n_50[0][0]]
    n90 = sorted_lengths[ind_n_90[0][0]]
    min_length = np.min(lengths)
    max_length = np.max(lengths)
    mean_length = int(np.mean(lengths))
    span = np.sum(lengths)
    return span, n50, n90, min_length, max_length, mean_length, cum_sum

def parse_fasta(fastas):
    seqs_by_fasta = {}
    for fasta in tqdm(sorted(fastas), total=len(fastas), desc="[%] ", ncols=80):
        if not os.path.isfile(fasta):
            print("[-] File %r could not be found" % (fasta))
            continue
        seqs = get_seqs(fasta)
        if len(seqs) < 1 or len(seqs[0]) == 0:
            print("[-] No sequences in %r" % (fasta))
            continue
        seqs_by_fasta[fasta] = seqs
    return seqs_by_fasta

def get_counts(seqs):
    seq_count = len(seqs)
    assembly = list("".join(seqs))
    char_counter = collections.Counter(assembly)
    N_count = char_counter['N']
    ATGC_count = char_counter['A'] + char_counter['T'] + char_counter['G'] + char_counter['C']
    nonATGCN_count = len(assembly) - char_counter['N'] - ATGC_count
    gc = "{:.3f}".format((char_counter['G'] + char_counter['C']) / ATGC_count) if ATGC_count else 'N/A'
    return seq_count, "{:.3f}".format(ATGC_count / len(assembly)), "{:.3f}".format(N_count / len(assembly)), "{:.3f}".format(nonATGCN_count / len(assembly)), gc

def collect_metrics(seqs_by_fasta, name_genomesize_by_fasta_f):
    results_by_fasta = collections.defaultdict(dict)
    for fasta, seqs in tqdm(seqs_by_fasta.items(), total=len(seqs_by_fasta), desc="[%] ", ncols=80):
        span, n50, n90, min_length, max_length, mean_length, cum_sum = get_stats(seqs)
        seq_count, ATGC_count, N_count, nonATGCN_count, gc = get_counts(seqs)
        results = {
                'fasta' : fasta,
                'ATGC' : ATGC_count,
                'N' : N_count,
                'nonATGCN' : nonATGCN_count,
                'gc' : gc,
                'x': np.arange(seq_count).astype(int),
                'y': cum_sum
                }
        if fmt == 'csv':
            results['count'] = seq_count
            results['raw_count'] = seq_count
            results['span'] = span
            results['n50'] = n50
            results['n90'] = n90
            results['min'] = min_length
            results['max'] = max_length
            results['mean'] = mean_length
        else:
            results['count'] = '{:,}'.format(seq_count)
            results['raw_count'] = seq_count
            results['span'] = '{:,}'.format(span)
            results['n50'] = '{:,}'.format(n50)
            results['n90'] = '{:,}'.format(n90)
            results['min'] = '{:,}'.format(min_length)
            results['max'] = '{:,}'.format(max_length)
            results['mean'] = '{:,}'.format(mean_length)
        if name_genomesize_by_fasta_f is None:
            results['name'] = fasta
            results['y_scaled'] = list(cum_sum / span)
        else:
            name, genomesize = name_genomesize_by_fasta_f[fasta]
            results['name'] = name
            results['y_scaled'] = list(cum_sum / genomesize)
        results_by_fasta[fasta] = results
    return results_by_fasta

def get_max_widths(rows):
    max_widths = [len(key) for key in KEYS]
    for line in rows:
        for idx, string in enumerate(line):
            length = len(string)
            if length > max_widths[idx]:
                max_widths[idx] = length
    return max_widths

def generate_output(results_by_fasta, fmt, prefix, name_genomesize_by_fasta_f):
    rows = []
    for fasta, result_dict in results_by_fasta.items():
        rows.append([result_dict[key] for key in KEYS])
    output = []
    if fmt == 'csv':
        output.append(",".join(KEYS)) 
        for row in rows:
            output.append(",".join(row)) 
    if fmt == 'tsv':
        output.append("\t".join(KEYS)) 
        for row in rows:
            output.append("\t".join(row)) 
    elif fmt == 'md':
        max_widths = get_max_widths(rows)
        output.append(format_table(max_widths, KEYS))
        output.append("|%s|" % "|".join([(":%s" % ("-" * (max_width - 1)) if idx == 0 else "%s:" % ("-" * (max_width - 1))) for idx, max_width in enumerate(max_widths)]))
        for row in rows:
            output.append(format_table(max_widths, row))
    else:
        sys.exit("[X] Output format %r not supported" % fmt)
    with open('%s.seqstats.%s' % (prefix, fmt), 'w') as output_fh:
        output_fh.write("%s\n" % ("\n".join(output)))
    plot_cumulative_length(results_by_fasta, prefix, name_genomesize_by_fasta_f)

def format_table(max_widths, row):
    return "|%s|" % "|".join(("%s" % word.ljust(max_widths[idx]) if idx == 0 else "%s" % word.rjust(max_widths[idx])) for idx, word in enumerate(row))

def plot_cumulative_length(results_by_fasta, prefix, name_genomesize_by_fasta_f):
    n = len(results_by_fasta)
    if n <= 10:
        cmap = 'tab10'
    else:
        cmap = 'tab20'
    cmap = plt.cm.get_cmap(cmap)
    colours = cmap(np.arange(n))
    x_max = 0
    fig, ax = plt.subplots(figsize=(12, 12))
    for idx, (fasta, results) in enumerate(results_by_fasta.items()):
        if results['raw_count'] > x_max:
           x_max = results['raw_count']
        ax.plot(results['x'], results['y'], color=colours[idx], label=results['name'], lw=1)
    sns.despine(ax=ax)
    ax.set_xlabel('Contigs')
    ax.ticklabel_format(useOffset=False, style='plain')
    ax.get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.set_ylabel('Bases')
    plt.legend(frameon=False)
    outfile = "%s.cumulative_length.png" % prefix
    print("[+] Generating %r" % outfile)
    fig.savefig(outfile, format="png")

    x_max = 0
    fig, ax = plt.subplots(figsize=(12, 12))
    for idx, (fasta, results) in enumerate(results_by_fasta.items()):
        if results['raw_count'] > x_max:
           x_max = results['raw_count']
        ax.plot(results['x'], results['y_scaled'], color=colours[idx], label=results['name'], lw=1)
    sns.despine(ax=ax)
    ax.set_xlabel('Contigs')
    ax.axhline(y=1, xmin=0, xmax=x_max, lw=3, linestyle='--', color='darkgrey')
    ax.set_ylabel('Bases')
    if not name_genomesize_by_fasta_f is None:
        ax.set_ylabel('Proportion of genome size')
    else:
        ax.set_ylabel('Proportion of assembly size')
    plt.legend(frameon=False)
    outfile = "%s.cumulative_proportional_length.png" % prefix
    print("[+] Generating %r" % outfile)
    fig.savefig(outfile, format="png")


def get_fmt(args):
    fmts_input = {
        'csv' : args['--csv'],
        'tsv' : args['--tsv'],
        'md' : args['--md']
        }
    fmts = [fmt for fmt, value in fmts_input.items() if value == True]
    if len(fmts) == 0:
        return 'tsv'
    elif len(fmts) > 1:
        sys.exit("[X] Please specify ONE output fmt")
    else:
        return fmts[0]

def parse_config(config_f):
    genomes_df = pd.read_csv(
        config_f,
        sep=",",
        names=['fasta_f', 'name', 'genome_size'],
        header=None)
    name_genomesize_by_fasta_f = {}
    for fasta_f, name, genome_size in genomes_df.values.tolist():
        name_genomesize_by_fasta_f[fasta_f] = (name, genome_size)
    return name_genomesize_by_fasta_f


if __name__ == "__main__":
    __version__ = '0.1'
    start_time = timer()
    KEYS = ['fasta', 'count', 'span', 'n50', 'n90', 'max', 'min', 'gc', 'ATGC', 'N', 'nonATGCN', 'mean']
    args = docopt(__doc__)
    fmt = get_fmt(args)
    seqs_by_fasta = parse_fasta(args['<FILE>'])
    if args['--config']:
        name_genomesize_by_fasta_f = parse_config(args['--config'])
    else:
        name_genomesize_by_fasta_f = None
    results_by_fasta = collect_metrics(seqs_by_fasta, name_genomesize_by_fasta_f)
    generate_output(results_by_fasta, fmt, args['--outprefix'], name_genomesize_by_fasta_f)
    #generate_plot(results_by_fasta, args['--outprefix'])
    print("[*] Total runtime: %.3fs" % (timer() - start_time))