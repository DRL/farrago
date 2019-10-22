#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""usage: ./protax.py -f <FILE> -b <FILE> -o <STR> [-n <FILE> -t <INT> -r <STR> -i <INT -h]
    
    -f, --fasta <FILE>                  Fasta file
    -b, --blast <FILE>                  BLAST file
    -o, --outprefix <STR>               Outprefix

    -t, --taxids <INT>                  Boundary Tax IDs
    -r, --tax_rule <STR>                Tax rule for taxonomy annotation 'best' or 'bestsum'
    -i, --ignore <INT>                  Ignore taxid
    -n, --nodesdb <FILE>                nodesdb

    -h, --help                          Help

--------------
diamond blastp:
    diamond blastp --query proteins.faa --db /ceph/software/databases/uniprot_2019_08/full/reference_proteomes.dmnd --outfmt 6 qseqid staxids bitscore qlen slen qseqid --sensitive --max-target-seqs 5 --evalue 1e-25 --threads 30 > /scratch/dlaetsch/protein.vs.uniprot.out && rsync /scratch/dlaetsch/protein.vs.uniprot.out . && touch protein.vs.uniprot.out.done
protax:
    ./protax.py -n ncbi_taxdump_20191021.nodesdb -r best -o proteins.vs.uniprot -f proteins.faa -b proteins.vs.uniprot.out ; \
"""

"""
To do: 
- Mode without taxonomy, just plot
- add new taxrule: lca
    - bob-rule
- iron out plotting taxonomy 
    ./protax.py -f pseudococcus_longispinus.v1.braker.aa.fa -b pseudococcus_longispinus.v1.braker.aa.vs.uniprot_2019_08.mts_5.eval_1e25.out -n ncbi_taxdump_20191021.nodesdb -r best -o pseudococcus_longispinus.v1.braker.aa.vs.uniprot
- fix bestsum float decimals

"""

from timeit import default_timer as timer
from docopt import docopt

import numpy as np
import matplotlib.pyplot as plt
#import matplotlib 
import seaborn as sns
import pandas as pd
sns.set_style('white')
sns.set_style('ticks')
sns.set_context('notebook')
import sys
import os
from tqdm import tqdm
import collections
import itertools
import networkx as nx

def parse_sequence_ids(fasta_f):
    if not os.path.isfile(fasta_f):
        sys.exit("[-] File %r could not be found" % (fasta_f))
    seq_ids = []
    with open(fasta_f) as fasta_fh:
        for l in fasta_fh:
            if l[0] == '>':
                seq_ids.append(l[1:].split()[0])
    if len(seq_ids) == 0:
        sys.exit("[-] No sequences in %r" % (fasta_f))
    return seq_ids

def get_taxgraph(nodesdb_f):
    print("[+] Parsing nodes in %r" % nodesdb_f)
    tax_count = int(open(nodesdb_f).readline().lstrip("# ").rstrip("\n"))
    nodesdb_df = pd.read_csv(
        nodesdb_f,
        sep="\t",
        names=['taxid', 'rank', 'name', 'parent'],
        skiprows=1)
    tax_graph = nx.DiGraph()
    for taxid, rank, name, parent in tqdm(nodesdb_df.values.tolist(), total=tax_count, desc="[%] ", ncols=80):
        tax_graph.add_node(taxid, rank=rank, name=name)
        tax_graph.add_edge(taxid, parent)
    return tax_graph

def split_csv(string):
    return int(string.split(";")[0])

def parse_blast(blast_f, ignore=None, tax_rule='bestsum'):
    ignore_string = '' 
    if ignore is not None:
        ignore_string = '(ignoring taxID %s)' % ignore
    print("[+] Parsing results in %r %s" % (blast_f, ignore_string))
    blast_df = pd.read_csv(
        blast_f,
        sep="\t",
        usecols=[0, 1, 2, 3, 4],
        names=['seq_id', 'taxid', 'bitscore', 'qlen', 'slen'],
        header=None,
        dtype={ 
            'seq_id': 'category', 
            'bitscore': np.float, 
            'qlen': np.int,
            'slen': np.int},
        converters={'taxid': split_csv}) # split on ";"
    if ignore is not None:
        blast_df = blast_df[blast_df['taxid'] != ignore]
    if tax_rule == 'best':
        blast_df = blast_df.drop_duplicates(subset=['seq_id'])
    hitObjs_by_seq_id = collections.defaultdict(list)
    for seq_id, taxid, bitscore, qlen, slen in tqdm(blast_df.values.tolist(), total=len(blast_df), desc="[%] ", ncols=80):
        hitObjs_by_seq_id[seq_id].append(HitObj(taxid, bitscore, qlen, slen))
    return hitObjs_by_seq_id

class HitObj(object):
    def __init__(self, tax_id, bitscore, qlen, slen):
        self.tax_id = tax_id
        self.bitscore = bitscore
        self.qlen = qlen
        self.slen = slen

def process_blast_results(hitObjs_by_seq_id, tax_graph, boundary_taxa):
    print("[+] Processing blast results ...")
    taxonomy_by_seq_id = {}
    hits_by_seq_id = {}
    deltas_by_boundary_taxon = collections.defaultdict(list)
    warnings = []
    for seq_id, hitObjs in tqdm(hitObjs_by_seq_id.items(), total=len(hitObjs_by_seq_id), desc="[%] ", ncols=80):
        score_by_taxon_by_rank = collections.defaultdict(lambda: collections.Counter())
        for hit_idx, hitObj in enumerate(hitObjs):
            score = hitObj.bitscore
            taxon_by_rank = {}
            if not hitObj.tax_id in tax_graph.nodes:
                warnings.append("[-] Hit to unknown taxon %r" % hitObj.tax_id)
            else:
                best_hit_taxonomy = []
                for path in nx.all_simple_paths(tax_graph, source=hitObj.tax_id, target=1):
                    for node_idx in path:
                        rank = tax_graph.nodes[node_idx]['rank']
                        if rank in RANKS:
                            taxon = tax_graph.nodes[node_idx]['name']
                            taxon_by_rank[rank] = taxon
                            if hit_idx == 0:
                                best_hit_taxonomy.append(taxon)
                    taxon_defined = None
                    for idx, rank in enumerate(RANK_ORDER):
                        if rank not in taxon_by_rank:
                            if RANK_ORDER[idx-1] in taxon_by_rank:
                                taxon = "%s-undef" % taxon_by_rank[RANK_ORDER[idx-1]]
                            else:
                                taxon = taxon_defined
                        else:
                            taxon = taxon_by_rank[rank]
                        taxon_defined = taxon
                        score_by_taxon_by_rank[rank][taxon] += score
                if hit_idx == 0:
                    taxonomy_taxa = set(best_hit_taxonomy)
                    deltas_by_boundary_taxon['root'].append(hitObj.qlen / hitObj.slen)
                    for boundary_taxon in boundary_taxa:
                        if boundary_taxon in taxonomy_taxa:
                            deltas_by_boundary_taxon[boundary_taxon].append(hitObj.qlen / hitObj.slen)
        try:
            taxonomy_by_seq_id[seq_id] = [score_by_taxon_by_rank[rank].most_common(1)[0][0] for rank in RANK_ORDER]       
        except IndexError:
            print(seq_id)
            for rank in RANK_ORDER:
                print("[W] %s" % score_by_taxon_by_rank[rank])
        hits_by_seq_id[seq_id] = [";".join([":".join([_taxon, str(_score)]) for _taxon, _score in score_by_taxon_by_rank[rank].most_common()]) for rank in RANK_ORDER]
    if warnings:
        print("\t\n".join(warnings))
    return taxonomy_by_seq_id, hits_by_seq_id, deltas_by_boundary_taxon

def generate_report(seq_ids, taxonomy_by_seq_id, hits_by_seq_id, outprefix):
    print("[+] Writing output ...")
    header = "\t".join(["#seq_id"] + ["%s_%s" % (idx+2, rank) for idx, rank in enumerate(RANK_ORDER)])
    rows_taxonomy = []
    rows_taxonomy.append(header)
    rows_hits = []
    rows_hits.append(header)
    for seq_id in seq_ids:
        if seq_id in taxonomy_by_seq_id:
            rows_taxonomy.append("%s\t%s" % (seq_id, "\t".join(taxonomy_by_seq_id[seq_id])))
            rows_hits.append("%s\t%s" % (seq_id, "\t".join(hits_by_seq_id[seq_id])))
        else:
            rows_taxonomy.append("%s\t%s" % (seq_id, "\t".join(['no-hit' for i in range(len(RANKS))])))
            rows_hits.append("%s\t%s" % (seq_id, "\t".join(['no-hit:0.0' for i in range(len(RANKS))])))
    report_f = '%s.report.tsv' % (outprefix) 
    with open(report_f, 'w') as report_fh:
        report_fh.write("%s\n" % ("\n".join(rows_taxonomy)))
        print("[+]\t%r" % report_f) 
    hits_f = '%s.hits.tsv' % (outprefix) 
    with open(hits_f, 'w') as hits_fh:
        hits_fh.write("%s\n" % ("\n".join(rows_hits)))
        print("[+]\t%r" % hits_f) 
    print("[+] %.2f%% of sequences received taxonomic annotation" % (len(taxonomy_by_seq_id) * 100/ len(seq_ids)))

def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

def plot_distribution(deltas_by_boundary_taxon, outprefix):
    colours = plt.cm.get_cmap('Paired')(np.arange(len(deltas_by_boundary_taxon)))
    fig, ax = plt.subplots(figsize=(12, 12))
    min_x, max_x, steps = 0, 2, 52
    bins = np.linspace(min_x, max_x, num=steps)
    #print(bins)
    ax.axvline(x=1.0, lw=3, linestyle='--', color='darkgrey')
    for idx, (boundary_taxon, deltas) in enumerate(deltas_by_boundary_taxon.items()):
        count_y, count_bins = np.histogram(np.clip(deltas, min_x, max_x), bins=bins)
        x = [np.mean([bin1, bin2]) for bin1, bin2 in pairwise(count_bins)]
        x[0], x[-1] = min_x, max_x
        ax.plot(x, (count_y/len(deltas))*100, color=colours[idx], label=boundary_taxon, lw=3, marker='o')
    ax.axhline(y=0.0, xmin=x[0], xmax=x[-1], lw=3, linestyle='--', color='darkgrey')
    sns.despine(ax=ax)
    ax.set_xlabel('Length fraction (qlen/slen of best hit)')
    ax.set_ylabel('Percentage of hits (within taxon boundary)')
    xlabels = [str(xlabel) for xlabel in ax.get_xticks().tolist()]
    xlabels[1] = "≤ 0" 
    xlabels[-2] = "≥ %s" % "{:.0f}".format(max_x)
    ax.set_ylim((0, 100 ))
    ax.set_xticklabels(xlabels)
    plt.legend(frameon=False)
    outfile = "%s.distribution.png" % outprefix
    print("[+] Generating %r" % outfile)
    fig.savefig(outfile, format="png")

def get_boundary_taxa(boundary_string, tax_graph):
    if args['--taxids'] is None:
        return set([])
    else:
        boundary_taxa = []
        for boundary_tax_id in [int(tax_id) for tax_id in boundary_string.replace(" ", "").split(",")]:
            if tax_graph.nodes[boundary_tax_id]['rank'] in RANKS:
                boundary_taxa.append(tax_graph.nodes[boundary_tax_id]['name'])
            else:
                sys.exit("[X] Boundery taxon %r has non-canonical rank %r must be: %s" % (
                                                                                    boundary_tax_id, 
                                                                                    tax_graph.nodes[boundary_tax_id]['rank'],
                                                                                    ", ".join(RANK_ORDER)))
    return set(boundary_taxa)

if __name__ == "__main__":
    __version__ = '0.1'
    start_time = timer()
    args = docopt(__doc__)
    
    RANK_ORDER = ['superkingdom', 'kingdom', 'subphylum', 'phylum', 'class', 'order', 'suborder', 'superfamily', 'family', 'genus', 'species']
    RANKS = set(RANK_ORDER)

    tax_rule = args['--tax_rule']
    ignore_tax_id = int(args['--ignore']) if not args['--ignore'] is None else None
    tax_graph = get_taxgraph(args['--nodesdb'])
    boundary_taxa = get_boundary_taxa(args['--taxids'], tax_graph)
    seq_ids = parse_sequence_ids(args['--fasta'])
    hitObjs_by_seq_id = parse_blast(args['--blast'], ignore=ignore_tax_id, tax_rule=tax_rule)
    taxonomy_by_seq_id, hits_by_seq_id, deltas_by_boundary_taxon = process_blast_results(hitObjs_by_seq_id, tax_graph, boundary_taxa)
    generate_report(seq_ids, taxonomy_by_seq_id, hits_by_seq_id, args['--outprefix'])
    plot_distribution(deltas_by_boundary_taxon, args['--outprefix'])
    print("[*] Total runtime: %.3fs" % (timer() - start_time))