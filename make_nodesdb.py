#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""usage: ./make_nodesdb.py --taxdump <DIR> -o <STR> [-h]
    
    -t, --taxdump <FILE>                TaxDump folder
    -o, --outprefix <STR>               Outprefix
    -h, --help                          Help

    ./make_nodesdb.py -t . -o ncbi_taxdump_20191021

"""

from timeit import default_timer as timer
from docopt import docopt
import sys
import os
from tqdm import tqdm
import collections

def get_files_by_filename(folder):
    file_by_filename = {}
    for f in os.listdir(folder):
        if f == 'nodes.dmp':
            file_by_filename['nodes'] = os.path.join(folder, f)
        elif f == 'names.dmp':
            file_by_filename['names'] = os.path.join(folder, f)
    if 'names' not in file_by_filename:
        print("[X] names.dmp not found in %r" % os.path.abspath(folder))
    if 'nodes' not in file_by_filename:
        print("[X] nodes.dmp not found in %r" % os.path.abspath(folder))
    if len(file_by_filename) != 2:
        sys.exit()
    return file_by_filename

def generate_nodesdb(file_by_filename):
    void = set(['|', ''])
    taxonomy_by_tax_id = collections.defaultdict(dict)
    print("[+] Parsing %r" % file_by_filename['nodes'])
    with open(file_by_filename['nodes']) as nodes_fh:
        for line in nodes_fh:
            tax_id, parent, rank = [string for string in line.rstrip('\t|\n').split("\t|\t") if not string in void][0:3]
            taxonomy_by_tax_id[tax_id]['parent'] = parent
            taxonomy_by_tax_id[tax_id]['rank'] = rank
    print("[+] Parsing %r" % file_by_filename['names'])
    with open('names.dmp') as names_fh:
        for line in names_fh:
            cols = [string for string in line.rstrip("\t|\n").split("\t|\t") if not string in void]
            tax_id, taxon, taxon_type = cols[0], cols[1], cols[-1]
            if taxon_type == "scientific name":
                taxonomy_by_tax_id[tax_id]['name'] = taxon
    print("[+] Generating nodesdb...")
    nodes_row = ["# %s " % len(taxonomy_by_tax_id)]
    for tax_id, taxonomy in tqdm(taxonomy_by_tax_id.items(), total=len(taxonomy_by_tax_id), desc="[%] ", ncols=80):
        nodes_row.append("\t".join([tax_id, taxonomy['rank'], taxonomy['name'], taxonomy['parent']]))
    return "\n".join(nodes_row)
    
def write_output(nodes_db, outprefix):
    out_f = "%s.nodesdb" % outprefix
    with open(out_f, 'w') as out_fh:
        out_fh.write(nodes_db)
    print("[+] Wrote %r" % out_f)

if __name__ == "__main__":
    __version__ = '0.1'
    start_time = timer()
    args = docopt(__doc__)
    print(args)
    
    file_by_filename = get_files_by_filename(args['--taxdump'])
    nodes_db = generate_nodesdb(file_by_filename)
    write_output(nodes_db, args['--outprefix'])
    print("[*] Total runtime: %.3fs" % (timer() - start_time))