#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""usage: ./vcf2zarr.py -v <FILE> -s <FILE> [-h]
    
    -v, --vcf <FILE>                            VCF file
    -s, --sample <FILE>                         Sample CSV file
    -h, --help                                  Help

"""

from timeit import default_timer as timer
from docopt import docopt
import pathlib
import allel
import pandas as pd
import numpy as np
import os
import shutil
import zarr

def vcf2zarr(vcf_f, sample_f):
    # Read samples
    pop_id_by_sample_id = pd.read_csv(sample_f, index_col=0, squeeze=True, header=None).to_dict()
    # Read vcf header
    header = [(header_line.split(",")) for header_line in allel.read_vcf_headers(vcf_f).headers if header_line.startswith("##contig")]
    length_by_seq_id = {_seq_id.lstrip("##contig=<ID="): int(_length.lstrip("length=").rstrip(">\n")) for (_seq_id, _length) in header}
    sample_ids = allel.read_vcf_headers(vcf_f).samples
    zarr_dir = str(pathlib.Path(vcf_f).with_suffix('.zarr'))
    if os.path.isdir(zarr_dir):
        print("[!] ZARR store %r exists. Deleting ..." % zarr_dir)
        shutil.rmtree(zarr_dir)
    print("[+] Generating ZARR store %r" % zarr_dir)
    for seq_id in length_by_seq_id.keys():
        allel.vcf_to_zarr( 
            vcf_f, 
            zarr_dir, 
            #group="%s=%s/" % (seq_id, length),
            group="%s" % seq_id,
            region=seq_id,
            fields=[
                'variants/POS',
                'variants/REF',
                'variants/ALT',
                'variants/QUAL',
                'variants/is_snp',
                'variants/numalt',
                'variants/DP',
                'calldata/GT',
                'calldata/DP'
            ], 
            overwrite=True)
    zarr_store = zarr.open(zarr_dir, mode='a')
    zarr_store.create_dataset('sample_ids', data=np.array(sample_ids))
    zarr_store.create_dataset('seq_ids', data=np.array(list(length_by_seq_id.keys())))
    zarr_store.create_dataset('seq_lengths', data=np.array(list(length_by_seq_id.values())))
    zarr_store.create_dataset('pop_ids', data=np.array([pop_id_by_sample_id[sample_id] for sample_id in sample_ids]))
    print(zarr_store.tree())

if __name__ == "__main__":
    __version__ = 0.2
    start_time = timer()
    args = docopt(__doc__)
    vcf2zarr(args['--vcf'], args['--sample'])
    print("[*] Total runtime: %.3fs" % (timer() - start_time))