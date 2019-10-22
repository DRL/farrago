#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""usage: ./zarrqc.py -z <FILE> -o <STR> [--indels -h]
    
    -z, --zarr <FILE>                           ZARR file
    -o, --prefix <STR>                          Output prefix
    --indels                                    Calculate indel distribution 

    -h, --help                                  Help

"""

'''
conda install -c conda-forge parallel pandas numpy tqdm docopt scikit-allel zarr scipy seaborn matplotlib
'''

from timeit import default_timer as timer
from docopt import docopt
import numpy as np
import zarr
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
sns.set_style('ticks')
sns.set_context('notebook')
import allel
import itertools
import collections
import pandas as pd
from tqdm import tqdm
import scipy
# import logging

COLOURS = ['orange', 'cornflowerblue', 'lightskyblue', 'gold']
SHADES = ['black', 'darkgrey', 'lightgrey', 'white'] 

'''
import numpy as np; 
from timeit import default_timer as timer; 
starts = np.arange(0, 10**8, 100); 
ends = starts + 50; 
a = np.array((starts, ends)).T; 
print('# create_ranges()'); 
t1=timer(); 
cr = create_ranges(a); 
print(timer()-t1) ; 
print('# list_comprehension()'); 
t2=timer(); 
lc = np.concatenate([np.arange(x[0], x[1]) for x in a]); 
print(timer()-t2); 
print("Same = %s" % all(cr==lc))
# create_ranges()
0.46567643700001327
# list_comprehension()
2.9621128510000005
Same = True
'''
def get_accessability_matrix_from_bed_file(bed_file, samples):
    ''' samples -> [array_like, bool]'''
    def read_bed(bedfile):
        bed_df = pd.read_csv(
        bedfile,
        sep="\t",
        usecols=[0, 1, 2, 4],
        names=['sequence_id', 'start', 'end', 'samples'],
        skiprows=1,
        header=None,
        dtype={
            'sequence_id': 'category',
            'start': np.int,
            'end': np.int,
            'samples': 'category'})
        return bed_df

def create_ranges(aranges):
    # https://stackoverflow.com/a/47126435
    l = aranges[:,1] - aranges[:,0]
    clens = l.cumsum()
    ids = np.ones(clens[-1], dtype=int)
    ids[0] = aranges[0,0]
    ids[clens[:-1]] = aranges[1:, 0] - aranges[:-1, 1]+1
    return ids.cumsum()

def get_zarr_callset(zarr_f):
    callset = zarr.open_group(zarr_f, mode='r')
    return callset

def plot_indel_bars(variant_type_counter_by_idx, metadataObj):
    fig, ax = plt.subplots(figsize=(14, 5))
    max_idx = max(variant_type_counter_by_idx.keys())
    x, y = [], []
    for idx in range(-max_idx, max_idx + 1):
        x.append(idx)
        snps = variant_type_counter_by_idx[idx]['snps']
        sites = variant_type_counter_by_idx[idx]['sites']
        if sites == 0:
            y.append(0)
        else:
            y.append(snps / sites)
    xlabels = [str(label) for label in range(-max_idx, max_idx + 1)]
    
    sns.despine(ax=ax)
    ax.bar(x, y, color='lightgrey')
    ax.set_ylabel('Probability of SNP')
    ax.set_xlabel('Distance from nonSNP')
    ax.set_ylim((0, 0.5))
    plt.xticks(x, xlabels, rotation=90)
    fig.savefig('%s.indel_snp_prob.png' % metadataObj.prefix, format="png")

def plot_snp_density(snp_densities, windowsize, metadataObj):
    # generalise ...
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.despine(ax=ax)
    # ax.hist(snp_densities, bins=50, color='lightgrey', align='left', density=True) # densities
    ax.hist(snp_densities, bins=50, color='lightskyblue', align='left', density=False)
    ax.set_xlabel('SNPs / %s kb' % int(windowsize / 1000))
    ax.set_ylabel('Count')
    fig.savefig('%s.snp_density.png' % metadataObj.prefix, format="png")

def plot_snp_numalt(snp_numalt_counter, metadataObj):
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.despine(ax=ax)
    xlabels = [key for key in snp_numalt_counter.keys()]
    y = [val for val in snp_numalt_counter.values()]
    ax.bar(snp_numalt_counter.keys(), y, color='grey')
    plt.xticks(list(snp_numalt_counter.keys()), xlabels)
    ax.set_ylabel('SNPs')
    ax.set_ylim(1, max(y))
    ax.set_yscale('log')
    labels = ['{:,}'.format(val) for val in snp_numalt_counter.values()]
    for rect, label in zip(ax.patches, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2, height + 5, label,
                ha='center', va='bottom')
    fig.savefig('%s.snp_numalt.png' % metadataObj.prefix, format="png")    

def plot_missingness_bars(counter_by_label, metadataObj): # works
    fig, ax = plt.subplots(figsize=(7, 5))
    max_idx = max([max(counter.keys()) for state, counter in counter_by_label.items()])
    y_by_state = collections.defaultdict(list)
    for state, counter in counter_by_label.items():
        for idx in range(max_idx + 1):
            y_by_state[state].append(counter[idx])
    bottom = np.zeros(max_idx+1)
    xlabels = [str(label) for label in list(range(max_idx + 1))]
    x = range(max_idx + 1)
    color_by_state = {
        'variant SNP': 'orange',
        'variant nonSNP': 'gold',
        'monomorphic SNP': 'lightskyblue',
        'monomorphic nonSNP': 'cornflowerblue',
        }
    for state, color in color_by_state.items():
        if state in y_by_state:
            ax.bar(x, y_by_state[state], color=color, label=state, bottom=bottom)
            bottom += np.array(y_by_state[state])
    ax.set_ylabel('Number of VCF records')
    ax.set_xlabel('Number of samples w/ missing GTs')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    sns.despine(ax=ax)
    plt.xticks(x, xlabels)
    plt.legend(frameon=False)
    fig.savefig('%s.missingness.png' % metadataObj.prefix, format="png")

def plot_sample_counts_by_gt(sample_counts_by_gt, metadataObj):
    genotype_labels = ['HOMREF', 'HET', 'HOMALT', 'MISS']
    fig, ax = plt.subplots(figsize=(10, 5))
    n = len(metadataObj.sample_ids)
    x = np.arange(n)
    colour_by_label = {label: COLOURS[idx] for idx, label in enumerate(genotype_labels)}
    bottom = np.zeros(n)
    for genotype_label in genotype_labels:
        y = sample_counts_by_gt[genotype_label] / sample_counts_by_gt['TOTAL']
        ax.bar(x, y, color=colour_by_label[genotype_label], label=genotype_label, bottom=bottom)
        bottom += np.array(y)
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.5, n)
    ax.set_ylabel('Proportion of SNP')
    ax.set_xlabel('Sample IDs')
    sns.despine(ax=ax)
    handles_genotypes = [mpl.patches.Patch(color=colour_by_label[label]) for label in genotype_labels]
    if metadataObj.pop_ids_order:
        plt.gca().add_artist(plt.legend(handles=handles_genotypes, labels=genotype_labels, title='Genotypes', frameon=False, bbox_to_anchor=(0, 1.02, 0.5, 0.2), loc="center", borderaxespad=0, ncol=len(genotype_labels)))
        colour_by_pop_id = {pop_id: SHADES[idx] for idx, pop_id in enumerate(metadataObj.pop_ids_order)}
        pop_colours = [colour_by_pop_id[pop_id] for pop_id in metadataObj.pop_ids]
        plt.xticks(x, metadataObj.sample_ids, rotation=90)
        handles_populations = [mpl.patches.Patch(color=colour_by_pop_id[pop_id]) for pop_id in metadataObj.pop_ids_order]
        plt.legend(handles=handles_populations, labels=metadataObj.pop_ids_order, title='Population IDs', frameon=False, bbox_to_anchor=(0.5, 1.02, 0.5, 0.2), loc="center", borderaxespad=0, ncol=len(metadataObj.pop_ids_order))
        for (colour, tick) in zip(pop_colours, ax.xaxis.get_ticklabels()):
            tick.set_color(colour)
    else:
        plt.legend(handles=handles_genotypes, labels=genotype_labels, title='Genotypes', frameon=False, bbox_to_anchor=(0, 1.02, 1, 0.2), loc="center", borderaxespad=0, ncol=len(genotype_labels))
    fig.tight_layout()
    fig.savefig('%s.sample_genotype_count.png' % metadataObj.prefix, format="png")

def locate_transitions(x):
    x = np.asarray(x)
    return (x == b'AG') | (x == b'GA') | (x == b'CT') | (x == b'TC')

def ti_tv(x):
    if len(x) == 0:
        return np.nan
    is_ti = locate_transitions(x)
    n_ti = np.count_nonzero(is_ti)
    n_tv = np.count_nonzero(~is_ti)
    if n_tv > 0:
        return n_ti / n_tv
    else:
        return np.nan

def plot_sfs(segregating_biallelic_snp_acs_by_pop_id, metadataObj):
    # joint_sfs for two populations (works)
    for (pop_id_A, pop_id_B) in itertools.combinations([pop_id for pop_id in metadataObj.pop_ids_order], 2):
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.despine(ax=ax)
        sfs1 = allel.sfs_folded(segregating_biallelic_snp_acs_by_pop_id[pop_id_A])
        #allel.plot_sfs_folded(sfs1, ax=ax, label=pop_id_A, n=segregating_biallelic_snp_acs_by_pop_id[pop_id_A].sum(axis=1).max())
        allel.plot_sfs_folded(sfs1, ax=ax, label=pop_id_A)
        sfs2 = allel.sfs_folded(segregating_biallelic_snp_acs_by_pop_id[pop_id_B])
        #allel.plot_sfs_folded(sfs2, ax=ax, label=pop_id_B, n=segregating_biallelic_snp_acs_by_pop_id[pop_id_B].sum(axis=1).max())
        allel.plot_sfs_folded(sfs2, ax=ax, label=pop_id_B)
        ax.legend()
        ax.set_title('Folded site frequency spectra')
        # workaround bug in scikit-allel re axis naming
        ax.set_xlabel('Minor allele frequency');
        ax.set_ylim(min([min(sfs1), min(sfs2)]), max([max(sfs1), max(sfs2)]))
        fig.tight_layout()
        fig.savefig('%s.sfsfs.%s_%s.png' % (metadataObj.prefix, pop_id_A, pop_id_B), format="png")
        #n : int, optional, Number of chromosomes sampled. If provided, X axis will be plotted as allele frequency, otherwise as allele count.
        jsfs = allel.joint_sfs_folded(
            segregating_biallelic_snp_acs_by_pop_id[pop_id_A][:], 
            segregating_biallelic_snp_acs_by_pop_id[pop_id_B][:])
        fig, ax = plt.subplots(figsize=(6, 6))
        ax = allel.plot_joint_sfs_folded(jsfs, ax=ax, imshow_kwargs={'cmap': 'Blues'})
        total = np.sum(jsfs)
        for i in range(len(jsfs)):
            for j in range(len(jsfs[i])):
                ax.text(j, i, "{:.2f}".format(jsfs[i, j]/total),
                    ha="center", va="center", color="black", fontsize=12)
        ax.set_ylabel('Alternate allele count, %s' % pop_id_A)
        ax.set_xlabel('Alternate allele count, %s' % pop_id_B)
        fig.tight_layout()
        fig.savefig('%s.jsfs.%s_%s.png' % (metadataObj.prefix, pop_id_A, pop_id_B), format="png")

def plot_ti_tv(x, biallelic_snps_mutations, xlabel, metadataObj, bins):
    fig, ax = plt.subplots(figsize=(7, 5))
    # plot a histogram
    ax.hist(x, bins=bins, color='lightskyblue')
    sns.despine(ax=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('biallelic SNPs')
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    # plot Ti/Tv
    ax2 = ax.twinx()
    sns.despine(ax=ax2, top=True, right=False, left=False, bottom=False)
    values = biallelic_snps_mutations#[::downsample]
    with np.errstate(over='ignore'):
        # binned_statistic generates an annoying overflow warning which we can ignore
        #y1, _, _ = scipy.stats.binned_statistic(x[::downsample], values, statistic=ti_tv, bins=bins)
        y1, _, _ = scipy.stats.binned_statistic(x, values, statistic=ti_tv, bins=bins)
    bx = (bins[1:] + bins[:-1]) / 2
    ax2.plot(bx, y1, color='k')
    ax2.set_ylabel('Ti/Tv')
    ax2.set_ylim(0.6, 1.8)
    fig.savefig('%s.ti_tv_vs_%s.png' % (metadataObj.prefix, xlabel), format="png")

def get_missingness_counter_by_label(gts, is_var, is_snp):
    missingness_counter_by_label = {}
    missingness_rows = gts.count_missing(axis=1).compute()
    missingness_counter_by_label['variant SNP'] = collections.Counter(missingness_rows[(is_snp & is_var)])
    missingness_counter_by_label['variant nonSNP'] = collections.Counter(missingness_rows[(~is_snp & is_var)])
    missingness_counter_by_label['monomorphic SNP'] = collections.Counter(missingness_rows[(is_snp & ~is_var)])
    missingness_counter_by_label['monomorphic nonSNP'] = collections.Counter(missingness_rows[(~is_snp & ~is_var)])
    return missingness_counter_by_label

def get_variant_counter_by_idx(reflen, pos, is_snp):
    up_nonsnp_pos = pos[~is_snp]
    down_nonsnp_pos = np.add(pos[~is_snp], reflen[~is_snp]) - 1 # see reason_for_minus_1_in_indels/
    idxs = np.concatenate((np.negative(np.arange(1, MAX_NON_SNP_DISTANCE+1)), np.arange(1, MAX_NON_SNP_DISTANCE+1)))
    variant_type_counter_by_idx = collections.defaultdict(collections.Counter)
    for idx in idxs:
        if idx > 0:
            nonsnp_pos = down_nonsnp_pos
        else:
            nonsnp_pos = up_nonsnp_pos
        focus_idx = np.add(nonsnp_pos, idx)
        snp_idx_pos = np.intersect1d(focus_idx, pos[is_snp])
        variant_type_counter_by_idx[idx]['snps'] = len(snp_idx_pos)
        variant_type_counter_by_idx[idx]['sites'] = len(focus_idx)
    return variant_type_counter_by_idx

class MetadataObj(object):
    '''Stores VCF metadata'''
    def __init__(self, dataset, prefix):
        self.seq_ids = dataset['/seq_ids']
        self.seq_lengths = dataset['/seq_lengths']
        self.pop_ids = dataset['/pop_ids']
        self.pop_ids_order = sorted(set(self.pop_ids))
        self.sample_ids = dataset['/sample_ids']
        self.sample_ids_by_pop_id = self._get_sample_ids_by_pop_id()
        self.prefix = prefix

    def _get_sample_ids_by_pop_id(self):
        sample_ids_by_pop_id = collections.defaultdict(list)
        sample_ids_by_pop_id['all'] = list(range(len(self.sample_ids)))
        for sample_id, pop_id in zip(range(len(self.sample_ids)), self.pop_ids):
            sample_ids_by_pop_id[pop_id].append(sample_id)
        return sample_ids_by_pop_id

    def yield_seq_tuple(self):
        for seq_id, seq_length in zip(self.seq_ids, self.seq_lengths):
            yield (seq_id, seq_length)

def get_sample_counts_by_gt(gts, metadataObj, is_snp):
    #sample_counts_by_gt = collections.defaultdict(lambda: np.zeros(len(metadataObj.sample_ids)))
    sample_counts_by_gt = {}
    sample_counts_by_gt['TOTAL'] = np.full(len(metadataObj.sample_ids), len(gts[is_snp]))
    sample_counts_by_gt['MISS'] = gts[is_snp].count_missing(axis=0).compute()[:]
    sample_counts_by_gt['HET'] = gts[is_snp].count_het(axis=0).compute()[:] 
    sample_counts_by_gt['HOMALT'] = gts[is_snp].count_hom_alt(axis=0).compute()[:]
    sample_counts_by_gt['HOMREF'] = gts[is_snp].count_hom_ref(axis=0).compute()[:]
    return sample_counts_by_gt

def get_snp_numalt_counter(numalts, is_snp):
    allelic_labels_by_numalt = {
        1: 'biallelic', 
        2: 'triallelic', 
        3: 'tetrallelic'
    }
    snp_numalt_counter = collections.Counter(
        {
        allelic_labels_by_numalt[numalt]: count 
            for numalt, count in collections.Counter(numalts[is_snp]).items()
        }
    )
    return snp_numalt_counter

def get_sample_snp_dp_counter_by_sample_id(sample_snp_dps, metadataObj):
    sample_snp_dp_counter_by_sample_id = {}
    for sample_idx, sample_id in enumerate(metadataObj.sample_ids):
        sample_snp_dp_counter_by_sample_id[sample_id] = collections.Counter(sample_snp_dps[sample_idx][:])
    return sample_snp_dp_counter_by_sample_id

class MetricObj(object):
    """Stores metrics for each sequence"""
    def __init__(self, seq_id, seq_length):
        self.seq_id = seq_id
        self.seq_length = seq_length
        self.valid = False

        self.missingness_counter_by_label = None
        self.variant_type_counter_by_idx = None
        self.sample_counts_by_gt = None
        self.snp_numalt_counter = None
        self.sample_snp_dp_counter_by_sample_id = None
        self.snp_densities = None

        self.variant_counter = collections.Counter()
        self.biallelic_snps_count = None
        self.biallelic_snps_seg_count_by_pop_id = {}
        self.biallelic_snps_QUAL_DPs = None
        self.biallelic_snps_mutations = None
        self.biallelic_snp_singletons_count = None


    def collect_metrics(self, dataset, metadataObj, indels_flag):
        pos = dataset[self.seq_id]['variants']['POS'][:] # numpy.ndarray
        gts = allel.GenotypeDaskArray(dataset[self.seq_id]['calldata']['GT']) # allel.model.dask.GenotypeDaskArray
        acs = gts.count_alleles(max_allele=3).compute() # allel.model.ndarray.AlleleCountsArray
        is_snp = np.array(dataset[self.seq_id]['variants']['is_snp']) # zarr.core.array ; len(REF) == 1 && len(ALT) == 1 (excludes SNP+other)
        is_var = acs.is_variant() # numpy.ndarray; AlternateCall >= 1
        # Missingness of SNP/nonSNP vars/invars (works)
        self.missingness_counter_by_label = get_missingness_counter_by_label(gts, is_var, is_snp)
        # snps numalts
        numalts = dataset[self.seq_id]['variants']['numalt'][:]
        self.snp_numalt_counter = get_snp_numalt_counter(numalts, is_snp)
        # variant_type_counter_by_idx
        if indels_flag and np.any(~is_snp & is_var):
            reflen = np.frompyfunc(len, 1, 1)(dataset[self.seq_id]['variants']['REF'][:])
            self.variant_type_counter_by_idx = get_variant_counter_by_idx(reflen, pos, is_snp)
        # sample_counts_by_gt
        self.sample_counts_by_gt = get_sample_counts_by_gt(gts, metadataObj, is_snp)
        # sample_snp_dp_counter_by_sample_id
        # https://matplotlib.org/3.1.0/gallery/statistics/multiple_histograms_side_by_side.html#sphx-glr-gallery-statistics-multiple-histograms-side-by-side-py
        sample_snp_dps = dataset[self.seq_id]['calldata']['DP'][:][is_snp]
        self.sample_snp_dp_counter_by_sample_id = get_sample_snp_dp_counter_by_sample_id(sample_snp_dps, metadataObj)
        # SNP density over windows
        counts, windows = allel.windowed_count(pos[:][is_snp], size=WINDOWSIZE, start=1, stop=self.seq_length) # seq_length
        self.snp_densities = np.round(counts / WINDOWSIZE, 4)
        # biallelic SNPs
        is_biallelic = acs.is_biallelic()[:]
        is_biallelic_snp = (is_biallelic & is_snp)
        self.biallelic_snp_singletons_count = np.count_nonzero((acs.max_allele() == 1) & acs.is_singleton(1))
        # TS/TV
        biallelic_snps_REF = np.array(dataset[self.seq_id]['variants']['REF'][:][is_biallelic_snp], dtype="|S2")
        biallelic_snps_ALT = np.array(dataset[self.seq_id]['variants']['ALT'][:, 0][is_biallelic_snp], dtype="|S2")
        
        biallelic_snps_DP = dataset[self.seq_id]['variants']['DP'][:][is_biallelic_snp]
        biallelic_snps_QUAL = dataset[self.seq_id]['variants']['QUAL'][:][is_biallelic_snp]
        biallelic_snps_acs = acs[is_biallelic_snp]
        self.biallelic_snps_mutations = np.char.add(biallelic_snps_REF, biallelic_snps_ALT)
        self.biallelic_snps_QUAL_DPs = biallelic_snps_QUAL / biallelic_snps_DP
        self.biallelic_snps_count = len(biallelic_snps_acs)
        biallelic_snps_gts = gts[is_biallelic_snp].compute()
        biallelic_snps_allelecounts_subpops = biallelic_snps_gts.count_alleles_subpops(metadataObj.sample_ids_by_pop_id, max_allele=1) # max_allele=1, otherwise error in allel.stats.sf._check_ac_n()
        self.biallelic_snps_seg_count_by_pop_id = {pop_id: biallelic_snps_allelecounts.count_segregating() for pop_id, biallelic_snps_allelecounts in biallelic_snps_allelecounts_subpops.items()}
        is_segregating_biallelic_snp = biallelic_snps_allelecounts_subpops['all'].is_segregating()[:]
        self.segregating_biallelic_snp_acs_by_pop_id = {pop_id: biallelic_snps_allelecounts_subpops[pop_id][is_segregating_biallelic_snp] for pop_id in metadataObj.pop_ids_order}


def get_metricObjs(dataset, metadataObj, indels_flag):
    metricObjs = []
    fails = []
    for seq_id, seq_length in tqdm(metadataObj.yield_seq_tuple(), total=len(metadataObj.seq_ids), desc="[%] ", ncols=100):
        metricObj = MetricObj(seq_id, seq_length)
        metricObj.valid = (seq_id in dataset)
        if metricObj.valid:
            metricObj.collect_metrics(dataset, metadataObj, indels_flag)
            metricObjs.append(metricObj)
        else:
            fails.append(seq_id)
    for seq_id in fails:
        print("[-] No variation data found for Seq ID %r" % seq_id)
    return metricObjs

def combine_missingness_counter_by_label(metricObjs):
    missingness_counter_by_label = collections.defaultdict(collections.Counter)
    for metricObj in metricObjs:
        #print(metricObj.seq_id)
        for label, counter in metricObj.missingness_counter_by_label.items():
            missingness_counter_by_label[label] += counter
            #print(label, counter)
    return missingness_counter_by_label

def combine_snp_numalt(metricObjs):
    return sum([metricObj.snp_numalt_counter for metricObj in metricObjs], collections.Counter())

def combine_variant_type_counter_by_idx(metricObjs):
    variant_type_counter_by_idx = collections.defaultdict(collections.Counter)
    for metricObj in metricObjs:
        for idx, counter in metricObj.variant_type_counter_by_idx.items():
            variant_type_counter_by_idx[idx] += counter
    return variant_type_counter_by_idx

def combine_sample_counts_by_gt(metricObjs):
    sample_counts_by_gt = {}
    for gt_label in GT_LABELS:
        sample_counts_by_gt[gt_label] = np.sum(np.array([metricObj.sample_counts_by_gt[gt_label] for metricObj in metricObjs]), axis=0)
    return sample_counts_by_gt

def plot_ti_tv_data(metricObjs, metadataObj):
    biallelic_snps_QUAL_DPs = np.concatenate([metricObj.biallelic_snps_QUAL_DPs for metricObj in metricObjs])
    biallelic_snps_mutations = np.concatenate([metricObj.biallelic_snps_mutations for metricObj in metricObjs])
    plot_ti_tv(biallelic_snps_QUAL_DPs, biallelic_snps_mutations, 'QUAL_DP', metadataObj, bins=np.arange(0, 60, 1))

def combine_segregating_biallelic_snp_acs_by_pop_id(metricObjs, metadataObj):
    segregating_biallelic_snp_acs_by_pop_id = {}
    for metricObj in metricObjs:
        for pop_id in metadataObj.pop_ids_order:
            if not pop_id in segregating_biallelic_snp_acs_by_pop_id:
                segregating_biallelic_snp_acs_by_pop_id[pop_id] = metricObj.segregating_biallelic_snp_acs_by_pop_id[pop_id]
            else:
                segregating_biallelic_snp_acs_by_pop_id[pop_id].concatenate(metricObj.segregating_biallelic_snp_acs_by_pop_id[pop_id])
    return segregating_biallelic_snp_acs_by_pop_id

def generate_results(metadataObj, metricObjs):
    # summarise data
    missingness_counter_by_label = combine_missingness_counter_by_label(metricObjs)
    plot_missingness_bars(missingness_counter_by_label, metadataObj)  
    snp_numalt_counter = combine_snp_numalt(metricObjs)
    plot_snp_numalt(snp_numalt_counter, metadataObj)
    if args['--indels']:
        variant_type_counter_by_idx = combine_variant_type_counter_by_idx(metricObjs)
        plot_indel_bars(variant_type_counter_by_idx, metadataObj)
    sample_counts_by_gt = combine_sample_counts_by_gt(metricObjs)
    plot_sample_counts_by_gt(sample_counts_by_gt, metadataObj)
    plot_ti_tv_data(metricObjs, metadataObj)
    segregating_biallelic_snp_acs_by_pop_id = combine_segregating_biallelic_snp_acs_by_pop_id(metricObjs, metadataObj)
    plot_sfs(segregating_biallelic_snp_acs_by_pop_id, metadataObj)
    # # plot_biallelic_heatmap(biallelic_snps_QA_QR, biallelic_snps_QUAL_DPs, 'QA/QR', 'QUAL/DP', "biallelic_snp.QA_QR_vs_QUAL_DP")
    # # plot_biallelic_heatmap(biallelic_snps_QUAL_DPs, biallelic_snps_MQMRs, 'QUAL/DP', 'MQMR', "biallelic_snp.QUAL_DP_vs_MQMR")
    # # plot_biallelic_heatmap(biallelic_snps_DPs, biallelic_snps_QUALs, 'DP', 'QUAL', "biallelic_snp.DP_vs_QUAL", vmin=None, vmax=None)
    # # plot_biallelic_heatmap(biallelic_snps_DPs, biallelic_snps_MQMs, 'DP', 'MQM', "biallelic_snp.DP_vs_MQM")
    # plot_snp_density(snp_densities, WINDOWSIZE)

def process_zarr(zarr_f, indels_flag, prefix):
    dataset = get_zarr_callset(zarr_f)
    print(dataset.tree())
    metadataObj = MetadataObj(dataset, prefix)
    metricObjs = get_metricObjs(dataset, metadataObj, indels_flag)
    return metadataObj, metricObjs

if __name__ == "__main__":
    '''
    -> GFF/ZARR : https://www.biostars.org/p/335077/
    '''
    start_time = timer()
    args = docopt(__doc__)
    print(args)
    # MAIN
    MAX_NON_SNP_DISTANCE = 25
    WINDOWSIZE = 10000
    GT_LABELS = ['TOTAL', 'MISS', 'HET', 'HOMALT', 'HOMREF']
    metadataObj, metricObjs = process_zarr(args['--zarr'], args['--indels'], args['--prefix'])
    generate_results(metadataObj, metricObjs)