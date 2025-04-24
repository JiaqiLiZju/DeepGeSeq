"""
Description:
    This script parses the IGAP age at onset survival GWAS p-values
    (Huang et al., 2017) and creates 2 VCF files--one containing
    nominally significant SNPs (p-value < 0.05) and one containing
    nonsignificant SNPs (p-value > 0.50).

Usage:
    group_snps_by_pval.py <gwas-pvalues> <output-dir>
    group_snps_by_pval.py -h | --help

Options:
    -h --help        Show this screen.

    <gwas-pvalues>   The path to the IGAP p-values file.
    <output-dir>     The path to the desired output directory.
                     If it does not exist, it will be automatically
                     created for you.
"""
import os

from docopt import docopt
import pandas as pd


if __name__ == "__main__":
    arguments = docopt(
        __doc__,
        version="1.0")
    gwas_pvals_file = arguments["<gwas-pvalues>"]
    output_dir = arguments["<output-dir>"]
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    df = pd.read_csv(gwas_pvals_file, sep='\t')

    nominally_sig_fh = open(os.path.join(
        output_dir, "lt0.05.vcf"), 'w+')
    nonsig_fh = open(os.path.join(
        output_dir, "gt0.50.vcf"), 'w+')
    nominally_sig_fh.write("#CHROM\tPOS\tID\tREF\tALT")
    nonsig_fh.write("#CHROM\tPOS\tID\tREF\tALT")

    for row in df.itertuples():
        ref = row.effect_allele
        alt = row.other_allele
        if "rs" not in row.SNP_ID:
            variant = row.SNP_ID.split(':')[-1]
            if variant == 'D' or variant == 'I':
                continue
            variant = variant.split('_')
            if len(variant) == 1 and ref == 'i':
                ref, alt = '', variant[0]
            elif len(variant) == 1 and ref == 'd':
                ref, alt = variant[0], ''
            elif len(variant) == 1:
                continue
        if row.p_value < 0.05:
            nominally_sig_fh.write("chr{0}\t{1}\t{2}\t{3}\t{4}\n".format(
                row.chromosome, row.base_pair_location, row.SNP_ID, ref, alt))
        elif row.p_value > 0.50:
            nonsig_fh.write("chr{0}\t{1}\t{2}\t{3}\t{4}\n".format(
                row.chromosome, row.base_pair_location, row.SNP_ID, ref, alt))

    nominally_sig_fh.close()
    nonsig_fh.close()


