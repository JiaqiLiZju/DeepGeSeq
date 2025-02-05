import pandas as pd

def read_vcf(filename):
	"""Read a VCF file into a pandas DataFrame

	This function takes in the name of a file that is VCF formatted and returns
	a pandas DataFrame with the comments filtered out. This will only return the
	columns that are most commonly provided in VCF files.


	Parameters
	----------
	filename: str


	Returns
	-------
	vcf: pandas.DataFrame
		A pandas DataFrame containing the rows.
	"""

	names = ["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", 
		"FORMAT"]
	dtypes = {name: str for name in names}
	dtypes['POS'] = int

	vcf = pd.read_csv(filename, delimiter='\t', comment='#', names=names, 
		dtype=dtypes, usecols=range(9))
	return vcf

