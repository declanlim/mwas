import sys
import pickle
if ('--debug' in sys.argv):
	import pdb
	pdb.set_trace()

with open('/home/ubuntu/s3_downloads/bioprojects/PRJEB28138.pickle', 'rb') as f:
	bioproj_df = pickle.load(f)

print("loaded bioproject")

bioproj_df.to_csv('pickle_out.csv', index=False)
