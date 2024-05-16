import sys
import pickle
if ('--debug' in sys.argv):
	import pdb
	pdb.set_trace()

with open('/home/ubuntu/s3_downloads/bioprojects/PRJDB10241.pickle', 'rb') as f:
	bioproj_df = pickle.load(f)

print("loaded bioproject")

bioproj_df.to_csv('pickle_out.csv', index=False)
