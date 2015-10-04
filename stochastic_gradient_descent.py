import pandas as pd

from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier

clf = SGDClassifier(loss = "log", n_iter = 1, shuffle = True, random_state = 42)
all_classes = [0,1]

def manipulate_train_data(list_of_dicts):
	for d in list_of_dicts:
		for key,value in d.iteritems():
			if type(value) == str:
				d[key] = int(value,16)
	return list_of_dicts

def read_data_file(filename, no_of_rows):
	df = pd.read_csv(filename, nrows = no_of_rows)
	return df

def perform_hashing(data_frame):
	raw_X_train = [dict(row[1]) for row in data_frame.iterrows()]
	fh = FeatureHasher(n_features = 2**20)
	train_data = manipulate_train_data(raw_X_train)
	X_enc_train = fh.transform(train_data)
	return X_enc_train


def train_classifier():
	clf.partial_fit(classes = all_classes)

if __name__ == '__main__':

	nrows = 1000
	X_train = read_data_file('train',nrows)
	X_enc_train = perform_hashing(X_train)

	train_classifier()
	print 'done'