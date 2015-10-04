from itertools import islice
from sklearn import tree, preprocessing
import numpy as np
from StringIO import StringIO
from inspect import getmembers
import struct

clf = tree.DecisionTreeClassifier()
min_max_scaler = preprocessing.MinMaxScaler(feature_range = (0,100))

def manipulate_train_data(data_row):
	line_array = data_row.strip().split(',')
	#trainingx_portion_1 = int(line_array[0])
	trainingx_portion_2 = line_array[2:5]
	trainingx_portion_3 = line_array[5:14]
	trainingx_portion_4 = line_array[14:]

	temp_list = []

	#temp_list.append(trainingx_portion_1)

	for i in xrange(0,len(trainingx_portion_2)):
		trainingx_portion_2[i] = float(trainingx_portion_2[i])
		temp_list.append(trainingx_portion_2[i])
	for i in xrange(0,len(trainingx_portion_3)):
		trainingx_portion_3[i] = struct.unpack(">f", struct.pack(">I",int(trainingx_portion_3[i],16)))[0]
		#trainingx_portion_3[i] = int(trainingx_portion_3[i], 16)
		temp_list.append(trainingx_portion_3[i])
	for i in xrange(0,len(trainingx_portion_4)):
		trainingx_portion_4[i] = float(trainingx_portion_4[i])
		temp_list.append(trainingx_portion_4[i])	

	#print temp_list
	if np.isnan(temp_list).any():
		return 0
	
	temp_list_nomalised = min_max_scaler.fit_transform(temp_list)	
	return (temp_list_nomalised,  int(line_array[1]))
			

def manipulate_test_data(data_row):
	line_array = data_row.strip().split(',')
	testingx_portion1 = line_array[1:4]
	testingx_portion2 = line_array[4:13]
	testingx_portion3 = line_array[13:]

	temp_list = []
	for i in xrange(0,len(testingx_portion1)):
		testingx_portion1[i] = float(testingx_portion1[i])
		temp_list.append(testingx_portion1[i])

	for i in xrange(0,len(testingx_portion2)):
		testingx_portion2[i] = struct.unpack(">f", struct.pack(">I",int(testingx_portion2[i],16)))[0]
		#testingx_portion2[i] = int(testingx_portion2[i], 16)
		temp_list.append(testingx_portion2[i])

	for i in xrange(0,len(testingx_portion3)):
		testingx_portion3[i] = float(testingx_portion3[i])
		temp_list.append(testingx_portion3[i])

	if np.isnan(temp_list).any():
	 	return 0
	#print temp_list
	temp_list_nomalised = min_max_scaler.fit_transform(temp_list)
	return temp_list_nomalised

def read_data(file_name, size_of_file):
	count = 0
	chunk_size = 10
	header = True
	with open(file_name) as f:
		while count < (size_of_file * chunk_size):
			data_listx = []
			data_listy = []
			if header:
				next = list(islice(f,1))
				header = False
				continue
			count = count + 1
			next = list(islice(f,chunk_size))
			if not next:
				break
			for i in xrange(0,chunk_size):
				if file_name == 'train':
					a = manipulate_train_data(next[i])
					if a!=0:
						data_listx.append(a[0])
						data_listy.append(a[1])
				else:
					a = manipulate_test_data(next[i])
					if a!= 0:
						data_listx.append(a)
			if file_name == 'train':
				#print data_listx
				yield (data_listx, data_listy)
			else:
				yield data_listx 	


def read_cv_data(size_of_file, train_size):
	chunk_size = 10
	start_point = (train_size * chunk_size) + 2
	end_point = start_point + chunk_size
	count  = 0
	with open('train') as f:
		while count < size_of_file:
			cvx_list = []
			cvy_list = []
			next = list(islice(f,start_point,end_point))

			for i in xrange(0,chunk_size):
				a = manipulate_train_data(next[i])
				if a != 0:
					cvx_list.append(a[0])
					cvy_list.append(a[1])
			start_point = end_point
			end_point = end_point + chunk_size
			count = count + 1
			yield (cvx_list, cvy_list)


def train_classifier(chunk):
	clf.fit(chunk[0], chunk[1])

def cv_classifier(chunk):
	answer = clf.predict(chunk[0])
	correct = 0
	for i,val in enumerate(answer):
		if val == chunk[1][i]:
			correct = correct + 1
	return correct

def test_classifier(chunk):
	print clf.predict(chunk)

def plot_trees():
	"""
	Retrieving the content of the tree as a string
	out = StringIO()
	tree.export_graphviz(clf, out_file = out)
	print out.getvalue()
	out.close()  """
	out_file = tree.export_graphviz(clf, out_file = 'out.dot')



if __name__ == '__main__':

	total_correct = 0
	total_wrong = 0

	size_of_train_data_to_read = 1000			#no. of blocks

	for chunk in read_data('train', size_of_train_data_to_read):
		train_classifier(chunk)

	plot_trees()

	size_of_cv_data_to_read = 1000
	for chunk in read_cv_data(size_of_cv_data_to_read, size_of_train_data_to_read):
		correct = cv_classifier(chunk)
		total_correct = total_correct + correct

	print "Accuracy: ", float(total_correct)/float(size_of_cv_data_to_read * 10)

	size_of_test_data_to_read = 10
	for chunk in read_data('test', size_of_test_data_to_read):
		test_classifier(chunk)

	print "\n\ndone"


"""
id,click,hour,C1,banner_pos,site_id,site_domain,site_category,app_id,app_domain,app_category,device_id,
device_ip,device_model,device_type,device_conn_type,C14,C15,C16,C17,C18,C19,C20,C21

[7336555168587348325, 14102100, 1005, 0, 1413108638, 3351916808, 1048658224, 3970769798, 2013391065, 131587874, 
2845778250, 772675055, 2119908332, 1, 0, 20366, 320, 50, 2333, 0, 39, -1, 157]

"""