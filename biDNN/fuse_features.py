import numpy as np
import h5py



def get_fused_data():
	print ('starting....')

	resnet = h5py.File('data/fc_resnet_max.hdf5', 'r')
	print ('resnet shape', np.array(list(resnet.values())).shape)

	flow = h5py.File('data/average_global_flow.h5', 'r')
	print ('flow shape', np.array(list(flow.values())).shape)

	flow_value_array = np.array(list(flow.values()))
	flow_value_array =np.delete(flow_value_array, 478, 0)
	print ('new flow shape', flow_value_array.shape)


	flow_keys = list(flow.keys())
	rn_keys = list(resnet.keys())


	flow_keys.remove('12090392@N02_13482799053_87ef417396.mov')

	flag = 1
	print ('sanity check for flow and rn keys')
	for i, flow_key in enumerate(flow_keys):
	    if rn_keys[i] != flow_key:
	        print (i, 'bachaaaaaao')
	        flag = 0

	if flag == 1:
		print ('all passed')


		print ('Getting rn features....')
		rn_features = []
		for array in list(resnet.values()):
		    array = np.array(array)
		    [rn_features.append(val) for val in array]
		    
		rn_features = np.array(rn_features)
		print (rn_features.shape)


		print ('Getting Flow features....')
		flow_features = []
		for array in flow_value_array:
		#     print (type(array))
		    [flow_features.append(val) for val in array]
		    
		flow_features = np.array(flow_features)
		print (flow_features.shape)


		print ('concatenating.....')
		fused_features = np.concatenate((rn_features, flow_features), axis=1)

		print ('fused features shape', fused_features.shape)

		return fused_features

