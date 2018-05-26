import os
import numpy as np
import h5py
import glob
import re

def main():
    with h5py.File("fc_i3d_global_pred.hdf5", "w") as f:
        for vname in glob.glob('../VIDEO_DIRECTORY/*'):

            vid_name = re.sub(r"\.mp4$", "", vname[19:])
          
            #print (vid_name)
            feat_file = os.path.join('i3d_global_predictions', vid_name + '.npz')
            feat = np.load(feat_file)['arr_0']

            f.create_dataset(vid_name, data=feat)

if __name__=='__main__':
    main()
