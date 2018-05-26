import os
import numpy as np
import h5py
import glob
import re
import pdb

def main():
    with h5py.File("fc_i3d_local_pred_left.hdf5", "w") as f:
        for vname in glob.glob('i3d_predictions_left/*'):

            vid_name = vname
            #print(vid_name)
            #re.sub(r"\.mp4$", "", vname[19:])
          
            #print (vid_name)
            #feat_file = os.path.join('i3d_local_predictions', vid_name)
            feat = np.load(vid_name)['arr_0']
            vid_name = re.sub(r"\.npz$", "", vid_name[21:])
            f.create_dataset(vid_name, data=feat)

if __name__=='__main__':
    main()
