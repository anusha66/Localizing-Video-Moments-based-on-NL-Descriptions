
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import numpy as np
import tensorflow as tf
import pdb
import i3d
import os
import numpy as np
import h5py
import glob
import re

rn_path = 'i3d_features'
rn_path2 = 'i3d_predictions'

_IMAGE_SIZE = 224
_NUM_CLASSES = 400

_SAMPLE_VIDEO_FRAMES = 40
_SAMPLE_PATHS = {
    'rgb': 'data/v_CricketShot_g04_c01_rgb.npy',
    'flow': 'data/v_CricketShot_g04_c01_flow.npy',
}

_CHECKPOINT_PATHS = {
    'rgb': 'data/checkpoints/rgb_scratch/model.ckpt',
    'flow': 'data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': 'data/checkpoints/flow_imagenet/model.ckpt',
}
_LABEL_MAP_PATH = 'data/label_map.txt'

FLAGS = tf.flags.FLAGS
#tf.flags.DEFINE_string('eval_type', 'rgb')

tf.flags.DEFINE_string('eval_type', 'rgb', 'rgb, flow, or joint')
tf.flags.DEFINE_boolean('imagenet_pretrained', True, '')
#with h5py.File("fc_i3d_max_pred.hdf5", "w") as f_pred:

def evaluate_sample():
  # with tf.device('/device:GPU:0'):
  NUMM = 1
 #f_pred = h5py.File("fc_i3d_max_pred_xaa.hdf5", "w")
 #with h5py.File("fc_i3d_max_fea_xaa.hdf5", "w") as f_fea:
  dl_path = '../VIDEO_DIRECTORY/'
  data_file = 'xaa'
  fff = open(data_file, 'r')
  for vname in fff:
  #for vname in glob.glob('../VIDEO_DIRECTORY/*'):
   vid_name = re.sub(r"\.mp4$", "", vname).strip()
   #vid_name ='13549533@N02_3355384254_1eef134b0c.mov'
   print (vid_name)
   rn_filename = os.path.join(rn_path, vid_name)
   rn_filename2 = os.path.join(rn_path2, vid_name)
   if os.path.exists(rn_filename + '.npz') and os.path.exists(rn_filename2 + '.npz'):
      continue

   feat_file = os.path.join('../video_frames', vid_name + '.npz')
   feat = np.load(feat_file)['arr_0']
   feat = ((feat - 0)/(255-0))*(1 + 1) -1
   all_fea = []
   all_pred = []
   check = math.floor(feat.shape[1]/40)
   
   for moment in range(min(check,6)):

    curr_moment = moment * 40
    next_moment = curr_moment + 40
    rgb_input_sample = feat[:,curr_moment:next_moment]
    tf.logging.set_verbosity(tf.logging.INFO)
    eval_type = FLAGS.eval_type
    imagenet_pretrained = FLAGS.imagenet_pretrained

    if eval_type not in ['rgb', 'flow', 'joint']:
      raise ValueError('Bad `eval_type`, must be one of rgb, flow, joint')

    kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH)]

    if eval_type in ['rgb', 'joint']:
    # RGB input has 3 channels.
     rgb_input = tf.placeholder(
        tf.float32,
        shape=(1, rgb_input_sample.shape[1], _IMAGE_SIZE, _IMAGE_SIZE, 3))
     with tf.variable_scope('RGB'):
      rgb_model = i3d.InceptionI3d(
          _NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
      rgb_logits, my_fea_temp2, n_ = rgb_model(
          rgb_input, is_training=False, dropout_keep_prob=1.0)
     rgb_variable_map = {}
     for variable in tf.global_variables():
      if variable.name.split('/')[0] == 'RGB':
        rgb_variable_map[variable.name.replace(':0', '')] = variable
     rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)	
    if eval_type in ['flow', 'joint']:
    # Flow input has only 2 channels.
     flow_input = tf.placeholder(
        tf.float32,
        shape=(1, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 2))
     with tf.variable_scope('Flow'):
      flow_model = i3d.InceptionI3d(
          _NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
      flow_logits, _ = flow_model(
          flow_input, is_training=False, dropout_keep_prob=1.0)
     flow_variable_map = {}
     for variable in tf.global_variables():
      if variable.name.split('/')[0] == 'Flow':
        flow_variable_map[variable.name.replace(':0', '')] = variable
     flow_saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)

    if eval_type == 'rgb':
     model_logits = rgb_logits
    elif eval_type == 'flow':
     model_logits = flow_logits
    else:
     model_logits = rgb_logits + flow_logits
    model_predictions = tf.nn.softmax(model_logits)
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
     tf.global_variables_initializer()
     feed_dict = {}
     if eval_type in ['rgb', 'joint']:
      if imagenet_pretrained:
        rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb_imagenet'])
      else:
        rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb'])
      tf.logging.info('RGB checkpoint restored')
      rgb_sample = rgb_input_sample
      #np.load(_SAMPLE_PATHS['rgb'])
      tf.logging.info('RGB data loaded, shape=%s', str(rgb_sample.shape))
      feed_dict[rgb_input] = rgb_sample

     if eval_type in ['flow', 'joint']:
      if imagenet_pretrained:
        flow_saver.restore(sess, _CHECKPOINT_PATHS['flow_imagenet'])
      else:
        flow_saver.restore(sess, _CHECKPOINT_PATHS['flow'])
      tf.logging.info('Flow checkpoint restored')
      flow_sample = np.load(_SAMPLE_PATHS['flow'])
      tf.logging.info('Flow data loaded, shape=%s', str(flow_sample.shape))
      feed_dict[flow_input] = flow_sample

     featuress, out_logits= sess.run(
        [my_fea_temp2, model_predictions],
        feed_dict=feed_dict)
     print("MOMENT", moment, out_logits[0].shape)
     all_pred.append(out_logits[0])
     all_fea.append(featuress[0])
     #pdb.set_trace()
     #print(out_logits[0])
    tf.reset_default_graph()
   if len(all_fea) < 6:
        diff = 6 - len(all_fea)
        for iii in range(diff):
           all_fea.append(np.zeros(1024))
           all_pred.append(np.zeros(400))

   #curr_moment = 0 * 40
   #next_moment =  40 * 6
   #rgb_input_sample = feat[:,curr_moment:next_moment]

   '''
   if len(all_pred) < 6:
        diff = 6 - len(all_pred)
        for iii in range(diff):
           all_pred.append(np.zeros(400))
   '''
   temp_fea = np.stack(all_fea) 
   temp_pred = np.stack(all_pred)
   np.savez(rn_filename, temp_fea)
   np.savez(rn_filename2, temp_pred)
   #f_fea.create_dataset(vid_name, data=temp_fea)
   #f_pred.create_dataset(vid_name, data=temp_pred)
   #print(len(all_fea)," NUMM ", NUMM)
   NUMM = NUMM + 1
   all_fea = []
   '''    
    out_logits = out_logits[0]

    out_predictions = out_predictions[0]
    sorted_indices = np.argsort(out_predictions)[::-1]

    print('Norm of logits: %f' % np.linalg.norm(out_logits))
    print('\nTop classes and probabilities')
    for index in sorted_indices[:20]:
      print(out_predictions[index], out_logits[index], kinetics_classes[index])
   '''
def main():
    dl_path = '../VIDEO_DIRECTORY/'
    data_file = 'xaa'
    with open(data_file, 'r') as file:
        for vname in file:
        #for vname in glob.glob('../VIDEO_DIRECTORY/*'):
            vid_name = re.sub(r"\.mp4$", "", vname[19:])
            #if vid_name == '25467267@N00_5451629458_1387d0c696.mov':
            #   continue
            print (vid_name)
            feat_file = os.path.join('../video_frames', vid_name + '.npz')
            feat = np.load(feat_file)['arr_0']
            for moment in range(6):
		
                curr_moment = moment * 40
                next_moment = curr_moment + 40
                
                tf.app.run(main=evaluate_sample, argv=[feat[:,curr_moment:next_moment]])
                print(all_fea) 
                #tf.app.run(evaluate_sample(feat[curr_moment:next_moment]))

if __name__ == '__main__':
	evaluate_sample()
