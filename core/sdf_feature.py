import os
import torch
import numpy as np
import tensorflow.compat.v1 as tf
from core.features import Features
from core.modules.model import *
from utils.utils import *
from core.data import normal_points

RESULT_DIR = 'Save_PC_Result'

class SDF(object):
    def __init__(self, image_size, BS, POINT_NUM, ckpt_dir, save_idx, save_pc):
        self.image_size = image_size
        self.BS = BS
        self.POINT_NUM = POINT_NUM
        self.ckpt_dir = ckpt_dir
        self.save_idx = save_idx
        self.save_pc = save_pc
        self.points_target = tf.placeholder(tf.float32, shape=[self.BS, self.POINT_NUM, 3])
        self.input_points_3d = tf.placeholder(tf.float32, shape=[self.BS, self.POINT_NUM, 3])
        self.feature = load_encoder(self.points_target, self.POINT_NUM, self.BS)
        self.point_feature = tf.tile(tf.expand_dims(self.feature,1),[1,self.POINT_NUM,1])
        self.sdf, self.grad_norm = local_decoder(self.point_feature, self.input_points_3d)
        self.g_points = self.input_points_3d - self.sdf * self.grad_norm
        self.loss_g = tf.reduce_mean(tf.norm((self.points_target - self.g_points), axis=-1))

        # create tensorflow configuration
        config = tf.ConfigProto()
        config.allow_soft_placement = False
        config.gpu_options.per_process_gpu_memory_fraction = 0.9  # allocate 70% memory of gpu
        config.gpu_options.allow_growth = True

        # Launch the session
        self.sess = tf.Session(config=config) 
        self.sess.run(tf.global_variables_initializer())

        # load check point
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(self.ckpt_dir).all_model_checkpoint_paths
        self.saver.restore(self.sess, checkpoint[self.save_idx])
        self.log_file = None
        print('Pretrain Model:', checkpoint[self.save_idx])

    def get_feature(self, sample, data_id, split, output_dir):
        total_feature = []
        points_all = sample[3]
        points_tran_all = sample[5]
        loss_total = 0.0
        valid_patch_num = 0
        for patch in range(len(points_all)):

            if points_all[patch].shape[1] != self.POINT_NUM:
                print('Error!', points_all[patch].shape)
                continue

            valid_patch_num += 1
            points = points_all[patch].reshape(-1, self.POINT_NUM, 3)
            G_points = []
            point = points[0,:].reshape(self.BS, self.POINT_NUM, 3)
            loss_c, feature_c, g_points_c, = self.sess.run([self.loss_g, self.feature, self.g_points], feed_dict={self.points_target:point, self.input_points_3d:point})
            
            loss_total += loss_c
            G_points.extend(g_points_c)
            total_feature.extend(feature_c)
            
            # Debug
            if split == 'test':
                self.log_file.write('Patch: %2d, Loss: %04f\n' % (patch, loss_c))

            # store the g_point (feature) of data (For Debug)
            if self.save_pc and split == 'test' and data_id <= 4:
                #origin_pc = points
                #predict_pc = np.array(G_points).reshape(-1,3)
                origin_pc = denormal_points(points, points_tran_all[patch])
                predict_pc = denormal_points(torch.tensor(np.array(G_points)),points_tran_all[patch])
                save_pc(origin_pc, os.path.join(output_dir, RESULT_DIR, split), '_' + str(data_id) + '_origin_patch' + str(patch))
                save_pc(predict_pc, os.path.join(output_dir, RESULT_DIR, split), '_' + str(data_id) + '_predict_patch' + str(patch))

        if split == 'test':
            self.log_file.write('\nAverage Loss: %04f' % (loss_total / valid_patch_num))
            self.log_file.write('\n')
        return total_feature

    def get_score_map(self, feature, sample, data_id, output_dir):

        points_all = sample[3]
        idx_all = sample[4]
        points_tran_all = sample[5]
        # sample_points = create_plane_sample_point(side_length=0.75,part_size=50).reshape(-1, self.POINT_NUM, 3)

        sample_sdf_total = 0.0
        valid_patch_num = 0
        s_map = [-1 for _ in range(self.image_size*self.image_size)]
        
        for patch in range(len(points_all)):

            if points_all[patch].shape[1] != self.POINT_NUM:
                print('Error!',points_all[patch].shape)
                continue

            valid_patch_num += 1
            points = points_all[patch].reshape(-1, self.POINT_NUM, 3)
            indices = idx_all[patch].reshape(-1, self.POINT_NUM)

            G_points = []
            point_feature = np.tile(np.expand_dims(feature[patch], 0), [1, self.POINT_NUM, 1])
            point = points[0,:].reshape(self.BS, self.POINT_NUM, 3)
            index = indices[0].reshape(self.POINT_NUM)
            sdf_c , g_points_c= self.sess.run([self.sdf, self.g_points], feed_dict={self.point_feature:point_feature, self.points_target:point, self.input_points_3d:point})
            
            G_points.extend(g_points_c)
            sdf_c = np.abs(np.asarray(sdf_c).reshape(-1))
            index = index.detach().cpu().numpy()


            #### Test Sampling Query point from space ####
            S_G_points = []
            s_sdf_total = 0.0
            #sample_points = random_sample_points(point, 10).reshape(-1,3)
            #sample_points , _ = normal_points(sample_points, True)
            #sample_points = sample_points.reshape(-1, self.POINT_NUM, 3)
            #for rt in range(sample_points.shape[0]):
            #    sample_point = sample_points[rt,:,:].reshape(1, self.POINT_NUM, 3)
            #    sg_points_c =  self.sess.run([self.g_points], feed_dict={self.point_feature:point_feature,  self.points_target:point, self.input_points_3d:sample_point})
            #    sample_point = np.array(sg_points_c).reshape(1, self.POINT_NUM, 3)
            #    s_sdf_c, sg_points_c = self.sess.run([self.sdf, self.g_points], feed_dict={self.point_feature:point_feature, self.input_points_3d:sample_point})
            #    sample_point = np.array(sg_points_c).reshape(1, self.POINT_NUM, 3)
            #    s_sdf_c, sg_points_c = self.sess.run([self.sdf, self.g_points], feed_dict={self.point_feature:point_feature, self.input_points_3d:sample_point})
                
            #    S_G_points.extend(sg_points_c)
            #    s_sdf_total += np.sum(s_sdf_c)
            
            #self.log_file.write('Patch: %2d, NN Sampled SDF: %04f\n\n' % (patch, (s_sdf_total / sample_points.shape[0])))
            #sample_sdf_total += (s_sdf_total / sample_points.shape[0])
      
            for L in range(sdf_c.shape[0]):
                if(s_map[index[L]] == -1):
                    s_map[index[L]] = sdf_c[L]

                elif(s_map[index[L]] > sdf_c[L]):
                    s_map[index[L]] = sdf_c[L]

            # store the g_point (feature) of data (For Debug)
            if self.save_pc and data_id <= 4:
                #predict_pc = np.array(G_points).reshape(-1,3)
                #sample_predict_pc = np.array(S_G_points).reshape(-1,3)
                #sample_pc = np.array(sample_points).reshape(-1,3)
                predict_pc = denormal_points(torch.tensor(np.array(S_G_points)),points_tran_all[patch])
                #sample_pc = denormal_points(torch.tensor(np.array(sample_points)),points_tran_all[patch])
                save_pc(predict_pc, os.path.join(output_dir, RESULT_DIR, 'test'), '_' + str(data_id) + '_NN_predict_patch' + str(patch))

        s_map = np.asarray(s_map)
        s_map[s_map == -1] = 0.0 
        s_map = s_map.reshape(1, 1, self.image_size, self.image_size)
        s_map = torch.tensor(s_map)
        self.log_file.write('\nAverage NN Sampled SDF: %04f' % (sample_sdf_total / valid_patch_num))
        self.log_file.write('\n----------------------------------------------------------------\n')
        return s_map

class SDFFeatures(Features):
    def __init__(self, image_size=224, f_coreset=0.5, coreset_eps=1):
        super().__init__(image_size, f_coreset, coreset_eps)
        
    def add_sample_to_mem_bank(self, sdf, sample, train_data_id, output_dir):
        feature = sdf.get_feature(sample, train_data_id, 'train', output_dir)
        feature = np.array(feature)
        patch = torch.tensor(feature)
        self.patch_lib.append(patch)

    def predict(self, sdf, sample, mask, label, test_data_id, output_dir, cls=None):
        feature = sdf.get_feature(sample, test_data_id, 'test', output_dir)
        feature = np.array(feature)
        feature = torch.tensor(feature)
        NN_feature, sdf_s = self.find_NNFeature(feature)
        score_map = sdf.get_score_map(NN_feature, sample, test_data_id, output_dir)
        self.record_score(score_map, sdf_s, label, mask)