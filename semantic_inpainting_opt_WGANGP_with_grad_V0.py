
# coding: utf-8

# ## Optimize z for inpainting


import numpy as np
import tensorflow as tf
import os
import time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import h5py
from sklearn import preprocessing
from sklearn import decomposition
from skimage.measure import compare_ssim as ssim
from G_D_model import generator, discriminator
import pandas as pd
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
import sys


data_dir = './datasets_with_KL/'

## KL data
code_version = 1 
code_version = 2 
# code_version = 3 
# code_version = 4 
# code_version = 5 

repeat_num = np.int32(sys.argv[3])



# 提取训练数据与测试数据  
if data_dir == './datasets_with_KL/':
    output_dir = './semantic_inpainting_WGANGP_with_physics/outputs_with_KL_V{0}/'.format(code_version)
    output_model_dir = output_dir + 'saved_model/'
    output_figure_dir = output_dir + 'inpainting_figures/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_model_dir):
        os.makedirs(output_model_dir)
    if not os.path.exists(output_figure_dir):
        os.makedirs(output_figure_dir)
    
    with h5py.File(data_dir+'kle512_lhs1000_test.hdf5','r') as f:
        test_input = f['input'][:]
        test_output = f['output'][:]


elif data_dir == './datasets_with_nonGaussian/':
    if continuous_nonGaussian:
        output_dir = './semantic_inpainting_WGANGP_with_physics/outputs_with_contin_nonGaussian_V{0}/'.format(code_version)
        with h5py.File(data_dir+'continuous_nonGaussian_10000_train.hdf5','r') as f: 
            test_input = f['input'][:]
            test_output = f['output'][:]
    else:
        output_dir = './semantic_inpainting_WGANGP_with_physics/outputs_with_nonGaussian_V{0}/'.format(code_version)
        with h5py.File(data_dir+'channel_ng64_n4096_train.hdf5','r') as f:
            test_input = f['input'][:]
            test_output = f['output'][:]

    output_model_dir = output_dir + 'saved_model/'
    output_figure_dir = output_dir + 'inpainting_figures/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_model_dir):
        os.makedirs(output_model_dir)
    if not os.path.exists(output_figure_dir):
        os.makedirs(output_figure_dir)




## reformate the data shape to [N, H, W, C], C 为4，分别是(K, h, flux_x, flux_y)
imsize = test_input.shape[-1]
test_data = np.zeros((test_input.shape[0],imsize,imsize,4))
for i in range(test_input.shape[0]):
    if test_input[i,0,:,:].min() < 0: # 如果最小值小于0，就说明保存的是logk
        test_data[i,:,:,0] = test_input[i,0,:,:]
    else:
        test_data[i,:,:,0] = np.log(test_input[i,0,:,:])
    test_data[i,:,:,1] = test_output[i,0,:,:]  # h
    test_data[i,:,:,2] = test_output[i,1,:,:]  # flux_x
    test_data[i,:,:,3] = test_output[i,2,:,:]  # flux_y
print('test data shape: ',test_data.shape)

itera_num = 100000  ## 这个要跟训练WGANGP的iteration一致，用于restore保存的model
batch_size = 50
noise_dim = 100   ## 也需要与WGANGP训练一致
lr = 0.0002
epsilon = 1e-14
numta = 1e-1
LAMBDA = 10
channels = 4
output_dim = imsize * imsize * channels
decay_step = 5000
decay_rate = 0.9

Lx = 2
Ly = 2
Ncol = imsize
Nrow = imsize
delta = Lx/Ncol

x_dis_value = np.linspace(0,Lx,Ncol)
y_dis_value = np.linspace(0,Ly,Nrow)
X_dis,Y_dis = np.meshgrid(x_dis_value,y_dis_value)
xy_cor = np.concatenate((X_dis.reshape(-1,1),Y_dis.reshape(-1,1)),axis=1)


def reformulate_data(data): # the input data is [width * height, sample_num], output is [batch_size, width, height,1]
    sample_num = data.shape[1]
    data_refor = np.zeros((sample_num,imsize,imsize,1))
    for i in range(data.shape[1]): 
        data_refor[i,:,:,0] = data[:,i].reshape(imsize,imsize)
    return data_refor


if imsize == 64:
    def generator(noise):
        with tf.variable_scope('generator',reuse=tf.AUTO_REUSE):
            net = tf.layers.dense(noise,4*4*256,activation=tf.nn.leaky_relu,use_bias=False,name='dense_layer')
            net = tf.layers.batch_normalization(net,name='BN_1')

            net = tf.reshape(net,(-1,4,4,256))  ## [None,4,4,256]
            net = tf.layers.conv2d_transpose(net,128,kernel_size=(3,3),strides=(1,1),padding='same',
                                            use_bias=False,activation=tf.nn.relu,name='conv1_layer')
            net = tf.layers.batch_normalization(net,name='BN_2')  ## [None,4,4,128]

            net = tf.layers.conv2d_transpose(net,64,kernel_size=(3,3),strides=(2,2),padding='same',
                                            use_bias=False,activation=tf.nn.relu,name='conv2_layer')
            net = tf.layers.batch_normalization(net,name='BN_3')  ## [None,8,8,64]

            net = tf.layers.conv2d_transpose(net,32,kernel_size=(3,3),strides=(2,2),padding='same',
                                            use_bias=False,activation=tf.nn.relu,name='conv3_layer')
            net = tf.layers.batch_normalization(net,name='BN_4')  ## [None,16,16,32]
            
            net = tf.layers.conv2d_transpose(net,16,kernel_size=(3,3),strides=(2,2),padding='same',
                                            use_bias=False,activation=tf.nn.relu,name='conv4_layer')
            net = tf.layers.batch_normalization(net,name='BN_5')  ## [None,32,32,16]

            net = tf.layers.conv2d_transpose(net,channels,kernel_size=(3,3),strides=(2,2),padding='same',
                                        use_bias=False,name='conv5_layer')  ## [None,64,64,4]

        return tf.reshape(net,[-1,output_dim])



    def discriminator(image):
        with tf.variable_scope('discriminator',reuse=tf.AUTO_REUSE):  # image [None,64,64,4]
            image = tf.reshape(image,[-1,imsize,imsize,channels])
            net = tf.layers.conv2d(image,64,(3,3),strides=(2,2),padding='same',activation=tf.nn.relu,name='conv1')
            net = tf.layers.batch_normalization(net,name='BN_1')  # [None,32,32,64]
            net = tf.layers.dropout(net,rate=0.3,name='drop_1')

            net = tf.layers.conv2d(net,128,(3,3),strides=(2,2),padding='same',activation=tf.nn.relu,name='conv2')
            net = tf.layers.dropout(net,rate=0.3,name='drop_2') # [None,16,16,128]
            
            net = tf.layers.conv2d(net,256,(3,3),strides=(2,2),padding='same',activation=tf.nn.relu,name='conv3')
            net = tf.layers.dropout(net,rate=0.3,name='drop_3')  # [None,8,8,256]
            
            net = tf.layers.conv2d(net,256,(3,3),strides=(1,1),padding='same',activation=tf.nn.relu,name='conv4')
            net = tf.layers.dropout(net,rate=0.3,name='drop_4')  # [None,8,8,256]

            net = tf.layers.flatten(net,name='flatten')  # [None,8*8*256]
            net = tf.layers.dense(net,1,name='dense')   # [None,1]
        return net

else:
    ## 当数据是40x40的时候，就用这个G和D
    def generator(noise):
        with tf.variable_scope('generator',reuse=tf.AUTO_REUSE):
            net = tf.layers.dense(noise,5*5*256,activation=tf.nn.leaky_relu,use_bias=False,name='dense_layer')
            net = tf.layers.batch_normalization(net,name='BN_1')

            net = tf.reshape(net,(-1,5,5,256))  ## [None,5,5,256]
            net = tf.layers.conv2d_transpose(net,128,kernel_size=(3,3),strides=(1,1),padding='same',
                                            use_bias=False,activation=tf.nn.relu,name='conv1_layer')
            net = tf.layers.batch_normalization(net,name='BN_2')  ## [None,5,5,128]

            net = tf.layers.conv2d_transpose(net,64,kernel_size=(3,3),strides=(2,2),padding='same',
                                            use_bias=False,activation=tf.nn.relu,name='conv2_layer')
            net = tf.layers.batch_normalization(net,name='BN_3')  ## [None,10,10,64]

            net = tf.layers.conv2d_transpose(net,32,kernel_size=(3,3),strides=(2,2),padding='same',
                                            use_bias=False,activation=tf.nn.relu,name='conv3_layer')
            net = tf.layers.batch_normalization(net,name='BN_4')  ## [None,20,20,32]
            
            net = tf.layers.conv2d_transpose(net,channels,kernel_size=(3,3),strides=(2,2),padding='same',
                                           use_bias=False,name='conv4_layer')  ## [None,40,40,4]

        return tf.reshape(net,[-1,output_dim])



    def discriminator(image):
        with tf.variable_scope('discriminator',reuse=tf.AUTO_REUSE):  # image [None,40,40,4]
            image = tf.reshape(image,[-1,imsize,imsize,channels])
            net = tf.layers.conv2d(image,64,(3,3),strides=(2,2),padding='same',activation=tf.nn.relu,name='conv1')
            net = tf.layers.batch_normalization(net,name='BN_1')  # [None,20,20,64]
            net = tf.layers.dropout(net,rate=0.3,name='drop_1')

            net = tf.layers.conv2d(net,128,(3,3),strides=(2,2),padding='same',activation=tf.nn.relu,name='conv2')
            net = tf.layers.dropout(net,rate=0.3,name='drop_2') # [None,10,10,128]
            
            net = tf.layers.conv2d(net,256,(3,3),strides=(2,2),padding='same',activation=tf.nn.relu,name='conv3')
            net = tf.layers.dropout(net,rate=0.3,name='drop_3')  # [None,5,5,256]

            net = tf.layers.flatten(net,name='flatten')  # [None,5*5*256]
            net = tf.layers.dense(net,1,name='dense')   # [None,1]
        return net



k_real_image = tf.placeholder(tf.float32,[1,imsize,imsize,1])
k_mask = tf.placeholder(tf.float32,[1,imsize,imsize,1])
y_k = k_real_image * k_mask   # mask 在有real_img有值的地方是1，missiong的地方是0

h_real_image = tf.placeholder(tf.float32,[1,imsize,imsize,1])
h_mask = tf.placeholder(tf.float32,[1,imsize,imsize,1])
y_h = h_real_image * h_mask

with tf.variable_scope('z_optimizer',reuse=tf.AUTO_REUSE): 
    z = tf.get_variable('z',[1,noise_dim],initializer=tf.random_normal_initializer())


G_output = generator(z)   
logits = discriminator(G_output)   

G_output = tf.reshape(G_output,[-1,imsize,imsize,channels])
G_output_k = G_output[:,:,:,0:1]
G_output_h = G_output[:,:,:,1:2]
# prior_loss = numta * tf.log(1-tf.sigmoid(logits) + epsilon)
prior_loss = - numta * tf.reduce_mean(logits)
context_loss = tf.reduce_sum(tf.abs(G_output_k - y_k) * k_mask) + tf.reduce_sum(tf.abs(G_output_h - y_h) * h_mask)
# context_loss = tf.reduce_sum(tf.abs(G_output_h - y_h) * h_mask)
loss =  prior_loss + context_loss * 10.

global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
learning_rate = tf.train.exponential_decay(1e-1, global_step, decay_step, decay_rate, staircase=False)
# learning_rate = 1e-1
with tf.variable_scope('z_optimizer',reuse=tf.AUTO_REUSE):   ## adam中也有需要学习的参数，所以需要让其在一个scope中，设置reuse
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss,var_list=z)


config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"generator")+tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"discriminator"))
saver.restore(sess,output_model_dir+'model-{0}'.format(itera_num))


## prepare data
test_idx = 8  # KL的测试用的样本8 
real_img_data_k = test_data[test_idx:test_idx+1,:,:,0:1]
real_img_data_h = test_data[test_idx:test_idx+1,:,:,1:2]

k_obs_percent = np.float(sys.argv[1])
h_obs_percent = np.float(sys.argv[2])

if 'nonGaussian' in data_dir:
    k_obs_name = f'./hard_data/k_idx_{k_obs_percent}_nonGaussian.txt'
    h_obs_name = f'./hard_data/h_idx_{h_obs_percent}_nonGaussian.txt'
else:
    k_obs_name = f'./hard_data/k_idx_{k_obs_percent}.txt'
    h_obs_name = f'./hard_data/h_idx_{h_obs_percent}.txt'

output_figure_save_dir = output_figure_dir + f'saved_figures_sample_{test_idx}_k_{k_obs_percent}_h_{h_obs_percent}/'
if not os.path.exists(output_figure_save_dir):
    os.makedirs(output_figure_save_dir)

if 'nonGaussian' in data_dir:
    k_obs_save_name = output_figure_save_dir + f'k_obs_idx_{k_obs_percent}_nonGaussian.txt'
    h_obs_save_name = output_figure_save_dir + f'h_obs_idx_{h_obs_percent}_nonGaussian.txt'
else:
    k_obs_save_name = output_figure_save_dir + f'k_obs_idx_{k_obs_percent}.txt'
    h_obs_save_name = output_figure_save_dir + f'h_obs_idx_{h_obs_percent}.txt'

mask_mat_k = np.zeros((imsize*imsize,1)).astype(np.float32)
# if repeat_num == 0:  
#     filling_idx_k = np.random.choice(imsize*imsize,np.int32(k_obs_percent*imsize*imsize),replace=False)  
#     np.savetxt(k_obs_name,filling_idx_k)  
# else:
filling_idx_k = np.int32(np.loadtxt(k_obs_name))  
np.savetxt(k_obs_save_name,filling_idx_k)  

mask_mat_k[filling_idx_k] = 1
mask_value_k = np.reshape(mask_mat_k,(1,imsize,imsize,1))


mask_mat_h = np.zeros((imsize*imsize,1)).astype(np.float32)
# if repeat_num == 0:  
#     filling_idx_h = np.random.choice(imsize*imsize,np.int32(h_obs_percent*imsize*imsize),replace=False)   ## 随机选取有数据的点
#     np.savetxt(h_obs_name,filling_idx_h)
# else:
filling_idx_h = np.int32(np.loadtxt(h_obs_name))
np.savetxt(h_obs_save_name,filling_idx_h)

mask_mat_h[filling_idx_h] = 1
mask_value_h = np.reshape(mask_mat_h,(1,imsize,imsize,1))


## 优化
loss_value_ensem = []
p_loss_value_ensem = []
c_loss_value_ensem = []
tf_dict = {k_real_image:real_img_data_k,k_mask:mask_value_k,h_real_image:real_img_data_h,h_mask:mask_value_h}

for i in range(1500):
    sess.run(optimizer,feed_dict=tf_dict)
    sess.run(tf.clip_by_value(z,-1,1))
    
    if i%10 == 0:
        [loss_value, conditional_output] = sess.run([loss,G_output],feed_dict=tf_dict)
        [p_loss_value,c_loss_value] = sess.run([prior_loss,context_loss],feed_dict=tf_dict)
        print("Step: {}, Loss: {:.3f}, prior loss: {:.3f}, context_loss: {:.3f}".format(i,loss_value,p_loss_value,c_loss_value))
        loss_value_ensem.append(loss_value)
        p_loss_value_ensem.append(p_loss_value)
        c_loss_value_ensem.append(c_loss_value)
        
        blend_k = conditional_output[:,:,:,0:1] * (1-mask_value_k) + real_img_data_k * mask_value_k
        # blend_k = conditional_output[:,:,:,0:1]
        blend_h = conditional_output[:,:,:,1:2] * (1-mask_value_h) + real_img_data_h * mask_value_h
        
sess.close()     




np.savetxt(output_figure_save_dir + f'{repeat_num}_generated_k_model_{itera_num}.txt',blend_k.ravel())
np.savetxt(output_figure_save_dir + f'{repeat_num}_generated_h_model_{itera_num}.txt',blend_h.ravel())
np.savetxt(output_figure_save_dir + 'k_ground_truth.txt',real_img_data_k.ravel())
np.savetxt(output_figure_save_dir + 'h_ground_truth.txt',real_img_data_h.ravel())


fig,ax = plt.subplots()
ax.set_xlabel('Iterations')
ax.set_ylabel('Loss')
ax.plot(np.arange(len(loss_value_ensem)),np.array(loss_value_ensem).ravel())
_ = ax.set_xticklabels(np.int32(ax.get_xticks())*10)
plt.savefig(output_figure_save_dir + 'loss_convergence.jpg')


## 绘制k的对比图
fig,ax = plt.subplots(1,3,figsize=(20,5))
gci = ax[0].contourf(X_dis,Y_dis,real_img_data_k[0,:,:,0],cmap=plt.cm.Blues)
plt.colorbar(gci,ax=ax[0],shrink=0.8)
ax[0].set_title('Ground truth')

fill_data = real_img_data_k.ravel()[filling_idx_k]
obs_point = xy_cor[filling_idx_k,:]
ax[1].scatter(obs_point[:,0],obs_point[:,1],c=fill_data,cmap=plt.cm.Blues)
plt.colorbar(gci,ax=ax[1],shrink=0.8)
ax[1].set_xlim(0,Lx)
ax[1].set_ylim(0,Ly)
ax[1].set_title('Input data ({0}% grid data)'.format(np.int32(k_obs_percent*100)))

gci = ax[2].contourf(X_dis,Y_dis,blend_k[0,:,:,0],cmap=plt.cm.Blues)
plt.colorbar(gci,ax=ax[2],shrink=0.8)
ax[2].set_title('Generated data')
plt.savefig(output_figure_save_dir + f'{repeat_num}_k_comparisons.jpg')

ssim_score = ssim(real_img_data_k[0,:,:,0],blend_k[0,:,:,0])
print('SSIM score is: ',ssim_score)


# 也可计算PSNR指标


fig,ax = plt.subplots(1,3,figsize=(20,5))
gci = ax[0].contourf(X_dis,Y_dis,real_img_data_h[0,:,:,0],cmap=plt.cm.Blues)
plt.colorbar(gci,ax=ax[0],shrink=0.8)
ax[0].set_title('Ground truth')


fill_data = real_img_data_h.ravel()[filling_idx_h]
obs_point = xy_cor[filling_idx_h,:]
ax[1].scatter(obs_point[:,0],obs_point[:,1],c=fill_data,cmap=plt.cm.Blues)
plt.colorbar(gci,ax=ax[1],shrink=0.8)
ax[1].set_xlim(0,Lx)
ax[1].set_ylim(0,Ly)
ax[1].set_title('Input data ({0}% grid data)'.format(np.int32(h_obs_percent*100)))

level=np.linspace(0.,1.,9)
blend_h[blend_h>1] = 1
blend_h[blend_h<0] = 0
gci = ax[2].contourf(X_dis,Y_dis,blend_h[0,:,:,0],cmap=plt.cm.Blues)
plt.colorbar(gci,ax=ax[2],shrink=0.8)
ax[2].set_title('Generated data')
plt.savefig(output_figure_save_dir +  f'{repeat_num}_h_comparisons.jpg')


ssim_score = ssim(real_img_data_h[0,:,:,0],blend_h[0,:,:,0])
print('SSIM score is: ',ssim_score)



## 绘制用于保存的图

fig,ax = plt.subplots(figsize=(6,5))
gci = ax.contourf(X_dis,Y_dis,real_img_data_k[0,:,:,0],cmap=plt.cm.Blues)
plt.colorbar(gci,ax=ax,shrink=0.8)
ax.set_title('Ground truth')
plt.tight_layout()
plt.savefig(output_figure_save_dir + 'k_Ground_truth.jpg')

fig,ax = plt.subplots(figsize=(6,5))
fill_data = real_img_data_k.ravel()[filling_idx_k]
obs_point = xy_cor[filling_idx_k,:]
ax.scatter(obs_point[:,0],obs_point[:,1],c=fill_data,cmap=plt.cm.Blues)
ax.set_title('Point measurements of $K$')
plt.colorbar(gci,ax=ax,shrink=0.8)
ax.set_xlim(0,Lx)
ax.set_ylim(0,Ly)
plt.tight_layout()
plt.savefig(output_figure_save_dir + 'k_point_obs.jpg')

fig,ax = plt.subplots(figsize=(6,5))
gci = ax.contourf(X_dis,Y_dis,blend_k[0,:,:,0],cmap=plt.cm.Blues)
plt.colorbar(gci,ax=ax,shrink=0.8)
ax.set_title('Generated data')
plt.savefig(output_figure_save_dir + f'{repeat_num}_k_predictions.jpg')


fig,ax = plt.subplots(figsize=(6,5))
gci = ax.contourf(X_dis,Y_dis,real_img_data_h[0,:,:,0],cmap=plt.cm.Blues)
plt.colorbar(gci,ax=ax,shrink=0.8)
ax.set_title('Ground truth')
plt.tight_layout()
plt.savefig(output_figure_save_dir + 'h_Ground_truth.jpg')

fig,ax = plt.subplots(figsize=(6,5))
fill_data = real_img_data_h.ravel()[filling_idx_h]
obs_point = xy_cor[filling_idx_h,:]
ax.scatter(obs_point[:,0],obs_point[:,1],c=fill_data,cmap=plt.cm.Blues)
ax.set_title('Point measurements of $h$')
plt.colorbar(gci,ax=ax,shrink=0.8)
ax.set_xlim(0,Lx)
ax.set_ylim(0,Ly)
plt.tight_layout()
plt.savefig(output_figure_save_dir + 'h_point_obs.jpg')

fig,ax = plt.subplots(figsize=(6,5))
gci = ax.contourf(X_dis,Y_dis,blend_h[0,:,:,0],cmap=plt.cm.Blues)
plt.colorbar(gci,ax=ax,shrink=0.8)
ax.set_title('Generated data')
plt.savefig(output_figure_save_dir + f'{repeat_num}_h_predictions.jpg')




## 绘制pdf匹配图
k_truth_series = pd.Series(real_img_data_k.ravel())
k_pred_series = pd.Series(blend_k.ravel())
plt.figure(figsize=(6,4))
k_truth_series.plot('kde',color='r',label='Ground truth')
k_pred_series.plot('kde',color='g',label='Prediction')
plt.xlabel('$LogK$',fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig(output_figure_save_dir + f'{repeat_num}_k_pdf_matching.jpg')


h_truth_series = pd.Series(real_img_data_h.ravel())
h_pred_series = pd.Series(blend_h.ravel())
plt.figure(figsize=(6,4))
h_truth_series.plot('kde',color='r',label='Ground truth')
h_pred_series.plot('kde',color='g',label='Prediction')
plt.legend()
plt.xlabel('$h$',fontsize=12)
plt.tight_layout()
plt.savefig(output_figure_save_dir + f'{repeat_num}_h_pdf_matching.jpg')




## 绘制散点图
plt.figure(figsize=(6,5))
plt.scatter(real_img_data_k.ravel(),blend_k.ravel(),s=2)
min_val = np.min([np.min(real_img_data_k),np.min(blend_k)]) * 1.05
max_val = np.max([np.max(real_img_data_k),np.max(blend_k)]) * 1.05
plt.plot([min_val,max_val],[min_val,max_val],'r',lw=2)
plt.xlabel('Ground truth of $LogK$',fontsize=12)
plt.ylabel('Predictions of $LogK$',fontsize=12)
coef = round(np.corrcoef(real_img_data_k.ravel(),blend_k.ravel())[0,1],3)
plt.text(min_val*0.8,max_val*0.8,f'$R^2={coef}$',fontsize=14)
plt.tight_layout()
plt.savefig(output_figure_save_dir + f'{repeat_num}_k_scatter_plot.jpg')
np.savetxt(output_figure_save_dir + f'{repeat_num}_k_corrcoef_model_{itera_num}.txt',np.array(coef).reshape(-1,1))



plt.figure(figsize=(6,5))
plt.scatter(real_img_data_h.ravel(),blend_h.ravel(),s=2)
plt.plot([0,1],[0,1],'r',lw=2)
plt.xlabel('Ground truth of $h$',fontsize=12)
plt.ylabel('Predictions of $h$',fontsize=12)
coef = round(np.corrcoef(real_img_data_h.ravel(),blend_h.ravel())[0,1],3)
plt.text(0.1,0.9,f'$R^2={coef}$',fontsize=14)
plt.tight_layout()
plt.savefig(output_figure_save_dir + f'{repeat_num}_h_scatter_plot.jpg')


k_rmse = np.sqrt(np.mean((real_img_data_k.ravel() - blend_k.ravel())**2))
print('RMSE of K:',k_rmse)
k_rmse = np.array(k_rmse).reshape(-1,1)
np.savetxt(output_figure_save_dir + f'{repeat_num}_k_rmse_model_{itera_num}.txt',k_rmse)





