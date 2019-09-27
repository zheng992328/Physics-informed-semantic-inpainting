
# coding: utf-8

import numpy as np
import tensorflow as tf
import os
import time
import h5py
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import decomposition
from skimage.measure import compare_ssim as ssim
from SobelFilter import SobelFilter_tf
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


seed = 0
tf.set_random_seed(seed)
np.random.seed(seed)

train = False

data_dir = './datasets_with_KL/'

## KL data
code_version = 1  #用的是40x40的数据
code_version = 2 


# 提取训练数据与测试数据  
if data_dir == './datasets_with_KL/':
    output_dir = './semantic_inpainting_WGANGP_with_physics/outputs_with_KL_V{0}/'.format(code_version)
    output_model_dir = output_dir + 'saved_model/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_model_dir):
        os.makedirs(output_model_dir)
    
    with h5py.File(data_dir+'kle512_lhs10000_train.hdf5','r') as f:
        train_input = f['input'][:]
        train_output = f['output'][:]

# non-Gaussian的数据可能需要修改
elif data_dir == './datasets_with_nonGaussian/':
    if continuous_nonGaussian:
        output_dir = './semantic_inpainting_WGANGP_with_physics/outputs_with_contin_nonGaussian_V{0}/'.format(code_version)
        with h5py.File(data_dir+'continuous_nonGaussian_10000_with_multimodal_lnk.hdf5','r') as f:
            train_input = f['input'][:]
            train_output = f['output'][:]
    else:
        output_dir = './semantic_inpainting_WGANGP_with_physics/outputs_with_nonGaussian_V{0}/'.format(code_version)
        with h5py.File(data_dir+'channel_ng64_n4096_train.hdf5','r') as f:
            train_input = f['input'][:]
            train_output = f['output'][:]
    
    output_model_dir = output_dir + 'saved_model/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_model_dir):
        os.makedirs(output_model_dir)

    

## reformate the data shape to [N, H, W, C], C 为4，分别是(K, h, flux_x, flux_y)
imsize = train_input.shape[-1]
train_data = np.zeros((train_input.shape[0],imsize,imsize,4))
for i in range(train_input.shape[0]):
    if train_input[i,0,:,:].min() < 0: # 如果最小值小于0，就说明保存的是logk
        train_data[i,:,:,0] = train_input[i,0,:,:]
    else:
        train_data[i,:,:,0] = np.log(train_input[i,0,:,:])  ## 
    train_data[i,:,:,1] = train_output[i,0,:,:]  # h
    train_data[i,:,:,2] = train_output[i,1,:,:]  # flux_x
    train_data[i,:,:,3] = train_output[i,2,:,:]  # flux_y

print('train data shape:',train_data.shape)


batch_size = 50  ## maybe can be enlarged
noise_dim = 100  ## need to be modified
lr = 0.0002
epsilon = 1e-14
numta = 3e-1
learning_rate = 4e-4
decay_step = 50000
decay_rate = 0.9
Iters = 150000
Iters_completed = 0  #以防训练到中途断了
part_completed = False
LAMBDA = 10
channels = 4
output_dim = imsize * imsize * channels

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

def squeeze_data(data):  # the input data is [batch_size, width, height,1], output is [width * height, sample_num]
    sample_num = data.shape[0]
    imsize = data.shape[1]*data.shape[2]
    data_refor = np.zeros((imsize,sample_num))
    for i in range(data.shape[0]):
        data_refor[:,i] = data[i,:,:,0].ravel()
    return data_refor



class mydata():
    def __init__(self,data):
        self.data = data
        self.all_num = self.data.shape[0]
        self.current_index = 0
    
    def refresh(self):
        s = np.arange(self.all_num)
        np.random.shuffle(s)

        self.data = self.data[s,:,:,:]
        self.current_index = 0
    
    def next_batch(self,batch_s):
        if self.current_index + batch_s > self.all_num:
            self.refresh()
        batch_data = self.data[self.current_index:self.current_index+batch_s,:,:,:]
        self.current_index += batch_s
        return batch_data




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

else:  ## imsize = 40
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




img = tf.placeholder(tf.float32,[None,imsize*imsize*4])
z = tf.placeholder(tf.float32,[None,noise_dim])

fake_image = generator(z)  # [None,imsize*imsize*4]
real_logits = discriminator(img)
fake_logits = discriminator(fake_image)


## gradient penalty
alpha = tf.random_uniform(shape=[batch_size,1],minval=0.,maxval=1.,dtype=tf.float32)
differences = fake_image - img
interpolates = img + alpha * differences
gradients = tf.gradients(discriminator(interpolates),[interpolates])[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients),reduction_indices=[1]))
gradient_penalty = tf.reduce_mean((slopes-1.)**2)


D_loss = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits) + LAMBDA * gradient_penalty
W_distance = tf.reduce_mean(real_logits) - tf.reduce_mean(fake_logits)

## PDE constraint
# constitutive constraint
fake_image_re = tf.reshape(fake_image,[-1,imsize,imsize,channels])
sobel_filter = SobelFilter_tf(imsize)
grad_h = sobel_filter.grad_h(fake_image_re[:,:,:,1:2])  # grad_h操作的数据是(N, H, W, 1)
grad_v = sobel_filter.grad_v(fake_image_re[:,:,:,1:2])
flux_h_pred = - tf.exp(fake_image_re[:,:,:,0:1]) * grad_h
flux_v_pred = - tf.exp(fake_image_re[:,:,:,0:1]) * grad_v
loss_1 = tf.reduce_mean(tf.square(flux_h_pred - fake_image_re[:,:,:,2:3])) + tf.reduce_mean(tf.square(flux_v_pred - fake_image_re[:,:,:,3:4]))   # 生成的数据满足本构关系

# continuity constraint
div_h = sobel_filter.grad_h(fake_image_re[:,:,:,2:3])
div_v = sobel_filter.grad_v(fake_image_re[:,:,:,3:4])
loss_2 = tf.reduce_mean(tf.square(div_h+div_v))

# boundary constraint
left_bound, right_bound = fake_image_re[:,:,0,1], fake_image_re[:,:,-1,1]
top_flux, down_flux = fake_image_re[:,0,:,3], fake_image_re[:,-1,:,3]
loss_dirchlet = tf.reduce_mean(tf.square(left_bound - tf.ones_like(left_bound))) + \
                tf.reduce_mean(tf.square(right_bound - tf.zeros_like(right_bound)))
loss_neumann = tf.reduce_mean(tf.square(top_flux - tf.zeros_like(top_flux))) + \
                tf.reduce_mean(tf.square(down_flux - tf.zeros_like(down_flux)))
loss_boundary = loss_dirchlet + loss_neumann


G_loss = -tf.reduce_mean(fake_logits) + loss_1 + loss_2 + 10.0 * loss_boundary

G_var = [var for var in tf.trainable_variables() if var.name.startswith('generator')]
D_var = [var for var in tf.trainable_variables() if var.name.startswith('discriminator')]


G_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4,beta1=0.,beta2=0.9).minimize(G_loss,var_list=G_var,colocate_gradients_with_ops=True)
D_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4,beta1=0.,beta2=0.9).minimize(D_loss,var_list=D_var,colocate_gradients_with_ops=True)

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver(max_to_keep=20)

## 恢复训练到中途的 model 继续训练
if part_completed:
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"generator")+tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"discriminator"))
    saver.restore(sess,output_model_dir+'model-{0}'.format(Iters_completed))


start = time.time()
G_loss_ensem = []
W_distance_ensem = []  ## W distance就是D-loss的相反数
constitutive_loss_ensem = []
continuity_loss_ensem = []
bound_loss_ensem = []

disc_iters = 5
batch_num_per_epoch = train_data.shape[0] // batch_size
data_for_training = mydata(train_data)
if train:
    for iteration in range(Iters-Iters_completed+1):
        
        ## train generator
        if iteration > 0:
            z_input = np.random.standard_normal([batch_size,noise_dim])
            _ = sess.run(G_optimizer,feed_dict={z:z_input})

        for i in range(disc_iters):
            batch_data = data_for_training.next_batch(batch_size)
            batch_data = np.reshape(batch_data,(-1,imsize*imsize*channels))
            z_input = np.random.standard_normal([batch_size,noise_dim])
            _ = sess.run(D_optimizer,feed_dict={img:batch_data,z:z_input})
        
        if iteration % 100 == 0:
            [W_distance_value,G_loss_value,fake_img_value,loss_1_value,loss_2_value,loss_bound_value] = sess.run([W_distance,
                                G_loss,fake_image_re,loss_1,loss_2,loss_boundary],feed_dict={img:batch_data,z:z_input})
            
            print('Step: {}, W_distance: {:.3f}, constitutive loss: {:.3f}, continuity loss: {:.3f}, bound loss: {:.3f}'.format(iteration,
            W_distance_value,loss_1_value,loss_2_value,loss_bound_value))
            
            G_loss_ensem.append(G_loss_value)
            W_distance_ensem.append(W_distance_value)
            constitutive_loss_ensem.append(loss_1_value)
            continuity_loss_ensem.append(loss_2_value)
            bound_loss_ensem.append(loss_bound_value)
            
            ## 保留中间过程的生成结果
            fig_idx = np.random.choice(fake_img_value.shape[0],4,replace=False)
            fake_k_sel = fake_img_value[fig_idx,:,:,0]
            fake_h_sel = fake_img_value[fig_idx,:,:,1]
            
            if iteration % 20000 == 0:

                fig,ax = plt.subplots(2,2,figsize=(10,8))
                for i in range(4):
                    plt.subplot(2,2,i+1)
                    gci = plt.contourf(X_dis,Y_dis,fake_k_sel[i,:,:],cmap=plt.cm.Blues)
                    plt.colorbar(gci,shrink=0.8)
                plt.savefig(output_dir+'fake_k_img_{0}.jpg'.format(iteration))

                fig,ax = plt.subplots(2,2,figsize=(10,8))
                for i in range(4):
                    plt.subplot(2,2,i+1)
                    gci = plt.contourf(X_dis,Y_dis,fake_h_sel[i,:,:],cmap=plt.cm.Blues)
                    plt.colorbar(gci,shrink=0.8)
                plt.savefig(output_dir+'fake_h_img_{0}.jpg'.format(iteration))
            

        if iteration % 10000 == 0:
            if part_completed:
                save_iteration = iteration + Iters_completed
            else:
                save_iteration = iteration

            checkpoint_path = os.path.join(output_model_dir, 'model')
            saver.save(sess,checkpoint_path,global_step=save_iteration)

else:
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"generator")+tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"discriminator"))
    saver.restore(sess,output_model_dir+'model-{0}'.format(100000))
    G_loss_ensem = np.loadtxt(output_dir + 'G_loss_ensem.txt')
    W_distance_ensem = np.loadtxt(output_dir + 'W_distance_ensem.txt')


print(f'Elapsed time: {(time.time()-start)/60.} min')


## plot some figures
plt.figure()
plt.plot(G_loss_ensem[10:],'r',label='Generator loss')  # loss是每100个保存一次，所以第10个对应着原始迭代的第1000步
plt.plot(W_distance_ensem[10:],'b',label='W distance')
plt.legend()
plt.savefig(output_dir+'loss_convergence.jpeg')
np.savetxt(output_dir + 'G_loss_ensem.txt',G_loss_ensem)
np.savetxt(output_dir + 'W_distance_ensem.txt',W_distance_ensem)
np.savetxt(output_dir + 'constitutive_loss_ensem.txt',constitutive_loss_ensem)
np.savetxt(output_dir + 'continuity_loss_ensem.txt',continuity_loss_ensem)
np.savetxt(output_dir + 'bound_loss_ensem.txt',bound_loss_ensem)


## 生成fake images
fake_img_num = train_data.shape[0]
z_new = np.random.standard_normal([fake_img_num,noise_dim])
new_fake_image = sess.run(fake_image_re,feed_dict={z:z_new})
sess.close()


K_pred = new_fake_image[:,:,:,0:1]
h_pred = new_fake_image[:,:,:,1:2]
flux_h_pred = new_fake_image[:,:,:,2:3]
flux_v_pred = new_fake_image[:,:,:,3:4]

K_pred_flatten = squeeze_data(K_pred)
h_pred_flatten = squeeze_data(h_pred)
flux_h_pred_flatten = squeeze_data(flux_h_pred)
flux_v_pred_flatten = squeeze_data(flux_v_pred)
np.savetxt(output_dir + 'K_pred.txt',K_pred_flatten)
np.savetxt(output_dir + 'h_pred.txt',h_pred_flatten)
np.savetxt(output_dir + 'flux_h_pred.txt',flux_h_pred_flatten)
np.savetxt(output_dir + 'flux_v_pred.txt',flux_v_pred_flatten)

# ## 与真实场对比统计信息
## 绘制real images 和 generated images的spectra

pca_terms = 100

pca_ref_K = decomposition.PCA()
K_train_flatten = squeeze_data(train_data[:,:,:,0:1])
pca_ref_K.fit(K_train_flatten.T)
eigen_vals_ref_K = pca_ref_K.explained_variance_
print('{0} terms for K reference:'.format(pca_terms),sum(eigen_vals_ref_K[:pca_terms])/sum(eigen_vals_ref_K))

pca_gene_K = decomposition.PCA()
K_pred_flatten = squeeze_data(K_pred)
pca_gene_K.fit(K_pred_flatten.T)
eigen_vals_gene_K = pca_gene_K.explained_variance_
print('{0} terms for K generation:'.format(pca_terms),sum(eigen_vals_gene_K[:pca_terms])/sum(eigen_vals_ref_K))

plt.figure(figsize=(8,6))
plt.scatter(np.arange(0,pca_terms),np.real(eigen_vals_gene_K)[:pca_terms],marker='o',color='r')
plt.plot(np.arange(pca_terms),np.real(eigen_vals_gene_K)[:pca_terms],label='Generated predictions',color='r') 
plt.scatter(np.arange(0,pca_terms),np.real(eigen_vals_ref_K)[:pca_terms],marker='s',color='b')
plt.plot(np.arange(pca_terms),np.real(eigen_vals_ref_K)[:pca_terms],label='Training dataset',color='b') 
plt.xlabel('Component',fontsize=15)
plt.ylabel('Eigenvalue',fontsize=15)
plt.setp(plt.gca().get_xticklabels(),fontsize=15) 
plt.setp(plt.gca().get_yticklabels(),fontsize=15) 
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig(output_dir+'logk_spectra.jpeg')

plt.figure(figsize=(8,6))
plt.semilogy(np.arange(0,pca_terms),np.real(eigen_vals_gene_K)[:pca_terms],marker='o',color='r')
plt.semilogy(np.arange(pca_terms),np.real(eigen_vals_gene_K)[:pca_terms],label='Generated predictions',color='r') 
plt.semilogy(np.arange(0,pca_terms),np.real(eigen_vals_ref_K)[:pca_terms],marker='s',color='b')
plt.semilogy(np.arange(pca_terms),np.real(eigen_vals_ref_K)[:pca_terms],label='Training dataset',color='b') 
plt.xlabel('Component',fontsize=15)
plt.ylabel('Log eigenvalue',fontsize=15)
plt.setp(plt.gca().get_xticklabels(),fontsize=15) 
plt.setp(plt.gca().get_yticklabels(),fontsize=15) 
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig(output_dir+'log_trans_logk_spectra.jpeg')


## 绘制h的spectra
pca_terms = 40

pca_ref_h = decomposition.PCA()
h_train_flatten = squeeze_data(train_data[:,:,:,1:2])
pca_ref_h.fit(h_train_flatten.T)
eigen_vals_ref_h = pca_ref_h.explained_variance_
print('{0} terms for h reference:'.format(pca_terms),sum(eigen_vals_ref_h[:pca_terms])/sum(eigen_vals_ref_h))

pca_gene_h = decomposition.PCA()
h_pred_flatten = squeeze_data(h_pred)
pca_gene_h.fit(h_pred_flatten.T)
eigen_vals_gene_h = pca_gene_h.explained_variance_
print('{0} terms for h generation:'.format(pca_terms),sum(eigen_vals_gene_h[:pca_terms])/sum(eigen_vals_ref_h))

plt.figure(figsize=(8,6))
plt.scatter(np.arange(0,pca_terms),np.real(eigen_vals_gene_h)[:pca_terms],marker='o',color='r')
plt.plot(np.arange(pca_terms),np.real(eigen_vals_gene_h)[:pca_terms],label='Generated predictions',color='r') 
plt.scatter(np.arange(0,pca_terms),np.real(eigen_vals_ref_h)[:pca_terms],marker='s',color='b')
plt.plot(np.arange(pca_terms),np.real(eigen_vals_ref_h)[:pca_terms],label='Training dataset',color='b') 
plt.xlabel('Component',fontsize=15)
plt.ylabel('Eigenvalue',fontsize=15)
plt.setp(plt.gca().get_xticklabels(),fontsize=15) 
plt.setp(plt.gca().get_yticklabels(),fontsize=15) 
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig(output_dir+'h_spectra.jpeg')

plt.figure(figsize=(8,6))
plt.semilogy(np.arange(0,pca_terms),np.real(eigen_vals_gene_h)[:pca_terms],marker='o',color='r')
plt.semilogy(np.arange(pca_terms),np.real(eigen_vals_gene_h)[:pca_terms],label='Generated predictions',color='r') 
plt.semilogy(np.arange(0,pca_terms),np.real(eigen_vals_ref_h)[:pca_terms],marker='s',color='b')
plt.semilogy(np.arange(pca_terms),np.real(eigen_vals_ref_h)[:pca_terms],label='Training dataset',color='b') 
plt.xlabel('Component',fontsize=15)
plt.ylabel('Log eigenvalue',fontsize=15)
plt.setp(plt.gca().get_xticklabels(),fontsize=15) 
plt.setp(plt.gca().get_yticklabels(),fontsize=15) 
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig(output_dir+'log_trans_h_spectra.jpeg')


## 绘制flux_x的spectra
pca_terms = 100

pca_ref_flux_x = decomposition.PCA()
flux_x_train_flatten = squeeze_data(train_data[:,:,:,2:3])
pca_ref_flux_x.fit(flux_x_train_flatten.T)
eigen_vals_ref_flux_x = pca_ref_flux_x.explained_variance_
print('{0} terms for flux x reference:'.format(pca_terms),sum(eigen_vals_ref_flux_x[:pca_terms])/sum(eigen_vals_ref_flux_x))

pca_gene_flux_x = decomposition.PCA()
flux_x_pred_flatten = squeeze_data(flux_h_pred)
pca_gene_flux_x.fit(flux_x_pred_flatten.T)
eigen_vals_gene_flux_x = pca_gene_flux_x.explained_variance_
print('{0} terms for flux x generation:'.format(pca_terms),sum(eigen_vals_gene_flux_x[:pca_terms])/sum(eigen_vals_ref_flux_x))

plt.figure(figsize=(8,6))
plt.scatter(np.arange(0,pca_terms),np.real(eigen_vals_gene_flux_x)[:pca_terms],marker='o',color='r')
plt.plot(np.arange(pca_terms),np.real(eigen_vals_gene_flux_x)[:pca_terms],label='Generated predictions',color='r') 
plt.scatter(np.arange(0,pca_terms),np.real(eigen_vals_ref_flux_x)[:pca_terms],marker='s',color='b')
plt.plot(np.arange(pca_terms),np.real(eigen_vals_ref_flux_x)[:pca_terms],label='Training dataset',color='b') 
plt.xlabel('Component',fontsize=15)
plt.ylabel('Eigenvalue',fontsize=15)
plt.setp(plt.gca().get_xticklabels(),fontsize=15) 
plt.setp(plt.gca().get_yticklabels(),fontsize=15) 
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig(output_dir+'flux_x_spectra.jpeg')

plt.figure(figsize=(8,6))
plt.semilogy(np.arange(0,pca_terms),np.real(eigen_vals_gene_flux_x)[:pca_terms],marker='o',color='r')
plt.semilogy(np.arange(pca_terms),np.real(eigen_vals_gene_flux_x)[:pca_terms],label='Generated predictions',color='r') 
plt.semilogy(np.arange(0,pca_terms),np.real(eigen_vals_ref_flux_x)[:pca_terms],marker='s',color='b')
plt.semilogy(np.arange(pca_terms),np.real(eigen_vals_ref_flux_x)[:pca_terms],label='Training dataset',color='b') 
plt.xlabel('Component',fontsize=15)
plt.ylabel('Log eigenvalue',fontsize=15)
plt.setp(plt.gca().get_xticklabels(),fontsize=15) 
plt.setp(plt.gca().get_yticklabels(),fontsize=15) 
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig(output_dir+'log_trans_flux_x_spectra.jpeg')


## 绘制flux_y的spectra
pca_terms = 100

pca_ref_flux_y = decomposition.PCA()
flux_y_train_flatten = squeeze_data(train_data[:,:,:,0:1])
pca_ref_flux_y.fit(flux_y_train_flatten.T)
eigen_vals_ref_flux_y = pca_ref_flux_y.explained_variance_
print('{0} terms for flux y reference:'.format(pca_terms),sum(eigen_vals_ref_flux_y[:pca_terms])/sum(eigen_vals_ref_flux_y))

pca_gene_flux_y = decomposition.PCA()
flux_y_pred_flatten = squeeze_data(flux_v_pred)
pca_gene_flux_y.fit(flux_y_pred_flatten.T)
eigen_vals_gene_flux_y = pca_gene_flux_y.explained_variance_
print('{0} terms for flux y generation:'.format(pca_terms),sum(eigen_vals_gene_flux_y[:pca_terms])/sum(eigen_vals_ref_flux_y))

plt.figure(figsize=(8,6))
plt.scatter(np.arange(0,pca_terms),np.real(eigen_vals_gene_flux_y)[:pca_terms],marker='o',color='r')
plt.plot(np.arange(pca_terms),np.real(eigen_vals_gene_flux_y)[:pca_terms],label='Generated predictions',color='r') 
plt.scatter(np.arange(0,pca_terms),np.real(eigen_vals_ref_flux_y)[:pca_terms],marker='s',color='b')
plt.plot(np.arange(pca_terms),np.real(eigen_vals_ref_flux_y)[:pca_terms],label='Training dataset',color='b') 
plt.xlabel('Component',fontsize=15)
plt.ylabel('Eigenvalue',fontsize=15)
plt.setp(plt.gca().get_xticklabels(),fontsize=15) 
plt.setp(plt.gca().get_yticklabels(),fontsize=15) 
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig(output_dir+'flux_y_spectra.jpeg')

plt.figure(figsize=(8,6))
plt.semilogy(np.arange(0,pca_terms),np.real(eigen_vals_gene_flux_y)[:pca_terms],marker='o',color='r')
plt.semilogy(np.arange(pca_terms),np.real(eigen_vals_gene_flux_y)[:pca_terms],label='Generated predictions',color='r') 
plt.semilogy(np.arange(0,pca_terms),np.real(eigen_vals_ref_flux_y)[:pca_terms],marker='s',color='b')
plt.semilogy(np.arange(pca_terms),np.real(eigen_vals_ref_flux_y)[:pca_terms],label='Training dataset',color='b') 
plt.xlabel('Component',fontsize=15)
plt.ylabel('Log eigenvalue',fontsize=15)
plt.setp(plt.gca().get_xticklabels(),fontsize=15) 
plt.setp(plt.gca().get_yticklabels(),fontsize=15) 
plt.legend(fontsize=15)
plt.tight_layout()
plt.savefig(output_dir+'log_trans_flux_y_spectra.jpeg')



## 6 random fake para images
# level = np.linspace(-1.1,1.1,15)
fig,ax = plt.subplots(3,2,figsize=(12,15))
fake_idx = np.random.choice(K_pred_flatten.shape[1],6,replace=False)
fake_K_sel = K_pred_flatten[:,fake_idx]
for i in range(6):
    plt.subplot(3,2,i+1)
    gci = plt.contourf(X_dis,Y_dis,fake_K_sel[:,i].reshape(imsize,imsize),cmap=plt.cm.Blues)
    plt.colorbar(gci,shrink=0.8)
plt.savefig(output_dir+'fake_K_images.jpeg')

## real image
fig,ax = plt.subplots(1,2,figsize=(12,4.5))
real_idx = np.random.choice(K_train_flatten.shape[1],2,replace=False)
real_K_sel = K_train_flatten[:,real_idx]
for i in range(2):
    plt.subplot(1,2,i+1)
    gci = plt.contourf(X_dis,Y_dis,real_K_sel[:,i].reshape(imsize,imsize),cmap=plt.cm.Blues)
    plt.colorbar(gci,shrink=0.8)
plt.savefig(output_dir+'real_K_images.jpeg')


## 6 random fake h images
level = np.linspace(-1.1,1.1,15)
fig,ax = plt.subplots(3,2,figsize=(12,15))
fake_idx = np.random.choice(h_pred_flatten.shape[1],6,replace=False)
fake_h_sel = h_pred_flatten[:,fake_idx]
for i in range(6):
    plt.subplot(3,2,i+1)
    gci = plt.contourf(X_dis,Y_dis,fake_h_sel[:,i].reshape(imsize,imsize),cmap=plt.cm.Blues)
    plt.colorbar(gci,shrink=0.8)
plt.savefig(output_dir+'fake_h_images.jpeg')

## real h images
fig,ax = plt.subplots(1,2,figsize=(12,4.5))
real_idx = np.random.choice(h_train_flatten.shape[1],2,replace=False)
real_h_sel = h_train_flatten[:,real_idx]
for i in range(2):
    plt.subplot(1,2,i+1)
    gci = plt.contourf(X_dis,Y_dis,real_h_sel[:,i].reshape(imsize,imsize),cmap=plt.cm.Blues)
    plt.colorbar(gci,shrink=0.8)
plt.savefig(output_dir+'real_h_images.jpeg')







# ## 绘制K的mean和std的匹配
# fake_K_mean = np.mean(K_pred_flatten,axis=1)
# fake_K_std = np.std(K_pred_flatten,axis=1)

# real_K_mean = np.mean(K_train_flatten,axis=1)
# real_K_std = np.std(K_train_flatten,axis=1)

# plt.figure()
# plt.plot(fake_K_mean,'r',label="Generated para data")
# plt.plot(real_K_mean,'b',label="Reference para data")
# plt.title("Comparisons of mean values")
# plt.legend()
# plt.savefig(output_dir+'Comparsion_K_mean.jpeg')


# plt.figure()
# plt.plot(fake_K_std,'r',label="Generated para data")
# plt.plot(real_K_std,'b',label="Reference para data")
# plt.title("Comparisons of std values")
# plt.legend()
# plt.savefig(output_dir+'Comparsion_K_std.jpeg')


# ## 绘制h的mean和std的匹配

# fake_h_mean = np.mean(h_pred_flatten,axis=1)
# fake_h_std = np.std(h_pred_flatten,axis=1)

# real_h_mean = np.mean(h_train_flatten,axis=1)
# real_h_std = np.std(h_train_flatten,axis=1)

# plt.figure()
# plt.plot(fake_h_mean,'r',label="Generated state data")
# plt.plot(real_h_mean,'b',label="Reference state data")
# plt.title("Comparisons of mean values")
# plt.legend()
# plt.savefig(output_dir+'Comparsion_h_mean.jpeg')


# plt.figure()
# plt.plot(fake_h_std,'r',label="Generated state data")
# plt.plot(real_h_std,'b',label="Reference state data")
# plt.title("Comparisons of std values")
# plt.legend()
# plt.savefig(output_dir+'Comparsion_h_std.jpeg')


# # ### fake整体上的mean和std的变化幅度要大一些

# # real 和 fake的均值的对比
# fig,ax = plt.subplots(1,2,figsize=(11,5))
# gci = ax[0].contourf(X_dis,Y_dis,fake_K_mean.reshape(imsize,imsize),cmap=plt.cm.Blues)
# plt.colorbar(gci,ax=ax[0],shrink=0.8)

# gci = ax[1].contourf(X_dis,Y_dis,real_K_mean.reshape(imsize,imsize),cmap=plt.cm.Blues)
# plt.colorbar(gci,ax=ax[1],shrink=0.8)
# plt.savefig(output_dir+'comparison_K_mean_image.jpeg')


# # K的real和fake的标准差的全场对比
# fig,ax = plt.subplots(1,2,figsize=(11,5))
# gci = ax[0].contourf(X_dis,Y_dis,fake_K_std.reshape(imsize,imsize),cmap=plt.cm.Blues)
# plt.colorbar(gci,ax=ax[0],shrink=0.8)

# gci = plt.contourf(X_dis,Y_dis,real_K_std.reshape(imsize,imsize),cmap=plt.cm.Blues)
# plt.colorbar(gci,ax=ax[1],shrink=0.8)
# plt.savefig(output_dir+'comparison_K_std_image.jpeg')


# # h的real 和 fake的均值的全场对比
# fig,ax = plt.subplots(1,2,figsize=(11,5))
# gci = ax[0].contourf(X_dis,Y_dis,fake_h_mean.reshape(imsize,imsize),cmap=plt.cm.Blues)
# plt.colorbar(gci,ax=ax[0],shrink=0.8)

# gci = ax[1].contourf(X_dis,Y_dis,real_h_mean.reshape(imsize,imsize),cmap=plt.cm.Blues)
# plt.colorbar(gci,ax=ax[1],shrink=0.8)
# plt.savefig(output_dir+'comparison_h_mean_image.jpeg')



# # h的real和fake的标准差的全场对比
# fig,ax = plt.subplots(1,2,figsize=(11,5))
# gci = ax[0].contourf(X_dis,Y_dis,fake_h_std.reshape(imsize,imsize),cmap=plt.cm.Blues)
# plt.colorbar(gci,ax=ax[0],shrink=0.8)

# plt.figure()
# gci = ax[1].contourf(X_dis,Y_dis,real_h_std.reshape(imsize,imsize),cmap=plt.cm.Blues)
# plt.colorbar(gci,ax=ax[1],shrink=0.8)
# plt.savefig(output_dir+'comparison_h_std_image.jpeg')


