
epochs = 10
#epochs
net_mode = 'ir'
#network
net_depth = 50
# number of layer
lr = 1e-3
# learning rategit
batch_size = 32
# batchsize
num_workers = 3
# num workers
data_mode = 'ms1m'
# data root_dir paths
rec_path ='faces_emore'

# paths in local
#<<<<<<< HEAD
# root_dir = '/home/minglee/Documents/aiProjects/generator_datav2'
# out_dir  = '/home/minglee/Documents/aiProjects/generator_datav2'

#=======
#>>>>>>> 602e539152a1eca437ac098844ffc06cf9905896
#root_dir = '/home/minglee/Documents/aiProjects/generator_datav2'
#out_dir  = '/home/minglee/Documents/aiProjects/generator_datav2'

# paths in sever
out_dir = '/mnt/DATA/duydmFabbi/dataFace/generator_datav2'  #paths load data bin file and read in folder faces_emore
root_dir = '/mnt/DATA/duydmFabbi/dataFace/generator_datav2' # save all folder status of models training

#<<<<<<< HEAD
#only  mxnet==1.7.0.post1 support for both of load_bin and load_mx_rec functions 
#only  mxnet==1.7.0.post1 support for both of load_bin and load_mx_rec functions 
#=======
#only  mxnet==1.7.0.post1 support for both of load_bin and load_mx_rec functions .
#>>>>>>> 602e539152a1eca437ac098844ffc06cf9905896
