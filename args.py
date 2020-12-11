
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
data_mode = 'emore'
# data root_dir paths
rec_path ='faces_emore'
# paths in local
root_dir = '/home/minglee/Documents/aiProjects/generator_datav2'
out_dir  = '/home/minglee/Documents/aiProjects/generator_datav2'
# paths in sever
out_dir = '/mnt/DATA/duydmFabbi/dataFace/generator_datav2'
#paths load data bin file and read in folder faces_emore
root_dir = '/mnt/DATA/duydmFabbi/dataFace/generator_datav2'
 # save all folder status of models training

#only  mxnet==1.7.0.post1 support for both of load_bin and load_mx_rec functions .  
