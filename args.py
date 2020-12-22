
epochs = 10
#epochs
net_mode = 'ir'
#network
net_depth = 50
# number of layer
lr = 1e-3
# learning rategit
batch_size = 64
# batchsize
num_workers = 4
# num workers
data_mode = 'ms1m'
# data root_dir paths
rec_path ='faces_emore'
<<<<<<< HEAD
model_save_interval = 2
load_pretrained_paths = 'model_ir_se50.pth'
save = True
threshold = 1.0
update = True
tta = True
score = True
file_name = 'Sherlock.avi'
save_name = 'recording'
begin = 1
duration = 0
th = 0.1  #threshold of probility pnet mtcnn 

# paths in local
# root_dir = '/home/minglee/Documents/aiProjects/git_clone/InsightFace_Pytorch'
# out_dir  = '/home/minglee/Documents/aiProjects/generator_datav2'
=======
model_save_interval = 1
load_pretrained_paths = ''

# paths in local
#<<<<<<< HEAD
# root_dir = '/home/minglee/Documents/aiProjects/generator_datav2'
# out_dir  = '/home/minglee/Documents/aiProjects/generator_datav2'

#=======
#>>>>>>> 602e539152a1eca437ac098844ffc06cf9905896
#root_dir = '/home/minglee/Documents/aiProjects/generator_datav2'
#out_dir  = '/home/minglee/Documents/aiProjects/generator_datav2'
>>>>>>> c1c619bb109a1cc53084a23bb4f4aa5d5dae03be

# paths in sever
out_dir = '/mnt/DATA/duydmFabbi/dataFace/generator_datav2'  #paths load data bin file and read in folder faces_emore
root_dir = '/mnt/DATA/duydmFabbi/dataFace/generator_datav2' # save all folder status of models training

#<<<<<<< HEAD
#only  mxnet==1.7.0.post1 support for both of load_bin and load_mx_rec functions 
#only  mxnet==1.7.0.post1 support for both of load_bin and load_mx_rec functions 
#=======
#only  mxnet==1.7.0.post1 support for both of load_bin and load_mx_rec functions .
#>>>>>>> 602e539152a1eca437ac098844ffc06cf9905896
