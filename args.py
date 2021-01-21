
epochs = 50
#epochs
net_mode = 'ir_se'
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
# save model of number epochs
model_save_interval = 1
pretrain_paths = 'model_ir_se50.pth'
save = True
threshold = 1.0
update = False
tta = True
score = True
file_name = 'Sherlock.avi'
save_name = 'recording'
begin = 1
duration = 0
th = 0.1  #threshold of probility pnet mtcnn

# paths in local
root_dir = '/mnt/DATA/duydmFabbi/dataFace/generator_datav2'
out_dir  = '/mnt/DATA/duydmFabbi/dataFace/trained-face-mask'
model_save_interval = 1
load_pretrained_paths = ''



# paths in sever
# out_dir = '/mnt/DATA/duydmFabbi/dataFace/generator_datav2'  #paths load data bin file and read in folder faces_emore
# root_dir = '/mnt/DATA/duydmFabbi/dataFace/generator_datav2' # save all folder status of models training
