
epochs = 300
#epochs
net_mode = 'ir_se'
#network
net_depth = 50
# number of layer
lr = 1e-3
# learning rategit
batch_size = 100
# batchsize
num_workers = 4
# num workers
data_mode = 'emore'
# data root_dir paths
rec_path ='faces_emore'
# save model of number epochs
model_save_interval = 1
pretrain_paths = 'ir_se_50_opochs_90.pth'
save = True
threshold = 2.0
update = True
tta = True
score = True
file_name = 'Sherlock.avi'
save_name = 'recording'
begin = 1
duration = 0
th = 0.1  #threshold of probility pnet mtcnn

# paths in local
evaluate_dataset = '/home/minglee/Documents/aiProjects/VN-celeb'
root_dir = '/home/minglee/Documents/aiProjects/git_clone/InsightFace_Pytorch/save_embedding'
out_dir  = '/mnt/DATA/duydmFabbi/model_train'
model_save_interval = 1
load_pretrained_paths = ''



# paths in sever
# out_dir = '/mnt/DATA/duydmFabbi/dataFace/generator_datav2'  #paths load data bin file and read in folder faces_emore
# root_dir = '/mnt/DATA/duydmFabbi/dataFace/generator_datav2' # save all folder status of models training
