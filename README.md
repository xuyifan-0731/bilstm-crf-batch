# bilstm-crf-batch

以batch形式，进行bilstm-crf的NER任务。　　
建议在训练阶段使用GPU服务器并行，在预测阶段使用CPU，这是因为预测所使用的维特比算法似乎并不能被GPU很好的加速。　　

进行训练：  
python train.py  
可选参数：  
parser.add_argument('--DATA_PATH', type=str,   default='data/')  
parser.add_argument('--SAVE_CPT_PATH', type=str, default='checkpoint_tmp/')  
parser.add_argument('--EMBEDDING_DIM', type=int, default=300)  
parser.add_argument('--HIDDEN_DIM', type=int, default=200)  
parser.add_argument('--BATCH_SIZE', type=int, default=16)  
parser.add_argument('--DEV_BATCH_SIZE', type=int, default=256)  
parser.add_argument('--EPOCHS', type=int, default=10)  
parser.add_argument('--LR', type=float, default=0.01)  
parser.add_argument('--Weight_Decay', type=float, default=1e-4)  
parser.add_argument('--GPU_DEVICE', type=str, default='0')  
进行预测：  
python predict.py  
可选参数：  
parser.add_argument('--DATA_PATH', type=str, default='data/')  
parser.add_argument('--CPT_PATH', type=str, default='use_checkpoint/')  
parser.add_argument('--SAVE_PATH', type=str, default='answer/')  
parser.add_argument('--DEV_BATCH_SIZE', type=int, default=1)  
parser.add_argument('--GPU_DEVICE', type=str, default='0')  


model.py:模型文件   
dataloader.py:数据处理文件  
 
