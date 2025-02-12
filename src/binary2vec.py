import torch
import random
import numpy
from tqdm import tqdm, trange
from torch import nn
import torch.nn.functional as F
import datetime
from torch_geometric.data import Batch
import os
from layers import GPS
import csv
from utils import tab_printer,to_cuda




class Binary2vec(torch.nn.Module):

    def __init__(self, args):

        super(Binary2vec, self).__init__()
        self.args = args
        self.setup_layers()

    def setup_layers(self):
        """
        Creating the layers.
        """
        attn_kwargs = {'dropout': self.args.dropout}

        self.GPS = GPS(channels=64, num_layers=self.args.num_layers, heads=self.args.heads,attn_type=self.args.attn_type,
            attn_kwargs=attn_kwargs)

#如果batch_g2==None  则是在做predit 
    def forward(self, batch_g1,batch_g2=None):
        if batch_g2 == None:
            batch_g1=to_cuda(batch_g1)
            batch_g1_g_vec=self.GPS(batch_g1.x,batch_g1.edge_index,batch_g1.edge_attr,batch_g1.batch)
            return batch_g1_g_vec
        else:
            batch_g1=to_cuda(batch_g1)
            batch_g2=to_cuda(batch_g2)
            batch_g1_g_vec=self.GPS(batch_g1.x,batch_g1.edge_index,batch_g1.edge_attr,batch_g1.batch)
            batch_g2_g_vec=self.GPS(batch_g2.x,batch_g2.edge_index,batch_g2.edge_attr,batch_g2.batch)

        cosine_similarities = F.cosine_similarity(batch_g1_g_vec, batch_g2_g_vec, dim=1)
          
        return cosine_similarities



class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.1):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, sim, labels):
        positive_loss = labels * (1-sim)**2
        negative_loss =  (1 - labels) * F.relu(sim - self.margin) ** 2 * 2
        loss = positive_loss + negative_loss
        return loss.mean()




class Binary2vecTrainer(object):

    def __init__(self, args):

        self.args = args
        self.initial()
        self.setup_model()

    def setup_model(self):
        self.model = Binary2vec(self.args)
        self.loss_fn = ContrastiveLoss(margin=0.1)

    def load_data(self):
        pass

    def initial(self):
        self.dataset_path=self.args.dataset
        self.train_g_pairs=torch.load(os.path.join(self.dataset_path,'train.pth'))
        self.test_g_pairs=torch.load(os.path.join(self.dataset_path,'test.pth'))
        self.valid_g_pairs=torch.load(os.path.join(self.dataset_path,'valid.pth'))


    def create_batches(self,g_pairs):
        batches = []
        positive_g_pairs = g_pairs['positive']
        negative_g_pairs = g_pairs['negative']
        all_pairs = positive_g_pairs + negative_g_pairs
        random.shuffle(all_pairs)

        batches_num = len(all_pairs) // self.args.batch_pairs_size

        for i in range(batches_num):
            batch = all_pairs[i * self.args.batch_pairs_size: (i + 1) * self.args.batch_pairs_size]
            batches.append(batch)

        remaining_pairs = all_pairs[batches_num * self.args.batch_pairs_size:]
        if remaining_pairs != []:
            batches.append(remaining_pairs)

        return batches
    

    def process_batch_g_pair(self, batch_g_pairs,mode):
        self.optimizer.zero_grad()  
        batch_g1_list = []
        batch_g2_list = []
        batch_g_labels = []

        for g_pair in batch_g_pairs:
            g1 = torch.load(g_pair[0])
            g2 = torch.load(g_pair[1])
            batch_g1_list.append(g1)
            batch_g2_list.append(g2)
            batch_g_labels.append(1 if g1.g_label == g2.g_label else 0)

        batch_g_labels = torch.tensor(batch_g_labels, dtype=torch.float).cuda()

        batch_g1 = Batch.from_data_list(batch_g1_list,exclude_keys=['g_label','pe'])
        batch_g2 = Batch.from_data_list(batch_g2_list,exclude_keys=['g_label','pe'])


        batch_g_sim = self.model(batch_g1,batch_g2)  
        batch_avg_loss_per_pair=self.loss_fn(batch_g_sim,batch_g_labels)

        if mode=='train':
            batch_avg_loss_per_pair.backward()
            self.optimizer.step()

        return batch_avg_loss_per_pair.detach().item()




    def fit(self):
        """
        Fitting a model.
        """

        self.model = self.model.cuda()  
        
        print("\nModel training.\n")


        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.args.learning_rate,
                                          weight_decay=self.args.weight_decay)
        start_epoch=0

        best_metric = float('inf')  # 用于存储最佳模型的验证误差
        if self.args.load_path:
            load_dict=self.load()
            start_epoch=load_dict['epoch']+1
            best_metric=load_dict['metric']
            print('best_metrci: ',best_metric)
   

        tab_printer(self.args)

        epochs = trange(start_epoch,self.args.epochs, leave=True, desc="Epoch")

        for epoch in epochs:

            self.model.train()
            batches = self.create_batches(self.train_g_pairs)
            epoch_loss_sum = 0 #统计这个epoch的总loss
            g_pairs_num = 0

            
            for batch_index, batch_g_pair in tqdm(enumerate(batches), total=len(batches), desc="Batches"):
                batch_avg_loss_per_pair = self.process_batch_g_pair(batch_g_pair,mode='train') #得到batch中每对的平均loss
                g_pairs_num=g_pairs_num+len(batch_g_pair)
                epoch_loss_sum = epoch_loss_sum + batch_avg_loss_per_pair*len(batch_g_pair)
                epoch_avg_loss = epoch_loss_sum/g_pairs_num
                epochs.set_description("Epoch %d (Loss_Per_Pair=%g)" % (epoch,round(epoch_avg_loss, 10))) #得到当前每对的平均loss

            
            print(f"\nEpoch {epoch} completed, now evaluating on the validation set.")
            metric = self.score(mode='eval')  # 验证集评估
            print(f'eval metric: {str(round(metric, 10))}')

            if metric < best_metric:
                print(f"New best model found at epoch {epoch} with metric {str(round(metric, 10))}. Saving model.")
                best_metric=metric
                self.save(epoch,metric)
           
    def score(self,mode):
        """
        Scoring on the test set.
        """
        
        self.model = self.model.cuda()  

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.args.learning_rate,
                                          weight_decay=self.args.weight_decay)
        if self.args.load_path:
            _=self.load()


        self.model.eval()

        if mode =='test':
            g_pairs=self.test_g_pairs  
        elif mode == 'eval':   
            g_pairs=self.valid_g_pairs
        else:
            print('error mode .please use test or eval')
            import sys
            sys.exit(1)

        with torch.no_grad():
            loss_sum = 0 #统计总loss
            batches = self.create_batches(g_pairs)
            g_pairs_num = 0
            for index, batch_g_pair in tqdm(enumerate(batches), total=len(batches), desc="Batches"):
                batch_avg_loss_per_pair = self.process_batch_g_pair(batch_g_pair,mode) #得到batch中每对的平均loss
                g_pairs_num=g_pairs_num+len(batch_g_pair)
                loss_sum = loss_sum + batch_avg_loss_per_pair*len(batch_g_pair)

            
            return loss_sum/g_pairs_num  #返回验证集/测试集中每对的平均loss




    def save(self,epoch,metric):
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # 时间格式: 年月日_时分秒

        filename = f"{current_time}_epoch={epoch}_metric={metric:.5f}.pth"
        
        save_path = f"{self.args.save_dir}/{filename}"
        checkpoint = {
        'model_state_dict': self.model.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        'epoch': epoch,
        'metric': metric
        }
        torch.save(checkpoint, save_path)
        print(f"Model and optimizer state saved to {save_path}.")

    def load(self):
        checkpoint = torch.load(self.args.load_path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.args.learning_rate     

        rt={
            'epoch':checkpoint['epoch'],
            'metric':checkpoint['metric']
        }
        return rt


def predit(args,predit_homoG_dir,predit_result_dir):
    
    model = Binary2vec(args)
            
    model = model.cuda()  
    checkpoint = torch.load(args.load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()

    
    with torch.no_grad():
        for root, dirs, files in os.walk(predit_homoG_dir):
            for id,file in enumerate(files):
                homoG=torch.load(os.path.join(root, file))
                bin_name=file.rsplit('.strip.preprocessed.pth', 1)[0]
                batch_g = Batch.from_data_list([homoG],exclude_keys=['g_label','pe'])
                g_vec=model(batch_g)
                result_file_path=os.path.join(predit_result_dir,bin_name+'.pth')
                torch.save(g_vec.cpu(), result_file_path)





    #exp_saving
#    with torch.no_grad():
#        with open(predit_result_file, 'w') as f:
#            writer = csv.writer(f)
#            for root, dirs, files in os.walk(predit_homoG_dir):
#                files.sort()
#                for id,file in enumerate(files):
#                    homoG=torch.load(os.path.join(root, file))
#                    batch_g = Batch.from_data_list([homoG],exclude_keys=['g_label','pe'])
#                    g_vec=model(batch_g)
#                    g_vec_flat=g_vec.cpu().flatten()
#                    g_vec_list = [val.item() for val in g_vec_flat]
#                    row = [id + 1] + g_vec_list
#                    writer.writerow(row)