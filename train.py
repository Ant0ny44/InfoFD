from sklearn.metrics import average_precision_score, accuracy_score, f1_score, roc_auc_score,roc_curve
from torch.utils.data import TensorDataset, DataLoader
from data.datasets import PickleDataset
from utils.argument_parsers import TrainArgs
from torch.optim import Optimizer, SGD, Adam
from torch.optim.lr_scheduler import LRScheduler, PolynomialLR
from utils.logger import WandbLogger
from utils.preprocess import preprocess_test_data, preprocess_train_data,preprocess_val_data
from models.IBEncoder import IBEncoder
from models import load_model
from tqdm import tqdm
import configparser 
import numpy as np
import argparse
import datetime
import random
import pickle
import torch
import yaml
import os

class IBTrainer():

    def __init__(self) -> None:
        
        self.args:argparse.Namespace = TrainArgs().generate()
        self.config = self.load_config()
        self.setseed()
        self.best_acc:float = -0.01
        self.dataloader,self.val_dataloader, self.test_loaders = self.get_dataloader()

        self.model:IBEncoder = load_model(self.config)

        self.model.to(self.args.device)

        self.optimizer, self.scheduler = self.load_optimizer()

        self.wandb_token = self.load_ini()

        self.logger = self.get_logger()

    def load_ini(self) -> str:
        '''
        Load env.ini.
        '''
        config = configparser.ConfigParser()
        config.read('env.ini')
        wandb_token:str = config.get('WANDB','TOKEN')
        config.clear()
        return wandb_token
    
    def get_logger(self) -> WandbLogger:

        if self.args.exp_name == 'auto':

            exp_name = 'exp-'+str(datetime.datetime.now().timestamp())

        else:

            exp_name = self.args.exp_name

        logger = WandbLogger(project_name=self.args.project_name, 
                             experiment_name=exp_name, 
                             user_token=self.wandb_token, 
                             project_configs={**self.config, **self.args.__dict__},
                             model = self.model,
                             loss_func=self.model.cls_loss_func,
                             offline=self.args.wandb_offline,
                             log_freq=self.args.log_freq,
                             tags=self.args.tags)
        
        return logger
    

    def setseed(self):

        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)

    def load_optimizer(self) ->tuple[Optimizer,LRScheduler]:
        
        if self.config['optimizer']['opt'] == 'Adam':

            optimizer = Adam(params=self.model.parameters(), lr = self.config['optimizer']['base_lr'])

        else:

            optimizer = SGD(params=self.model.parameters(), lr = self.config['optimizer']['base_lr'])

        scheduler = PolynomialLR(optimizer, total_iters=self.config['train']['max_epoch'] - self.config['optimizer']['decay_epoch'] , verbose=False, power=self.config['optimizer']['scheduler_power'])

        return optimizer, scheduler
    
    def save_model(self)->None:
        
        torch.save(self.model.state_dict(),  self.config['train']['final_save_path'])

    def load_model_state(self)->None:
        
        self.model.load_state_dict(torch.load(self.args.ckpg_path))

    def get_dataloader(self) -> DataLoader:


        print("Using cached data...")

        print('Loading train features...')

        if os.path.exists(self.config['data']['train_root_cache']) and os.path.exists(self.config['data']['val_root_cache']) and os.path.exists(self.config['data']['test_root_cache']):

            print("Cached data found, loading...")
            
        else:
                
            print("Cached data not found, generating...")
            preprocess_val_data(batch_size=512,
                                     device=self.args.device,
                                     model_name='CLIP:ViT-L/14', 
                                     val_data_path=self.config['data']['val_root'],
                                     save_path=self.config['data']['val_root_cache'],
                            )
            preprocess_test_data(batch_size=512,
                                     device=self.args.device,
                                    
                                     model_name='CLIP:ViT-L/14', 
                                     test_data_path=self.config['data']['test_root'],
                                     save_path=self.config['data']['test_root_cache'],
                            )

            preprocess_train_data(batch_size=512,
                                      device=self.args.device,
                                      model_name='CLIP:ViT-L/14',
                                      train_captions_path=self.config['data']['train_captions_path'],
                                      train_img_root=self.config['data']['train_root'],
                                      save_path=self.config['data']['train_root_cache'])

        dataset = PickleDataset(self.config['data']['train_root_cache'])

        train_dataloader:DataLoader = DataLoader(dataset=dataset, 
                                            batch_size=self.config['data']['batch_size'], 
                                            shuffle=self.config['data']['shuffle'],
                                           num_workers=self.config['data']['num_workers'], drop_last = True)
        
        print("Loading val pickle features...")
       
        with open(self.config['data']['val_root_cache'], 'rb') as f:

            features, labels = pickle.load(f)

        val_dataset = TensorDataset(features, labels)
        val_dataloader:DataLoader = DataLoader(dataset=val_dataset, 
                                            batch_size=self.config['data']['batch_size'], 
                                            shuffle=self.config['data']['shuffle'],
                                           num_workers=self.config['data']['num_workers'], drop_last = True)
        
        print("Loading test pickle features...")
        
        with open(self.config['data']['test_root_cache'], 'rb') as f:

            datasets = pickle.load(f)

        print(list(datasets.keys()))
        test_loaders:dict[str, DataLoader] = {}
            
        for dataset_name, data in datasets.items():
            temp_dataset = TensorDataset(data[0], data[1])
            temp_loader = DataLoader(dataset=temp_dataset, 
                                            batch_size=self.config['data']['batch_size'], 
                                            num_workers=self.config['data']['num_workers'])
            test_loaders.update({dataset_name:temp_loader})

        return train_dataloader, val_dataloader, test_loaders
    
    def load_config(self)->dict[str:str|float|int]:

        with open(self.args.config_path, 'r') as f:

            file_content:str = f.read()
        
        configs:dict[str:str|float|int] = yaml.load(file_content, yaml.Loader)

        return configs

    def start(self) -> None:

        for epoch in range(self.config['train']['max_epoch']):
            
            self.model.train()
            num_processed:int = 0
            y_gt_all:torch.Tensor = torch.zeros(len(self.dataloader.dataset))
            y_pred_all:torch.Tensor = torch.zeros(len(self.dataloader.dataset))

            for batch in tqdm(self.dataloader):
                
                if self.config['data']['prompts'] == False:

                    X, y_gt = batch
                    y_gt = torch.Tensor(y_gt)
                    prompts = torch.randn_like(X)

                else:

                    prompts, X, y_gt = batch
                    y_gt = torch.Tensor(y_gt[0])

                prompts = prompts.to(self.args.device)
                X = X.to(self.args.device)
                y_gt_all[num_processed:num_processed + y_gt.shape[0]] = y_gt
                y_gt = y_gt.to(self.args.device)

                loss_total, pred_labels, cls_loss ,mmd_loss ,l2_z_mean = self.model(X, prompts, y_gt, True)

                y_pred_all[num_processed:num_processed +  y_gt.shape[0]] = pred_labels.detach().cpu()
                loss_total.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                num_processed += y_gt.shape[0]
                self.logger.add_log({'Train cls loss':cls_loss.detach().cpu()})

                if mmd_loss == 0:

                    self.logger.add_log({'Train mmd loss':0})

                else:

                    self.logger.add_log({'Train mmd loss':mmd_loss.detach().cpu()})

                if l2_z_mean == 0:

                    self.logger.add_log({'Train l2_z_mean':0})

                else:

                    self.logger.add_log({'Train l2_z_mean':l2_z_mean.detach().cpu()})
                
            # Adjusts LR.
            if epoch >= self.config['optimizer']['decay_epoch']:

                self.scheduler.step()

            AP:float = average_precision_score(y_gt_all.numpy(), y_pred_all.numpy())

            train_acc = accuracy_score(y_gt_all.numpy(), y_pred_all.numpy() > 0.5)
            self.logger.add_log({'Train ACC':round(train_acc *100, 2)})
            self.logger.add_log({'Train AP':round(AP *100, 2)})
            print("Train ACC: {:.2f}".format(train_acc *100))
            
            self.logger.upload()

            if epoch % self.config['train']['val_interval'] == 0:

                self.validate(epoch)

            self.logger.upload()

        if self.config['train']['test_after_train']:

            self.test(epoch)
            
            self.test_on_best(epoch)

        self.logger.upload()
        self.logger.upload_table()

        self.save_model()

    def validate(self, epoch:int) -> None:
        
        self.model.eval()
        results:str = f'Validation after epoch {epoch}: '

        data_name = 'Validation set'
        fpr95, auroc, f1, ap, r_acc0, f_acc0, acc0 = self.compute_sores(self.val_dataloader)
        results += "{0}-{1:.2f}-{2:.2f}, ".format(data_name, ap * 100, acc0 * 100)

        results += f"AP-{round(ap * 100, 2)}-"
        results += f"ACC-{round(acc0 * 100, 2)}-" 
        results += f"F1-{round(f1, 2)}-"
        results += f"AUROC-{round(auroc, 2)}-"
        results += f"FPR95-{round(fpr95, 2)}"

        self.logger.add_log({'Valid AP':round(ap * 100, 2)})
        self.logger.add_log({'Valid F1':round(f1, 2)})
        self.logger.add_log({'Valid Acc':round(acc0 * 100, 2)})
        self.logger.add_log({'Valid rAcc':round(r_acc0 * 100, 2)})
        self.logger.add_log({'Valid fAcc':round(f_acc0 * 100, 2)})
        self.logger.add_log({'Valid AUROC':round(auroc, 2)})
        self.logger.add_log({'Valid FPR95':round(fpr95, 2)})

        print(results)


        # Save the best.
        if acc0 * 100 > self.best_acc:
            print(f'Validation accuracy improved from {self.best_acc:.2f}% to {acc0 * 100 :.2f}%. Saving model...')
            self.best_acc = acc0 * 100
            torch.save(self.model.state_dict(), self.config['train']['best_save_path'])

        # Save the last.
        self.save_model()

    def test(self, epoch:int) -> None:
        
        self.model.eval()
        results:str = f'Test after epoch {epoch}: '
        ap_dict:dict[str:float] = dict()
        acc_dict:dict[str:float] = dict()
        real_acc_dict:dict[str:float] = dict()
        fake_acc_dict:dict[str:float] = dict()
        f1_dict:dict[str:float] = dict()
        auroc_dict:dict[str:float] = dict()
        fpr95_dict:dict[str:float] = dict()
 

        for data_name,loader in tqdm(self.test_loaders.items()):

            fpr95, auroc, f1, ap, r_acc0, f_acc0, acc0 = self.compute_sores(loader)
            results += "{0}-{1:.2f}-{2:.2f}, ".format(data_name, ap * 100, acc0 * 100)

            ap_dict.update({'Test-' + data_name: ap * 100})
            auroc_dict.update({'Test-' + data_name: auroc * 100})
            fpr95_dict.update({'Test-' + data_name: fpr95 * 100})
            f1_dict.update({'Test-' + data_name: f1 * 100}) 
            acc_dict.update({'Test-' + data_name: acc0 * 100})
            real_acc_dict.update({'Test-' + data_name: r_acc0 * 100}), 
            fake_acc_dict.update({'Test-' + data_name: f_acc0 * 100})


        results += f"mAP-{round(np.mean(list(ap_dict.values())), 2)}-"
        results += f"mACC-{round(np.mean(list(acc_dict.values())), 2)}-" 
        results += f"mF1-{round(np.mean(list(f1_dict.values())), 2)}-"
        results += f"mAUROC-{round(np.mean(list(auroc_dict.values())), 2)}-"
        results += f"mFPR95-{round(np.mean(list(fpr95_dict.values())), 2)}"
        ap_dict.update({'mAP':round(np.mean(list(ap_dict.values())), 2)})
        acc_dict.update({'mACC':round(np.mean(list(acc_dict.values())), 2)})
        f1_dict.update({'F1 score':round(np.mean(list(f1_dict.values())), 2)})
        self.logger.add_multi_log('Test AP',ap_dict)
        self.logger.add_multi_log('Test F1 score',f1_dict)
        self.logger.add_multi_log('Test Accuracy',acc_dict)
        self.logger.add_multi_log('Test real accuracy',real_acc_dict)
        self.logger.add_multi_log('Test fake accuracy',fake_acc_dict)
        self.logger.add_multi_log('Test AUROC',auroc_dict)
        self.logger.add_multi_log('TEst FPR95',fpr95_dict)
        self.logger.upload()
        print(results)


    def test_on_best(self, epoch:int) -> None:

        self.model.load_state_dict(torch.load(self.config['train']['best_save_path']))
        self.model.eval()
        results:str = f'Test after epoch {epoch}: '
        ap_dict:dict[str:float] = dict()
        acc_dict:dict[str:float] = dict()
        real_acc_dict:dict[str:float] = dict()
        fake_acc_dict:dict[str:float] = dict()
        f1_dict:dict[str:float] = dict()
        auroc_dict:dict[str:float] = dict()
        fpr95_dict:dict[str:float] = dict()
 

        for data_name,loader in tqdm(self.test_loaders.items()):

            fpr95, auroc, f1, ap, r_acc0, f_acc0, acc0 = self.compute_sores(loader)
            results += "{0}-{1:.2f}-{2:.2f}, ".format(data_name, ap * 100, acc0 * 100)

            ap_dict.update({'Best-' + data_name: ap * 100})
            auroc_dict.update({'Best-' + data_name: auroc * 100})
            fpr95_dict.update({'Best-' + data_name: fpr95 * 100})
            f1_dict.update({'Best-' + data_name: f1 * 100}) 
            acc_dict.update({'Best-' + data_name: acc0 * 100})
            real_acc_dict.update({'Best-' + data_name: r_acc0 * 100}), 
            fake_acc_dict.update({'Best-' + data_name: f_acc0 * 100})


        results += f"mAP-{round(np.mean(list(ap_dict.values())), 2)}-"
        results += f"mACC-{round(np.mean(list(acc_dict.values())), 2)}-" 
        results += f"mF1-{round(np.mean(list(f1_dict.values())), 2)}-"
        results += f"mAUROC-{round(np.mean(list(auroc_dict.values())), 2)}-"
        results += f"mFPR95-{round(np.mean(list(fpr95_dict.values())), 2)}"
        ap_dict.update({'mAP':round(np.mean(list(ap_dict.values())), 2)})
        acc_dict.update({'mACC':round(np.mean(list(acc_dict.values())), 2)})
        f1_dict.update({'F1 score':round(np.mean(list(f1_dict.values())), 2)})
        self.logger.add_multi_log('Best AP',ap_dict)
        self.logger.add_multi_log('Best F1 score',f1_dict)
        self.logger.add_multi_log('Best Accuracy',acc_dict)
        self.logger.add_multi_log('Best real accuracy',real_acc_dict)
        self.logger.add_multi_log('Best fake accuracy',fake_acc_dict)
        self.logger.add_multi_log('Best AUROC',auroc_dict)
        self.logger.add_multi_log('Best FPR95',fpr95_dict)
        self.logger.upload_table()
        print(results)

    def compute_sores(self, loader):

        with torch.no_grad():
            y_true, y_pred = [], []
            for batch in loader:
                img,label = batch
                in_tens = img.to(self.args.device)
                labels_cuda = label.to(self.args.device)
                _, pred_labels = self.model(in_tens,None, labels_cuda)
        
                y_pred.extend(pred_labels.detach().cpu().tolist())
                y_true.extend(label.flatten().tolist())

        y_true, y_pred = np.array(y_true), np.array(y_pred)
        
        # Get AP 
        ap = average_precision_score(y_true, y_pred)
        f1 = f1_score(y_true.astype(int), np.round(y_pred))

        # Calculate AUROC
        auroc = self.calculate_auroc(y_true, y_pred)

        # Calculate FPR95
        fpr95 = self.caculate_false_positive_rate(y_true, y_pred, 0.95)

        # Acc based on 0.5
        r_acc0, f_acc0, acc0 = self.calculate_acc(y_true, y_pred, 0.5)

        return fpr95, auroc, f1, ap, r_acc0, f_acc0, acc0

    def caculate_false_positive_rate(self, y_true, y_pred, tpr_at:float = 0.95):

        fpr_list, tpr_list, _ = roc_curve(y_true, y_pred)
        fpr = fpr_list[tpr_list >= tpr_at][0]
        return fpr
    
    def calculate_auroc(self, y_true, y_pred):

        auroc = roc_auc_score(y_true, y_pred)

        return auroc
    

    def calculate_acc(self, y_true, y_pred, thres)->tuple[float,float,float]:

        r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > thres)
        f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > thres)
        acc = accuracy_score(y_true, y_pred > thres)
        return r_acc, f_acc, acc    

    
if __name__ == '__main__':
    
    trainer = IBTrainer()

    trainer.start()
    
