from train import IBTrainer
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, accuracy_score
import numpy as np
import torch
import pickle
from models.IBEncoder import IBEncoder
from models import load_model

from utils.argument_parsers import EvalArgs

class IBEvaluator(IBTrainer):

    def __init__(self):

        self.args = EvalArgs().generate()
        self.config = self.load_config()
        self.setseed()
        _, _, self.test_loaders = self.get_dataloader()

        self.model:IBEncoder = load_model(self.config)

        self.model.to(self.args.device)
        self.load_model_state()

    def eval(self) -> None:
        
        self.model.eval()
        results:str = f"Test on {self.config['data']['test_root']}: \n"
        ap_dict:dict[str:float] = dict()
        acc_dict:dict[str:float] = dict()
        real_acc_dict:dict[str:float] = dict()
        fake_acc_dict:dict[str:float] = dict()
        f1_dict:dict[str:float] = dict()
        auroc_dict:dict[str:float] = dict()
        fpr95_dict:dict[str:float] = dict()

        for data_name,loader in self.test_loaders.items():

            single_result = f'{data_name}: '
    
            fpr95, auroc, f1, ap, r_acc0, f_acc0, acc0 = self.compute_sores(loader)
            
            single_result += f"AP-{round(ap*100, 2)}-"
            single_result += f"ACC-{round(acc0*100, 2)}-"
            single_result += f"rACC-{round(r_acc0*100, 2)}-"
            single_result += f"fACC-{round(f_acc0*100, 2)}-"
            single_result += f"F1-{round(f_acc0*100, 2)}-"
            single_result += f"AUROC-{round(auroc, 2)}-"
            single_result += f"FPR95-{round(fpr95 * 100, 2)}\n"
            print(single_result)

            with open(self.args.log_dir, 'a') as f:

                f.write(single_result)

            ap_dict.update({data_name: ap * 100})
            auroc_dict.update({data_name: auroc * 100})
            fpr95_dict.update({data_name: fpr95 * 100})
            f1_dict.update({data_name: f1 * 100}) 
            acc_dict.update({data_name: acc0 * 100})
            real_acc_dict.update({data_name: r_acc0 * 100}), 
            fake_acc_dict.update({data_name: f_acc0 * 100})



            ap_dict.update({'mAP':round(np.mean(list(ap_dict.values())), 2)})
            acc_dict.update({'mACC':round(np.mean(list(acc_dict.values())), 2)})
            f1_dict.update({'F1 score':round(np.mean(list(f1_dict.values())), 2)})
    
            ap_dict.update({data_name: ap * 100}), 
            acc_dict.update({data_name: acc0 * 100})
            real_acc_dict.update({data_name: r_acc0 * 100}), 
            fake_acc_dict.update({data_name: f_acc0 * 100})

        
        results += f"mAP-{round(np.mean(list(ap_dict.values())), 2)}-"
        results += f"mACC-{round(np.mean(list(acc_dict.values())), 2)}-" 
        results += f"mF1-{round(np.mean(list(f1_dict.values())), 2)}-"
        results += f"mAUROC-{round(np.mean(list(auroc_dict.values())), 2)}-"
        results += f"mFPR95-{round(np.mean(list(fpr95_dict.values())), 2)}\n"


        print(results)

        with open(self.args.log_dir, 'a') as f:

            f.write(results)

    def compute_sores(self, loader):

        with torch.no_grad():

            y_true, y_pred = [], []
            for img, label in loader:

                in_tens = img.to(self.args.device)
                labels_cuda = label.to(self.args.device)
                _,pred_labels,z = self.model(in_tens,None, labels_cuda,z_required = True)  
                y_pred.extend(pred_labels.detach().cpu().tolist())
                y_true.extend(label.flatten().tolist())
            

        y_true, y_pred = np.array(y_true), np.array(y_pred)

        # ================== save this if you want to plot the curves =========== # 
        # torch.save( torch.stack( [torch.tensor(y_true), torch.tensor(y_pred)] ),  'baseline_predication_for_pr_roc_curve.pth' )
        # exit()
        # =================================================================== #
        
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

    def start(self) -> None:

        self.eval()

        if self.args.z_required:

            with open(self.args.save_path, 'wb') as f:

                pickle.dump(self.z_dict, f)

if __name__ == '__main__':
    evaluator = IBEvaluator()
    evaluator.start()