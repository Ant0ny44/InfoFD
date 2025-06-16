import argparse

class TrainArgs():

    def __init__(self) -> None:
        pass
    
    def generate(self)-> argparse.Namespace:
        
        parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        
        parser.add_argument('--project_name', type=str, default='InfoFD')
        parser.add_argument('--exp_name', type=str, default='auto')
        parser.add_argument('--tags', nargs='+', default=None)
        parser.add_argument('--log_freq', type=int, default=50)
        parser.add_argument('--save_pth', action='store_true')
    
        parser.add_argument('--ckpg_path', default='./ckpgs/final.pth', type=str)
        parser.add_argument('--config_path', default='./configs/EP1.yml', type=str)
        parser.add_argument('--text_features', default=None, type=str)

        parser.add_argument('--device', type=str, default='cuda:0')
        parser.add_argument('--seed',type=int, default=7310)

        parser.add_argument('--wandb_offline', action='store_true')

        return parser.parse_args() 

class EvalArgs():

    def __init__(self) -> None:

        pass

    def generate(self) ->argparse.Namespace:

        parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument('--device', type=str, default='cuda:1')
        parser.add_argument('--seed',type=int, default=7310)
        
        parser.add_argument('--save_path', default='./z_hidden.pkl', type=str)
        
        parser.add_argument('--ckpg_path', default='./ckpgs/final.pth', type=str)
        parser.add_argument('--config_path', default='./configs/EP1.yml', type=str)
        parser.add_argument('--log_dir',type=str,default='./eval.log')
        
        return parser.parse_args()

class PreprocessArgs():

    def __init__(self) -> None:

        pass

    def generate(self) -> argparse.Namespace:

        parser:argparse.ArgumentParser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        
        parser.add_argument('--real_path', type=str, required=True)
        parser.add_argument('--fake_path', type=str, required=True)

        parser.add_argument('--save_dir', type=str, required=True)
        parser.add_argument('--max_sample', type=int, default=1000)
        
        parser.add_argument('--model_name', type=str, default='CLIP:ViT-L/14')
        parser.add_argument('--device', type=str, default='cuda:0')
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--num_workers', type=int, default=14)

        return parser.parse_args() 
