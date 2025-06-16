import torch
from .clip.clip import load
from .IBEncoder import IBEncoder

def load_model(config:dict[str:str|float|int]) -> torch.nn.Module:

    model:IBEncoder = IBEncoder(config['model']['hidden_dim'], 
            input_dim=config['model']['input_dim'],
            num_classes=config['model']['num_classes'],
            lambda_0=config['optimizer']['lambda_0'], 
            lambda_1=config['optimizer']['lambda_1'], 
            prior_type =config['model']['prior_type'],
            projection_type=config['model']['proj_type'],
            aug_strategy =config['model']['aug_strategy'],
            aug_intensity =config['model']['aug_intensity'],
            shift_intensity=config['model']['shift_intensity'],)
        
    return model 