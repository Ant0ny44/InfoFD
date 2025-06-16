import torch
import wandb
class WandbLogger():

    def __init__(self, 
                 user_token:str,
                 project_name:str, 
                 experiment_name:str, 
                 project_configs:dict[str:str|float|int],
                 model:torch.nn.Module = None,
                 loss_func:torch.nn.Module = None,
                 offline = False, 
                 watch_type = 'all',
                 log_freq:int = 50,
                 tags:list[str] = None) -> None:

        wandb.login(key=user_token)
        wandb.init(project=project_name, 
                   name=experiment_name, 
                   tags=tags,
                   config=project_configs,
                   mode='offline' if offline else 'online')
        self.metrics_cache:dict[str:float] = dict()
        self.table_cache:dict[str:wandb.Table] = dict()
        
        if model and loss_func is not None:

            wandb.watch(model, loss_func, log_freq=log_freq, log=watch_type)

    def add_log(self, log:dict[str:str|float|int]) -> None:
        '''
        Add log to the metrics_cache. 
        '''
        self.metrics_cache.update(log)
    
    def add_multi_log(self, table_name:str, logs:dict[str:str|float|int]) -> None:
        '''
        Add multi-line to one chart.
        '''
        if table_name in list(self.table_cache.keys()):
            
            self.table_cache[table_name].add_data(*list(logs.values()))

        else:

            table = wandb.Table(columns=list(logs.keys()))
            table.add_data(*list(logs.values()))
            self.table_cache.update({table_name:table})

    def upload_table(self) -> None:
        '''
        Only update table data.
        '''
        for name, table in self.table_cache.items():
            
             wandb.log({name:wandb.plot.line(
            table, "step", "Value", title="Test result"

        )})
             
    def upload(self) -> None:
        '''
        Upload the metrics and clean the metrics_cache.
        '''
       
        wandb.log(self.metrics_cache)
        self.metrics_cache = dict()
        
    def exit(self) -> None:
        '''
        Stop logging.
        '''
        self.upload()
        wandb.finish()