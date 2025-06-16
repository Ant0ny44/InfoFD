import torch.nn
import torch
import pickle
import torch.nn.functional as F
from torch import linalg as LA
from sympy.matrices import Matrix, GramSchmidt
from  utils import aug_strategies
from typing import Tuple
from sklearn.decomposition import PCA

class IBEncoder(torch.nn.Module):


    def __init__(self, hidden_dim:int, 
                 input_dim:int = 768, 
                 num_classes:int = 2,
                 lambda_0 = 1e-1,lambda_1 = 1e-1,
                 prior_type:str = 'both',
                 projection_type:str = 'cut',
                 aug_strategy:str = 'gaussian',
                 aug_intensity:float = 0.01,
                 shift_intensity:float = 0.3,
                 lp:int = 4096):
        '''
        `prior_type`: Should be `no-cond`,`normal`,`random`, `fixed` or `both`.

        `projection_type`: Should be `cut` `pca` or  `linear`.
        '''
        super().__init__()
       
        self.num_classes:int = num_classes
        self.hidden_dim:int = hidden_dim
        self.mu_encoder:torch.nn.Linear = torch.nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.sigma_encoder:torch.nn.Linear = torch.nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.prior:torch.Tensor = torch.rand((self.num_classes, self.hidden_dim)) 
        self.projection_type:str = projection_type
        self.aug_intensity:float = aug_intensity
        self.shift_intensity:float = shift_intensity
        self.aug_strategies:dict[str|function] = self.generate_aug_strategies()[aug_strategy]
        self.batchnorm:torch.nn.BatchNorm1d = torch.nn.BatchNorm1d(num_features = input_dim)
        self.lp:int = lp
        self.prior_memory_bank:torch.Tensor = torch.zeros(2, self.hidden_dim)

        if self.projection_type == 'linear':

            self.proj:torch.nn.Linear = torch.nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=False)

        elif self.projection_type == 'pca':

            self.proj = PCA(n_components = self.hidden_dim)

        self.decoder:torch.nn.Linear = torch.nn.Linear(in_features=hidden_dim , out_features=1) # To predict 0 or 1
        self.prior_type:str = prior_type

        if self.prior_type == 'fixed':

            with open('/home/qinhaotian/projects/TGIBE-FakeDetection/An image of real or fake.pkl', 'rb') as f:
                
                self.prior:torch.Tensor = pickle.load(f).float()
                # self.prior[0] = torch.randn_like(self.prior[0])
                self.prior[0], self.prior[1] = self.prior[0],self.prior[1]

        elif self.prior_type == 'random':
            
            with open('/home/qinhaotian/projects/TGIBE-FakeDetection/random_charcters_768.pkl', 'rb') as f:
                
                rand_feature:torch.Tensor = torch.nn.functional.normalize(pickle.load(f).float(),dim = 0)
                
                self.prior:torch.Tensor = torch.stack([rand_feature, torch.randn_like(rand_feature)], dim = 0) # Use dummy vector.
                
        elif self.prior_type == 'no-cond':
            
            self.prior:torch.Tensor = torch.randn((1,self.hidden_dim))
            
        self.regulation_coef:list[float, float] = [lambda_0, lambda_1]

        self.cls_loss_func:torch.nn.Module = torch.nn.BCEWithLogitsLoss()

        self.initialize(hidden_dim)

    def generate_aug_strategies(self):

        return  {'gaussian':aug_strategies.gaussian,'fcn':aug_strategies.fcn}

    def reparameterize(self, mu:torch.Tensor, sigma:torch.Tensor, noise_ratio:int = 0.1)-> torch.Tensor:
        '''
        Reparameterization trick.
        .. Inputs::
            `mu`:[B, hidden_dim].

            `sigma`:[B, hidden_dim].
        .. Returns::
            sample: [B, hidden_dim].
        '''
        std = sigma.mul(noise_ratio).exp()
        eps = torch.rand_like(std)

        return std.mul(eps) + mu
    
    def initialize(self, hidden_dim:int) -> None:
            
        torch.nn.init.xavier_uniform_(self.mu_encoder.weight, gain=torch.nn.init.calculate_gain('linear'))
        torch.nn.init.xavier_uniform_(self.sigma_encoder.weight, gain=torch.nn.init.calculate_gain('linear'))
        torch.nn.init.xavier_uniform_(self.decoder.weight, gain=torch.nn.init.calculate_gain('linear'))

        torch.nn.init.orthogonal_(self.prior)

        if self.projection_type == 'linear':

            torch.nn.init.xavier_uniform_(self.proj.weight, gain=torch.nn.init.calculate_gain('linear')) 

    def forward(self, img_features:torch.FloatTensor,
                prompts:torch.FloatTensor, 
                y_gt:torch.FloatTensor,
                debug_mode:float = False,
                z_required = False) -> Tuple[torch.Tensor,torch.Tensor]|Tuple[torch.Tensor,torch.Tensor,torch.Tensor]|Tuple[torch.Tensor,torch.Tensor, torch.Tensor,torch.Tensor,torch.Tensor]|Tuple[torch.Tensor,torch.Tensor, torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:
        '''
        Sample a feature based on reparameterization trick. `y_gt` consists of `0` or `1`, which drives the encoder to fit, and calculate loss.

        .. Inputs::
            `img_features`: Shape is required as [B, `input_dim`].

            `prompts`: Shape is required as [B, `input_dim`].

            `y_gt`: Shape is required as [B, 1].

            `debug_mode`: `total_loss`, `pred_labels`, `cls_loss`, `mmd_loss` and `l2_z_mean` will be returned if `debug_mode` is `True`, otherwise, only returns `total_loss` and `pred_labels`.

            `z_required`: whether to return `z_mean`.

        ..Outputs::
            `total_loss` and `pred_labels`, `pred_labels` `s shape is [B, 1].
        '''
        # 1. Calculate mean and variance.
        if self.training:

            if torch.rand(1) <2:

                img_features, prompts, y_gt = self.aug_strategies(img_features, prompts, y_gt, self.shift_intensity, self.aug_intensity)
            
            img_features = self.batchnorm(img_features)
            prompts = self.batchnorm(prompts)


        mu = self.mu_encoder(img_features)
        var = self.sigma_encoder(img_features)
        var = torch.clamp(var, max=100)

        # 2. Sample.(Only in training.)
        if self.training:
        
            z = self.reparameterize(mu, var)

        else:

            z = mu

        # 3. Predict.
        y_pred = self.decoder(mu) # [B, num_classes]

        if self.training:

            self.update_memory_bank(prompts, y_gt)

        # 4. Calculate MMD loss with prior.
        mmd_loss, l2_z_mean, z_mean = self.get_mmd_loss(z, prompts, y_gt, self.num_classes)

        # 5. Calculate classification loss.
        cls_loss = self.cls_loss_func(y_pred.squeeze(1), y_gt)

        # 6. Calculate total loss and predicted labels.
        loss_total:torch.Tensor = self.regulation_coef[0] * cls_loss +\
                                  self.regulation_coef[1] * mmd_loss
        
        with torch.no_grad():

            pred_labels:torch.Tensor = y_pred.sigmoid().flatten()

        return_list = [loss_total, pred_labels]

        if debug_mode:

            return_list = [*return_list, cls_loss ,mmd_loss ,l2_z_mean]
        
        if z_required:

            return_list.append(z)

        return tuple(return_list)
    
    def update_memory_bank(self, prompts:torch.Tensor, y_gt:torch.Tensor)-> None:

        if self.projection_type == 'cut':

            prompts = torch.topk(prompts, k = self.hidden_dim, dim=1,)[0]
            
        elif self.projection_type == 'linear':

            prompts = self.proj(prompts)

        elif self.projection_type == 'pca':

            prompts = torch.Tensor(self.proj.fit_transform(prompts.cpu().detach().numpy())).to(prompts.device) # [B, N]

        if self.prior_type == 'both':    # Two side guided.
            
            prior:torch.Tensor = torch.stack([prompts[y_gt == 0].to(prompts.device).detach().sum(dim = 0), prompts[y_gt == 1].to(prompts.device).detach().sum(dim = 0)], dim = 0) # [2, hidden]
            real_len = prompts[y_gt == 0].shape[0]
            fake_len = prompts[y_gt == 1].shape[0]

        elif self.prior_type == 'single': # Single side.
            
            prior:torch.Tensor = torch.stack([prompts[y_gt == 0].to(prompts.device).detach().sum(dim = 0), torch.randn_like(prompts[y_gt == 0].to(prompts.device).detach().sum(dim = 0)).to(prompts.device)], dim = 0) # [2, hidden]
            real_len = prompts[y_gt == 0].shape[0]
            fake_len = prompts[y_gt == 1].shape[0]
        else:

            return 
        
        self.prior_memory_bank = self.prior_memory_bank.to(prompts.device)
        
        if self.prior_memory_bank[0].sum() == 0 and self.prior_memory_bank[1].sum() == 0:
             
            self.prior_memory_bank[0] = prior[0] / real_len 
            self.prior_memory_bank[1] = prior[1] / fake_len 

        else:
    
            self.prior_memory_bank[0] = (self.prior_memory_bank[0] * self.lp + prior[0]) /(self.lp + real_len)  # Eq. 7.1
            self.prior_memory_bank[1] = (self.prior_memory_bank[1] * self.lp + prior[1]) /(self.lp + fake_len)  # Eq. 7.2
 
    def get_mmd_loss(self, z:torch.FloatTensor, prompts:torch.FloatTensor, y_gt:torch.FloatTensor, num_cls:int):

        z_mean = torch.stack([z[y_gt==i_cls].mean(dim=0) for i_cls in range(num_cls)], dim=0)
        l2_z_mean= LA.norm(z.mean(dim=0), ord=2)
        
        if not self.training:

            return 0, l2_z_mean, z_mean
                    
        if self.prior_type == 'both':    # Two side guided.

            # mmd_loss = F.kl_div(z_mean, self.prior_memory_bank,reduction='sum') # Nan.
            prior = self.gram_schmidt(self.prior_memory_bank[0], self.prior_memory_bank[1])
            mmd_loss = F.mse_loss(z_mean, self.prior_memory_bank)

        elif self.prior_type == 'no-cond':  # Use N(0,1) for prior.

            # mmd_loss = F.mse_loss(z, self.prior.repeat(z.shape[0],1).to(z.device)) 
            mmd_loss = F.mse_loss(z.mean(dim = 0), (self.prior_memory_bank[0]+ self.prior_memory_bank[1]).to(z.device)/2) 


        elif self.prior_type == 'single':  # Single side.
            pri = self.prior_memory_bank        
            mmd_loss = F.mse_loss(z_mean, pri) 

        elif self.prior_type == 'only-t':  # Only text.
            
            mmd_loss = F.mse_loss(z.mean(dim=0), self.proj(prompts.to(prompts.device)).mean(dim=0)) 

        elif self.prior_type == 'random' or self.prior_type == 'fixed': # fixed, random.
            
            prior = torch.topk(self.prior,k = self.hidden_dim, dim=1,)[0].to(z.device)
            prior = self.gram_schmidt(prior[0], prior[1])
            # prior[0],prior[1] = prior[1], prior[0]
            mmd_loss = F.mse_loss(z_mean, prior) 
            
        else: # normal.

            # mmd_loss = F.mse_loss(z_mean, self.prior.to(z.device)) 
            mmd_loss = F.mse_loss(z, self.proj(prompts.to(z.device))) 
            
        return mmd_loss, l2_z_mean, z_mean
    
    def gram_schmidt(self, vector1:torch.FloatTensor,vector2:torch.FloatTensor):
    
        l = [Matrix(vector1.reshape(1, -1).cpu()), Matrix(vector2.reshape(1, -1).cpu())]
        # Normalize

        matrix = GramSchmidt(l,orthonormal=True)
    
        return torch.Tensor(matrix).to(vector1.device)