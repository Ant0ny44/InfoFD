import torch

def gaussian(features:torch.Tensor, prompts:torch.Tensor, gt:torch.Tensor, shift_intensity:float,intensity:float):

    noisy =  torch.rand_like(features[0]) * shift_intensity* 2 - shift_intensity + torch.randn_like(features) * intensity
    noisy = noisy.cpu()
    noisy[gt == 0] = torch.zeros(features.shape[1])
    features_aug = features + noisy.to(features.device)
    prompts_aug = prompts
    gt_aug = gt

    return features_aug, prompts_aug, gt_aug

def fcn(features:torch.Tensor, prompts:torch.Tensor, gt:torch.Tensor, shift_intensity:float,intensity:float):
    
    features_normalized = torch.nn.functional.normalize(features, dim=0)
    domain_shift = torch.randn(features_normalized.shape[0],features_normalized.shape[1]) * shift_intensity
    aug_intensity = torch.rand(features_normalized.shape[0],features_normalized.shape[1]) * intensity
    noisy = torch.normal(mean=domain_shift, std=aug_intensity)
   
    noisy[gt == 0] = torch.zeros(features_normalized.shape[1])
    features_aug = features_normalized + noisy.to(features_normalized.device)
    prompts_aug = prompts
    gt_aug = gt

    return features_aug, prompts_aug, gt_aug