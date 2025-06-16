from torch.utils.data import Dataset, DataLoader,TensorDataset
import torch
import pickle
import os
import numpy as np
from random import shuffle
from torchvision.transforms import transforms
from PIL import Image
from enum import Enum
from io import BytesIO
from scipy.ndimage.filters import gaussian_filter
import torchvision.transforms.functional as TF
from torchvision.transforms import transforms,InterpolationMode

class AugStrategies(Enum):

    @classmethod
    def jpeg_compression(dump, img:Image.Image, quality:int = 80) -> Image.Image:

        out = BytesIO()
        img.save(out, format='jpeg', quality=quality)
        img = Image.open(out).copy()
        out.close()
        
        return img
    
    @classmethod
    def gaussian_blur(dump, img:Image.Image, sigma:float = 1.0) -> Image.Image:
        
        img_aug:np.ndarray = np.array(img)
        gaussian_filter(img_aug[:,:,0], output=img_aug[:,:,0], sigma=sigma)
        gaussian_filter(img_aug[:,:,1], output=img_aug[:,:,1], sigma=sigma)
        gaussian_filter(img_aug[:,:,2], output=img_aug[:,:,2], sigma=sigma)
        return Image.fromarray(img_aug)
    
    @classmethod
    def resizing(dump, img:Image.Image, ratio:float = None, interpolation:Enum = InterpolationMode.BILINEAR) -> Image.Image:
        '''
        `interpolation` should be one of:
        `InterpolationMode.NEAREST`, 
        `InterpolationMode.NEAREST_EXACT`, 
        `InterpolationMode.BILINEAR`, 
        `InterpolationMode.BICUBIC`,  
        `InterpolationMode.BOX`, 
        `InterpolationMode.HAMMING`, 
        `InterpolationMode.LANCZOS`.
        '''
        if not ratio:

            ratio = np.random.rand(1)
            
        size = [int(img.size[0] * ratio), int(img.size[1] * ratio)]
        
        img_aug:Image.Image = TF.resize(img, size, interpolation = interpolation)
        
        return img_aug
    
class PickleDataset(Dataset):

    def __init__(self, pickle_path:str):

        super().__init__()

        with open(pickle_path, 'rb') as f:

            features, labels = pickle.load(f)

        self.features = TensorDataset(features)
        self.labels = TensorDataset(labels)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        prompt = self.features[index][0][0]
        img = self.features[index][0][1]
        label = self.labels[index] 
        return prompt, img, label
    
    def __len__(self)-> int:
        
        return len(self.features)
    
class ProGanDataset(Dataset):
    
    def __init__(self, root_path:str, loading_classes:list[str] = None, aug_strategies:list[tuple[AugStrategies, dict]] = None, raw_output = False):
        '''
        `root_path` should be like `aaaa/.../bbbb/`.
        ```
            bbbb			
            ├── airplane
                ├── 0_real
                ├── 1_fake
            │── bird
                ├── 0_real
                ├── 1_fake
            │── boat
                ├── 0_real
                ├── 1_fake
            │      ...
        ```
        '''
        super().__init__()
        
        real_list = self.__generate_file_paths( os.path.join(root_path), must_contain='0_real', loading_classes = loading_classes)
        fake_list = self.__generate_file_paths( os.path.join(root_path), must_contain='1_fake', loading_classes = loading_classes )
        self.raw_output:bool = raw_output
        self.file_paths = np.array(real_list + fake_list)
        self.labels = np.array([0] * len(real_list) + [1] * len(fake_list))

        random_idxes:list[int] = list(range(len(self.labels)))
        shuffle(random_idxes)
        self.file_paths = self.file_paths[random_idxes]
        self.labels = self.labels[random_idxes]
        self.file_paths = self.file_paths[:100000]
        self.labels = self.labels[:100000]
        self.__load_aug_strategies(aug_strategies)
    
    def __len__(self):

        return len(self.file_paths)
    
    def __load_aug_strategies(self, aug_strategies:list[tuple[AugStrategies, dict]] = None) -> None:
       
        aug_list:list = []
        if not aug_strategies:
            self.transform = transforms.Compose([
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
                ])
    
            
        else:
    
            for strategy, params in aug_strategies:
                
                aug_list.append(transforms.Lambda(lambda img: strategy(img, **params)))
            
            aug_list += [transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])]
        
            self.transform = transforms.Compose(aug_list)
        

    def __generate_file_paths(self, rootdir:str, must_contain:str, exts=["png", "jpg", "JPEG", "jpeg"], loading_classes:list[str] = None)->list[str]:

        out:list[str] = [] 
        for r, d, f in os.walk(rootdir):
            for file in f:
                if (file.split('.')[1] in exts)  and  (must_contain in os.path.join(r, file)):
                    if loading_classes == None or r.split('/')[-3] in loading_classes:
                        out.append(os.path.join(r, file))

        return out

    def __getitem__(self, idx:int) -> tuple[torch.Tensor, int]:

        img_pil:Image.Image = Image.open(self.file_paths[idx]).convert('RGB')
        label:int = self.labels[idx]
        img:torch.Tensor|Image.Image = img_pil if self.raw_output else self.transform(img_pil)

        return img, label
    

# Unit test PASS!
if __name__ == '__main__':
    
    dataset = PickleDataset('/mnt/sdb/data/qinhaotian/train_features.pkl')
    prompt, img, label = dataset[0]
    print(prompt.shape)
    print(img.shape)