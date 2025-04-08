r""" FSS-1000 few-shot semantic segmentation dataset """
import os
import glob

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np


class DatasetFSS(Dataset):
    def __init__(self, datapath, query_file, fold, transform, split, shot, use_original_imgsize):
        self.split = split
        self.benchmark = 'fss'
        self.shot = shot

        self.base_path = os.path.join(datapath, 'FSS-1000/data')

        self.query_file = os.path.join(datapath, f'FSS-1000/{query_file}')

        # Given predefined test split, load randomly generated training/val splits:
        # (reference regarding trn/val/test splits: https://github.com/HKUSTCV/FSS-1000/issues/7))
        with open(os.path.join(datapath, 'FSS-1000/splits/%s.txt' % split), 'r') as f:
            self.categories = f.read().split('\n')[:-1]
        self.categories = sorted(self.categories)

        self.class_ids = self.build_class_ids()
        self.img_metadata = self.build_img_metadata()

        self.transform = transform

    def __len__(self):
        return len(self.img_metadata)

    def __getitem__(self, idx):
        query_name, support_names, class_sample = self.sample_episode(idx)
        query_img, query_mask, support_imgs, support_masks = self.load_frame(query_name, support_names)

        query_img = self.transform(query_img)
        query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()

        support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])

        support_masks_tmp = []
        for smask in support_masks:
            smask = F.interpolate(smask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
            support_masks_tmp.append(smask)
        support_masks = torch.stack(support_masks_tmp)

        batch = {'query_img': query_img,
                 'query_mask': query_mask,
                 'query_name': query_name,

                 'support_imgs': support_imgs,
                 'support_masks': support_masks,
                 'support_names': support_names,

                 'class_id': torch.tensor(class_sample)}

        return batch

    def load_frame(self, query_name, support_names):
        query_img = Image.open(query_name).convert('RGB')
        support_imgs = [Image.open(name).convert('RGB') for name in support_names]

        query_id = query_name.split('/')[-1].split('.')[0]
        query_name = os.path.join(os.path.dirname(query_name), query_id) + '.png'
        support_ids = [name.split('/')[-1].split('.')[0] for name in support_names]
        support_names = [os.path.join(os.path.dirname(name), sid) + '.png' for name, sid in zip(support_names, support_ids)]

        query_mask = self.read_mask(query_name)
        support_masks = [self.read_mask(name) for name in support_names]

        return query_img, query_mask, support_imgs, support_masks

    def read_mask(self, img_name):
        mask = torch.tensor(np.array(Image.open(img_name).convert('L')))
        mask[mask < 128] = 0
        mask[mask >= 128] = 1
        return mask

    def sample_episode(self, idx):
        query_name = self.img_metadata[idx]
        class_sample = self.categories.index(query_name.split('/')[-2])
        if self.split == 'val':
            class_sample += 520
        elif self.split == 'test':
            class_sample += 760

        # Get all possible candidate names (1-10.jpg in same directory)
        all_candidates = [str(i)+'.jpg' for i in range(1, 11)]
        candidates = [f for f in all_candidates 
                    if f != os.path.basename(query_name)]  # Exclude query image
        
        # Randomly select unique support samples
        support_names = []
        if len(candidates) >= self.shot:  # Ensure enough candidates exist
            selected = np.random.choice(candidates, self.shot, replace=False)
            support_names = [os.path.join(os.path.dirname(query_name), name) 
                        for name in selected]
        else:
            raise ValueError(f"Not enough unique samples available (need {self.shot}, have {len(candidates)})")
        
        return query_name, support_names, class_sample

    def build_class_ids(self):
        if self.split == 'trn':
            class_ids = range(0, 520)
        elif self.split == 'val':
            class_ids = range(520, 760)
        elif self.split == 'test':
            class_ids = range(760, 1000)
        return class_ids

    def build_img_metadata(self):

    # def build_img_metadata(self):
        img_metadata = []
        
        # Case 1: Query file specified
        if hasattr(self, 'query_file') and self.query_file and os.path.exists(self.query_file):
            with open(self.query_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:  # Need at least folder + 1 image
                        folder = parts[0]
                        images = parts[1:]
                        
                        # Check if folder exists in dataset
                        folder_path = os.path.join(self.base_path, folder)
                        if not os.path.exists(folder_path):
                            print(f"Warning: Folder '{folder}' not found in dataset")
                            continue
                            
                        # Add specified images
                        for img_name in images:
                            img_path = os.path.join(folder_path, img_name+".jpg")
                            if os.path.exists(img_path):
                                img_metadata.append(img_path)
                            else:
                                print(f"Warning: Image '{img_name}' not found in folder '{folder}'")
        
        # Case 2: No query file - use all categories and all images
        else:
            for cat in self.categories:
                folder_path = os.path.join(self.base_path, cat)
                if not os.path.exists(folder_path):
                    continue
                    
                # Get all images in category folder
                img_paths = glob.glob(os.path.join(folder_path, '*'))
                for img_path in img_paths:
                    ext = os.path.splitext(img_path)[1].lower()
                    if ext in ['.jpg', '.jpeg', '.png']:
                        img_metadata.append(img_path)
        
        return sorted(img_metadata)  # Return sorted list for consistency