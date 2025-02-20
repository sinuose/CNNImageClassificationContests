# torch
import torch
import torchvision
from torch.utils.data import Dataset
# images
from PIL import Image
import matplotlib.pyplot as plt
# utils
from tqdm import tqdm
import numpy as np
import os

# Dataset Class
class DiabeticRetinopathyDataset(Dataset):
    '''
    This class just assumes the files have image content and an ID which is denoted in the title
    Each image needs 2 things
        - image content (2 images per patient)
        - patient ID
    '''
    def __init__(self, data_path,  transform = None):
        super(DiabeticRetinopathyDataset, self).__init__()
        # each dataset needs a path to ALL the images
        self.data_path = data_path # path to data
        self.transform = transform # transform to normalize the data

        # List all image files in the data_path
        self.image_files = [f for f in os.listdir(data_path) if f.endswith('.jpeg')]
        # Extract patient IDs from the image filenames
        self.patient_ids = list(set([f.split('_')[0] for f in self.image_files]))
        # Group images by patient ID
        self.patient_images = {pid: [f for f in self.image_files if f.startswith(pid)] for pid in self.patient_ids}
        
    def __len__(self):
        '''length of the dataset'''
        return len(self.patient_ids)
    
    def __getitem__(self, index):
        ''' get a single item in the dataset'''
        patient_id = self.patient_ids[index]
        image_files = self.patient_images[patient_id]
        
        # Load the two images for the patient
        images = []

        for img_file in tqdm(image_files):
            # data path to the image directly
            img_path = os.path.join(self.data_path, img_file)
            # open the image up
            img = Image.open(img_path)
            
            if self.transform is not None:
                img = self.transform(img)
            
            images.append(img)
        
        # Assuming the label is the same for both images of the same patient
        # You may need to adjust this based on your actual data structure
        label = torch.tensor(int(patient_id))  # Replace with actual label logic if needed
        
        return images, label, patient_id
    
#------------------------------------------------------------------------------------------
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.figure(figsize = (12, 12))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    
#-----------------------------------------------------------------------------------------
