import random
import torch
import torch.nn as nn
from torchvision.transforms.functional import rotate
from torchvision.transforms import InterpolationMode


def inference_time_augmentation(data,i):
    """
    Apply inference-time augmentation to input data.

    Args:
        data (torch.Tensor): Input data with shape (channels, height, width).
        i (int): Index specifying the augmentation type:
            0: No flip
            1: Flip along the last dimension
            2: Flip along the second-to-last dimension
            3: Flip along both dimensions

    Returns:
        torch.Tensor: Augmented data.
    """
    if i == 0:
        x = data
    elif i == 1:
        x = torch.flip(data.clone(),dims=(-1,))
    elif i == 2:
        x = torch.flip(data.clone(),dims=(-2,))
    elif i == 3:
        x = torch.flip(data.clone(),dims=(-2,-1))
    
    return x


class TorchRandomRotate(nn.Module):
    """
    Apply random rotation to input images and masks. In contrast to the implementation in the Kornia library, this will apply a rotation by
    different degrees to each element instead of rotating all images in batch by the same degree.

    Attributes:
        degrees (tuple or list): Range of possible rotation angles.
        probability (float, default=1): Probability of applying the rotation.
        interpolation (InterpolationMode, default=BILINEAR): Interpolation mode for rotation.
        center (tuple, Optional): Center of rotation. If None, the center is the center of the image.
        fill (float, default=0): Value to fill the image during rotation.
        mask_fill (float, default=0): Value to fill the mask during rotation.

    Returns:
        tuple: Tuple containing the rotated image and mask (if provided).
        
    """
    
    def __init__(self, degrees, probability=1.0,interpolation=InterpolationMode.BILINEAR, center=None, fill=0,mask_fill=0):
        super().__init__()
        if not isinstance(degrees,(list,tuple)):
            degrees = (-abs(degrees),abs(degrees))

        self.degrees = degrees
        self.interpolation = interpolation
        self.center = center
        self.fill_value = fill
        self.mask_fill_value = mask_fill
        self.proba = probability

    @staticmethod
    def get_params(degrees) -> float:
        """
        Get parameters for a random rotation. Given a range of degrees "0 to 30 deg", it will generate a random degree in this range

        Parameters:
            degrees (tuple): A tuple representing the range of degrees for the rotation.

        Returns:
            float: A random angle within the specified degree range.

        """
        angle = float(torch.empty(1).uniform_(float(degrees[0]), float(degrees[1])).item())
        return angle
    

    @torch.no_grad()
    def __call__(self,img,mask=None):
        """
        Apply random rotation to the input image and mask. for each element in batch that isn't skipped, we generate an angle in a range,
        rotate the image by this angle, and apply rotation by the same angle to the mask (if it's not None),
        at the end we return the rotated image with the rotated mask (if provided)

        Parameters:
            img (torch.Tensor): Input image tensor of shape (batch_size, height, width, channels).
            mask (torch.Tensor, optional): Input mask tensor with same shape as image. Default is None.

        Returns:
            tuple: A tuple containing the rotated image tensor and the rotated mask tensor (if provided).

        Note:
            >>> If the mask is provided, both the image and mask are rotated by the same angle.
        """
        batch_size = img.shape[0]

        for i in range(batch_size):
            
            if random.random() > self.proba:
                continue

            angle = self.get_params(self.degrees)
            img[i,...] = rotate(img[i,...], angle, self.interpolation, False, self.center, self.fill_value)

            if mask is not None:
                mask[i,...] =  rotate(mask[i,...], angle, self.interpolation, False, self.center, self.mask_fill_value)
                mask = mask.float()
        if mask is not None:
            mask[mask<0] = self.mask_fill_value
        return img,mask


class RandomMaskIgnore(nn.Module):
    """
    RandomMaskIgnore class generates random masks within specified lengths with a given probability.
    It modifies the input mask tensor in-place, and the modified tensor is returned.

    Attributes:
        min_length (int, default=50): Minimum length of the randomly generated bounding box.
        max_length (int, default=100): Maximum length of the randomly generated bounding box.
        proba (float, default=0.5): Probability of applying the random mask modification.
        ignore_index (int, default=-10): Value used to fill the masked region.
    """

    def __init__(self,min_length=50,max_length=100,proba=0.5,ignore_index=-10):
        super().__init__()

        self.min_length = min_length
        self.max_length = max_length
        self.proba = proba
        self.ignore_index = ignore_index
    

    def generate_random_bbox(self,shape):
        """
        Generate a random bounding box within the specified shape.

        Args:
            shape (tuple): Height and width of the bounding box.

        Returns:
            tuple: Coordinates (top, left, bottom, right) of the bounding box.
        """
        H,W = shape
        L = random.randint(self.min_length,self.max_length)

        t = random.randint(0,H-L)
        b = t + L

        l = random.randint(0,W-L)
        r = l + L

        return (t,l,b,r)
    
    def mask_channel(self,bbox,channel):
        """
        Modify the specified channel by applying a mask within the bounding box.

        Args:
            bbox (tuple): Coordinates (top, left, bottom, right) of the bounding box.
            channel (tensor): Input channel to be modified.

        Returns:
            tensor: Modified channel.
        """
        (t,l,b,r) = bbox
        channel[:,t:b,l:r] = self.ignore_index
        return channel
    
    @torch.no_grad()
    def __call__(self,mask):
        """
        Apply random mask modifications to the input mask tensor with shape (Batch Size, Number of Channels, Height, Width)

        Args:
            mask (tensor): Input mask tensor.

        Returns:
            tensor: Modified mask tensor.
        """
        B,C,H,W = mask.shape
        for i in range(B):
            if random.random() > self.proba:
                continue
            bbox = self.generate_random_bbox((H,W))
            mask[i,...] = self.mask_channel(bbox,mask[i,...])
        
        return mask


class MaskPixelDrop(nn.Module):
    """
    MaskPixelDrop randomly drops pixels in the input mask tensor based on specified probabilities for positive and negative drops.
    
    Args:
        neg_drop (int or tuple, default for int =50): Probability range for dropping negative pixels. Default is (0, 50).
        pos_drop (int or tuple, default for int =50): Probability range for dropping positive pixels. Default is (0, 50).
        ignore_index (int, default=-10): Value used to fill the dropped pixels. Default is -10.

    Returns:
        torch.Tensor: Modified mask tensor with dropped pixels.
    """
    def __init__(self,neg_drop=50,pos_drop=50,ignore_index=-10):
        super().__init__()

        if not isinstance(neg_drop,tuple):
            neg_drop = (0,neg_drop)
        if not isinstance(pos_drop,tuple):
            pos_drop = (0,pos_drop)
        
        self.neg_drop = neg_drop
        self.pos_drop = pos_drop

        self.ignore_index = ignore_index
    
    @staticmethod
    def get_drop_proba(_range):
        """
        Get a random drop probability within the specified range.
        
        Args:
            _range (tuple): A tuple representing the range specified for the proba.
        
        Returns:
            int: A randomly generated probability in the range specified
        """
        return random.randint(_range[0],_range[1]) / 100
    
    def random_pixel_drop(self,gt,mask,_range):
        """
        Randomly drop pixels in the input mask tensor. Cs, Rs, Ws are column indices, row indices and depth indices respectively
        of non-zero elemtns in mask tensor.

        Args:
            gt (torch.Tensor): Ground truth tensor.
            mask (torch.Tensor): Input mask tensor.
            _range (tuple): Range for dropping pixels.

        Returns:
            torch.Tensor: Modified ground truth tensor with dropped pixels.
        """
        Cs,Hs,Ws = mask.nonzero(as_tuple=True)
        proba = self.get_drop_proba(_range)
        max_num = Cs.shape[0]
        drop_count = min(max_num,int(proba * max_num))
        
        if drop_count == 0 or max_num == 0:
            return gt

        indexes = random.sample(range(0, max_num), drop_count)
        Cs,Hs,Ws = Cs[indexes].tolist(),Hs[indexes].tolist(),Ws[indexes].tolist()
        gt[Cs,Hs,Ws] = self.ignore_index
        return gt

    @torch.no_grad()
    def __call__(self,mask):
        """
        Apply random pixel drops to the input mask tensor.

        Args:
            mask (torch.Tensor): Input mask tensor.

        Returns:
            torch.Tensor: Modified mask tensor with dropped pixels.
        """
        B,C,H,W = mask.shape
        pos_mask = mask.gt(0)
        neg_mask = mask.eq(0)
        for i in range(B):
            mask[i] = self.random_pixel_drop(mask[i],pos_mask[i],self.pos_drop)
            mask[i] = self.random_pixel_drop(mask[i],neg_mask[i],self.neg_drop)
        return mask
