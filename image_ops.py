import numpy as np
from scipy import ndimage, signal, optimize
import pywt
from tqdm import tqdm

# import ctf functions
import ctf_code

# pass in ctf_list (size 5), then create ctf_array (size 10000)
class Dataset_Operations:
    def __init__(self, images, ctf_list, snr, metric='wemd', level=6):
        self.images = images #mask(images, images[0].shape)
        self.ctf_list = ctf_list
        self.ctf_array = np.repeat(ctf_list, 10000/5, axis=0)
        self.snr = snr # snr param
        self.metric = metric
        if metric == 'wemd':
            self.wavelet_space = wave_transform_volumes(self.images, level)
        self.n = self.images.shape[0]
        self.level = level

    def batch_distance_to(self, image):
        if self.metric == 'l2':
            distances = [((self.images[i] - image)**2).sum() for i in range(len(self.images))]
            return np.array(distances)

        elif self.metric == 'wemd':
            wavelet_img =  wave_transform_volumes([image], self.level)[0]
            distances = [np.abs(self.wavelet_space[i] - wavelet_img).sum() for i in range(len(self.wavelet_space))]
            return np.array(distances)

    # batch_distance_to_ctf: distance from each entry to properly ctf'd image
    def batch_distance_to_ctf(self, image):
        # only compute image with 5 ctfs, then np.repeat
        image_ctfs_list = [ctf.apply(image) for ctf in self.ctf_list]
        image_ctfs = np.repeat(image_ctfs_list, 10000/5, axis=0)

        if self.metric == 'l2':
            distances = [((self.images[i] - image_ctfs[i])**2).sum() for i in range(len(self.images))]
            return np.array(distances)
        
        # only compute wavelet transforms for 5, then np.repeat
        elif self.metric == 'wemd':
            wavelet_imgs_list = [wave_transform_volumes([img], self.level)[0] for img in image_ctfs_list]
            wavelet_imgs = np.repeat(wavelet_imgs_list, 10000/5, axis=0)  
            distances = [np.abs(self.wavelet_space[i] - wavelet_imgs[i]).sum() for i in range(len(self.wavelet_space))]
            return np.array(distances)

    def distance(self, i, j):
        if self.metric == 'l2':
            return np.sqrt(((self.images[i] - self.images[j])**2).sum())

        elif self.metric == 'wemd':
            return np.abs(self.wavelet_space[i] - self.wavelet_space[j]).sum()


    def batch_oriented_average(self, idxs, orientation_lists, strategy='mean', ncores = 1, **kwargs):
        ''' orientations will be opposite the direction that the image will be rotated in'''
        rotated_image_sets = []
        for idx, orientations in zip(idxs, orientation_lists):
            rotated_images = [ndimage.rotate(im, -1 * orientations[i], reshape=False) for i, im in enumerate(self.images[idx])]
            rotated_image_sets.append(np.asarray(rotated_images))

        if strategy == 'mean':
            centers = []
            for rotated in rotated_image_sets:
                if rotated.shape[0] != 0:
                    centers.append(np.sum(rotated, axis=0) / rotated.shape[0])
                else:
                    centers.append(np.ones(self.images[0].shape))
            return np.asarray(centers)

        elif strategy == 'emd-bary':
            reg = kwargs['reg']
            numItermax = 15000 if 'numItermax' not in kwargs else kwargs['numItermax']
            return [barycenter(rotated, reg, numItermax=numItermax) for rotated in rotated_image_sets]

    # batch_oriented_average_ctf: average of images in cluster according to LS formula/CTFAveragingFilter
    def batch_oriented_average_ctf(self, idxs, orientation_lists, strategy='mean', ncores = 1, **kwargs):
        ''' orientations will be opposite the direction that the image will be rotated in'''
        rotated_image_sets = []
        for idx, orientations in zip(idxs, orientation_lists):
            rotated_images = [ndimage.rotate(im, -1 * orientations[i], mode='wrap', reshape=False) for i, im in enumerate(self.images[idx])]
            rotated_image_sets.append(np.asarray(rotated_images))

        # modified using CTFAveragingFilter
        if strategy == 'mean':
            centers = []
            for i, rotated in enumerate(rotated_image_sets):
                if rotated.shape[0] != 0:
                    ctf_array_masked = np.array([self.ctf_array[j] for j in idxs[i]])
                    ctf_avg_filter = ctf_code.CTFAveragingFilter(ctf_array_masked, self.snr) # added snr param
                    centers.append(ctf_avg_filter.apply(rotated))
                else:
                    centers.append(np.ones(self.images[0].shape))
            return np.asarray(centers)

    def __getitem__(self, idx):
        return self.images[idx]

    def __getslice__(self, i, j):
        return self.images[i:j]

## Computing wEMD

DIMENSION = 2
WAVELET_NAME = 'sym5'

def wave_emd(p1,p2):
    p = np.asarray(p1)-np.asarray(p2)
    p = np.abs(p)
    emd = np.sum(p)
    return emd

def volume_to_wavelet_domain(volume, level, wavelet=pywt.Wavelet(WAVELET_NAME)):
    """
    This function computes an embedding of non-negative 3D Numpy arrays such that the L_1 distance
    between the resulting embeddings is approximately equal to the Earthmover distance of the arrays.
    It implements the weighting scheme in Eq. (20) of the Technical report by Shirdhonkar, Sameer, and David W. Jacobs. "CAR-TR-1025 CS-TR-4908 UMIACS-TR-2008-06." (2008).
    """

    assert len(volume.shape) == DIMENSION

    volume_dwt = pywt.wavedecn(volume, wavelet, mode='zero', level=level)

    detail_coefs = volume_dwt[1:]
    n_levels = len(detail_coefs)

    weighted_coefs = []
    for (j, details_level_j) in enumerate(volume_dwt[1:]):
        for coefs in details_level_j.values():
            multiplier = 2**((n_levels-1-j)*(1+(DIMENSION/2.0)))
            weighted_coefs.append(coefs.flatten()*multiplier)

    return np.concatenate(weighted_coefs)

def wave_transform_volumes(volumes, level):
    wavelet = pywt.Wavelet(WAVELET_NAME)
    return [volume_to_wavelet_domain(vol, level, wavelet) for vol in volumes]

def mask(data, size):
    size = size[0]
    mask = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if ((i - ((size-1)/2))**2 + (j - ((size-1)/2))**2 <= ((size-1)/2)**2):
                mask[i,j] = 1
    return data * mask
