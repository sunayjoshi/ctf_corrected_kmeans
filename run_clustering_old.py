import numpy as np
import logging
import os
import multiprocessing
import argparse
from tqdm import tqdm 
from scipy import ndimage
from skimage.transform import rescale

from image_ops import Dataset_Operations

# for pickling
import pickle

# new imports
import ctf_code
import snr_estimator

logger = logging.getLogger('clustering')
logging.basicConfig(level = logging.INFO)

## Experiment Parameters

parser = argparse.ArgumentParser()
parser.add_argument('--snr', type=int, default=0)
parser.add_argument('--k', type=int,  default=100)
parser.add_argument('--n_angles', type=int,  default=200)
parser.add_argument('--niter', type=int, default=50) # changed to 50
parser.add_argument('--ncores', type=int, default=4)

parser.add_argument('--data_file_prefix', type=str, default='ribosome')
parser.add_argument('--clustering_type', type=str,  default='l2')

                                                                              
# added effects arg that can be none, -ctf, -transl, -ctf-transl, etc.
# ex: for ctf, run with --effects=-ctf                                              
parser.add_argument('--effects', type=str, default='')

args = parser.parse_args()

# Global Variables
data_file = args.data_file_prefix
snr = args.snr
clustering_type = args.clustering_type
k = args.k
n_angles = args.n_angles
angles = [360/n_angles * i for i in range(n_angles)]
n_iter = args.niter
experiment_name = args.clustering_type + "-" + str(args.snr)
ncores = args.ncores
centers = None
labels = None

# added effects
effects = args.effects

downsampled_resolution = 64

## Data Loading
def add_noise(images, snr):
    power_clean = (images**2).sum()/np.size(images)
    noise_var = power_clean/snr
    return images + np.sqrt(noise_var)*np.random.normal(0, 1, images.shape)
logger.info("Loading" + data_file + " data at snr level " + str(snr))
ref_angles = np.load("data/" + data_file + "_angles_centered.npy")

# added clean data
clean_data = np.load("data/" + data_file + "_images_centered.npy")
data = np.load("data/" + data_file + "_images_centered.npy") # to be affected by ctf

# initialize array of ctfs to be applied, then pass into Dataset_Operations constructor
ctf_list = [ctf_code.RadialCTFFilter(defocus=d) for d in np.linspace(1.5e4, 2.5e4, 5)]  # list of 5 radial ctf objects
ctf_array = np.repeat(ctf_list, 10000/5, axis=0) # duplicate until 10,000 ctfs

# apply ctfs to clean data
data = np.array([ctf.apply(data[j]) for j, ctf in enumerate(ctf_array)])

# add noise
if snr == 0:
    noisy_data = data
else:
    noisy_data = add_noise(data, 1/snr)

noisy_data = noisy_data.astype('float32')
downsampling_ratio = downsampled_resolution / noisy_data.shape[1]

logger.info("Beginning experiment ... ")
logger.info("Using " + clustering_type + " clustering with " + str(k) + " centers and " + str(n_angles) + "angles")
logger.info("Requested " + str(ncores) + " out of " + str(multiprocessing.cpu_count()))

image_dataset = Dataset_Operations(noisy_data, ctf_list, snr, metric=clustering_type) # have ctf_list, snr params

## Clustering Logic

# use known snr; later, replace with snr estimator
#dataset_noise = snr_estimator.estimate_noise_batch(noisy_data)
dataset_snr = 1/snr #snr_estimator.estimate_snr_batch(noisy_data[0], dataset_noise)

def update_distance_for(center):
    dists = []
    for j, angle in enumerate(angles):
        dists.append(image_dataset.batch_distance_to(ndimage.rotate(center, angle, mode='wrap', reshape=False))) # batch_distance_to
    return dists

def save():
    centers_name = "pickles/" + experiment_name + "-centers" + effects + ".pickle"
    labels_name = "pickles/" + experiment_name + "-labels" + effects + ".pickle"
    with open(centers_name, 'wb') as f:
        pickle.dump(centers, f)
    with open(labels_name, 'wb') as f:
        pickle.dump(labels, f)

def initialize_centers(init='random_selected'):
    global centers
    global labels

    if centers is not None:
        return

    if init == 'random_selected':
        centers = []
        center_idxs = np.random.choice([i for i in range(image_dataset.n)], k, replace=False)
        for i in center_idxs:
            shape = image_dataset[i].shape
            centers.append(image_dataset[i])
        centers = np.array(centers)

    if init == 'k++':
        centers, center_idxs = _k_plus_plus() # two arrays, 2nd named center_idxs for consistency with 'random_selected' case

def _k_plus_plus():
    low_res_noise_data = np.array([rescale(im, downsampling_ratio) for im in noisy_data]).astype('float32')
    low_res_data = Dataset_Operations(low_res_noise_data, ctf_list, snr, metric=clustering_type) # added ctf_list, snr params

    chosen_centers_idx = [np.random.randint(low_res_data.n)]
    initialization_angles = angles[::int(1/downsampling_ratio)]
    distances = np.zeros((low_res_data.n, len(chosen_centers_idx),len(initialization_angles)))

    for _ in tqdm(range(k-1)):
        if len(chosen_centers_idx) > 1:
            new_distances = np.zeros((low_res_data.n, len(chosen_centers_idx),len(initialization_angles)))
            new_distances[:, :len(chosen_centers_idx) - 1, :] = old_distances
            distances = new_distances
        for j, angle in enumerate(initialization_angles):
            dist = low_res_data.batch_distance_to(ndimage.rotate(low_res_data[chosen_centers_idx[-1]], angle, mode='wrap', reshape=False)) # batch_distance_to
            distances[:, -1, j] = dist
        old_distances = distances.copy()
        distances = distances.reshape(low_res_data.n, len(chosen_centers_idx) *len(initialization_angles))
        distances = distances.min(axis=1)**2
        distances[chosen_centers_idx] = 0
        probabilities = distances/distances.sum()
        probabilities = probabilities.reshape(image_dataset.n)
        next_center = np.random.choice(low_res_data.n, 1, probabilities.tolist())[0]
        chosen_centers_idx.append(next_center)
    centers = []
    for idx in chosen_centers_idx:
        centers.append(image_dataset[idx])
    return centers, chosen_centers_idx

def cluster(niter = 25, ncores = 1, init='random_selected', tolerance = 100):
    global centers
    global labels
    initialize_centers(init=init)
    pool = multiprocessing.Pool(processes=ncores)

    old_loss = np.inf
    losses = []
    for _ in tqdm(range(niter)):
        save()
        dists = np.array(pool.map(update_distance_for, centers))
        distances = np.transpose(dists, (2,0,1))
        min_distance_idxs = distances.reshape(image_dataset.n, k * len(angles)).argmin(axis=1)

        min_dists = distances.reshape(image_dataset.n, k * len(angles)).min(axis=1)
        loss = np.sum(min_dists**2)
        losses.append(loss)
        print(loss)
        if np.abs(loss - old_loss) < tolerance:
            print('converged')
            break
        else:
            old_loss = loss

        labels = np.floor(min_distance_idxs / len(angles))
        orientations = np.array([angles[i] for i in min_distance_idxs % len(angles)])

        idxs = [[] for i in range(k)]
        orientation_lists = [[] for i in range(k)]
        for idx, label in enumerate(labels):
            label = int(label)
            idxs[label].append(idx)
            orientation_lists[label].append(orientations[idx])
        centers = image_dataset.batch_oriented_average(idxs, orientation_lists) # batch_oriented_average
    print(losses)

if __name__ == "__main__":
    cluster(niter=n_iter, ncores=ncores, init='k++')
