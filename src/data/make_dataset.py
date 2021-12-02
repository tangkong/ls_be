# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from pathlib import Path
from scipy.io import loadmat
import numpy as np

def ret_cwd():
    print(Path.cwd())

def generate_dataset(n_cpts=3, seed=42, 
            faces=None, xanes=None, supply_truth=False):
    """ Generate a dataset using faces for weights and xanes spectra as data.
    Creates a ground truth image by randomly rotating (n_cpts) face images.
    Each pixel represents a XANES spectra, which is a linear combination of 
        spectra with weights given by each component image

    To recover weights from face_image, reshape as (1024, 3)

    returns: 
        - face_image: (32, 32, n_cpts) array 
        - xanes_data: (1024, 230) array, each row representing one spectrum
        - xanes_energy: (230) array, x coordinates
        - (optional) xanes_cpts: (n_cpts, 230) array, component spectra, ground truth
    """
    if seed=='random':
        seed = np.random.randint(0, 1000)
    rng = np.random.default_rng(seed=seed)
    # load data if faces/xanes arrays aren't provided
    if faces is None:  
        fp = Path(__file__).parent.parent.parent / 'data/raw/Faces5000.mat'
        x = loadmat(fp)
        faces = x['Faces5000']
    if (xanes is None) or (xanes == 'xanes'):
        fp = Path(__file__).parent.parent.parent / 'data/raw/Fe_normMaster1.mat'
        y = loadmat(fp)
        xanes = y['Fe_normMaster1'] # n_energies x n_samples
    if xanes == 'trig':
        # generate sin/cos basis functions
        x = np.linspace(0, 1, 100)
        xanes = np.array([  
                    np.sin(2*5*np.pi*rng.random() * (x - rng.random()))
                    for i in range(n_cpts)
                    ])

        xanes=np.hstack([x.reshape(-1,1), xanes.T])
        

    face_inds = rng.integers(0,faces.shape[0], size=(n_cpts))
    xanes_inds = rng.integers(1,xanes.shape[1], size=(n_cpts)) # ignores energies for now

    # gather faces and rotate randomly
    face_data = np.array([np.rot90(faces[i].reshape(32,32), k=rng.integers(0,3)).reshape(-1) 
                            for i in face_inds])
    # take ith element of each image as a weight, i is length of face image array
    weights = np.array([face_data.take(i, axis=-1) for i in range(face_data.shape[-1])])

    # grab component xanes spectra and generate linear combinations
    xanes_cpts = np.array([xanes[:,i] for i in xanes_inds])
    xanes_data = np.array([np.dot(weights[i], xanes_cpts) / np.sum(weights[i]) for i in range(len(weights))])

    face_image = weights.reshape(32,32,n_cpts)

    xanes_energy = np.array(xanes[:,0])

    if supply_truth is False:
        return face_image, xanes_data, xanes_energy
    else: # return end components
        return face_image, xanes_data, xanes_energy, xanes_cpts

def read_xrf_map(fp):
    return [1]

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
