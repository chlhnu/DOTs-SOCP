from numpy import loadtxt as np_load_txt
from pathlib import Path

def get_mu(area_vertices = None, vertices = None):
    bdy_path = Path(__file__).parent / 'data_mu'
    mub0 = np_load_txt(bdy_path / 'sphere_puncture_data_mu0.txt')
    mub1 = np_load_txt(bdy_path / 'sphere_puncture_data_mu1.txt')

    return mub0, mub1