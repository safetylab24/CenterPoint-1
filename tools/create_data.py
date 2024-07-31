import copy
from pathlib import Path
import pickle

import fire, os

from det3d.datasets.nuscenes import nusc_common as nu_ds
from det3d.datasets.utils.create_gt_database import create_groundtruth_database

def nuscenes_data_prep(root_path, version, nsweeps=10, filter_zero=True, virtual=False):
    # nu_ds.create_nuscenes_infos(root_path, version=version, nsweeps=nsweeps, filter_zero=filter_zero)
    # if version == 'v1.0-trainval':
    create_groundtruth_database(
        "NUSC",
        root_path,
        Path(root_path) / "infos_val_{:02d}sweeps_withvelo_filter_True.pkl".format(nsweeps),
        nsweeps=nsweeps,
        virtual=virtual
    )
    

if __name__ == "__main__":
    fire.Fire()
