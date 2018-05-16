import numpy as np
import h5py


def get_pvec_indices(h5dat, h5key, pvec_tag, pvecs):
    assert len(pvecs[0]) == 3
    data = h5dat[h5key]
    plist = data.attrs[pvec_tag]
    pinds = []
    for p_v in pvecs:
        for index, p_l in enumerate(plist):
            if all(np.array(p_v) == p_l):
                pinds.append(index)
                break
    return pinds


def get_h5_data(h5dat, h5key, pinds):
    data = h5dat[h5key][...]
    data = data[:, pinds, ...]
    return data


def merge_data_sets(h5data_set, h5keys, pinds):
    data_set = [
        get_h5_data(h5data, h5key, pinds)
        for h5data, h5key in zip(h5data_set, h5keys)
    ]
    cfg_size_list = [
        dset.shape[0] for dset in data_set
    ]
    t_len = data_set[0].shape[-1]
    p_len = len(pinds)
    c_len = sum(cfg_size_list)
    d_shape = (c_len, p_len, t_len)
    newdata = np.empty(d_shape, dtype=data_set[0].dtype)
    newdata[:data_set[0].shape[0], ...] = data_set[0]
    newdata[data_set[0].shape[0]:, ...] = data_set[1]
    return newdata
