import itertools as it
import numpy as np
import h5py
import aff_hdf5


def norm_twopt(data):
    """ normalize two-point function """
    magsq = data*data.conj()
    return data/np.sqrt(magsq)


def arg_twopt(data):
    """ find log of two-point function """
    dtype = data.dtype
    assert_string = "dtype = {0}, not np.complex".format(dtype)
    assert dtype == np.complex, assert_string
    norm_data = norm_twopt(data)
    return np.log(norm_data)


def gen_phase_matrix(data):
    """ creates NxN phase-reweighted matrix from data """
    N = data.shape[0]
    shape = (N, N)
    dtype = data.dtype
    matrix = np.empty(shape=shape, dtype=dtype)
    norm_data = norm_twopt(data)
    for n in xrange(N):
        matrix[n] = data/np.roll(norm_data, -n)
    return matrix


def phase_reweight_twopt(paths, h5file, h5name):
    """ Phase Reweight Data """
    path_in, path_out = paths
    h5key_list = aff_hdf5.get_h5_key(path_in + h5file)
    h5f0 = h5py.File(path_in + h5file, 'r')
    h5f1 = h5py.File(path_out + h5name, 'w')
    hs_shape = None
    hs_dtype = None
    for key in h5key_list:
        h5key = "/".join(key)
        print h5key
        if hs_shape is None:
            hs_shape = list(h5f0[h5key].shape)
            hs_shape.append(hs_shape[-1])
            hs_shape = tuple(hs_shape)
        if hs_dtype is None:
            hs_dtype = h5f0[h5key].dtype
        dset = h5f1.require_dataset(h5key, hs_shape,
                                    dtype=hs_dtype,
                                    fletcher32=True)
        for cfg in xrange(hs_shape[0]):
            for piter in xrange(hs_shape[1]):
                matrix = gen_phase_matrix(h5f0[h5key][cfg, piter])
                dset[cfg, piter] = matrix
        for attrs_key, attrs_val in zip(h5f0[h5key].attrs.keys(),
                                        h5f0[h5key].attrs.values()):
            dset.attrs[attrs_key] = attrs_val
        dset.file.flush()
    h5f0.close()
    h5f1.close()
    print "File Closed"
