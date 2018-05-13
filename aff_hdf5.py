import os
import re
import functools as ft
import itertools as it
import numpy as np
import aff
import h5py


def label_to_vec(label, str_tup, delim=''):
    """
    Given a label (momentum or coordinate) it returns a list 
    of numbers corresponding to the position of momentum vector 
    denoted by the label 
    returns list
    """
    length = len(str_tup)
    if len(delim) != 0:
        label = re.sub(delim, '', label)
    vec = []
    for s_t in str_tup:
        val = label.split(s_t)
        a_val, label = val[0], val[1]
        if len(a_val) != 0 and len(vec) < length:
            vec.append(int(a_val))
    if len(vec) < length:
        vec.append(int(label))
        return vec


def vec_to_label(vector, str_tup, delim=''):
    """ Inverse of the above function """
    label = ''
    for j, (s_t, vec) in enumerate(zip(str_tup, vector)):
        label += s_t + str(vec)
        if j < len(vector):
            label += delim
    return label


def sort_labels(labels, str_tup, delim=''):
    """ Given list of labels, returns sorted labels with vectors as well """
    ltv = ft.partial(label_to_vec, str_tup, delim)
    vtl = ft.partial(vec_to_label, str_tup, delim)
    vecs = map(ltv, labels)
    vecs = sorted(vecs, key=lambda x: np.dot(x, x))
    lbls = map(vtl, vecs)
    return lbls, vecs


def get_files(path, *tags):
    """
    This function will return all the files in the specified 'path' 
    directory that contains all of the 'tag' strings
    returns file_list
    """
    files = os.listdir(path)
    for tag in tags:
        files = [fle for fle in files
                 if tag in fle]
    return files


def sort_files(file_list, csrc_ind):
    """
    This function is only used for AMA analysis. Two separate
    lists will be generated; one for 'ex' in file, and the other
    'sl' in the file. From there each 'ex' file will be paired
    with it's corresponding 'sl' file and be put in a separate
    list as a tuple called 'bias list'.
    returns bias_list, sl_list
    """
    ex_list, sl_list = [], []
    for fle in file_list:
        if 'ex' in fle:
            ex_list.append(fle)
        elif 'sl' in f:
            ex_list.append(fle)
        else:
            raise "BAD File Name"
    csrc_list = []
    for fle in ex_list:
        csrc_list.append(fle.split('.')[csrc_ind])
    bias_list = []
    for it, csrc in enumerate(csrc_list):
        for fsl in sl_list:
            if csrc in fsl:
                bias_list.append((ex_list[it], fsl))
            break

    return bias_list, sl_list


def get_aff_key(aff_file):
    """ Returns the tree structure of an aff-file """
    aff_r = aff.Reader(aff_file)
    loc, tree = '', []
    while bool(aff_r.ls(loc)):
        val = aff_r.ls(loc)
        loc = loc + "/" + val[0]
        tree.append(val)
    aff_r.close()
    return tree


def key_to_file(key, aff_file, csrc_ind=None, insert=None, delim=None):
    """ returns a modified key for a given file """
    def mod_csrc(csrc, delim):
        newcsrc = ''
        for j in xrange(len(csrc)-1):
            if csrc[j].isdigit() and csrc[j+1].isalpha():
                newcsrc += csrc[j] + delim
            else:
                newcsrc += csrc[j]
        newcsrc += csrc[-1]
        return newcsrc

    if csrc_ind is not None:
        assert insert is not None
        csrc = aff_file.split('.')[csrc_ind]
        if delim is not None:
            csrc = mod_csrc(csrc, delim)
        newkey = key[:insert] + [csrc] + key[insert:]
    else:
        newkey = key
    newkey = "/".join(newkey)
    return newkey


def key_to_ex_file_list(key, bias_list, sl_list, csrc_ind=None,
                        insert=None, delim=None):
    """ Gets key structure for the bias list in the form of ex_key """
    key_list = []
    for (ex, sl) in bias_list:
        ex_key = key_to_file(key, ex, csrc_ind, insert, delim)
        key_list.append(ex_key)
    return key_list


def key_to_sl_file_list(key, bias_list, sl_list, csrc_ind=None,
                        insert=None, delim=None):
    """ Gets key structure for all sloppy files """
    key_list = [
        key_to_file(key, fle, csrc_ind, insert, delim)
        for fle in sl_list
    ]
    return key_list


def key_to_bias_file_list(key, bias_list, sl_list, csrc_ind=None,
                          insert=None, delim=None):
    """ Gets key structure for the bias list in the form of (ex_key, sl_key) """
    key_list = []
    for (ex, sl) in bias_list:
        ex_key = key_to_file(key, ex, csrc_ind, insert, delim)
        sl_key = key_to_file(key, sl, csrc_ind, insert, delim)
        key_entry = (ex_key, sl_key)
        key_list.append(key_entry)
    return key_list


def key_to_ama_file_list(key, bias_list, sl_list, csrc_ind=None,
                         insert=None, delim=None):
    """
    Gets the full key structure for both bias and sloppy
    solves without introducing (as far as I can think of)
    no uneccesary function calls
    """
    bs_key_list = []
    for (ex, sl) in bias_list:
        ex_key = key_to_file(key, ex, csrc_ind, insert, delim)
        sl_key = key_to_file(key, sl, csrc_ind, insert, delim)
        key_entry = (ex_key, sl_key)
        bs_key_list.append(key_entry)
    sl_key_list = [
        key_to_file(key, fle, csrc_ind, insert, delim)
        for fle in file_list
    ]
    key_list = (bs_key_list, sl_key_list)
    return return key_list


def get_aff_data(aff_r, aff_key, mom_labels, phases=None):
    """
    Returns the data in an aff file of dimensions
    returns aff_data[momentum, time]
    """
    if aff_key[-1] is not '/':
        aff_key += '/'
    aff_data, n_t = None, None
    for it_mom, mom in enumerate(mom_labels):
        x_dat = aff_r.read('{0}{1}'.format(aff_key, mom))
        x_dat = np.asarray(x_dat)
        if aff_data is None:
            n_t = len(x_dat)
            aff_data = np.zeros((len(mom_labels), n_t), dtype=np.complex128)
        if phases is not None:
            n_t_sep = n_t / phases.shape[1]
            # assert n_t_sep == phases.shape[-1]
            for j in xrange(0, phases.shape[1]):
                ind1, ind2 = j*n_t_sep, (j+1)*n_t_sep
                x_dat[ind1:ind2] = phases[it_mom][j]*x_dat[ind1:ind2]
        aff_data[it_mom] = x_dat[:]
    return aff_data


def get_ex_data_per_config(bias_list, sl_list, key_list, mom_labels):
    """ gets data from ex-solve """
    ex_data = None
    for file_pairs, keys in zip(bias_list, key_list):
        file_ex = file_pairs[0]
        aff_ex = aff.Reader(file_ex)
        dat_ex = get_aff_data(aff_ex, keys[0], mom_labels)
        if ex_data is None:
            ex_data = dat_ex
        else:
            ex_data += dat_ex
        aff_ex.close()
    ex_data = ex_data/float(len(bias_list))

    return ex_data


def get_sl_data_per_config(bias_list, sl_list, key_list, mom_labels):
    """ gets data from sl-solve """
    aff_sl_list = [aff.Reader(f_sl) for f_sl in sl_list]
    aff_sl = np.asarray([
        get_aff_data(afr, ksl, mom_labels)
        for afr, ksl in zip(aff_sl_list, key_list)
    ])
    sl_data = aff_sl.mean(axis=0)
    for aff_r in aff_sl_list:
        aff_r.close()

    return sl_data


def get_bias_data_per_config(bias_list, sl_list, key_list, mom_labels):
    """ gets bias data from (ex and sl)-solves """
    bias_data = None
    for file_pairs, keys in zip(bias_list, key_list):
        file_ex, file_sl = file_pairs
        aff_ex, aff_sl = aff.Reader(file_ex), aff.Reader(file_sl)
        dat_ex = get_aff_data(aff_ex, keys[0], mom_labels)
        dat_sl = get_aff_data(aff_sl, keys[1], mom_labels)
        if bias_data is None:
            bias_data = dat_ex - dat_sl
        else:
            bias_data += (dat_ex - dat_sl)
        aff_ex.close()
        aff_sl.close()
    bias_data = bias_data/float(len(bias_list))

    return bias_data


def get_ama_data_per_config(bias_list, sl_list, key_list, mom_labels):
    """ Think of something to put here :^/ """
    bias_key_list, sl_key_list = key_list

    # First I will work on the bias
    bias_data = get_bias_data_per_config(bias_list, bias_key_list, mom_labels)
    dat_dtype = bias_data.dtype

    # Now for sloppy data
    sl_data = get_sl_dat_per_config(sl_list, sl_key_list, mom_labes)
    ama_data = bias_data + sl_data
    return ama_data


def aff_to_h5(paths, method, h5_name, key_tree, t_len,
              file_params, key_params, pos_mom_tuples,
              attrs_dict):
    """ Think of something to put here :^/ """
    # get appropriate write function
    func_key_call = {'ex': (get_ex_data_per_config, key_to_ex_file_list)
                 'sl': (get_sl_data_per_config, key_to_sl_file_list)
                 'bias': (get_bias_data_per_config, key_to_bias_file_list)
                 'ama': (get_ama_data_per_config, key_to_ama_file_list)}
    assert method in func_call.keys()
    get_func, key_func = func_call[method]
        
    # Preliminary Stuff
    config, tag = file_params
    x_tup, mom_tup = pos_mum_tuples
    csrc_ind, insert, delim = key_params
    assert len(paths) == 2
    path_in, path_out = paths
    if path_out[-1] is not '/':
        path_out += '/'

    h5_name = "_".join([h5_name, method, tag])
    h5_name += ".h5"
    print h5_name
    print "config: {0}".format(config)
    mom_labels = key_tree[-1]
    mom_len = len(mom_labels)
    mom_labels, mom_vec = sort_labels(mom_labels, mom_tup, '_')
    mom_vecs = map(np.asarray, mom_vecs)
    key_tree[-1] = mom_labels

    hs_shape = (mom_len, t_len)
    hs_dtype = np.complex128

    h5_file = h5py(path_out+h5_name, 'w')
    print "hdf5 file {0} created".format(h5_file.filename())
    
    key_tree_gen = it.product(*key_tree[:-1])

    for key in key_tree_gen:
        data = np.empty(hs_shape, dtype=hs_dtype)
        key = list(key)
        h5key = "/".join(key)
        print "current key: {0}".format(h5key)

        # get files
        files = get_files(path_in, config, tag)
        bfiles, sfiles = sort_files(files, csrc_ind)

        # get key
        key_list = key_func(key, bfiles, sfiles, 
                           csrc_ind, insert, delim)

        # get data
        data = get_func(bfiles, sfiles, key_list, mom_labels)
        dset = h5_file.require_dataset(h5key, hs_shape,
                                       dtype=hs_dtype,
                                       fletcher32=True)
        dset[:] = data

    for att_key, att_val in zip(attrs_dict.keys(), attrs_dict.values()):
        h5_file.attrs[att_key] = att_val
    h5_file.file.flush()
    h5_file.close()
    print "File Closed"


def get_h5_key(h5file):
    """ Get h5keys as an iterator """
    h5f1 = h5py.File(h5file, 'r')
    h5_list = [h5f1.keys()]
    h5_tag = h5_list[0][0]+"/"
    while type(h5f1[h5_tag]) == h5py._hl.group.Group:
        h5_list.append(h5f1[h5_tag].keys())
        h5_tag += h5f1[h5_tag].keys()[0] + "/"
    return it.product(h5_list)


def h5_convert_ama(paths, h5_name, config_list, tag):
    """
    Takes all the bias and sloppy data and 
    turns them into one ama file with all 
    configurations
    """
    path_in, path_out = paths
    files = [get_files(path_in, config, tag) for config in config_list]
    assert len(files) == config_list
    key_tree_gen = get_h5_key(files[0])
    return None
