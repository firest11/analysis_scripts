import functools as ft
import itertools as it
import numpy as np


def jk_blocks(data, jk_smp, axis=None):
    """
    Function that takes in data-set and
    creates jackknife blocks
    """
    if axis == None:
        axis = 0
    assert data.shape[axis] % jk_smp == 0
    if axis is not None:
        assert axis < data.shape[axis] and axis >= 0
    blocks = []
        
    size = len(data)
    N = size/jk_smp
    chunk = size/N
    inds = np.arange(0, size, jk_smp)
    block = np.delete(data, inds, axis=axis)
    blocks.append(block)
    inds += 1
    while inds[-1] < len(data):
        block = np.delete(data, inds, axis=axis)
        blocks.append(block)
        inds += 1

    return blocks


def calc_mean_var(data, axis=None):
    """
    returns mean and var. 
    A check for jk-functions below
    """
    dtype = data.dtype
    mean = np.mean(data, axis, dtype)
    diffs = [
        pow(mean - dat, 2) for dat in data
    ]
    variance = ft.reduce(lambda x, y: x + y, diffs)
    variance *= 1/float(len(diffs) - 0.0)

    return mean, np.sqrt(variance)


def calc_jk_mean_var(data, blocks, axis=None):
    """
    returns mean and var from data
    and jackknife blocks
    """
    jk_smp = float(len(blocks))
    dtype = data.dtype
    mean = np.mean(data, axis, dtype)
    blocks_mean = np.array([
        blk.mean(axis) for blk in blocks
    ])
    diff_blocks = [
        pow(mean - block, 2) for block in blocks_mean
    ]
    variance = ft.reduce(lambda x, y: x + y, diff_blocks)
    variance *= (jk_smp - 1.0)/jk_smp

    return mean, np.sqrt(variance)


def calc_jk_ratio_mean_var(data1, data2, blocks1, blocks2, d_t=None, axis=None):
    """
    Computes <A>/<B>, Var(<A>/<B>)
    """
    assert len(blocks1) == len(blocks2)
    if d_t is not None:
        data2 = data2[..., d_t]
        blocks2 = [
            blk[..., d_t] for blk in blocks2
        ]
    jk_smp = float(len(blocks1))
    dtype = data1.dtype
    mean = data1.mean(axis, dtype)/data2.mean(axis, dtype)
    blocks_mean = [
        blk1.mean(axis, dtype)/blk2.mean(axis, dtype)
        for blk1, blk2 in zip(blocks1, blocks2)
    ]
    diff_blocks = [
        pow(mean - block, 2) for block in blocks_mean
    ]
    variance = ft.reduce(lambda x, y: x + y, diff_blocks)
    variance *= (jk_smp - 1.0)/jk_smp
    return mean, np.sqrt(variance)


def calc_bias_corrected_jk_mean_var(data, blocks, axis=None):
    """
    returns mean and var from data
    and jackknife blocks with
    bias correction
    """
    jk_smp = float(len(blocks))
    mean = np.mean(data, axis)
    blocks_mean = np.array([
        blk.mean(axis) for blk in blocks
    ])
    block_mean = np.mean(blocks_mean, axis=0)
    estimate = jk_smp*mean - (jk_smp-1)*block_mean
    diff_blocks = [
        pow(estimate - block, 2) for block in blocks_mean
    ]
    variance = ft.reduce(lambda x, y: x + y, diff_blocks)
    variance *= (jk_smp - 1.0)/jk_smp

    return mean, np.sqrt(variance)


def calc_bias_corrected_jk_ratio_mean_var(data1, data2, blocks1, blocks2,
                                          d_t=None, axis=None):
    """
    Computes <A>/<B>, Var(<A>/<B>)
    with bias correction
    """
    assert len(blocks1) == len(blocks2)
    if d_t is not None:
        data2 = data2[..., d_t]
        blocks2 = [
            blk[..., d_t] for blk in blocks2
        ]
    jk_smp = float(len(blocks1))
    dtype = data1.dtype
    mean = data1.mean(axis, dtype)/data2.mean(axis, dtype)
    blocks_mean = [
        blk1.mean(axis, dtype)/blk2.mean(axis, dtype)
        for blk1, blk2 in zip(blocks1, blocks2)
    ]
    block_mean = np.mean(blocks_mean, axis=0)
    estimate = jk_smp*mean - (jk_smp-1)*block_mean
    diff_blocks = [
        pow(estimate - block, 2) for block in blocks_mean
    ]
    variance = ft.reduce(lambda x, y: x + y, diff_blocks)
    variance *= (jk_smp - 1.0)/jk_smp

    return mean, np.sqrt(variance)
