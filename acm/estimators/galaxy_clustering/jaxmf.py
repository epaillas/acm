import time
import logging
import numpy as np

from .base import BaseDensityMeshEstimator

import jax
import jax.numpy as jnp
from typing import Tuple
from jax import config; config.update('jax_enable_x64', True)

# JIT-compiled per-slice routine
@jax.jit
def minkowski_slice_jax(delta_slices: jnp.ndarray, thresholds: jnp.ndarray, thres_mask: float
                        ) -> Tuple[jnp.ndarray, jnp.int32]:
    """
    delta_slices: shape (2, Y, Z) float32
    thresholds: shape (T,) float32
    thres_mask: scalar float32
    returns: (MFs_Tx4, vol_slice)
    """
    # build 8 neighbor values exactly like original code
    # ds shape: (8, Y, Z)
    ds0 = delta_slices[0]
    ds1 = delta_slices[1]
    ds2 = jnp.roll(ds0, shift=-1, axis=0)
    ds3 = jnp.roll(ds1, shift=-1, axis=0)
    ds4 = jnp.roll(ds0, shift=-1, axis=1)
    ds5 = jnp.roll(ds1, shift=-1, axis=1)
    ds6 = jnp.roll(ds2, shift=-1, axis=1)
    ds7 = jnp.roll(ds3, shift=-1, axis=1)
    ds = jnp.stack([ds0, ds1, ds2, ds3, ds4, ds5, ds6, ds7], axis=0)  # (8,Y,Z)

    ds_min = jnp.min(ds, axis=0)  # (Y,Z)
    ds_max = jnp.max(ds, axis=0)  # (Y,Z)

    # valid pixels (same as original if ds_min > thres_mask)
    mask_valid = ds_min > thres_mask  # (Y,Z)
    vol_slice = jnp.sum(mask_valid).astype(jnp.int32)

    # thresholds broadcasted to (T,1,1)
    T = thresholds.shape[0]
    thr = thresholds[:, None, None]  # (T,1,1)

    # full (t < ds_min) and partial ((t >= ds_min) & (t < ds_max)) masks
    # both have shape (T,Y,Z)
    full_mask = (thr < ds_min[None, :, :]) & mask_valid[None, :, :]
    partial_mask = (thr >= ds_min[None, :, :]) & (thr < ds_max[None, :, :]) & mask_valid[None, :, :]

    # contributions for full_mask: M0 gains +1 for each (threshold,pixel) that is full
    M0_full = jnp.sum(full_mask, axis=(1, 2)).astype(jnp.float64)  # (T,)

    # For partial positions compute n3,n2,n1,n0 as in original code.
    # ds_bool shape (T,8,Y,Z): ds > threshold
    # To reduce memory we compute ds_bool as (8,T,Y,Z) or (T,8,Y,Z). Use (T,8,Y,Z) for clarity.
    ds_bool = (ds[None, :, :, :] > thr[:, None, :, :])  # (T,8,Y,Z), dtype=bool

    # n3 = ds0 > t  => ds_bool[:,0,:,:]
    n3 = ds_bool[:, 0, :, :].astype(jnp.int32)  # (T,Y,Z)

    # n2 = (ds0 or ds1) + (ds0 or ds2) + (ds0 or ds4)
    n2 = (
        (jnp.logical_or(ds_bool[:, 0, :, :], ds_bool[:, 1, :, :]).astype(jnp.int32)) +
        (jnp.logical_or(ds_bool[:, 0, :, :], ds_bool[:, 2, :, :]).astype(jnp.int32)) +
        (jnp.logical_or(ds_bool[:, 0, :, :], ds_bool[:, 4, :, :]).astype(jnp.int32))
    )  # (T,Y,Z)

    # n1 = (ds0 or ds1 or ds2 or ds3) + (ds0 or ds2 or ds4 or ds6) + (ds0 or ds4 or ds1 or ds5)
    n1 = (
        (jnp.logical_or(jnp.logical_or(ds_bool[:, 0, :, :], ds_bool[:, 1, :, :]),
                        jnp.logical_or(ds_bool[:, 2, :, :], ds_bool[:, 3, :, :])).astype(jnp.int32)) +
        (jnp.logical_or(jnp.logical_or(ds_bool[:, 0, :, :], ds_bool[:, 2, :, :]),
                        jnp.logical_or(ds_bool[:, 4, :, :], ds_bool[:, 6, :, :])).astype(jnp.int32)) +
        (jnp.logical_or(jnp.logical_or(ds_bool[:, 0, :, :], ds_bool[:, 4, :, :]),
                        jnp.logical_or(ds_bool[:, 1, :, :], ds_bool[:, 5, :, :])).astype(jnp.int32))
    )  # (T,Y,Z)

    # n0 = or of all 8
    n0 = jnp.any(ds_bool, axis=1).astype(jnp.int32)  # (T,Y,Z)

    # Zero-out values outside partial_mask (we only add these when partial_mask True)
    pm = partial_mask.astype(jnp.int32)  # (T,Y,Z)
    n3_p = (n3 * pm).astype(jnp.float64)
    n2_p = (n2 * pm).astype(jnp.float64)
    n1_p = (n1 * pm).astype(jnp.float64)
    n0_p = (n0 * pm).astype(jnp.float64)

    # Sum over pixels (Y,Z) to get threshold-wise totals
    sum_n3 = jnp.sum(n3_p, axis=(1, 2))  # (T,)
    sum_n2 = jnp.sum(n2_p, axis=(1, 2))
    sum_n1 = jnp.sum(n1_p, axis=(1, 2))
    sum_n0 = jnp.sum(n0_p, axis=(1, 2))

    # Build MFs (T x 4)
    MFs = jnp.zeros((T, 4), dtype=jnp.float64)
    # M0: full contributions + partial n3 contributions
    MFs = MFs.at[:, 0].set(M0_full + sum_n3)
    # M1 contribution: (-3*n3 + n2) * 2/9
    MFs = MFs.at[:, 1].set(((-3.0 * sum_n3 + sum_n2) * (2.0 / 9.0)))
    # M2 contribution: (3*n3 - 2*n2 + n1) * 2/9
    MFs = MFs.at[:, 2].set(((3.0 * sum_n3 - 2.0 * sum_n2 + sum_n1) * (2.0 / 9.0)))
    # M3 contribution: (-n3 + n2 - n1 + n0)
    MFs = MFs.at[:, 3].set((-sum_n3 + sum_n2 - sum_n1 + sum_n0))

    return MFs, vol_slice


class MinkowskiFunctionals(BaseDensityMeshEstimator):
    """
    Computes 3D Minkowski functionals using the JAX implementation of the slice routine.
    Usage is similar to the original MinkowskiFunctionals class.
    """

    def __init__(
        self,
        thres_mask: float,
        thresholds : np.ndarray,
        batch_slices: int = 32,
        **kwargs
        ):
        """
        batch_slices: how many slices to process per python loop iteration. Small batches
                      reduce peak memory and keep JAX compilation efficient.
        """
        self.logger = logging.getLogger('MinkowskiFunctionals')
        self.logger.info('Initializing MinkowskiFunctionals (Jax-Based).')

        self.thres_mask = thres_mask
        self.thresholds = thresholds
        self.batch_slices = batch_slices
        super().__init__(**kwargs)

    def run(self):
        query_positions = self.get_query_positions(self.delta_mesh, method='lattice')
        t0 = time.time()
        self.delta_query = self.delta_mesh.read(query_positions).reshape(self.data_mesh.meshsize)

        # ensure float32 input for memory (we still compute sums in float64 where needed)
        delta = self.delta_query.astype(np.float32)
        dims_x, dims_y, dims_z = delta.shape
        len_thres = len(self.thresholds) 
        thresholds_j = jnp.array(self.thresholds)
        delta_padded = np.concatenate((delta, delta[0:1, :, :]), axis=0)

        # Accumulators
        MFs3D = jnp.zeros((len_thres, 4), dtype=jnp.float64)
        vol = 0

        # Process slices in small batches to control memory and JIT overhead
        i = 0
        while i < dims_x:
            # prepare batch of at most batch_slices slices -> for each slice we need 2 consecutive planes
            batch_end = min(i + self.batch_slices, dims_x)
            # we will process slices i..batch_end-1
            # build stack of delta_slices for each slice: shape (batch_size, 2, Y, Z)
            # Using numpy then convert to jnp to avoid building huge Python lists
            batch_size = batch_end - i
            # gather the required pairs
            pair_indices = np.stack([np.arange(i, batch_end), np.arange(i, batch_end) + 1], axis=1)  # (B,2)
            # build array (B,2,Y,Z)
            delta_pairs = delta_padded[pair_indices]  # shape (B,2,Y,Z)
            # convert to jnp
            delta_pairs_j = jnp.array(delta_pairs)

            # vectorize the slice function over the batch dimension using vmap
            # minkowski_slice_jax takes (2,Y,Z), thresholds, mask -> returns (T,4), vol_slice
            vmapped = jax.vmap(lambda ds: minkowski_slice_jax(ds, thresholds_j, self.thres_mask), in_axes=0, out_axes=0)
            # outputs: MFs_batch shape (B,T,4), vols_batch shape (B,)
            MFs_batch, vols_batch = vmapped(delta_pairs_j)
            # sum across batch axis
            MFs3D = MFs3D + jnp.sum(MFs_batch, axis=0)
            vol += int(jnp.sum(vols_batch))

            i = batch_end

        # Normalize MFs same as original:
        l = float(vol)
        a = float(self.data_mesh.cellsize[0])
        # if vol is zero avoid division by zero
        if l == 0:
            norm = jnp.array([0.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
            self.MFs3D = np.zeros_like(np.array(MFs3D))
        else:
            factors = jnp.array([1.0 / l, 1.0 / (l * a), 1.0 / (l * a * a), 1.0 / (l * a * a * a)],
                                dtype=jnp.float64)
            MFs3D = MFs3D * factors[None, :]
            self.MFs3D = np.array(MFs3D)  # convert back to numpy for easy printing/consumption

        self.logger.info(f'Processed {dims_x} slices in {time.time() - t0:.2f} s. Volume (valid pixels): {vol}')
        return self.MFs3D
