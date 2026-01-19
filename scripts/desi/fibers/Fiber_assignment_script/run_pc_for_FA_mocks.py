import os 
from pycorr import TwoPointCorrelationFunction, setup_logging
import numpy as np
from mockfactory import Catalog
from matplotlib import pyplot as plt
import argparse
import logging


logger = logging.getLogger('F.A.')
setup_logging()

def get_2PCF_from_LC(cat, rd_cat, edges, nthreads=256, R1R2=None, weights=None,weights_rd=None, mask=None, mask_rd=None):

    if mask is None:
        mask = np.ones(cat.csize, dtype=bool)
    else:
        if weights is not None:
            weights = weights[mask]
    if mask_rd is None:
        mask_rd = np.ones(rd_cat.csize, dtype=bool)
    else:
        if weights_rd is not None:
            weights_rd = weights_rd[mask_rd]

    result = TwoPointCorrelationFunction('smu', edges,
                                        data_positions1=[cat['RA'][mask], cat['DEC'][mask], cat['Distance'][mask]],
                                        data_positions2=None, engine='corrfunc', position_type='rdd', data_weights1=weights,
                                        randoms_positions1=[rd_cat['RA'][mask_rd], rd_cat['DEC'][mask_rd], rd_cat['Distance'][mask_rd]], randoms_weights1=weights_rd,
                                        boxsize=None, nthreads=nthreads, R1R2=R1R2)

    return result

def get_wp_from_LC(cat, rd_cat, edges, nthreads=256, R1R2=None, weights=None,weights_rd=None, mask=None, mask_rd=None):

    if mask is None:
        mask = np.ones(cat.csize, dtype=bool)
    else:
        if weights is not None:
            weights = weights[mask]
    if mask_rd is None:
        mask_rd = np.ones(rd_cat.csize, dtype=bool)
    else:
        if weights_rd is not None:
            weights_rd = weights_rd[mask_rd]

    result = TwoPointCorrelationFunction('rppi', edges,
                                        data_positions1=[cat['RA'][mask], cat['DEC'][mask], cat['Distance'][mask]],
                                        data_positions2=None, engine='corrfunc', position_type='rdd', data_weights1=weights,
                                        randoms_positions1=[rd_cat['RA'][mask_rd], rd_cat['DEC'][mask_rd], rd_cat['Distance'][mask_rd]], randoms_weights1=weights_rd,
                                        boxsize=None, nthreads=nthreads, R1R2=R1R2)


    return result


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Compute 2PCF for fiber assign cutsky mock')

    argparser.add_argument('--input_catalog', type=str, help='Path to input cutsky catalog')
    argparser.add_argument('--input_rd_catalog', type=str, help='Path to input random cutsky catalog')
    argparser.add_argument('--tracer', type=str, help='Tracer used choices: ELG, LRG, QSO', choices=['LRG', 'ELG', 'QSO'], default='LRG')
    argparser.add_argument('--output_dir', type=str, help='directory to put output computation and plots', default='2PCF_FA_results')
    args = argparser.parse_args()


    cutsky = Catalog.read(args.input_catalog)
    if 'AVAILABLE' not in cutsky.columns():
        raise ValueError('The cutsky catalog must have an AVAILABLE column indicating which targets were assigned fibers')

    mask_tr = (cutsky['TRACER'] == args.tracer)
    cutsky_rd = Catalog.read(args.input_rd_catalog)
    dir_output = args.output_dir

    os.makedirs(dir_output, exist_ok=True)

    edges_smu = [np.linspace(1, 150, 151), np.linspace(-1, 1, 201)]
    result_smu_avail = get_2PCF_from_LC(cutsky, cutsky_rd, edges=edges_smu, nthreads=256, mask=mask_tr & cutsky['AVAILABLE'], mask_rd=cutsky_rd['AVAILABLE'])

    s_lc, (xi0_lc, xi2_lc) = result_smu_avail(return_sep=True, ells=[0,2])

    result_smu_avail.save(os.path.join(dir_output, 'result_smu_available.npy'))

    result_smu_FA = get_2PCF_from_LC(cutsky, cutsky_rd, edges=edges_smu, nthreads=256, mask=mask_tr & (cutsky['NUMOBS']>0), mask_rd=cutsky_rd['AVAILABLE'], R1R2=result_smu_avail.R1R2)

    s_fa, (xi0_fa, xi2_fa) = result_smu_FA(return_sep=True, ells=[0,2])
    result_smu_FA.save(os.path.join(dir_output, 'result_smu_FA.npy'))

    result_smu_FA_wcomp = get_2PCF_from_LC(cutsky, cutsky_rd, edges=edges_smu, nthreads=256, mask=mask_tr & (cutsky['NUMOBS']>0), mask_rd=cutsky_rd['AVAILABLE'], R1R2=result_smu_avail.R1R2, weights=cutsky['COMP_WEIGHT'])
    s_fa_wcomp, (xi0_fa_wcomp, xi2_fa_wcomp) = result_smu_FA_wcomp(return_sep=True, ells=[0,2])
    result_smu_FA_wcomp.save(os.path.join(dir_output, 'result_smu_FA_wcomp.npy'))


    plt.plot(s_lc, xi0_lc*s_lc**2, label='Available for FA')
    plt.plot(s_fa, xi0_fa*s_fa**2, label='Observed after FA')
    plt.plot(s_fa_wcomp, xi0_fa_wcomp*s_fa_wcomp**2, label='Observed after FA w/ comp weight')

    plt.legend()
    plt.xlabel('s [Mpc/h]')
    plt.ylabel('$s^2 \cdot \\xi_0$')
    plt.savefig(os.path.join(dir_output,'xi0_fa_vs_avail.png'))
    plt.close()

    edges_rppi = [np.geomspace(0.01, 100, 48), np.linspace(-40, 40, 81)]

    result_rppi_avail = get_wp_from_LC(cutsky, cutsky_rd, edges=edges_rppi, nthreads=256, mask=mask_tr & cutsky['AVAILABLE'], mask_rd=cutsky_rd['AVAILABLE'])

    rp_lc, wp_lc = result_rppi_avail(return_sep=True, pimax=40)

    result_rppi_avail.save(os.path.join(dir_output, 'result_rppi_available.npy'))

    result_rppi_FA = get_wp_from_LC(cutsky, cutsky_rd, edges=edges_rppi, nthreads=256, mask=mask_tr & (cutsky['NUMOBS']>0), mask_rd=cutsky_rd['AVAILABLE'], R1R2=result_rppi_avail.R1R2)

    rp_fa, wp_fa = result_rppi_FA(return_sep=True, pimax=40)
    result_rppi_FA.save(os.path.join(dir_output, 'result_rppi_FA.npy'))

    result_rppi_FA_wcomp = get_wp_from_LC(cutsky, cutsky_rd, edges=edges_rppi, nthreads=256, mask=mask_tr & (cutsky['NUMOBS']>0), mask_rd=cutsky_rd['AVAILABLE'], R1R2=result_rppi_avail.R1R2, weights=cutsky['COMP_WEIGHT'])
    rp_fa_wcomp, wp_fa_wcomp = result_rppi_FA_wcomp(return_sep=True, pimax=40)
    result_rppi_FA_wcomp.save(os.path.join(dir_output, 'result_rppi_FA_wcomp.npy'))


    plt.semilogx(rp_lc, wp_lc*rp_lc, label='Available for FA')
    plt.semilogx(rp_fa, wp_fa*rp_fa, label='Observed after FA')
    plt.semilogx(rp_fa_wcomp, wp_fa_wcomp*rp_fa_wcomp, label='Observed after FA w/ comp weight')

    plt.legend()
    plt.xlabel('rp [Mpc/h]')
    plt.ylabel('$r_p \cdot w_p$ [Mpc/$h$]')

    plt.savefig(os.path.join(dir_output,'wp_fa_vs_avail.png'))

