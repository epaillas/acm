#cython: language_level=3 
#cython: profile=False 
#cython: boundscheck=False
#cython: wraparound=False
import cython
from cython.parallel import prange, threadid
cimport openmp
import numpy as np
from libc.stdlib cimport malloc, free
from libc.math cimport fabs
from libc.stdio cimport FILE, fprintf, fopen, fclose, printf, fflush, stdout
from libcpp.vector cimport vector

#################################################### Definitions ###########################################################

DEF SPEED_OF_LIGHT=299792.458
DEF PI=3.1415926535
DEF PREC_DIGIT=10

cdef extern from "math.h":
    double sqrt(double x) nogil
    double cos(double x) nogil
    double sin(double x) nogil
    double pow(double x, double y) nogil
ctypedef int size_t
cdef extern from "gsl/gsl_block.h":
    ctypedef struct gsl_block:
        size_t size
        double * data
cdef extern from "gsl/gsl_permutation.h":
    ctypedef struct gsl_permutation:
        size_t size
        size_t * data
    gsl_permutation *  gsl_permutation_alloc(size_t n) nogil
    void gsl_permutation_free(gsl_permutation * p) nogil
cdef extern from "gsl/gsl_matrix.h":
    ctypedef struct gsl_matrix:
        size_t size1
        size_t size2
        size_t tda
        double * data
        gsl_block * block
        int owner
    ctypedef struct gsl_matrix_view:
        gsl_matrix matrix
    gsl_matrix_view_array (double * base,
                             const size_t n1, 
                             const size_t n2) nogil
    gsl_matrix *  gsl_matrix_alloc(size_t n1, size_t n2) nogil
    double  gsl_matrix_get(gsl_matrix * m, size_t i, size_t j) nogil
    void  gsl_matrix_set(gsl_matrix * m, size_t i, size_t j, double x) nogil
    void  gsl_matrix_free(gsl_matrix * m) nogil
    gsl_matrix_view  gsl_matrix_submatrix(gsl_matrix * m, size_t k1, size_t k2, size_t n1, size_t n2) nogil
    int  gsl_matrix_memcpy(gsl_matrix * dest, gsl_matrix * src) nogil

cdef extern from "gsl/gsl_vector.h":
    ctypedef struct gsl_vector:
        size_t size
        size_t stride
        double * data
        gsl_block * block
        int owner
    gsl_vector *gsl_vector_alloc (const size_t n) nogil
    void gsl_vector_set (gsl_vector * v, const size_t i, double x) nogil
    double gsl_vector_get(gsl_vector * v, size_t i) nogil
    void gsl_vector_free (gsl_vector * v) nogil
cdef extern from "gsl/gsl_blas.h":
    int gsl_blas_ddot (const gsl_vector * X,
                   const gsl_vector * Y,
                   double * result
                   ) nogil
    int gsl_blas_dgemv(CBLAS_TRANSPOSE_t TransA,
                        double alpha,
                        gsl_matrix * A, 
                        gsl_vector * x, 
                        double beta, 
                        gsl_vector * y) nogil
    ctypedef enum CBLAS_TRANSPOSE_t:
                    CblasNoTrans=111
                    CblasTrans=112
                    CblasConjTrans=113
cdef extern from "gsl/gsl_math.h":
    ctypedef struct gsl_function:
        double (* function) (double x, void * params) 
        void * params
cdef extern from "gsl/gsl_linalg.h":
    int gsl_linalg_LU_decomp (gsl_matrix * A, gsl_permutation * p, int *signum) nogil
    int gsl_linalg_LU_solve (const gsl_matrix * LU,
                         const gsl_permutation * p,
                         const gsl_vector * b,
                         gsl_vector * x) nogil
cdef extern from "gsl/gsl_integration.h":
    ctypedef struct gsl_integration_workspace:
        size_t limit
        size_t nrmax
        size_t i
        size_t maximum_level
        double *alist
        double *blist
        double *rlist
        double *elist
        size_t *order
        size_t *level
    gsl_integration_workspace * gsl_integration_workspace_alloc(const size_t n) nogil
    void gsl_integration_workspace_free(gsl_integration_workspace * w) nogil
    int gsl_integration_qag (const gsl_function *f, 
                         double a, double b,
                         double epsabs, double epsrel, size_t limit,
                         int key,
                         gsl_integration_workspace * workspace,
                         double *result, double *abserr) nogil
    DEF GSL_INTEG_GAUSS51 = 5
    DEF GSL_INTEG_GAUSS61 = 6



cdef extern from "gsl/gsl_linalg.h":
    int gsl_linalg_LU_decomp (gsl_matrix * A, gsl_permutation * p, int *signum) nogil
    double gsl_linalg_LU_det (gsl_matrix * LU, int signum) nogil
    int  gsl_linalg_LU_invert(gsl_matrix * LU, gsl_permutation * p, gsl_matrix * inverse) nogil



cdef extern from "delaunay_backend.cpp":
    cdef cppclass DelaunayOutput:
        vector[double] x, y, z, r, volume, dtfe, area, px, py, pz
        vector[size_t] vertices[4]
        size_t n_simplices
    DelaunayOutput cdelaunay(vector[double] X, vector[double] Y, vector[double] Z) nogil
    DelaunayOutput cdelaunay_periodic_extend(vector[double] X, vector[double] Y, vector[double] Z, vector[double] box_min, vector[double] box_max, double cpy_range) nogil
    DelaunayOutput cdelaunay_periodic(vector[double] X, vector[double] Y, vector[double] Z, vector[double] box_min, vector[double] box_max) nogil
    DelaunayOutput cdelaunay_full(vector[double] X, vector[double] Y, vector[double] Z) nogil
    DelaunayOutput cdelaunay_periodic_full(vector[double] X, vector[double] Y, vector[double] Z, vector[double] box_min, vector[double] box_max, double cpy_range) nogil

cdef void matrix_invert(gsl_matrix *matrix, gsl_matrix *matrix_inv, int size) nogil:

    cdef gsl_permutation *p = gsl_permutation_alloc(size);
    cdef int s;

    
    gsl_linalg_LU_decomp(matrix, p, &s);

    
    
    gsl_linalg_LU_invert(matrix, p, matrix_inv);

    gsl_permutation_free(p);


    
def get_void_catalog_cgal(double[:,:] points,
                        bint periodic = False,
                        int periodic_mode = 0,
                        list box_min = [0., 0., 0.],
                        list box_max = [0., 0., 0.],
                        double cpy_range = 0.,
                        ):

    cdef Py_ssize_t i,k
    cdef vector[double] in_x, in_y, in_z, _box_min, _box_max
    
    
    
    in_x.reserve(points.shape[0])
    in_y.reserve(points.shape[0])
    in_z.reserve(points.shape[0])
    
    for i in range(points.shape[0]):
        in_x.push_back(points[i,0])
        in_y.push_back(points[i,1])
        in_z.push_back(points[i,2])
    
    cdef DelaunayOutput voids
    if not periodic:
        voids = cdelaunay(in_x, in_y, in_z)
    else:
        for i in range(3):
            _box_min.push_back(box_min[i])
            _box_max.push_back(box_max[i])
        if periodic_mode == 0:
            voids = cdelaunay_periodic_extend(in_x, in_y, in_z, _box_min, _box_max, cpy_range)
        elif periodic_mode == 1:
            voids = cdelaunay_periodic(in_x, in_y, in_z, _box_min, _box_max)
        else:
            raise ValueError("periodic_mode must be 0 (extend bounds) or 1 (periodic data structures).")
        
    cdef size_t n_simplices
    in_x.clear()
    in_y.clear()
    in_z.clear()
    
    n_simplices = voids.n_simplices
    output = np.zeros((n_simplices, 4), dtype=np.double)
    for k in range(n_simplices):
        output[k,0] = voids.x[k]
        output[k,1] = voids.y[k]
        output[k,2] = voids.z[k]
        output[k,3] = voids.r[k]
    
    

    return output

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def get_void_catalog_full(double[:,:] points,
                        bint periodic=False,
                        int n_threads = 16,
                        double p = 2,
                        list box_min = [0., 0., 0.],
                        list box_max = [0., 0., 0.],
                        double cpy_range = 0.,):

    cdef Py_ssize_t i,k,j
    cdef vector[double] in_x, in_y, in_z, _box_min, _box_max
       
    in_x.reserve(points.shape[0])
    in_y.reserve(points.shape[0])
    in_z.reserve(points.shape[0])
    
    for i in range(points.shape[0]):
        in_x.push_back(points[i,0])
        in_y.push_back(points[i,1])
        in_z.push_back(points[i,2])
    
    cdef DelaunayOutput voids
    if not periodic:
        voids = cdelaunay_full(in_x, in_y, in_z)
    else:
        for i in range(3):
            _box_min.push_back(box_min[i])
            _box_max.push_back(box_max[i])
        voids = cdelaunay_periodic_full(in_x, in_y, in_z, _box_min, _box_max, cpy_range)
    
    cdef size_t n_simplices
    in_x.clear()
    in_y.clear()
    in_z.clear()
    
    cdef double w, numerator
    
    
    fflush(stdout)

    n_simplices = voids.n_simplices
    output = np.zeros((n_simplices, 7), dtype=np.double)
    dtfe = np.zeros(points.shape[0], dtype=np.double)
    cdef double [:,:] output_view = output
    cdef double [:] dtfe_view = dtfe

    printf("==> Computing DTFE\n")
    fflush(stdout)
    for k in range(points.shape[0]):
        
        dtfe_view[k] = 4. / voids.dtfe[k]
        
        
        
    printf("==> Copying voids and interpolating\n")
    fflush(stdout)
    
   
    for k in range(n_simplices):
        output_view[k,0] = voids.x[k]
        output_view[k,1] = voids.y[k]
        output_view[k,2] = voids.z[k]
        output_view[k,3] = voids.r[k]
        output_view[k,4] = voids.volume[k]
        w = 1. / voids.r[k]**p
        numerator = 0
        for i in range(4):
            if voids.vertices[i][k] < points.shape[0]:
                numerator = numerator + w * dtfe_view[<size_t> voids.vertices[i][k]]
        output_view[k,5] = numerator / (4 * w)
        
        output_view[k,6] = voids.area[k]
        
    

    printf("    Done\n")
    fflush(stdout)
        
    

    return output, dtfe


        
cpdef int allocate_to_grid(double[:,:] data, 
                    int[:,:,:] grid_void_count, 
                    int[:,:,:,:] grid_id_buffer, 
                    double box_size,
                    int n_threads) nogil except -1:
    
    cdef int n_grid = grid_void_count.shape[0]
    cdef double bin_size = box_size / n_grid
    cdef double inv_bin_size = n_grid / box_size
    cdef int n_buffer = grid_id_buffer.shape[3]
    cdef int idx, idy, idz
    cdef int i
    for i in prange(data.shape[0], nogil=True, num_threads=n_threads):
        
        idx = <int> ((data[i, 0] * inv_bin_size) + n_grid) % n_grid
        idy = <int> ((data[i, 1] * inv_bin_size) + n_grid) % n_grid
        idz = <int> ((data[i, 2] * inv_bin_size) + n_grid) % n_grid
        
        if n_buffer < grid_void_count[idx, idy, idz]: return -1
        grid_id_buffer[idx, idy, idz, grid_void_count[idx, idy, idz]] = i
        grid_void_count[idx, idy, idz] += 1
    
    return 0

def get_satellites(double[:,:] data, 
                    int[:,:,:] grid_void_count, 
                    int[:,:,:,:] grid_id_buffer, 
                    int[:] central_id, 
                    int[:] is_central, 
                    int[:] n_satellites, 
                    double[:] distances, 
                    double[:] central_radius, 
                    double box_size,
                    double box_pad):

    """
    Compute the number of satellites from a void catalog
    Parameters:
    data: ndarray shape (nvoids, 4)
        Array containing the void coordinates and radii (x,y,z,r)
    grid_void_count: ndarray shape (ngrid, ngrid, ngrid)
        Array containing the number of voids in the cell (ix, iy, iz)
        See func. allocate_to_grid
    grid_id_buffer: ndarray (ngrid, ngrid, ngrid, nbuffer)
        Array containing the ids of the voids in the cell (ix, iy, iz)
        See func. allocate_to_grid
    central_id: ndarray (nvoids,)
        Array to store the id of the central void
    is_central: ndarray (nvoids,)
        Array to store boolean value (True if central)
    n_satellites: ndarray (nvoids,)
        Array to store number of satellites for a central void,
        zero if satellite.
    distances: ndarray (nvoids,)
        Array to store de distance to the corresponding central void.
    central_radius: ndarray (nvoids,)
        Array to store the radius of the corresponding central void.
    box_size: double
        Size of the box in which the grid is embedded
    
    """
    if data.shape[1] < 4:
        raise ValueError
    cdef int n_grid = grid_void_count.shape[0]
    cdef double inv_bin_size = n_grid / (box_size + 2 * box_pad)
    cdef int xmin, xmax, ymin, ymax, zmin, zmax
    cdef Py_ssize_t i, j, k
    cdef double sqr, r

    for i in range(data.shape[0]):
        
        if is_central[i]:
            sqr = data[i,3]*data[i,3]
            central_id[i] = i
            central_radius[i] = data[i,3]
            r = data[i,3] 
            xmin = <int> (((data[i,0] + box_pad - r) * inv_bin_size) )
            xmax = <int> (((data[i,0] + box_pad + r) * inv_bin_size) )
            ymin = <int> (((data[i,1] + box_pad - r) * inv_bin_size) )
            ymax = <int> (((data[i,1] + box_pad + r) * inv_bin_size) )
            zmin = <int> (((data[i,2] + box_pad - r) * inv_bin_size) )
            zmax = <int> (((data[i,2] + box_pad + r) * inv_bin_size) )
            
            for xid in range(xmin, xmax+1):
                #if xid >= n_grid: continue
                xid += n_grid
                xid %= n_grid
                for yid in range(ymin, ymax+1):
                    #if yid >= n_grid: continue
                    yid += n_grid
                    yid %= n_grid
                    for zid in range(zmin, zmax+1):
                        #if zid >= n_grid: continue
                        zid += n_grid
                        zid %= n_grid

                        for j in range(grid_void_count[xid, yid, zid]):
                            k = grid_id_buffer[xid, yid, zid, j]
                            
                            if k <= i: continue
                            distance = (data[i, 0] - data[k, 0])**2 + (data[i, 1] - data[k, 1])**2 + (data[i, 2] - data[k, 2])**2
                            
                            if distance < sqr:
                                
                                if is_central[k]:    
                                    central_id[k] = i
                                    central_radius[k] = data[i,3]
                                    is_central[k] = 0
                                    n_satellites[i] += 1
                                    distances[k] = distance
                                elif not is_central[k] and distances[k] < distance:
                                    n_satellites[central_id[k]] -= 1
                                    central_id[k] = i
                                    central_radius[k] = data[i,3]
                                    is_central[k] = 0
                                    n_satellites[i] += 1
                                    distances[k] = distance


########################################## Sky to Cartesian Coordinate conversion ################################################

@cython.boundscheck(False)
cdef double comoving_dist_integrand(double x, void * params) nogil:

    # Signature must match what GSL asks for in the integration.
    # Extract parameters
    cdef double H0 =  (<double *> params)[0]
    cdef double OmegaL =  (<double *> params)[1]
    cdef double OmegaM =  (<double *> params)[2]
    cdef double c =  (<double *> params)[3]
    #printf("%lf %lf %lf %lf\n", H0, OmegaL, OmegaM, c)

    cdef double H = H0 * sqrt(OmegaL + OmegaM * pow( 1 + x , 3))

    return c / H
    
cdef double integrate_z(double z, double H0, double OmegaL, double OmegaM) nogil :

    cdef gsl_integration_workspace *w
    cdef double result, error
    cdef double c = SPEED_OF_LIGHT
    cdef int prec = PREC_DIGIT
    cdef gsl_function integrand
    cdef double* params = [ H0, OmegaL, OmegaM, c ]

    integrand.function = &comoving_dist_integrand
    integrand.params = params

    w = gsl_integration_workspace_alloc(1000)
    gsl_integration_qag(&integrand, 0, z, 0, pow(10, -prec), 1000, GSL_INTEG_GAUSS51, w, &result, &error)
    gsl_integration_workspace_free(w)

    return result

@cython.boundscheck(False)
def sky_to_cart_parallel(double[:,:] input, double[:,:] output, int n_lines, int n_threads, double H0=67.7, double OmegaM=0.307115):

    cdef double OmegaL = 1 - OmegaM
    cdef double dist, ra, dec, h
    h = H0 / 100
    cdef Py_ssize_t i
    for i in prange(n_lines, nogil=True, num_threads=n_threads):
        dist = integrate_z(input[i,2], H0, OmegaL, OmegaM)
        ra = input[i,0] * PI / 180
        dec = input[i,1] * PI / 180
        #X
        output[i,0] = dist * cos(dec) * cos(ra) * h
        #Y
        output[i,1] = dist * cos(dec) * sin(ra) * h
        #Z
        output[i,2] = dist * sin(dec) * h
        