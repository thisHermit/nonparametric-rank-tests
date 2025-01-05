import pyshearlab
import numpy
import matplotlib.pyplot as plt
import time
import pickle
import pywt
from numba import jit
import warnings
import os
import math
from numpy import matlib

def l2norm_der_ft(f, k):
    """
    Compute L2-norm of k-th derivative of f using Fourier transform
    """

    f = f.flatten('F')
    n = f.size
    w = 2*numpy.pi*numpy.roll(numpy.arange(-math.floor(n/2),math.ceil(n/2)).transpose(), n%2)
    w = numpy.fft.fftshift(numpy.power(w,2*k))
    
    ndf = numpy.abs( numpy.fft.fft(f) / n )
    ndf = numpy.power(ndf, 2)
    ndf = numpy.multiply(w, ndf)
    ndf = numpy.sqrt(numpy.sum(ndf))
    
    return ndf

def num_divergence(W_1, W_2):
    """
    Numerical divergence
    """

    div_W = numpy.zeros(W_1.shape)
    div_W[1:-1,:] = W_1[1:-1,:] - W_1[:-2,:]
    div_W[0,:] = W_1[0,:]
    div_W[-1,:] = -W_1[-2,:]
    div_W[:,1:-1] = div_W[:,1:-1] + W_2[:,1:-1] - W_2[:,:-2]
    div_W[:,0] += W_2[:,0]
    div_W[:,-1] -= W_2[:,-2]
    
    return div_W

def num_grad(g):
    """
    Numerical gradient
    """

    g_x = numpy.zeros(g.shape)
    g_x[:-1,:] = g[1:,:] - g[:-1,:]
    g_y = numpy.zeros(g.shape)
    g_y[:,:-1] = g[:,1:] - g[:,:-1]
    
    return [g_x, g_y]

def proxHk(v, pen, k, verbose = False):
    """
    Proximal operator of Hk
        minimize_{u} 1/2||u - v||_2^2  + lambda/2 ||D^ u||_2^2
    """

    m, n = v.shape
    wr = matlib.repmat(2*numpy.pi*numpy.roll(numpy.arange(-math.floor(m/2), math.ceil(m/2)).reshape(m, 1), m%2), 1, n)
    wc = matlib.repmat(2*numpy.pi*numpy.roll(numpy.arange(-math.floor(n/2), math.ceil(n/2)),n%2), m, 1)
    w = numpy.fft.fftshift(numpy.power(numpy.power(wr, 2)+numpy.power(wc, 2), k))
    
    if verbose:
        print("Sobolev inversion ...\n")
        tic = time.perf_counter()
        
    pen = pen / (m*n)
    u = numpy.fft.ifft2(numpy.divide(numpy.fft.fft2(v), 1+numpy.multiply(pen, w))).real
    
    if verbose:
        print("elapsed {} seconds\n".format(time.perf_counter() - tic))
        
    return u

def proxTV(v, pen, ui = None, maxit = 500, tol = 10e-3, verbose = False):
    """
    Proximal operator of TV
        minimize_{u} 1/2||u - v||_2^2  + lambda||\nabla u||_1
    """

    tau = 0.24
    if ui is None:
        ui = v
        
    w = ui
    W_1 = numpy.zeros(ui.shape)
    W_2 = numpy.zeros(ui.shape)
    v_x, v_y = num_grad(v)
    not_converged = True
    nit = 0
    
    while not_converged:
        w_x, w_y = num_grad(w)
        W_1 = W_1 + tau*(w_x + v_x)
        W_2 = W_2 + tau*(w_y + v_y)
        rescaling_factor = numpy.sqrt(W_1**2 + W_2**2) / pen
        rescaling_factor[rescaling_factor < 1] = 1
        W_1 = W_1 / rescaling_factor
        W_2 = W_2 / rescaling_factor
        w_new = num_divergence(W_1, W_2)

        if (nit >= maxit) or (numpy.max(abs(w_new - w)) <= tol):
            not_converged = False

        w = w_new
        nit += 1
        
    if verbose:
        print("iterations in \"proxTv\": ", nit)
        
    u = w+v
    
    return u


def chTV(g, pen, maxit = 500, tol = 10e-3, verbose = False):
    """
    Proximal operator of TV ('fast' algorithm)
        minimize_{u} 1/2||u - v||_2^2  + lambda||\nabla u||_1
    """

    tau = 0.24
    nit = 0

    p_1 = numpy.zeros(g.shape)
    p_2 = numpy.zeros(g.shape)
    w = g
    not_converged = True
    while not_converged:
        grad_1, grad_2 = num_grad(num_divergence(p_1, p_2) - g/pen)
        rescale = 1 + tau * numpy.sqrt(numpy.power(grad_1, 2) + numpy.power(grad_2, 2))
        p_1 = (p_1 + tau * grad_1) / rescale
        p_2 = (p_2 + tau * grad_2) / rescale
        # match stopping criterion
        w_new = pen * num_divergence(p_1, p_2)
        nit += 1

        if (nit > maxit) or (numpy.max(abs(w_new - w)) < tol):
            not_converged = False

        w = w_new

    if verbose:
        print("iterations in \"chTv\": ", nit)

    return g - w_new

def proxl1(v, pen):
    """
    Proximal operator of l1 norm
        minimize_{u} 1/2||u - v||_2^2  + lambda||u||_1
    """

    if isinstance(v, list):
        u = v.copy()
        for i in range(len(u)):
            u[i] = proxl1(v[i], pen)
    elif isinstance(v, tuple):
        return proxl1(v[0], pen), proxl1(v[1], pen), proxl1(v[2], pen)
    else:
        aux = numpy.abs(v) - pen
        aux = (aux + numpy.abs(aux)) / 2
        u = numpy.multiply(numpy.sign(v), aux)

    return u

"""
Interface for all the transforms
"""

warnings.filterwarnings('ignore')

class TransformInterface:
    adjoint = False


    def ctranspose(self):
        """
        Create the conjugate adjoint operator
        """
        self.adjoint = not self.adjoint


    def forward(self, img, normed=True):
        """
        Perform transform on an image.

        Usage:

            v = transform.forward(u)

        Input:

            img: The image to be transformed
            normed: optional (default normed = True)
                    Logical value that specifies whether or not the transform should be normed
        """
        pass


    def multiscale(self, img, th, maxIt=500, sigma=None, tau=None, theta=1, toDisp=2, check=50,
                   tol=1e-4, ctol=1e-2, rType='TV', nitTV=1e3, tolTV=1e-3, k_sob=1):
        """
        Multiscale regression via R minimization under multiscale constraints
            minimize_u R(u) subject to ||Phi (u-v)||_inf <= th

        Usage:

            x, stat = transform.multiscale(img, th)

        Input:

            img: data matrix
            th: threshold in multiscale constraint
            maxIt: optional (default maxIt = 500).
                   maximum number of iterations
            sigma: optional (default sigma = max(img.shape)) step size in dual problem

            tau: optional (default tau = 1/sigma) step size in primal problem

            theta: optional (default theta = 1) step size for extrapolation

            toDisp: optional (default toDisp = 2)
                    Logical value that controls information displayed to the user while the algorithm is running.
                    0: no information
                    1: only information in form of text on the console
                    2: additionally the current images are displayed periodically every <<check>> iteration
            check: optional (default check = 50)
                   period of displayed images
            tol: optional (default tol = 1e-4)
                 stopping criterion. threshold for weighted difference between this and last step's image
            ctol: optional (default ctol = 1e-2)
                  stopping criterion. threshold for difference between the highest values in this and last step's image
                  in the transformed space
            rType: optional (default rType = 'TV')
                   regression type using
                   'TV': proximal operator of TV ( minimize_{u} 1/2||u - v||_2^2  + lambda||\nabla u||_1 )
                   'HK': proximal operator of HK ( minimize_{u} 1/2||u - v||_2^2  + lambda/2 ||D^ u||_2^2 )
                   'chTV': proximal operator of TV (different 'fast' algorithm)
            nitTV: optional (default nitTV = 1e3)
                   if rType == 'TV' specifies the max number of iterations in each call of the proximal operator
            tolTV: optional (default tolTV = 1e-3)
                   stopping criterion threshold for the algorithm of the TV operator
            k_sob: optional (default k_sob = 1)
                   if rType == 'HK' specifies Sobolev inversion parameter

        Output:

            x: solution matrix
            stat: details of each outer iteration
                  'oVal': objective values R(u)
                  'cGap': gaps of the constraint || Phi (u-v) ||_inf - th
                      tm: computation cost until each iteration
        """
        # default values
        if sigma is None:
            sigma = max(img.shape)
        if tau is None:
            tau = 1 / sigma

        # initialization
        xo = img
        xbar = xo
        Kimg = self.forward(img)
        y = Kimg
        stat = dict()
        stat['oVal'] = numpy.zeros(maxIt)
        stat['cGap'] = numpy.zeros(maxIt)
        stat['tm'] = numpy.zeros(maxIt)

        if toDisp > 0:
            print('Iteration starts ... ({} in total)\n'.format(maxIt))
            tmAll = time.perf_counter()

        # iteration
        it = 0
        while it < maxIt:

            tmSt = time.perf_counter()

            it += 1

            if (it % check == 0) and (toDisp > 0):
                print(it, 'th iteration\n')

            Kxbar = self.forward(xbar)
            y = y + sigma * (Kxbar - Kimg)
            y = proxl1(y, sigma * th)
            self.ctranspose()
            x = xo - tau * numpy.real(self.forward(y))
            self.ctranspose()

            if rType == 'TV':
                x = proxTV(x, tau, x, nitTV, tolTV / it, (it % check == 0) and (toDisp > 0))
            elif rType == 'HK':
                x = proxHk(x, tau, k_sob, (it % check == 0) and (toDisp > 0))
            elif rType == 'chTV':
                x = chTV(x, tau, nitTV, tolTV / it, (it % check == 0) and (toDisp > 0))

            xbar = x + theta * (x - xo)
            xo = x

            # iteration details
            stat['tm'][it - 1] = time.perf_counter() - tmSt
            if rType == 'TV' or rType == 'ch':
                dx1, dx2 = num_grad(x)
                stat['oVal'][it - 1] = numpy.sum(numpy.sqrt(dx1[:]**2) + dx2[:]**2) / img.size
            elif rType == 'HK':
                stat['oVal'][it - 1] = 0.5 * l2norm_der_ft(x, k_sob)

            Kx = self.forward(x)
            maxVal = numpy.max(numpy.abs(Kx - Kimg))
            if abs(theta) > numpy.finfo(float).eps:
                stat['cGap'][it - 1] = (maxVal - th) / th
            else:
                stat['cGap'][it - 1] = maxVal - th

            if (it % check == 0):
                inc = numpy.linalg.norm(xbar - x) / theta / numpy.sqrt(x.size)
                gap = maxVal / th - 1
                if toDisp > 1:
                    # new plot for iteration
                    plt.figure(figsize=(15, 3.5))
                    plt.subplot(1, 3, 1)
                    plt.title('Data')
                    plt.axis('off')
                    plt.imshow(img)
                    plt.colorbar()
                    plt.subplot(1, 3, 2)
                    plt.title('Iter {}'.format(it))
                    plt.axis('off')
                    plt.imshow(x)
                    plt.colorbar()
                    plt.subplot(1, 3, 3)
                    plt.title('Residual')
                    plt.axis('off')
                    plt.imshow(x - img)
                    plt.colorbar()
                    plt.show()

                    print('\t changes {sinc} (tol {stol}), gap {sgap} (ctol {sctol})\n'.format(sinc=inc, stol=tol,
                                                                                               sgap=gap, sctol=ctol))
                if (gap < ctol) and (inc < tol):
                    break
        if toDisp > 0:
            print('Stop at {} iterations and {} sec elapsed!\n'.format(it, time.perf_counter() - tmAll))

        stat['oVal'] = stat['oVal'][:it]
        stat['cGap'] = stat['cGap'][:it]
        stat['tm'] = stat['tm'][:it]

        return x, stat


    def msQuantile(self, alpha=0.9, nDraw=5000, seed=100, toSave=True, toDisp=True, check=500, dir='./simulations/', filename=None):
        '''
        Simulate the lower alpha quantile of multiscale statistics

        Usage:

            q, mStat = transform.msQuantile()

        Input:

            alpha: optional (default alpha = 0.9)
                   significance level
            nDraw: optional (default nDraw = 5000)
                   number of draws in the simulation
            seed: optional (default seed = 100)
                  seed used in random draws
            toSave: optional (default toSave = True)
                    Logical value that specifies whether or not the simulation should be saved to a '.pkl' file.
            toDisp: optional (default toDisp = True)
                    Logical value that specifies whether or not information about iterations should be printed onto the
                    console while the function runs.
            check: optional (default check = 500)
                   If toDisp == True, an update about the current iteration is printed onto the console every <<check>>
                   iterations.
            dir: optional (default dir = './simulations/')
                 If toSave == True this specifies the directory in which the file should be stored in.
            filename: optional (default filename = 'simQ_<<type(transform)>>_r_<<nDraw>>_sz_<<self.imSize[0]>>_<<self.imSize[1]>>.pkl')
                      Name of the data file created

        Output:

            q: the simulated lower alpha quantile
            mStat: the simulated values of multiscale statistics
        '''

        if filename is None:
            filename = 'simQ_{}_r_{}_sz_{}_{}.pkl'.format(type(self).__name__, nDraw, self.imSize[0], self.imSize[1])

        mStat, simEx = self.checkFile(dir + filename)

        if not simEx:
            if toDisp:
                print('Simulate via {} draw ...\n'.format(nDraw))
                tmAll = time.perf_counter()
            mStat = numpy.zeros((nDraw, 1))
            numpy.random.seed(seed)
            for r in range(nDraw):
                if (r + 1) % check == 0 and toDisp:
                    print('\t # {} draws'.format(r + 1))
                u = numpy.random.normal(size=self.imSize)
                mStat[r] = numpy.max(numpy.abs(self.forward(u)))
            if toDisp:
                print('End of simulation: {} sec elapsed'.format(time.perf_counter() - tmAll))
            if toSave:
                if toDisp:
                    print('Simulation result is stored!\n')
                    self.storeSim(mStat, dir + filename)
        else:
            print('Load from previous simulation results!\n')

        q = numpy.quantile(mStat, alpha)
        return q, mStat

    # checks if file exists and transform is the same
    # also returns simulated values if the file exists
    def checkFile(self, filename):
        '''
        Helper function to check whether or not the given file contains data about the correct transform
        '''
        pass

    # stores simulation and transform in pkl file
    def storeSim(self, data, filename):
        '''
        Helper function to store correct data in a '.pkl' file
        '''
        pass


class Shearlet(TransformInterface):
    '''
    Class for the Shearlet operator
    '''
    useGPU = False

    def __init__(self, imSize, nScales=4, shearLevels=None, full=False, directionalFilter=None,
                 quadratureMirrorFilter=None):
        '''
        Initializes a Shearlet operator object

        Usage:

            myShearlet = pymind.transform.Shearlet(imSize)

        Input:

            imSize: Size of the images for which the transform is to be used on.
            nScales: optional (default nScales = 4)
                     number of scales
            shearLevels: optional (default shearLevels = numpy.arange(0, nScales) // 2 + 1)
                         number of shearlet levels
            full: optional (default full = False)
                  Logical value that determines whether a full shearlet system is computed or if shearlets lying on the
                  border of the second cone are omitted.
            directionalFilter: optional
                               A 2D directional filter that serves as the basis of the directional component of the
                               shearlets. For more information see pyshearlab documentation.
            quadratureMirrorFilter: optional (default quadratureMirrorFilter =
                                                  numpy.array([0.0104933261758410, -0.0263483047033631,
                                                  -0.0517766952966370, 0.276348304703363, 0.582566738241592,
                                                  0.276348304703363, -0.0517766952966369, -0.0263483047033631,
                                                  0.0104933261758408])
                                    A 1D quadrature mirror filter defining the wavelet component of the shearlets.
        '''
        self.imSize = imSize
        self.QMF = quadratureMirrorFilter
        self.DF = directionalFilter
        if shearLevels is None:
            shearLevels = numpy.arange(0, nScales) // 2 + 1
        if self.QMF is None:
            quadratureMirrorFilter = numpy.array([0.0104933261758410, -0.0263483047033631,
                                                  -0.0517766952966370, 0.276348304703363, 0.582566738241592,
                                                  0.276348304703363, -0.0517766952966369, -0.0263483047033631,
                                                  0.0104933261758408])
        if self.DF is None:
            directionalFilter = pyshearlab.modulate2(pyshearlab.dfilters('dmaxflat4', 'd')[0] / numpy.sqrt(2), 'c')

        self.shearletSystem = pyshearlab.SLgetShearletSystem2D(useGPU=self.useGPU, rows=self.imSize[0],
                                                               cols=self.imSize[1],
                                                               nScales=nScales, shearLevels=shearLevels, full=full,
                                                               directionalFilter=self.DF,
                                                               quadratureMirrorFilter=self.QMF)

        deltaFun = numpy.fft.fftshift(numpy.fft.ifft2(numpy.ones(imSize))) * numpy.sqrt(numpy.prod(imSize))
        slcoef = pyshearlab.SLsheardec2D(deltaFun, self.shearletSystem)
        self.normShe = numpy.zeros((self.shearletSystem['nShearlets'], 1))
        for s in range(self.shearletSystem['nShearlets']):
            self.normShe[s] = numpy.linalg.norm(slcoef[:, :, s]) / numpy.sqrt(slcoef[:, :, s].size)

    def forward(self, img, normed=True):
        v = img.copy()
        if normed:
            if self.adjoint:
                for s in range(self.shearletSystem['nShearlets']):
                    v[:, :, s] = v[:, :, s] * self.normShe[s]
                prodimg = pyshearlab.SLshearrec2D(v, self.shearletSystem)
            else:
                prodimg = pyshearlab.SLsheardec2D(v, self.shearletSystem)
                for s in range(self.shearletSystem['nShearlets']):
                    prodimg[:, :, s] = prodimg[:, :, s] / self.normShe[s]

        else:
            if self.adjoint:
                prodimg = pyshearlab.SLshearrec2D(v, self.shearletSystem)
            else:
                prodimg = pyshearlab.SLsheardec2D(v, self.shearletSystem)

        return prodimg

    def storeSim(self, data, filename):
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        shearletSim = {'mStat': data, 'shearLevels': self.shearletSystem['shearLevels'],
                       'full': self.shearletSystem['full'],
                       'QMF': self.QMF, 'DF': self.DF}
        pickle.dump(shearletSim, open(filename, 'wb'))

    def checkFile(self, filename):
        try:
            shearletSim = pickle.load(open(filename, 'rb'))
            if ((shearletSim['shearLevels'] == self.shearletSystem['shearLevels']).all() and
                    shearletSim['full'] == self.shearletSystem['full'] and
                    shearletSim['QMF'] == self.QMF and
                    shearletSim['DF'] == self.DF):
                return shearletSim['mStat'], True
            else:
                return None, False
        except:
            return None, False


class Wavelet(TransformInterface):
    """
    Class for the Wavelet operator
    """

    def __init__(self, imSize, filterType='sym6', wavScale=2):
        """
        Initializes a Wavelet operator object

        Usage:

            myWavelet = pymind.transform.Wavelet(imSize)

        Input:

            imSize: Size of the images for which the transform is to be used on.
            filterType: optional (default filterType = 'sym6')
                        Wavelet to use. To get a list of possible wavelets use pywt.families() and pywt.wavelist().
            wavScale: optional (default wavScale = 2)
                      Wavelet scale.
        """
        self.minScale = wavScale
        self.filterType = filterType
        self.imSize = imSize

    def forward(self, img, normed=True):
        if self.adjoint:
            return pywt.waverec2(pywt.array_to_coeffs(img, self.slice, 'wavedec2'), self.filterType,
                                 mode='periodization')
        else:
            Kimg, slice = pywt.coeffs_to_array(
                pywt.wavedec2(img, self.filterType, level=self.minScale, mode='periodization'))
            self.slice = slice
            return Kimg

    def storeSim(self, data, filename):
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        waveletSim = {'mStat': data, 'filterType': self.filterType, 'minScale': self.minScale, 'imSize': self.imSize}
        pickle.dump(waveletSim, open(filename, 'wb'))

    def checkFile(self, filename):
        try:
            waveletSim = pickle.load(open(filename, 'rb'))
            if waveletSim['filterType'] == self.filterType and waveletSim['minScale'] == self.minScale and waveletSim[
                'imSize'] == self.imSize:
                return waveletSim['mStat'], True
            else:
                return None, False
        except:
            return None, False


class Cube(TransformInterface):
    """
    Class for the Cube operator
    """

    def __init__(self, imSize, ctype='partition', param=None, seed=123, norm=None):
        """
        Intializes a Cube operator object

        Usage:

            myCube = pymind.transform.Cube(imSize)

        Input:

            imSize: Size of the images for which the transform is to be used on.
            ctype: optional (default ctype = 'partition')
                   Specifies how the cube operator is created.
                   'partition': Generates a param-partition cube system
                   'scale': Generates a cube system by specified scales
                   'custom': Takes param as cube system
            param: Input depends on choice of ctype.
                   'partition': optional (default param = 2)
                   'scale': optional (default param = numpy.power(2, numpy.arange(0, numpy.floor(numpy.log2(numpy.min(imSize))) + 1)))
                   'custom': Takes param as cube system (no default)
            seed: optional (default seed = 123)
                  If norm is None, seed to be used in the estimation of the norm
            norm: optional
                  Norm to be used in the normed transformation. If norm is not given, a norm is estimated via power method.
        """

        if ctype == 'partition':
            if param is None:
                param = 2
            st, ed = self.pPartitionCube(imSize, param)
        elif ctype == 'scale':
            if param is None:
                param = numpy.power(2, numpy.arange(0, numpy.floor(numpy.log2(numpy.min(imSize))) + 1))
            st, ed = self.scale2cube(param, imSize)
        elif ctype == 'custom':
            st, ed = param

        self.imSize = imSize
        self.type = ctype
        self.st = st - 1
        self.ed = ed - 1

        if norm is None:
            print('Estimate the norm (this might take some time) ...\n')
            tstart = time.perf_counter()
            tol = 1e-6
            nit = 1e3
            numpy.random.seed(seed)
            u = numpy.random.randn(imSize[0], imSize[1])
            u = u / numpy.linalg.norm(u)

            i = 0
            while i < nit:
                i += 1
                v = self.mrdualCube(imSize, self.mrcoefCube(u))
                en = numpy.sum(v * u)
                if numpy.linalg.norm(v - numpy.dot(en, u)) < tol:
                    break
                u = v / numpy.linalg.norm(v)
            self.norm = numpy.sqrt(en)*1.1

            print('Time cost is {t} s, # power iteration is {power}, and estimated norm is {norm}\n'.format(t=time.perf_counter() - tstart, power=i, norm=self.norm))
        else:
            self.norm = norm


    def pPartitionCube(self, sz, p=2):
        """
        Helper function to generate a p-partition cube system
        """

        m, n = sz
        maxTheta = int(numpy.ceil(numpy.log(numpy.max(sz)) / numpy.log(p)))
        nMaxCube = int((p**(2*(maxTheta + 1)) - 1) / (p**2 - 1))

        st = numpy.zeros((nMaxCube, 2))
        ed = numpy.zeros((nMaxCube, 2))

        cnt = 0
        for theta in range(maxTheta+1):
            str, edr = self.partitionFixLen(m, p**theta)
            nr = len(str)
            stc, edc = self.partitionFixLen(n, p**theta)
            nc = len(stc)

            nCube = nc * nr

            st[cnt:cnt+nCube, 0] = numpy.matlib.repmat(str, nc, 1).flatten(order='F')[:nCube]
            st[cnt:cnt+nCube, 1] = numpy.matlib.repmat(stc, nr, 1).flatten()
            ed[cnt:cnt + nCube, 0] = numpy.matlib.repmat(edr, nc, 1).flatten(order='F')[:nCube]
            ed[cnt:cnt + nCube, 1] = numpy.matlib.repmat(edc, nr, 1).flatten()

            cnt += nCube

        return st[:cnt, :], ed[:cnt, :]


    def partitionFixLen(self, n, len):
        """
        Helper function to partition intervals of a fixed length
        """

        lnd = numpy.ceil(numpy.arange(0, len) / len * n) + 1
        rnd = numpy.append(lnd[1:] - 1, n)

        ind = lnd <= rnd
        return lnd[ind], rnd[ind]


    def scale2cube(self, scl, sz):
        """
        Helper function to generate a cube system by specified scales
        """

        m, n = sz
        scl = scl[scl <= numpy.min(sz)]
        nMaxCube = int(numpy.sum((m - scl + 1) * (n - scl + 1)))

        st = numpy.zeros((nMaxCube, 2))
        ed = numpy.zeros((nMaxCube, 2))

        cnt = 0
        for s in scl:
            str = numpy.arange(1, m-s+2)
            edr = numpy.arange(s, m+1)
            nr = m-s+1
            stc = numpy.arange(1, n-s+2)
            edc = numpy.arange(s, n+1)
            nc = n-s+1

            nCube = nc * nr

            st[cnt:cnt + nCube, 0] = numpy.matlib.repmat(str, nc, 1).flatten(order='F')[:nCube]
            st[cnt:cnt + nCube, 1] = numpy.matlib.repmat(stc, nr, 1).flatten()
            ed[cnt:cnt + nCube, 0] = numpy.matlib.repmat(edr, nc, 1).flatten(order='F')[:nCube]
            ed[cnt:cnt + nCube, 1] = numpy.matlib.repmat(edc, nr, 1).flatten()

            cnt = cnt + nCube

        return st, ed

    @jit()
    def mrcoefCube(self, img):
        """
        Compute multiresolution transformation
        """

        m = img.shape[1] + 1
        cs = numpy.zeros((len(img), m))
        cs[:, 1:] = numpy.cumsum(img, axis=0).transpose()
        cs = cs.flatten()

        st = self.st.flatten(order='F')
        ed = self.ed.flatten(order='F')
        nCube = len(self.st)
        coef = numpy.zeros(nCube)
        for i in range(nCube):
            coef[i] = 0
            for j in numpy.arange(int(st[i + nCube]), int(ed[i + nCube] + 1)):
                coef[i] += cs[int(ed[i] + 1 + j*m)] - cs[int(st[i] + j*m)]
            coef[i] /= numpy.sqrt((ed[i] - st[i] + 1) * (ed[i + nCube] - st[i + nCube] + 1))


        return coef

    @jit()
    def mrdualCube(self, sz, y):
        """
        Dual multiresolution transform
        """

        m, n = sz
        u = numpy.zeros(m*n)
        st = self.st.flatten(order='F')
        ed = self.ed.flatten(order='F')
        x = y.flatten()
        nCube = len(self.st)
        for k in range(nCube):
            sc = x[k] / numpy.sqrt((ed[k] - st[k] + 1)*(ed[k + nCube] - st[k + nCube] + 1))
            for i in numpy.arange(int(st[k]), int(ed[k] + 1)):
                for j in numpy.arange(int(st[k + nCube]), int(ed[k + nCube] + 1)):
                    u[i + j*m] += sc

        return u.reshape(sz).transpose()

    def forward(self, img, normed=True):
        if self.adjoint:
            res = self.mrdualCube(self.imSize, img)
        else:
            res = self.mrcoefCube(img)

        if normed:
            return res / self.norm
        else:
            return res

    def storeSim(self, data, filename):
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        cubeSim = {'mStat': data, 'st': self.st, 'ed': self.ed, 'imSize': self.imSize}
        pickle.dump(cubeSim, open(filename, 'wb'))

    def checkFile(self, filename):
        try:
            cubeSim = pickle.load(open(filename, 'rb'))
            if (cubeSim['st'] == self.st).all() and (cubeSim['ed'] == self.ed).all() and cubeSim['imSize'] == self.imSize:
                return cubeSim['mStat'], True
            else:
                return None, False
        except:
            return None, False