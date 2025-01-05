import numpy
import scipy


def stdIm(img, diff='OD1'):
    """
    Estimate the standard deviation of an image
        OD1: ordinary difference of order 1
        OD2: ordinary difference of order 2
        LTO: Liu, Tanaka and Okutomi (ICIP '12; IEEE TIP '13)
    """

    if diff == 'OD1':
        m, n = img.shape
        sigma = numpy.sqrt((numpy.sum(numpy.power(numpy.abs(numpy.diff(img, axis=0)), 2))/(m-1)/n
                           +numpy.sum(numpy.power(numpy.abs(numpy.diff(img, axis=1)), 2))/(n-1)/m)/4)
    elif diff == 'OD2':
        m, n = img.shape
        sigma = numpy.sqrt((sum(numpy.power(abs(numpy.diff(img, n=2, axis=0)), 2))/(m-2)/n
                           +sum(numpy.power(abs(numpy.diff(img, n=2, axis=1)), 2))/(n-2)/m)/12)
    elif diff == 'LTO':
        sigma = NoiseLevel(img)
    return sigma


def NoiseLevel(img, patchsize=7, decim=0, conf=1-1e-6, itr=3):
    """
    Estimate noise level in an image
    """
    kh = [-1/2, 0, 1/2]
    imgh = scipy.ndimage.convolve(img, kh, mode='nearest')
    imgh = imgh[:, 1:imgh.shape[1]-2, :]
    imgh = numpy.multiply(imgh, imgh)
    
    kv = numpy.transpose(kh)
    imgv = scipy.ndimage.convolve(img, kv, mode='nearest')
    imgv = imgv[1:imgh.shape[0]-2, :, :]
    imgv = numpy.multiply(imgv, imgv)
    
    Dh = my_convmtx2(kh, patchsize, patchsize)
    Dv = my_convmtx2(kv, patchsize, patchsize)
    DD = numpy.matmul(numpy.transpose(Dh), Dh) + numpy.matmul(numpy.transpose(Dv), Dv)
    r = numpy.linalg.matrix_rank(DD)
    
    Dtr = DD.trace()
    tau0 = scipy.stats.invgamma.cdf(conf, r/2, scale=2*Dtr/r)
    
    nlevel, th, num = []
    
    for cha in range(img.shape[2]):
        X = im2col(img[:, :, cha], (patchsize, patchsize))
        Xh = im2col(imgh[:, :, cha], (patchsize, patchsize-2))
        Xv = im2col(imgv[:, :, cha], (patchsize-2, patchsize))
        
        Xtr = sum(numpy.vstack((Xh, Xv)))
        
        if decim > 0:
            XtrX = numpy.vstack(Xtr, X)
            ind = numpy.argsort(XtrX, axis=0)
            XtrX = numpy.take_along_axis(Xtr, ind, axis=0)
            p = numpy.floor(XtrX.shape[1]/(decim+1))
            p = numpy.arange(p)*(decim+1)
            Xtr = XtrX[1, p]
            X = XtrX[1:XtrX.shape[0], p]
            
        tau = float('inf')
        if X.shape[1] < X.shape[0]:
            sig2 = 0
        else:
            cov = numpy.matmul(X, numpy.transpose(X)) / (X.shape[1]-1)
            d = numpy.linalg.eig(cov)
            sig2 = d[0]
            
        for i in numpy.arange(1, itr):
            tau = sig2*tau0
            p = [Xtr < tau]
            Xtr = Xtr[:, p]
            X = X[:, p]
            
            if X.shape[1] < X.shape[0]:
                break
            
            cov = numpy.matmul(X, numpy.transpose(X)) / (X.shape[1]-1)
            d = numpy.linalg.eig(cov)
            sig2 = d[0]
            
        numpy.append(nlevel, numpy.sqrt(sig2))
        numpy.append(th[cha], tau)
        numpy.append(num[cha], X.shape[1])
        
    return nlevel, th, num


def my_convmtx2(H, m, n):
    """
    2D convolution matrix
    """
    s = H.shape
    T = numpy.zeros(((m-s[0]+1)*(n-s[1]+1), m*n))
    k = 0
    for i in range(m-s[0]+1):
        for j in range(n-s[1]+1):
            for p in range(s[0]):
                T[k, (i+p)*n+j:(i+p)*n+j+s[1]-1] = H[p, :]
            k += 1
    return T


def im2col(img, size):
    """
    Rearrange image blocks into columns
    """
    m, n = img.shape
    last_col = n-size[1]+1
    last_row = m-size[0]+1
    
    start_idx = numpy.arange(size[0])[:, None]*n + numpy.arange(size[1])
    offset_idx = numpy.arange(last_row)[:, None]*n + numpy.arange(last_col)
    
    return numpy.take(img, start_idx.ravel()[:, None] + offset_idx.ravel())