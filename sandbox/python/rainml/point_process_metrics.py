#! /usr/bin/env python
# Time-stamp: <2024-04-05 14:30:56 rytis>
import numpy as np
import functools

def filter_k(X,Y,tau):
    laplace_kernel = lambda X, tau: functools.reduce(lambda s, x: s + np.exp(-np.abs(x)/tau), X, 0)
    tmp = np.sum([laplace_kernel(X-y, tau) for y in Y])
    return 0.25*tmp/(len(X)*len(Y))/tau

def filter_d(X,Y,tau):
    tmp = filter_k(X,X,tau) + filter_k(Y,Y,tau) - 2.0*filter_k(X,Y,tau)
    return 0 if tmp<=0 else np.sqrt(tmp)

def filter_cov(X, Y, tau):
    x = np.sqrt(filter_k(X, X, tau))
    y = np.sqrt(filter_k(Y, Y, tau))
    return filter_k(X, Y, tau)/(x*y)


# given i, j,
# dX[i] = x[i+1]-x[i]
# dY[j] = y[j+1]-y[j]
# tau[ij] = 0.5*min(dX[i], dX[i-1], dY[j], dY[j-1])
# P[ij] = 1 if x[i]-y[j] < tau[ij] else (0 if x[i]-y[j]<0 else 0.5)


def cooc_k(X, Y):
    """ Co-occurrence metric
    """
    def c(U, V, mU, mV):
        s = 0
        for i in range(len(mU)):
            for j in range(len(mV)):
                t = 0.5*min(mU[i], mV[j])
                tmp = U[1+i] - V[1+j]
                s += (1.0 if ((tmp < t) and (tmp > 0)) else (0.5 if tmp == 0 else 0.0))
        return s
    dX = X[1:]-X[:-1]
    dY = Y[1:]-Y[:-1]
    mX = np.array([min(dX[i], dX[i-1]) for i in range(1, len(dX))])
    mY = np.array([min(dY[i], dY[i-1]) for i in range(1, len(dY))])

    return (c(X, Y, mX, mY) + c(Y, X, mY, mX))/np.sqrt(len(mX))/np.sqrt(len(mY))

def cooc_d(X, Y):
    return 1.0 - cooc_k(X, Y)



def my_stats(X, Y, times):
    """ X, Y: event lists
    """
    if type(X) is not np.ndarray:
        X = np.array(X)
    # if type(Y) is not np.ndarray:
    #     Y = np.array(Y)
    if not (np.any(X) and np.any(Y)):
        return [None]
    d = [np.min(np.abs(X-t)) for t in Y]
    d.sort()
    nrm = 1.0/len(d)            # len(d) = len(Y)
    return [nrm*np.searchsorted(d, t, side='left') for t in times]

def my_stats_multiple(Xs, Ys, times):
    """ Joint statistics for multiple pairs of event sets (useful to combine multiple recordings  into one)
    Xs, Ys: lists of event lists
    In this routine we join all of the data.
    """
    assert len(Xs) == len(Ys), "unequal list sizes"
    # if type(X) is not np.ndarray:
    #     X = np.array(X)
    # # if type(Y) is not np.ndarray:
    # #     Y = np.array(Y)
    data = []
    for X, Y in zip(Xs, Ys):
        if type(X) is not np.ndarray:
            X = np.array(X)
        data.append([np.min(np.abs(X-t)) for t in Y])
    # Joint statistics
    data = [item for row in data for item in row]
    data.sort()
    nrm = 1.0/len(data)
    return [nrm*np.searchsorted(data, t, side='left') for t in times]




# x = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010]
# i = 3
# j = 0

# j = 0
# for i in [0, 1, 2, 3, 4]:    
#     (t1, t2) = (v_events[i][j], r_events[i]) if len(r_events[i]) < len(v_events[i][j]) else (r_events[i], v_events[i][j])    
#     z = my_stats(t1, t2, x)
#     plt.plot(x, z, '.-')

# plt.xlim([0.001, 0.01])
# plt.ylim([0.5, 1.0])
# plt.show()


