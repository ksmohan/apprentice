import numpy as np

def mkRationalTestData(dim, N, order=(2,1), origin=0, noise=0, xmin=-1, xmax=1):
    """
    Generate N random dim-dimensional point X in the open interval (xmin, xmax).
    Then produce a monomial structure for polynials of order (m,n).
    Then set all coefficients to 1 and evaluate the polynomials at the points X.
    Optionally, origin can be used to put the pole away from 0. Also
    optionally, random noise can be added by setting noise >0.
    """
    X = xmin + np.random.rand(N, dim)*(xmax-xmin)
    if dim==1:
        X=sorted(X)
    from apprentice import monomial
    sg = monomial.monomialStructure(dim, order[0])
    sh = monomial.monomialStructure(dim, order[1])

    LVg = [monomial.recurrence(x, sg)        for x in X]
    LVh = [monomial.recurrence(x-origin, sh) for x in X]

    ac = np.ones(len(sg))
    bc = np.ones(len(sh))
    G = np.array([np.dot(ac, lv)*(1+ np.random.normal(0,noise)) for lv in LVg])
    H = np.array([np.dot(bc, lv)*(1+ np.random.normal(0,noise)) for lv in LVh])
    return X, G/(H)

def f1(P):
    x, y = P
    return np.exp(x*y)/(x**2-1.44)/(y**2-1.44)

def f2(P):
    x, y = P
    return np.log(2.25 - x**2 -y**2)

def f3(P):
    x, y = P
    return np.tanh(5*(x-y))

def f4(P):
    x, y = P
    return np.exp(-1.*(x**2+y**2)/1000.)

def f5(P):
    x, y = P
    return abs(x-y)**3

def f6(P):
    x, y = P
    return (x**3 - x*y + y**3)/(x**2-y**2 +x*y**2)

def f7(P):
    x, y = P
    return (x + y**3)/(y**2x)

def f8(P):
    x, y = P
    return (x**2 + y**2 + x - y - 1)/((x-1.1)*(y-1.1))

def mkRes(X_train, X_test, order, fn):
    import pyrapp
    if fn==1:
        Y_train = [f1(x) for x in X_train]
        Y_test  = [f1(x) for x in X_test]
    elif fn==2:
        Y_train = [f2(x) for x in X_train]
        Y_test  = [f2(x) for x in X_test]
    elif fn==3:
        Y_train = [f3(x) for x in X_train]
        Y_test  = [f3(x) for x in X_test]
    elif fn==4:
        Y_train = [f4(x) for x in X_train]
        Y_test  = [f4(x) for x in X_test]
    elif fn==5:
        Y_train = [f5(x) for x in X_train]
        Y_test  = [f5(x) for x in X_test]
    elif fn==6:
        Y_train = [f6(x) for x in X_train]
        Y_test  = [f6(x) for x in X_test]
    elif fn==7:
        Y_train = [f7(x) for x in X_train]
        Y_test  = [f7(x) for x in X_test]
    elif fn==8:
        Y_train = [f8(x) for x in X_train]
        Y_test  = [f8(x) for x in X_test]
    else:
        raise Exception("function {} not implemented, exiting".format(fn))

    R = pyrapp.Rapp(X_train,Y_train, order=order)
    return [abs(R(x)-Y_test[num]) for num, x in enumerate(X_test)]

def plotRes(X_test, res, order, fn):
    m, n=order


    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rc('text', usetex = True)
    mpl.rc('font', family = 'serif', size=12)
    mpl.style.use("ggplot")
    cmapname   = 'viridis'
    plt.clf()

    plt.scatter(X_test[:,0], X_test[:,1], marker = '.', c = np.log10(res), cmap = cmapname, alpha = 0.8)
    plt.vlines(-1, ymin=-1, ymax=1, linestyle="dashed")
    plt.vlines( 1, ymin=-1, ymax=1, linestyle="dashed")
    plt.hlines(-1, xmin=-1, xmax=1, linestyle="dashed")
    plt.hlines( 1, xmin=-1, xmax=1, linestyle="dashed")
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.ylim((-1.5,1.5))
    plt.xlim((-1.5,1.5))
    plt.title("Absolute error for $f_{}$ with $m={},~n={}$".format(fn,m,n))
    b=plt.colorbar()
    b.set_label("$\log_{10}$ (Resdiual)")
    plt.savefig('f{}-residual_{}_{}.jpg'.format(fn,m,n))

if __name__=="__main__":
    import sys
    NP=100000
    m=int(sys.argv[1])
    n=int(sys.argv[2])
    np.random.seed(554)

    X_train = np.random.rand(1000, 2)*2-1
    X_test  = np.random.rand(NP, 2)*3-1.5

    for i in range(6,7):
        res = mkRes(X_train, X_test, (m,n), i)
        plotRes(X_test, res, (m,n), i)