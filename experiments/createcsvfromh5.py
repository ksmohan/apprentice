import apprentice as app
import optparse, os, sys, h5py
import numpy as np
from shutil import copyfile

if __name__ == "__main__":

    op = optparse.OptionParser(usage=__doc__)
    op.add_option("--log", dest="ISLOG", action='store_true', default=False,
                  help="input data is logarithmic --- affects how we filter (default: %default)")
    op.add_option("-o", dest="OUTPUT", default=None, help="Output folder (default: %default)")
    op.add_option("-w", dest="WEIGHTS", default=None, help="Obervable file (default: %default)")
    opts, args = op.parse_args()

    if os.path.exists(opts.OUTPUT):
        uk = '_all'
        if uk in opts.OUTPUT:
            print("found {}".format(opts.OUTPUT))
            if opts.WEIGHTS is not None:
                weights = list(set(app.tools.readObs(opts.WEIGHTS)))
                for i in range(len(weights)):
                    weights[i] = weights[i].replace("_",'')
                    weights[i] = weights[i].replace("/", '')
                outdir2 = opts.OUTPUT.split(uk)[0]
                os.makedirs(outdir2,exist_ok=True)
                for file in os.listdir(opts.OUTPUT):
                    if file.split('#')[0].replace('_','') in weights:
                        pathin = os.path.join(opts.OUTPUT,file)
                        pathout = os.path.join(outdir2,file)
                        copyfile(pathin, pathout)
            sys.exit(0)

    os.makedirs(opts.OUTPUT, exist_ok=True)



    binids = app.tools.readIndex(args[0])

    for num, b in enumerate(binids):
        DATA = app.tools.readH53(args[0], [num])
        _X = DATA[0][0]
        _Y = DATA[0][1]
        _E = DATA[0][2]
        USE = np.where((_Y > 0)) if opts.ISLOG else np.where((_E >= 0))
        X = _X[USE]
        Y = np.log10(_Y[USE]) if opts.ISLOG else _Y[USE]
        Y = np.atleast_2d(Y)
        outfile = os.path.join(opts.OUTPUT,binids[num].replace('/','_'))
        np.savetxt(outfile, np.hstack((X, Y.T)), delimiter=",")





