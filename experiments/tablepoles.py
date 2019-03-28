import numpy as np
from apprentice import RationalApproximationSIP, RationalApproximationONB, PolynomialApproximation
from apprentice import tools, readData
import os

def tablepoles(farr,noisearr, tarr, ts, table_or_latex):
    print (farr)
    print (noisearr)
    print (thresholdarr)
    # print (testfilearr)
    # print (bottomallarr)

    thresholdvalarr = np.array([float(t) for t in tarr])
    thresholdvalarr = np.sort(thresholdvalarr)

    results = {}

    import glob
    import json
    import re
    if not os.path.exists("plots"):
        os.mkdir('plots')
    for num,fname in enumerate(farr):
        results[fname] = {}
        # testfile = testfilearr[num]
        # bottom_or_all = bottomallarr[num]
        testfile = "../benchmarkdata/"+fname+"_test.txt"
        # testfile = "../benchmarkdata/"+fname+".txt"
        print(testfile)
        bottom_or_all = all
        try:
            X, Y = readData(testfile)
        except:
            DATA = tools.readH5(testfile, [0])
            X, Y= DATA[0]

        if(bottom_or_all == "bottom"):
            testset = [i for i in range(trainingsize,len(X_test))]
            X_test = X[testset]
            Y_test = Y[testset]
        else:
            X_test = X
            Y_test = Y

        maxY_test = max(Y_test)
        for noise in noisearr:
            results[fname][noise] = {}
            noisestr = ""
            if(noise!="0"):
                noisestr = "_noisepct"+noise
            folder = "%s%s_%s"%(fname,noisestr,ts)
            filelist = np.array(glob.glob(folder+"/out/*.json"))
            filelist = np.sort(filelist)
            for file in filelist:
                if file:
                    with open(file, 'r') as fn:
                        datastore = json.load(fn)
                optm = datastore['m']
                optn = datastore['n']
                if(optm==1 or optn==1):
                    continue

            # optjsonfile = folder+"/plots/Joptdeg_"+fname+noisestr+"_jsdump_opt6.json"
            #
            # if not os.path.exists(optjsonfile):
            #     print("optjsonfile: " + optjsonfile+ " not found")
            #     exit(1)
            #
            # if optjsonfile:
            #     with open(optjsonfile, 'r') as fn:
            #         optjsondatastore = json.load(fn)

            # # optm = optjsondatastore['optdeg']['m']
            # # optn = optjsondatastore['optdeg']['n']
            # # index = -1
            # # while(optn ==0):
            # #     index+=1
            # #     if(index == 0 and "optdeg_p1" in optjsondatastore):
            # #         optm = optjsondatastore['optdeg_p1']['m']
            # #         optn = optjsondatastore['optdeg_p1']['n']
            # #     if(index ==1 and "optdeg_m1" in optjsondatastore):
            # #         optm = optjsondatastore['optdeg_m1']['m']
            # #         optn = optjsondatastore['optdeg_m1']['n']
            # #     if(index ==2):
            # #         print("setting to 0")
            # #         results[fname][noise] = {"rapp":{},"rappsip":{}}
            # #         for tval in thresholdvalarr:
            # #             tvalstr = str(int(tval))
            # #             results[fname][noise]["rapp"][tvalstr] = "0"
            # #             results[fname][noise]["rappsip"][tvalstr] = "0"
            # #         continue


                rappsipfile = "%s/out/%s%s_%s_p%d_q%d_ts%s.json"%(folder,fname,noisestr,ts,optm,optn,ts)
                rappfile = "%s/outra/%s%s_%s_p%d_q%d_ts%s.json"%(folder,fname,noisestr,ts,optm,optn,ts)
                # print(rappfile)
                if not os.path.exists(rappsipfile):
                    print("rappsipfile %s not found"%(rappsipfile))
                    exit(1)

                if not os.path.exists(rappfile):
                    print("rappfile %s not found"%(rappfile))
                    exit(1)

                rappsip = RationalApproximationSIP(rappsipfile)
                Y_pred_rappsip = rappsip.predictOverArray(X_test)
                rapp = RationalApproximationONB(fname=rappfile)
                Y_pred_rapp = np.array([rapp(x) for x in X_test])
                # results[fname][noise] = {"rapp":{},"rappsip":{}}
                # print(maxY_test)
                for tval in thresholdvalarr:
                    # print(fname, maxY_test)
                    # print(Y_pred_rappsip)

                    # rappsipcount = ((sum(abs(i)/abs(maxY_test) >= tval for i in Y_pred_rappsip))/float(len(Y_test))) *100
                    # rappcount = ((sum(abs(i)/abs(maxY_test) >= tval for i in Y_pred_rapp))/float(len(Y_test))) *100

                    l2allrappsip = np.sum((Y_pred_rappsip-Y_test)**2)
                    l2countrappsip = 0.
                    rappsipcount = 0
                    for num,yp in enumerate(Y_pred_rappsip):
                        if abs(yp)/abs(maxY_test) >= tval:
                            rappsipcount+=1
                            l2countrappsip += np.sum((yp-Y_test[num])**2)
                    l2notcountrappsip = l2allrappsip - l2countrappsip

                    l2countrappsip = np.sqrt(l2countrappsip)
                    l2notcountrappsip = np.sqrt(l2notcountrappsip)
                    l2allrappsip = np.sqrt(l2allrappsip)

                    l2allrapp = np.sum((Y_pred_rapp-Y_test)**2)
                    l2countrapp = 0.
                    rappcount = 0
                    for num,yp in enumerate(Y_pred_rapp):
                        if abs(yp)/abs(maxY_test) >= tval:
                            rappcount+=1
                            l2countrapp += np.sum((yp-Y_test[num])**2)
                    l2notcountrapp = l2allrapp - l2countrapp

                    l2countrapp = np.sqrt(l2countrapp)
                    l2notcountrapp = np.sqrt(l2notcountrapp)
                    l2allrapp = np.sqrt(l2allrapp)



                    # rappsipcount = sum(abs(i) >= tval for i in Y_pred_rappsip)
                    # rappcount = sum(abs(i) >= tval for i in Y_pred_rapp)

                    # print("----------------")
                    # print(maxY_test,tval)
                    # for i in Y_pred_rappsip:
                    #     if abs(i)/abs(maxY_test) >= tval:
                    #         print(abs(i))
                    # for i in Y_pred_rapp:
                    #     if abs(i)/abs(maxY_test) >= tval:
                    #         print(abs(i))

                    data = {
                        'm':optm,
                        'n':optn,
                        'rapp':str(int(rappcount)),
                        'rappsip':str(int(rappsipcount)),
                        'l2countrappsip' : l2countrappsip,
                        'l2notcountrappsip' : l2notcountrappsip,
                        'l2allrappsip' : l2allrappsip,
                        'l2countrapp' : l2countrapp,
                        'l2notcountrapp' : l2notcountrapp,
                        'l2allrapp' : l2allrapp
                    }
                    tvalstr = str(int(tval))
                    pq = "p%d_q%d"%(optm,optn)

                    if(pq in results[fname][noise]):
                        resultsdata = results[fname][noise][pq]
                        resultsdata[tvalstr] = data
                    else:
                        results[fname][noise][pq] = {tvalstr:data}


                    # results[fname][noise][tvalstr] = str(int(rappcount))
                    # results[fname][noise][tvalstr] = str(int(rappsipcount))

    # print(results)
    # print (json.dumps(results,indent=4, sort_keys=True))


    s = ""
    if(table_or_latex == "table"):
        s+= "\t\t\t"
        for noise in noisearr:
            s+= "%s\t\t\t\t\t\t\t"%(noise)
        s+="\n"
        for noise in noisearr:
            s += "\t\tRat Apprx\tRat Apprx SIP\t\t"
        s+="\n\n"
        for noise in noisearr:
            for tval in thresholdvalarr:
                s += "\t%s"%(int(tval))
            s+="\t"
            for tval in thresholdvalarr:
                s += "\t%s"%(int(tval))
            s+="\t"
        s += "\n"
        for fname in farr:
            s += "%s\n"%(fname)
            for pq in results[fname][noisearr[0]].keys():
                s += "%s"%(pq)
                for noise in noisearr:
                    for tval in thresholdvalarr:
                        tvalstr = str(int(tval))
                        sss = "-"
                        if(results[fname][noise][pq][tvalstr]["rapp"] != "0"):
                            sss= results[fname][noise][pq][tvalstr]["rapp"]
                        s += "\t%s"%(sss)
                    s+="\t"
                    for tval in thresholdvalarr:
                        tvalstr = str(int(tval))
                        sss = "-"
                        if(results[fname][noise][pq][tvalstr]["rappsip"] != "0"):
                            sss= results[fname][noise][pq][tvalstr]["rappsip"]
                        s += "\t%s"%(sss)
                    s+="\t"
                s+="\n"

    elif(table_or_latex =="latex"):
        for fname in farr:
            for pq in results[fname][noisearr[0]].keys():
                s+= "%s %s\n"%(fname,pq)
                s += "\\multirow{3}{*}{\\ref{fn:%s}}&$|W_t|$"%(fname)
                for noise in noisearr:
                    for tval in thresholdvalarr:
                        tvalstr = str(int(tval))
                        s+="&%s"%(results[fname][noise][pq][tvalstr]["rapp"])
                    for tval in thresholdvalarr:
                        tvalstr = str(int(tval))
                        s+="&%s"%(results[fname][noise][pq][tvalstr]["rappsip"])
                s+="\\\\\\cline{2-14}\n"
                s+="&$E_t$"
                for noise in noisearr:
                    for tval in thresholdvalarr:
                        tvalstr = str(int(tval))
                        s+="&%.1E"%(results[fname][noise][pq][tvalstr]["l2countrapp"])
                    for tval in thresholdvalarr:
                        tvalstr = str(int(tval))
                        s+="&%.1E"%(results[fname][noise][pq][tvalstr]["l2countrappsip"])
                s+="\\\\\\cline{2-14}\n"
                s+="&$E'_t$"
                for noise in noisearr:
                    for tval in thresholdvalarr:
                        tvalstr = str(int(tval))
                        s+="&%.1E"%(results[fname][noise][pq][tvalstr]["l2notcountrapp"])
                    for tval in thresholdvalarr:
                        tvalstr = str(int(tval))
                        s+="&%.1E"%(results[fname][noise][pq][tvalstr]["l2notcountrappsip"])
                s+="\\\\\\cline{2-14}\n"
                s+="\\hline\n\n"

    print(s)


if __name__ == "__main__":


 # python tablepoles.py f1,f2,f3,f4,f5,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f19,f20,f22  0,10-1 10,100,1000 2x  table
 # for fno in {1..5} {7..10} {12..20} 22; do  name="f"$fno; nohup python tablepoles.py $name 0,10-1 10,100,1000 2x  table> ../../debug/"tablepoles_"$name".log" 2>&1 & done
    import os, sys
    if len(sys.argv) != 6:
        print("Usage: {} function noise thresholds ts testfilelist bottom_or_all table_or_latex".format(sys.argv[0]))
        sys.exit(1)

    farr = sys.argv[1].split(',')
    if len(farr) == 0:
        print("please specify comma saperated functions")
        sys.exit(1)

    noisearr = sys.argv[2].split(',')
    if len(noisearr) == 0:
        print("please specify comma saperated noise levels")
        sys.exit(1)

    thresholdarr = sys.argv[3].split(',')
    if len(thresholdarr) == 0:
        print("please specify comma saperated threshold levels")
        sys.exit(1)

    # testfilearr = sys.argv[5].split(',')
    # if len(testfilearr) == 0:
    #     print("please specify comma saperated testfile paths")
    #     sys.exit(1)
    #
    # bottomallarr = sys.argv[6].split(',')
    # if len(bottomallarr) == 0:
    #     print("please specify comma saperated bottom or all options")
    #     sys.exit(1)


    # tablepoles(farr,noisearr, thresholdarr, testfilearr, bottomallarr,sys.argv[4],sys.argv[7])
    tablepoles(farr,noisearr, thresholdarr, sys.argv[4],sys.argv[5])
###########
