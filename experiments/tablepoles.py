import numpy as np
from apprentice import RationalApproximationSIP, RationalApproximationONB, PolynomialApproximation
from apprentice import tools, readData
import os

def getpqstr(fname):
    pq = ""
    if (fname=='f1'): pq = "p2_q4"
    if (fname=='f2'): pq = "p5_q2"
    if (fname=='f3'): pq = "p4_q3"
    if (fname=='f4'): pq = 'p3_q2'
    if (fname=='f5') : pq = "p2_q3"
    if (fname=='f7') : pq = 'p2_q7'
    if fname=='f8' : pq = "p3_q3"
    if fname=='f9' : pq = "p3_q7"
    if fname=='f10' : pq = 'p2_q4'
    if fname=='f12' : pq = 'p3_q3'
    if fname=='f13' : pq = "p2_q7"
    if fname=='f14' : pq = "p3_q6"
    if fname=='f15' : pq = "p2_q5"
    if fname=='f16' : pq = "p3_q7"
    if fname=='f17' : pq = 'p4_q6'
    if fname=='f18' : pq = "p2_q3"
    if fname=='f19' : pq = "p3_q3"
    if fname=='f21' : pq = "p5_q2"
    if fname=='f22' : pq = "p2_q4"
    return pq

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
                # if(optm==1 or optn==1):
                #     continue
                pq = "p%d_q%d"%(optm,optn)
                if(getpqstr(fname) != pq):
                    if(table_or_latex !='table'):
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
                pappfile = "%s/outpa/%s%s_%s_p%s_q%s_ts%s.json"%(folder,fname,noisestr,ts,optm,optn,ts)
                # print(rappfile)
                if not os.path.exists(rappsipfile):
                    print("rappsipfile %s not found"%(rappsipfile))
                    exit(1)

                if not os.path.exists(rappfile):
                    print("rappfile %s not found"%(rappfile))
                    exit(1)

                # if not os.path.exists(pappfile):
                #     print("pappfile %s not found"%(pappfile))
                #     exit(1)

                rappsip = RationalApproximationSIP(rappsipfile)
                Y_pred_rappsip = rappsip.predictOverArray(X_test)
                rapp = RationalApproximationONB(fname=rappfile)
                Y_pred_rapp = np.array([rapp(x) for x in X_test])
                # papp = PolynomialApproximation(fname=pappfile)
                # Y_pred_papp = np.array([papp(x) for x in X_test])
                # # results[fname][noise] = {"rapp":{},"rappsip":{}}
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

                    # l2allpapp = np.sum((Y_pred_papp-Y_test)**2)



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
                        # 'l2allpapp': l2allpapp
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
                sspecific = ""
                s+= '%'+" %s %s\n"%(fname,pq)
                s+= "\\multirow{4}{*}{\\ref{fn:%s}}&$|W_{r,t}|$"%(fname)
                sspecific+= '%'+" %s %s\n"%(fname,pq)
                sspecific += "\\multirow{4}{*}{\\ref{fn:%s}}&$|W_{r,t}|$"%(fname)
                for noise in noisearr:
                    for tval in thresholdvalarr:
                        tvalstr = str(int(tval))
                        s+="&%s"%(results[fname][noise][pq][tvalstr]["rapp"])
                        sspecific+="&%s"%(results[fname][noise][pq][tvalstr]["rapp"])
                    for tval in thresholdvalarr:
                        tvalstr = str(int(tval))
                        s+="&%s"%(results[fname][noise][pq][tvalstr]["rappsip"])
                        sspecific+="&%s"%(results[fname][noise][pq][tvalstr]["rappsip"])
                s+="\\\\\\cline{2-10}\n"
                s+="&$E_{r,t}$"
                sspecific+="\\\\\\cline{2-10}\n"
                sspecific+="&$E_{r,t}$"
                for noise in noisearr:
                    for tval in thresholdvalarr:
                        tvalstr = str(int(tval))
                        if(results[fname][noise][pq][tvalstr]["l2countrapp"] ==0):
                            s+="&0"
                            sspecific+="&0"
                        else:
                            s+="&%.1E"%(results[fname][noise][pq][tvalstr]["l2countrapp"])
                            sspecific+="&%.1E"%(results[fname][noise][pq][tvalstr]["l2countrapp"])
                    for tval in thresholdvalarr:
                        tvalstr = str(int(tval))
                        if(results[fname][noise][pq][tvalstr]["l2countrappsip"] ==0):
                            s+="&0"
                            sspecific+="&0"
                        else:
                            s+="&%.1E"%(results[fname][noise][pq][tvalstr]["l2countrappsip"])
                            sspecific+="&%.1E"%(results[fname][noise][pq][tvalstr]["l2countrappsip"])
                s+="\\\\\\cline{2-10}\n"
                s+="&$E'_{r,t}$"
                sspecific+="\\\\\\cline{2-10}\n"
                sspecific+="&$E'_{r,t}$"
                for noise in noisearr:
                    for tval in thresholdvalarr:
                        tvalstr = str(int(tval))
                        if(results[fname][noise][pq][tvalstr]["l2notcountrapp"] == 0):
                            s+="&0"
                            sspecific+="&0"
                        else:
                            s+="&%.1E"%(results[fname][noise][pq][tvalstr]["l2notcountrapp"])
                            sspecific+="&%.1E"%(results[fname][noise][pq][tvalstr]["l2notcountrapp"])
                    for tval in thresholdvalarr:
                        tvalstr = str(int(tval))
                        if(results[fname][noise][pq][tvalstr]["l2notcountrappsip"] == 0):
                            s+="&0"
                            sspecific+="&0"
                        else:
                            s+="&%.1E"%(results[fname][noise][pq][tvalstr]["l2notcountrappsip"])
                            sspecific+="&%.1E"%(results[fname][noise][pq][tvalstr]["l2notcountrappsip"])
                s+="\\\\\\cline{2-10}\n"
                s+="&$\\Delta_r$"
                sspecific+="\\\\\\cline{2-10}\n"
                sspecific+="&$\\Delta_r$"
                for noise in noisearr:
                    tvalstr = str(int(thresholdvalarr[0]))
                    if(results[fname][noise][pq][tvalstr]["l2allrapp"]==0):
                        s+="&\\multicolumn{2}{|c|}{0}"
                        sspecific+="&\\multicolumn{2}{|c|}{0}"
                    else:
                        s+="&\\multicolumn{2}{|c|}{%.1E}"%(results[fname][noise][pq][tvalstr]["l2allrapp"])
                        sspecific+="&\\multicolumn{2}{|c|}{%.1E}"%(results[fname][noise][pq][tvalstr]["l2allrapp"])
                    if(results[fname][noise][pq][tvalstr]["l2allrappsip"]==0):
                        s+="&\\multicolumn{2}{|c|}{0}"
                        sspecific+="&\\multicolumn{2}{|c|}{0}"
                    else:
                        s+="&\\multicolumn{2}{|c|}{%.1E}"%(results[fname][noise][pq][tvalstr]["l2allrappsip"])
                        sspecific+="&\\multicolumn{2}{|c|}{%.1E}"%(results[fname][noise][pq][tvalstr]["l2allrappsip"])
                s+="\\\\\\cline{2-10}\n"
                s+="\\hline\n\n"
                sspecific+="\\\\\\cline{2-10}\n"
                sspecific+="\\hline\n\n"
                if (fname=='f3' and pq == "p4_q3")\
                    or (fname=='f5' and pq == "p2_q3")\
                    or (fname=='f8' and pq == "p3_q3")\
                    or (fname=='f9' and pq == "p3_q7")\
                    or (fname=='f13' and pq == "p2_q7")\
                    or (fname=='f14' and pq == "p3_q6")\
                    or (fname=='f18' and pq == "p2_q3")\
                    or (fname=='f19' and pq == "p3_q3"):
                    print(sspecific)
                    exit(1)
    elif(table_or_latex =="latexall"):
        for fname in farr:
            for pq in results[fname][noisearr[0]].keys():
                sspecific = ""
                s+= '%'+" %s %s\n"%(fname,pq)
                s+= "\\multirow{3}{*}{\\ref{fn:%s}}&$r$~(Algorithm~\\ref{A:Polyak})"%(fname)
                sspecific+= '%'+" %s %s\n"%(fname,pq)
                sspecific+= "\\multirow{3}{*}{\\ref{fn:%s}}&$r$~(Algorithm~\\ref{A:Polyak})"%(fname)
                for noise in noisearr:
                    tvalstr = str(int(thresholdvalarr[0]))
                    if(results[fname][noise][pq][tvalstr]["l2allrappsip"]==0):
                        s+="&0"
                        sspecific+="&0"
                    else:
                        s+="&%.1E"%(results[fname][noise][pq][tvalstr]["l2allrappsip"])
                        sspecific+="&%.1E"%(results[fname][noise][pq][tvalstr]["l2allrappsip"])
                    for tval in thresholdvalarr:
                        tvalstr = str(int(tval))
                        if(results[fname][noise][pq][tvalstr]["l2countrappsip"]==0):
                            s+="&0"
                            sspecific+="&0"
                        else:
                            s+="&%.1E"%(results[fname][noise][pq][tvalstr]["l2countrappsip"])
                            sspecific+="&%.1E"%(results[fname][noise][pq][tvalstr]["l2countrappsip"])

                        if(results[fname][noise][pq][tvalstr]["l2notcountrappsip"] ==0):
                            s+="&0"
                            sspecific+="&0"
                        else:
                            s+="&%.1E"%(results[fname][noise][pq][tvalstr]["l2notcountrappsip"])
                            sspecific+="&%.1E"%(results[fname][noise][pq][tvalstr]["l2notcountrappsip"])

                s+="\\\\\\cline{2-12}\n"
                s+="&$r$ (Algorithm \\ref{ALG:MVVandQR})"
                sspecific+="\\\\\\cline{2-12}\n"
                sspecific+="&$r$ (Algorithm \\ref{ALG:MVVandQR})"
                for noise in noisearr:
                    tvalstr = str(int(thresholdvalarr[0]))
                    if(results[fname][noise][pq][tvalstr]["l2allrapp"] == 0):
                        s+="&0"
                        sspecific+="&0"
                    else:
                        s+="&%.1E"%(results[fname][noise][pq][tvalstr]["l2allrapp"])
                        sspecific+="&%.1E"%(results[fname][noise][pq][tvalstr]["l2allrapp"])

                    for tval in thresholdvalarr:
                        tvalstr = str(int(tval))
                        if(results[fname][noise][pq][tvalstr]["l2countrapp"] ==0):
                            s+="&0"
                            sspecific+="&0"
                        else:
                            s+="&%.1E"%(results[fname][noise][pq][tvalstr]["l2countrapp"])
                            sspecific+="&%.1E"%(results[fname][noise][pq][tvalstr]["l2countrapp"])
                        if(results[fname][noise][pq][tvalstr]["l2notcountrapp"] == 0):
                            s+="&0"
                            sspecific+="&0"
                        else:
                            s+="&%.1E"%(results[fname][noise][pq][tvalstr]["l2notcountrapp"])
                            sspecific+="&%.1E"%(results[fname][noise][pq][tvalstr]["l2notcountrapp"])
                s+="\\\\\\cline{2-12}\n"
                s+="&$r_{N=0}$ (Algorithm \\ref{A:Polyak})"
                sspecific+="\\\\\\cline{2-12}\n"
                sspecific+="&$r_{N=0}$ (Algorithm \\ref{A:Polyak})"
                for noise in noisearr:
                    tvalstr = str(int(thresholdvalarr[0]))
                    if(results[fname][noise][pq][tvalstr]["l2allpapp"] == 0):
                        s+="&0"
                        sspecific+="&0"
                    else:
                        s+="&%.1E"%(results[fname][noise][pq][tvalstr]["l2allpapp"])
                        sspecific+="&%.1E"%(results[fname][noise][pq][tvalstr]["l2allpapp"])
                    s+="&\\multicolumn{4}{c|}{}"
                    sspecific+="&\\multicolumn{4}{c|}{}"
                s+="\\\\\\cline{2-12}\n"
                s+="\\hline\n\n"
                sspecific+="\\\\\\cline{2-12}\n"
                sspecific+="\\hline\n\n"
                if (fname=='f3' and pq == "p4_q3")\
                    or (fname=='f5' and pq == "p2_q3")\
                    or (fname=='f8' and pq == "p3_q3")\
                    or (fname=='f9' and pq == "p3_q7")\
                    or (fname=='f13' and pq == "p2_q7")\
                    or (fname=='f14' and pq == "p3_q6")\
                    or (fname=='f18' and pq == "p2_q3")\
                    or (fname=='f19' and pq == "p3_q3"):
                    print(sspecific)

    print(s)


if __name__ == "__main__":


 # python tablepoles.py f1,f2,f3,f4,f5,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f19,f20,f22  0,10-1 10,100,1000 2x  table
 # for fno in {1..5} {7..10} {12..20} 22; do  name="f"$fno; nohup python tablepoles.py $name 0,10-1 10,100,1000 2x  table> ../../debug/"tablepoles_"$name".log" 2>&1 & done
 # for fno in 3 5 9 13 14 18 19; do  name="f"$fno; nohup python tablepoles.py $name 0,10-1 10,100,1000 2x  latex> ../../debug/"tablepoles_latex_"$name".log" 2>&1 & done
    import os, sys
    if len(sys.argv) != 6:
        print("Usage: {} function noise thresholds ts testfilelist bottom_or_all table_or_latex_or_latexall".format(sys.argv[0]))
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
