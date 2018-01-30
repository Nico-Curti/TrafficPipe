# snakemake -n # check workflow
# snakemake --dag | dot -Tpdf > traffic_pipe.pdf

from __future__ import division
import numpy as np
import pandas as pd

# pipeline of traffic
import os # local, sep, files, dirs
import sys # exit
import platform # check system
import datetime # date and time variables
from contextlib import suppress # makedir with no error 
from sklearn.model_selection import StratifiedKFold # RepeatedStratifiedKFold
#from tpot import TPOTClassifier # genetic pipeline

# plots of results
import seaborn as sns # pretty/easy plots with pandas
import matplotlib.pylab as plt # plots
import matplotlib.patches as mpatches # pretty labels

configfile: "config.yaml"

np.random.seed(123) # random number generator
sns.set_context("paper", font_scale=1.5)
sep             =   os.sep
local           =   os.path.abspath(".")
if platform.system() == "Windows":
    extension   =   ".exe"
elif platform.system() == "Linux" or platform.system()[:6] == "CYGWIN":
    extension   =   ""

db_dir          =   config["folders"]["data"]
scripts         =   config["folders"]["scripts"]
plot_dir        =   config["folders"]["plot"]
train_dir       =   config["folders"]["train"]
test_dir        =   config["folders"]["test"]
param_dir       =   config["folders"]["param"]
db_name         =   config["filename"]
coils           =   list(config["coil"])
n_input         =   list(config["n_input"])
Nit             =   int(config["Nit"])
K               =   int(config["K"])
fbp_train       =   os.path.join(local, "bin", config["softwares"]["train"] + extension)
fbp_test        =   os.path.join(local, "bin", config["softwares"]["test"] + extension)

max_iter        =   int(config["max_iter"])
n_population    =   int(config["n_population"])
elit_rate       =   float(config["elit_rate"])
mutation_rate   =   float(config["mutation_rate"])

nth_bin_db      =   int(config["NTH_binarization"])
nth_train       =   int(config["NTH_gentrain"])
nth_select      =   int(config["NTH_select"])
nth_gen_fold    =   int(config["NTH_gen_folds"])
nth_fbp_train   =   int(config["NTH_fbp_train"])
nth_fbp_test    =   int(config["NTH_fbp_test"])

lower_time      =    '05.59.59'
upper_time      =    '22.00.01'
Ncar            =    100
wd              = [[1,7,8], [14,15,21,22,28,29]]
weekend = sum([ ["0%d/05/2011"%i for i in wd[0]], ["%d/05/2011"%i for i in wd[1]] ], [])
weekdays = sum([ ["0%d/05/2011"%i for i in range(1, 10) if i not in wd[0]], ["%d/05/2011"%i for i in range(10, 32) if i not in wd[1]] ], [])



with(suppress(OSError)):
    os.makedirs(os.path.join(local, "log"))
    os.makedirs(os.path.join(local, param_dir))
    os.makedirs(os.path.join(local, "tex", plot_dir))
    os.makedirs(os.path.join(local, scripts))
    for coil in coils:
        os.makedirs(os.path.join(local, train_dir, "train_" + coil))
        os.makedirs(os.path.join(local, test_dir, "test_" + coil))


rule all:
    input:
        exe = fbp_train,
        datafile = expand(os.path.join(local, db_dir, train_dir, "train_{coil}",
                                        "{coil}_n_{n}_{fold}_{train}.csv"),
                                        fold = list(map(str, range(Nit))),
                                        train = list(map(str, range(K))),
                                        coil = coils,
                                        n = n_input,
                                        )
        #train    = expand(os.path.join(local, db_dir, train_dir, "train_{coil}",
        #                                "{coil}_n_{n}_{fold}_{train}.csv"),
        #                                 train=list(map(str, range(K))),
        #                                 coil=coils,
        #                                 n=n_input,
        #                                 fold=list(map(str, range(Nit)))
        #                ),

                    
        #test    = expand(os.path.join(local, db_dir, test_dir, "test_{coil}",
        #                                "{coil}_n_{n}_{fold}_{train}.csv"),
        #                                 train=list(map(str, range(K))),
        #                                 coil=coils,
        #                                 n=n_input,
        #                                 fold=list(map(str, range(Nit)))
        #                 ),
        #files    = expand([os.path.join(local, db_dir, db_name + "{coil}_n_{n}.weekdays"), os.path.join(local, db_dir, db_name + "{coil}_n_{n}.weekend")], coil=coils, n=n_input),
        #train_db = expand(os.path.join(local, db_dir, db_name + "{coil}_n_{n}.train"), coil = coils, n = n_input),
        #bin_db = expand(os.path.join(local, db_dir, db_name + "{coil}.binary"), coil=coils),
        #results = expand(os.path.join(local, db_dir, "results_{coil}_n_{n}.csv"),
        #                coil=coils, 
        #                n=n_input, 
        #                ),



rule binarize_db:
    input:
        db = os.path.join(local, db_dir, db_name + "{coil}.csv"),
    output:
        bin_db = os.path.join(local, db_dir, db_name + "{coil}.binary"),
    benchmark:
        os.path.join("benchmark", "benchmark_binarize_db_{coil}.dat")
    threads:
        nth_bin_db
    message:
        "Binarization of db for coil {wildcards.coil}"
    run:
        data = pd.read_csv(input.db, 
                   sep=' |\t',
                   engine='python',
                   skiprows=1,
                   names=["days", "time", "way", "direction", "lenght", "speed", "headway"],
                   dtype={"days": datetime.datetime, 
                          "time" : datetime.datetime,
                          "way" : np.int,
                          "direction" : np.int,
                          "lenght" : np.int,
                          "speed" : np.float,
                          "headway" : np.float},
                   usecols=[0, 1, 2, 5, 6])
        data = data.loc[(data['time'] > lower_time) & (data['time'] <= upper_time)]
        days = list(pd.unique(data.days))
        ways = list(pd.unique(data.way))
        
        ways = (data.groupby(data.way, sort=False))
        result = dict()

        for way, db in ways:
            days = (db.groupby(db.days, sort=False))  
            for day, tmp_db in days:  
                flux = np.asarray([ Ncar / (tmp_db.headway[i : i + Ncar].sum() / 36000)  for i in range(len(tmp_db.headway) - Ncar)])
                density = [flux[i] / tmp_db.speed[i : i + Ncar].mean() for i in range(len(tmp_db.speed) - Ncar)]
                density_bound = np.mean(density)
                binary = 2*np.logical_and(density[:-1] > density_bound, np.diff(flux) <= 0.) -1
                result[','.join([day, str(way)])] =  ';'.join(map(str, binary))
                
        pd.DataFrame.from_dict(result, orient="index").to_csv(output.bin_db, sep=",", index=True, header=False)

rule generate_train:
    input:
        bin_db = os.path.join(local, db_dir, db_name + "{coil}.binary"),
    output:
        train_db = os.path.join(local, db_dir, db_name + "{coil}_n_{n}.train"),

    benchmark:
        os.path.join("benchmark", "benchmark_train_db_{coil}.dat")
    threads:
        nth_train
    message:
        "Generation of training db for coil {wildcards.coil} with n = {wildcards.n}"
    params:
        n_input = "{n}", 
    run:
        data = pd.read_csv(input.bin_db, 
                           sep=',',
                           engine='python',
                           header=None,
                           names=["day-way", "bin"]
                           )
        train = []
        for key, row in data.iterrows():
            binary = list(map(int, row["bin"].split(";")))
            size = int(len(binary)/(int(params.n_input)))
            binary = np.reshape(binary[:int(params.n_input)*size], (size, int(params.n_input)))
            lbl = 2*(np.sum(np.roll(binary, 1, axis=0), 1) > 0)-1
            for b, t in zip(lbl, binary):
                train.append(sum([row['day-way'].split(","), [str(b), ';'.join(map(str, t))]], []))
        pd.DataFrame(data = train).to_csv(output.train_db, sep=",", index=False, header=False, mode = "w")


rule select_data:
    input:
        train_db   = os.path.join(local, db_dir, db_name + "{coil}_n_{n}.train"),
    output:
        files    = [os.path.join(local, db_dir, db_name + "{coil}_n_{n}.weekdays"), os.path.join(local, db_dir, db_name + "{coil}_n_{n}.weekend")],
    benchmark:
        os.path.join("benchmark", "benchmark_select_db_{coil}.dat")
    threads:
        nth_select
    message:
        "Select data from db {wildcards.coil} with n = {wildcards.n}"
    run:
        db = pd.read_csv(input.train_db, sep=',', engine='python', header=None, names=['day', 'way', 'label', 'train'])
        db[list(range(0,len(db['train'][0].split(';'))))] = db['train'].str.split(';', expand = True)
        db = db.drop(['train'], axis = 1)
        
        for types, outfile in zip([weekdays, weekend], output.files):
            tmp = db[db.day.isin(types)]
            tmp.index = [';'.join([str(d),str(w)]) for d,w in zip(tmp.day, tmp.way)]
            lbl = list(tmp.label)
            tmp = tmp.drop(["day", "way", "label"], axis=1)
            with open(outfile, "a") as out:
                out.write(",%s,%d\n"%(','.join(map(str, lbl[:-1])), lbl[-1]))
                tmp.to_csv(out, sep=',',header=False, index=True)


rule generate_fold:
    input:
        datafile = os.path.join(local, db_dir, db_name + "{coil}_n_{n}.weekdays"),
    output:
        train    = expand(os.path.join(local, db_dir, train_dir, "train_{{coil}}",
                                        "{{coil}}_n_{{n}}_{{fold}}_{train}.csv"),
                                         train=list(map(str, range(K)))
                         ),
        test    = expand(os.path.join(local, db_dir, test_dir, "test_{{coil}}",
                                        "{{coil}}_n_{{n}}_{{fold}}_{train}.csv"),
                                         train=list(map(str, range(K)))
                         ),
    benchmark:
        os.path.join("benchmark", "benchmark_generate_folds_{coil}_{n}_{fold}.dat")
    threads:
        nth_gen_fold
    message:
        "Generation of folds file for {wildcards.coil} : {wildcards.n} : {wildcards.fold}"
    run:
        db = pd.read_csv(input.datafile, sep=",", index_col=0)
        lbl = np.asarray(list(map(int, map(float, db.columns))))
        cv  = StratifiedKFold(n_splits = K, shuffle = True, random_state = int(wildcards.fold)) # K-Fold cross validation
        for train, test, (train_index, test_index) in zip(output.train, output.test, cv.split(np.zeros(len(lbl)), lbl)):
            tmp = db.iloc[train_index].dropna(how="all", axis=1)
            lbl_train = lbl[train_index]
            with open(train, "w") as f:
                f.write("%s\t%s\n"%('\t'.join(map(str, lbl_train[:-1])), str(lbl_train[-1])))
                tmp.to_csv(train, sep="\t", header=False, index=False)

            tmp = db.iloc[test_index].dropna(how="all", axis=1)
            lbl_test = lbl[test_index]
            with open(test, "w") as f:
                f.write("%s\t%s\n"%('\t'.join(map(str, lbl_test[:-1])), str(lbl_test[-1])))
                tmp.to_csv(test, sep="\t", header=False, index=False)


rule run_fbp:
    input: 
        exe = fbp_train,
        datafile = expand(os.path.join(local, db_dir, train_dir, "train_{{coil}}"
                                        "{{coil}}_n_{{n}}_{fold}_{train}.csv"),
                                        fold = list(map(str, range(Nit))),
                                        train = list(map(str, range(K)))
                                        )
    output:
        parameters = os.path.join(local, param_dir, "params_{coil}_n_{n}.csv"),
    benchmark:
        os.path.join("benchmark", "benchmark_fbp_train_{coil}_{n}_{fold}_{train}.dat")
    threads:
        nth_fbp_train
    message:
        "FBP with GA for {wildcards.coil} : {wildcards.n} : {wildcards.fold} : {wildcards.train}"
    run:
        for file in input.datafile:
            os.system(' '.join(["{input.exe}", 
                                "-f", file, 
                                "-o", "{output.parameters}", 
                                "-k", "10", 
                                "-r", "123", 
                                "-i", str(max_iter), 
                                "-n", str(n_population), 
                                "-e", str(elit_rate), 
                                "-m", str(mutation_rate)
                                ])
                        )

rule validation:
    input:
        exe = fbp_test,
        parameters = os.path.join(local, param_dir, "params_{coil}_n_{n}.csv"),
        trainfile = expand(os.path.join(local, db_dir, train_dir, "train_{{coil}}"
                                        "{{coil}}_n_{{n}}_{fold}_{train}.csv"),
                                        fold = list(map(str, range(Nit))),
                                        train = list(map(str, range(K)))
                                        ),
        testfile = expand(os.path.join(local, db_dir, test_dir, "test_{{coil}}"
                                        "{{coil}}_n_{{n}}_{fold}_{train}.csv"),
                                        fold = list(map(str, range(Nit))),
                                        train = list(map(str, range(K)))
                                        ),
    output:
        results = os.path.join(local, db_dir, "FBPresults_{coil}_n_{n}.csv"),
    benchmark:
        os.path.join("benchmark", "benchmark_fbp_test_{coil}_{n}.dat")
    threads:
        nth_fbp_test
    message:
        "FBP validation for {wildcards.coil} : {wildcards.n}"
    run:
        for train, test in zip(input.trainfile, input.testfile):
            os.system(' '.join(["{input.exe}", 
                                "-f", train, 
                                "-t", test,
                                "-o", "{output.results}", 
                                "-p", "{input.parameters}"
                                ])
                        )


#rule tpot:
#    input:
#        trainfile = expand(os.path.join(local, db_dir, train_dir, "train_{{coil}}"
#                                        "{{coil}}_n_{{n}}_{fold}_{train}.csv"),
#                                        fold = list(map(str, range(Nit))),
#                                        train = list(map(str, range(K)))
#                                        ),
#        testfile = expand(os.path.join(local, db_dir, test_dir, "test_{{coil}}"
#                                        "{{coil}}_n_{{n}}_{fold}_{train}.csv"),
#                                        fold = list(map(str, range(Nit))),
#                                        train = list(map(str, range(K)))
#                                        ),
#    output:
#        results = os.path.join(local, db_dir, "TPOTresults_{coil}_n_{n}.csv")
#    benchmark:
#        os.path.join("benchmark", "benchmark_tpot_{coil}_{n}.dat")
#    threads:
#        nth_tpot
#    message:
#        "TPOT pipeline for {wildcards.coil} : {wildcards.n}"
#    run:
#        tpot = TPOTClassifier(generations=max_iter, population_size=n_population, verbosity=0)
#        with open(output.results, "w") as f:
#            f.write("mcc\taccuracy")
#            for train, test in zip(input.trainfile, input.testfile):
#                train_data = pd.read_csv(train, sep=",", header=None)
#                test_data = pd.read_csv(test, sep=",", header=None)
#                train_lbl = list(train_data.columns)
#                test_lbl = list(test_data.columns)
#                tpot.fit(train_data, train_lbl)
#                lbl_predict = tpot.predict(test_data)
#                mcc = matthews_corrcoef(test_lbl, lbl_predict)
#                acc = accuracy_score(test_lbl, lbl_predict, normalize=True)
#                tpot.export(os.path.join(local, scripts, "tpot_" + test.split(os.sep)[-1].split(".")[0] + "_pipeline.py"))
#
#                f.write("%.3f\t%.3f\n"%(mcc, acc))


rule boxplots:
    input:
        fbp_res = expand(os.path.join(local, db_dir, "FBPresults_{coil}_n_{n}.csv"), coil=coils, n=n_input),
        tpot_res = expand(os.path.join(local, db_dir, "TPOTresults_{coil}_n_{n}.csv"), coil=coils, n=n_input),
    output:
        mcc_box = os.path.join(local, plot_dir, "boxplot_mcc.png"),
        acc_box = os.path.join(local, plot_dir, "boxplot_acc.png")
    benchmark:
        os.path.join("benchmark", "benchmark_boxplots.dat")
    message: 
        "Boxplots of results"
    run:
        fbp_db = pd.read_csv(input.fbp_res, sep="\t", header=None, names=["mcc", "accuracy"])
        tpot_db = pd.read_csv(input.tpot_res, sep="\t", header=1, names=["mcc", "accuracy"])
        palette = "hls"
        

        fig, ax = plt.subplots(figsize=(16,8))
        plt.subplots_adjust(left=0.15, right=0.9, top=0.8,  bottom=0.2)
        sns.boxplot( x = "cancer", 
                     y = "mcc", 
                     hue = "dtype",
                     data = db, 
                     palette = palette, 
                     ax = ax,
                     notch = True,
                     saturation = .9,
                     linewidth = 3)
        
        ax.hlines(.5, -0.5, len(params.cancer)*len(params.dtype), colors='r', linestyle='dashed', linewidth=4)
        ax.set_xlim(-0.5, len(params.cancer)-.5)
        ax.set_ylim(0, 1)
        ax.set_ylabel("AUC (Area Under the Curve)", fontsize=14)
        ax.set_xlabel("Cancer", fontsize=14)
        ax.set_xticklabels(params.cancer, rotation = 0, fontsize = 14)
        ax.set_yticklabels(np.arange(0, 1.1, .2), rotation = 0, fontsize = 14)
        for i,c in enumerate(params.cancer):
            ax.vlines(len(params.dtype)*.5*i-.5, 0, 1, color ='k', linestyle="dashed", linewidth=1.5)
        
        labels = [mpatches.Patch(color = color, label = t) for color, t in zip(sns.color_palette(palette, len(params.dtype)) , params.dtype)]
        plt.legend(handles=labels, fontsize=14, loc='lower left', prop={'weight' : 'semibold', 'size':14})
        plt.savefig(output.out)
