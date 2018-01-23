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
plot_dir        =   config["folders"]["plot"]
train_dir       =   config["folders"]["train"]
test_dir        =   config["folders"]["test"]
param_dir       =   config["folders"]["param"]
db_name         =   config["filename"]
coils           =   list(config["coil"])
n_input         =   list(map(int, list(config["n_input"])))
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

with(suppress(OSError)):
    os.makedirs(os.path.join(local, "log"))
    os.makedirs(os.path.join(local, param_dir))
    os.makedirs(os.path.join(local, "tex", plot_dir))
    for coil in coils:
        os.makedirs(os.path.join(local, train_dir, "train_" + coil))
        os.makedirs(os.path.join(local, train_dir, "test_" + coil))


rule all:
    input:
        results = expand(os.path.join(local, db_dir, "results_{coil}_n_{n}.csv"),
                        coil=coils, 
                        n=n_input, 
                        ),


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
    run:
        data = pd.read_csv(input.bin_db, 
                           sep=',',
                           engine='python',
                           header=None,
                           names=["day-way", "bin"]
                           )
        train = []
        n_input = int({n})
        for key, row in data.iterrows():
            binary = list(map(int, row["bin"].split(";")))
            size = int(len(binary)/n_input)
            binary = np.reshape(binary[:n_input*size], (size, n_input))
            lbl = 2*(np.sum(np.roll(binary, 1, axis=0), 1) > 0)-1
            for b, t in zip(lbl, binary):
                train.append(sum([row['day-way'].split(","), [str(b), ';'.join(map(str, t))]], []))
        pd.DataFrame(data = train).to_csv(output.train_db, sep=",", index=False, header=False, mode = "w")


rule select_data:
    input:
        train_db   = os.path.join(local, db_dir, db_name + "{coil}_n_{n}.train"),
    output:
        week_db    = os.path.join(local, db_dir, db_name + "{coil}_n_{n}.week"),
        weekend_db = os.path.join(local, db_dir, db_name + "{coil}_n_{n}.weekend"),
    benchmark:
        os.path.join("benchmark", "benchmark_select_db_{coil}.dat")
    threads:
        nth_select
    message:
        "Select data from db {wildcards.coil} with n = {wildcards.n}"
    run:
        print("MISS!!!!!!!!!!!!!!!!!!!!")

rule generate_fold:
    input:
        datafile = os.path.join(local, db_dir, db_name + "{coil}_n_{n}.week"),
    output:
        train    = expand(os.path.join(local, db_dir, train_dir, "train_{{coil}}"
                                        "{{coil}}_n_{{n}}_{{fold}}_{train}.csv"),
                                         train=list(map(str, range(K)))
                         ),
        test    = expand(os.path.join(local, db_dir, train_dir, "test_{{coil}}"
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
        db  = pd.read_csv(input.datafile, sep=",", names=["day", "way", "lbl", "train"], header=None, engine='python')
        lbl = list(db["lbl"])
        cv  = StratifiedKFold(n_splits = K, shuffle = True, random_state = int("{fold}")) # K-Fold cross validation
        for train_index, test_index in cv.split(np.zeros(len(lbl)), lbl):
            # TO FIX
            tmp         = db.iloc[:, train_index]
            tmp.columns = lbl[train_index]
            tmp.to_csv(output.train, header = False, index = False, sep = ",", mode = "w")

            tmp         = db.iloc[:, test_index]
            tmp.columns = lbl[test_index]
            tmp.to_csv(output.test, header = False, index = False, sep = ",", mode = "w")

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
                                        )
        testfile = expand(os.path.join(local, db_dir, test_dir, "test_{{coil}}"
                                        "{{coil}}_n_{{n}}_{fold}_{train}.csv"),
                                        fold = list(map(str, range(Nit))),
                                        train = list(map(str, range(K)))
                                        ),
    output:
        results = os.path.join(local, db_dir, "results_{coil}_n_{n}.csv"),
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
