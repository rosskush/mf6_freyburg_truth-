import os
import sys
import shutil
import numpy as np
import pandas as pd
import pyemu



def setup_mf6_freyberg():
    import numpy as np
    import pandas as pd
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    
    import flopy

    org_model_ws = os.path.join('freyberg_mf6')
    tmp_model_ws = "temp_pst_from"
    if os.path.exists(tmp_model_ws):
        shutil.rmtree(tmp_model_ws)
    os.mkdir(tmp_model_ws)

    # load in model
    sim = flopy.mf6.MFSimulation.load(sim_ws=org_model_ws)
    # change path of model
    sim.simulation_data.mfpath.set_sim_path(tmp_model_ws)
    m = sim.get_model("freyberg6")
    # write to external file format
    sim.set_all_data_external(check_data=False)
    sim.write_simulation()
    # run mf6 simulation
    pyemu.os_utils.run("mf6",cwd=tmp_model_ws)


    template_ws = "new_temp"
    # assign spatial refrence
    sr = m.modelgrid
    # create PstFrom instance 
    pf = pyemu.utils.PstFrom(original_d=tmp_model_ws, new_d=template_ws,
                 remove_existing=True,
                 longnames=True, spatial_reference=sr,
                 zero_based=False, start_datetime="1-1-2018")
    # set up variogram and geostructures
    v = pyemu.geostats.ExpVario(contribution=1.0,a=1000)
    gr_gs = pyemu.geostats.GeoStruct(variograms=v)
    rch_temporal_gs = pyemu.geostats.GeoStruct(variograms=pyemu.geostats.ExpVario(contribution=1.0,a=60))

    # add imports to forward run
    pf.extra_py_imports.append('flopy')

    # add paramters to PstFrom
    ib = m.dis.idomain[0].array
    tags = {"npf_k_":[0.1,10.],"sto_ss":[.1,10],"sto_sy":[.9,1.1],"rch_recharge":[.5,1.5]}
    dts = pd.to_datetime("1-1-2018") + pd.to_timedelta(np.cumsum(sim.tdis.perioddata.array["perlen"]),unit="d")
    print(dts)
    for tag, bnd in tags.items():
        lb, ub = bnd[0], bnd[1]
        arr_files = [f for f in os.listdir(tmp_model_ws) if tag in f and f.endswith(".txt")]
        if "rch" in tag:
            for arr_file in arr_files:
                kper = int(arr_file.split('.')[1].split('_')[-1]) - 1
                pf.add_parameters(filenames=arr_file,par_type="constant",par_name_base=arr_file.split('.')[1]+"_cn",
                                  pargp="rch_const",zone_array=ib,upper_bound=ub,lower_bound=lb,geostruct=rch_temporal_gs,
                                  datetime=dts[kper])
        else:
            for arr_file in arr_files:
                ult_lb = None
                ult_ub = None
                if "npf_k_" in arr_file:
                   ult_ub = 100.0
                   ult_lb = 0.01
                   pf.add_parameters(filenames=arr_file,par_type="grid",par_name_base=arr_file.split('.')[1]+"_gr",
                                  pargp=arr_file.split('.')[1]+"_gr",zone_array=ib,upper_bound=ub,lower_bound=lb,
                                  geostruct=gr_gs,ult_ubound=ult_ub,
                                  ult_lbound=ult_lb)
                # use a slightly lower ult bound here
                pf.add_parameters(filenames=arr_file, par_type="pilotpoints", par_name_base=arr_file.split('.')[1]+"_pp",
                                  pargp=arr_file.split('.')[1]+"_pp", zone_array=ib,upper_bound=ub,lower_bound=lb,
                                  ult_ubound=ult_ub,
                                  ult_lbound=ult_lb,geostruct=gr_gs)

                # use a slightly lower ult bound here
                pf.add_parameters(filenames=arr_file, par_type="constant",
                                  par_name_base=arr_file.split('.')[1] + "_cn",
                                  pargp=arr_file.split('.')[1] + "_cn", zone_array=ib,
                                  upper_bound=ub, lower_bound=lb,geostruct=gr_gs)

    
    # add model run command
    pf.mod_sys_cmds.append("mf6")
    print(pf.mult_files)
    print(pf.org_files)

    df = pd.read_csv(os.path.join(tmp_model_ws, "heads.csv"), index_col=0)
    pf.add_observations("heads.csv", insfile="heads.csv.ins", index_cols="time", use_cols=list(df.columns.values),
                        prefix="hds", rebuild_pst=True)

    # build pest
    pst = pf.build_pst('freyberg.pst')
    pst.control_data.noptmax = 0
    pst.pestpp_options['ies_parameter_ensemble'] = "prior.jcb"
    pst.write(os.path.join(template_ws,"freyberg.pst"))
    pyemu.os_utils.run("pestpp-ies freyberg.pst",cwd=template_ws)

    # set up prior ensemble
    num_reals = 100
    pe = pf.draw(num_reals, use_specsim=True)
    pe.enforce()
    pe.to_binary(os.path.join(template_ws, "prior.jcb"))
    

def run_prior_mc(template_ws,num_reals=100,num_workers=10):
    pst = pyemu.Pst(os.path.join(template_ws,"freyberg.pst"))
    pst.control_data.noptmax = -1
    pst.pestpp_options["ies_num_reals"] = num_reals
    pst.write(os.path.join(template_ws,"freyberg.pst"))
    pyemu.os_utils.start_workers(template_ws,"pestpp-ies","freyberg.pst",num_workers=num_workers,
                                 worker_root=".",master_dir="master_prior_mc")


def set_truth(template_ws):
	# read through prior ensamble and set one run as the "truth" 
    m_d = "master_prior_mc"
    pst = pyemu.Pst(os.path.join(m_d,"freyberg.pst"))
    oe = pd.read_csv(os.path.join(m_d,"freyberg.0.obs.csv"),index_col=0)
    pe = pd.read_csv(os.path.join(m_d, "freyberg.0.par.csv"), index_col=0)
    truth_real = "1"
    obs = pst.observation_data
    obs.loc[:,"obsval"] = oe.loc[truth_real,pst.obs_names].values
    obs.loc[:,"weight"] = 0.0
    obs.loc[:,"time"] = obs.time.apply(np.float)
    sites = ["trgw_0_9_1","trgw_2_33_7","trgw_0_22_15"]
    for site in sites:
        site_obs = obs.loc[obs.apply(lambda x: site in x.obsnme and x.time < 365,axis=1),"obsnme"]
        obs.loc[site_obs,"weight"] = 100.0
    print(pst.nnz_obs)
    pst.write(os.path.join(template_ws,"freyberg_run.pst"))

def run_ies(template_ws,num_reals=100,num_workers=10,noptmax=2):
    pst = pyemu.Pst(os.path.join(template_ws,"freyberg_run.pst"))
    pst.control_data.noptmax = noptmax
    pst.pestpp_options["ies_num_reals"] = num_reals
    pst.write(os.path.join(template_ws,"freyberg_run.pst"))
    pyemu.os_utils.start_workers(template_ws,"pestpp-ies","freyberg_run.pst",num_workers=num_workers,
                                 worker_root=".",master_dir="master_ies")


if __name__ == "__main__":
    setup_mf6_freyberg()
    #run_prior_mc("new_temp",num_reals=10,num_workers=5)
    #set_truth("new_temp")
    #run_ies("new_temp",num_reals=10,num_workers=5)



