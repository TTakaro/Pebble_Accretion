__author__ = 'michaelrosenthal'

import drag_functions_turb as fn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import ScalarFormatter
from scipy.optimize import fsolve
from scipy.integrate import odeint

def TvsR(a = [1e-9,1e-5,1e-2,1e-1],a_au=1,m_suns=1,m_earths=1,pnts=5e3,verb=0):

    #Function to plot the terminal velocity of an object vs. radius, using parameters from Ruth's Student's code

    #Flags
    verbose = verb #Print values of parameters at each step

    #Number of points to use
    # pnts = 5e3

    #Define parameters specific to this test case
    # a_au = 1
    # m_suns = 1
    # m_earths = 1e-1

    if verbose:
        print "a = %.3g AU" %a_au
        print "m_star = %.3g m_sun" %m_suns
        print "m_core = %.3g" %m_earths



    a_core = fn.au*a_au
    m_star = fn.m_sun*m_suns
    r_core = fn.r_earth*((m_earths)**(1./3.))
    m_core = fn.m_earth*m_earths


    rho_obj = 2.0



    enc_ang = 0 #Angle for energy dissipation between particle and eddy
    vel_ang = 0 #Angle of the particles velocity due to interation with turbulent eddies. 0 denotes completely in the phi direction
    #Alpha - parameterization of turbulence
    # a = .1

    #Array of alphas for plotting, alpha is the parameterization of turbulence
    # a = np.logspace(-6,-0.69314718055994529,num=6)
    # a = [1e-9,1e-5,1e-2,1e-1]
    #Return derived parameters
    # m_obj = fn.obj_mass(r_obj,rho_obj)

    # deriv_params = {"sig": fn.surf_dens(a_core),"t": fn.temp(a_core),"om": fn.omega(m_star,a_core),"cs": fn.sound_speed(t),\
    #                 "h": fn.scale_height(cs,om), "rho_g": fn.gas_density(sig,h), "mfp": fn.mean_free_path(fn.mu,rho_g,fn.cross),\
    #                 "vth": fn.therm_vel(cs), "v_core": fn.vkep(m_star,a_core), "vg": fn.v_gas(v_core,cs),\
    #                 "v_cg": np.absolute(v_core - vg)}

    sig = fn.surf_dens(a_core)
    t =  fn.temp(a_core)
    om = fn.omega(m_star,a_core)
    cs = fn.sound_speed(t)
    h = fn.scale_height(cs,om)
    rho_g = fn.gas_density(sig,h)
    mfp = fn.mean_free_path(fn.mu,rho_g,fn.cross)
    vth = fn.therm_vel(cs)
    v_core = fn.vkep(m_star,a_core)
    vg = fn.v_gas(v_core,cs)
    vkep = v_core
    v_cg = np.absolute(v_core - vg)
    vrel_i = v_cg
    v_g_phi = v_cg
    eta = fn.eta(cs,vkep)

    d = {}
    d["sig"] = fn.surf_dens(a_core)
    d["t"] =  fn.temp(a_core)
    d["om"] = fn.omega(m_star,a_core)
    d["cs"] = fn.sound_speed(t)
    d["h"] = fn.scale_height(cs,om)
    d["rho_g"] = fn.gas_density(sig,h)
    d["mfp"] = fn.mean_free_path(fn.mu,rho_g,fn.cross)
    d["vth"] = fn.therm_vel(cs)
    d["v_core"] = fn.vkep(m_star,a_core)
    d["vg"] = fn.v_gas(v_core,cs)
    d["v_cg"] = np.absolute(v_core - vg)
    d["vrel_i"] = v_cg

    #Turbulent Parameters
    d["kv"] = vth*mfp
    # kv_t = a*cs*h
    # per = 2*np.pi/om
    # re_f = fn.re_f(a,cs,h,vth,mfp)
    # v_turb = fn.turb_vel(kv_t,om)

    #Build array of Hill Radii
    h_r_arr = np.zeros(pnts)
    h_r = fn.hill_rad(m_core,a_core,m_star)
    h_r_arr.fill(fn.hill_rad(m_core,a_core,m_star))

    #Build array of GF radius
    GF_r_arr = np.zeros(pnts)
    GF_r = np.sqrt(h_r*r_core)
    GF_r_arr.fill(GF_r)

    #Build array of Bondi Radii
    b_rad_arr = np.zeros(pnts)
    b_r = fn.bondi_rad(m_core,cs)
    b_rad_arr.fill(fn.bondi_rad(m_core,cs))
    b_rad = fn.bondi_rad(m_core,cs)


    if verbose:
        print "sig = %s" %sig
        print "T = %s" %t
        print "om = %s" %om
        print "cs = %s" %cs
        print "H = %s" %h
        print "rho_g = %s" %rho_g
        print "mfp = %s" %mfp
        print "vth = %s" %vth
        print "v_cg = %s" %v_cg


    #Generate array of radii for plots
    r = np.logspace(-5,7,num=pnts)

    #Mass for each radius
    mass_objects = fn.obj_mass(r,rho_obj)

    #Intialize arrays for plots
    v_new = []
    v_pg_arr = []
    rey_nums = []
    forces_obj_cap = []
    forces_obj_main = []
    forces_core = []
    v_turb_arr = []

    #Master arrays for plotting parameterized by alpha
    v_pg_master = []
    forces_cap_master = []
    forces_core_master = []
    forces_main_master = []
    v_cg_master = []
    stl_master = []
    v_p_master = []
    v_part_turb_master = []
    v_p_turb_master = []
    v_enc_gas_master = []
    v_gas_tot_arr = []



    for i,alph in enumerate(a):
        #Intialize arrays for this value of alpha
        v_pg_arr = []
        forces_obj_cap = []
        forces_obj_main = []
        forces_core = []
        v_p_arr = []
        v_part_turb = []
        v_p_turb_arr = []
        stl_arr = []
        v_cg_tmp = np.zeros(pnts)


        #Calculate turbulent parameters dependent on alpha
        kv_t = alph*cs*h
        per = 2*np.pi/om
        re_f = fn.re_f(alph,cs,h,vth,mfp)
        v_turb = fn.turb_vel(alph,cs)
        v_gas_tot = np.sqrt(v_turb**2 + (vg-v_core)**2)

        v_turb_arr.append([v_turb])
        v_gas_tot_arr.append(v_gas_tot)
        # v_cg_phi = v_core - v_turb * np.cos(enc_ang)
        # v_cg_r = v_turb * np.sin(enc_ang)
        # v_cg = np.sqrt(v_cg_phi**2 + v_cg_r**2)
        v_cg = v_gas_tot

        if verbose:
            print "v_turb = %s" %v_turb
            print "v_cg = %s" %v_cg
            print "re_f = %s" %re_f
        v_cg_tmp.fill(v_cg)

        for i,rad in enumerate(r):
            d["r"] = rad
            # re = fn.rey(rad,vrel_i,vth,mfp)
            if rad > 9.*mfp/4:
                #Calculate terminal velocity and stopping time by iterating over force law. I've ignored angle here...
                delta = 1
                v_i = vrel_i
                while np.abs(delta) > .001:
                    re = fn.rey(rad,v_i,vth,mfp)
                    mobj = fn.obj_mass(rad,rho_obj)
                    dc = fn.drag_c(re)
                    fd = fn.stokes_ram(rad,dc,rho_g,v_i)
                    t_s = mobj*v_i/fd
                    v_L_r = fn.v_new_r(eta,vkep,t_s,om)
                    v_L_phi = fn.v_new_phi(eta,vkep,t_s,om)
                    v_L = np.sqrt(v_L_r**2 + v_L_phi**2)
                    t_eddy = (om*(1 + (v_L/v_turb)**2)**.5)**-1
                    stl = t_s/t_eddy
                    if stl > 10:
                        v_new_turb = v_turb*np.sqrt(1-(1 + stl)**-1)
                    else:
                        v_new_turb = fn.v_pg(v_turb,stl,re_f)
                    # v_new_phi = v_L_phi + v_new_turb*np.cos(vel_ang)
                    # v_new_r = v_L_r + v_new_turb*np.sin(vel_ang)
                    # v_new = np.sqrt(v_new_phi**2 + v_new_r**2)

                    v_new = np.sqrt(v_L**2 + v_new_turb**2)
                    delta = v_new - v_i
                    # print delta
                    v_i = v_new
            else:
                t_s = fn.ts_eps(rho_obj,rho_g,rad,vth)

            v_L_r = fn.v_new_r(eta,vkep,t_s,om)
            v_L_phi = fn.v_new_phi(eta,vkep,t_s,om)
            v_L = np.sqrt(v_L_r**2 + v_L_phi**2)
            t_eddy = (om*(1 + (v_L/v_turb)**2)**.5)**-1
            stl_turb = t_s/t_eddy
            stl_arr.append(t_s*om)
            # print stl
            if stl_turb > 10:
                v_new_turb = v_turb*np.sqrt(1-(1 + stl_turb)**-1)
            else:
                v_new_turb = fn.v_pg(v_turb,stl_turb,re_f)
            v_part_turb.append(v_new_turb)
            # v_new_phi = v_L_phi + v_new_turb*np.cos(vel_ang)
            # v_new_r = v_L_r + v_new_turb*np.sin(vel_ang)
            # v_pg = np.sqrt(v_new_phi**2 + v_new_r**2)

            v_pg = np.sqrt(v_L**2 + v_new_turb**2)
            v_pg_arr.append(v_pg)
            v_p_turb = np.sqrt(v_turb**2 - v_new_turb**2)
            v_obj_phi = np.abs(v_L_phi + vg)
            v_L_iner = np.sqrt((v_obj_phi - v_core)**2 + v_L_r**2)
            v_p = np.sqrt(v_L_iner**2 + v_p_turb**2)
            # print (np.abs(v_obj_phi - v_core) + v_p_turb*np.cos(vel_ang))**2 + ((v_L_r + v_p_turb*np.sin(vel_ang))**2)
            # v_p = np.sqrt((np.abs(v_obj_phi - v_core) + v_p_turb*np.cos(vel_ang))**2 + (v_L_r + v_p_turb*np.sin(vel_ang))**2)
            v_p_arr.append(v_p)
            v_p_turb_arr.append(v_p_turb)
            re = fn.rey(rad,v_pg,vth,mfp)

            # if i==3340:
            #     print "Re = %.3g" %re
            dc = fn.drag_c(re)
            forces_obj_cap.append(fn.drag_force(rad,v_pg,dc,rho_g,mfp,vth))
            forces_core.append(fn.drag_force(r_core,v_cg,fn.drag_c(fn.rey(r_core,v_cg,vth,mfp)),rho_g,mfp,vth))
            v_cap = v_gas_tot #Set relevant velocity for orbit maintain
            re_m = fn.rey(rad,v_cap,vth,mfp)
            dc_m = fn.drag_c(re_m)
            forces_obj_main.append(fn.drag_force(rad,v_cap,dc_m,rho_g,mfp,vth))

        v_part_turb_master.append(v_part_turb)
        v_p_turb_master.append(v_p_turb_arr)
        v_p_master.append(v_p_arr)
        v_pg_master.append(v_pg_arr)
        forces_cap_master.append(forces_obj_cap)
        forces_core_master.append(forces_core)
        forces_main_master.append(forces_obj_main)
        stl_master.append(stl_arr)
        v_cg_master.append(v_cg_tmp)


    #Calculate differential acceleration between the core and object for both orbit capture and orbit maintain
    # print forces_cap_master/mass_objects
    delta_a_cap = np.abs(forces_cap_master/mass_objects - np.true_divide(forces_core_master,m_core)) #Is this dividing the forces by mass_objs correctly?
    delta_a_main = np.abs(forces_main_master/mass_objects - np.true_divide(forces_core_master,m_core)) # Same as above

    # #Caclualte WISH radii
    r_ws_cap = fn.wish_radius(delta_a_cap,m_core,mass_objects)
    r_ws_main = fn.wish_radius(delta_a_main,m_core,mass_objects)

    #Cacluate stabilility radius as the minimum of r_h, r_cap, r_maintain
    min_1 = np.minimum(r_ws_cap,r_ws_main)
    r_stab_arr = np.minimum(min_1,h_r_arr)
    # print np.shape(r_stab_arr)

    #Calculate the crossing radius as the maximum of the bondi and stable radii
    r_acc_arr = np.zeros((len(a),int(pnts)))
    en_regime = np.zeros((len(a),int(pnts)))
    for i in range (0,len(a)):
        for j in range(0,int(pnts)):
            if r_stab_arr[i][j]>b_rad:
                r_acc_arr[i][j] = r_stab_arr[i][j]
                en_regime[i][j] = 1
            else:
                r_acc_arr[i][j] = b_rad
                en_regime[i][j] = 0
    # r_cross_arr = np.maximum(r_stab_arr,b_r)
    #
    # #Calculate the orbit velocity of particle at a distance r_cross from the core
    v_cross_arr = fn.vkep(m_core,r_acc_arr)
    #
    #Relative velocity of the particle at r_cross relative to the gas
    # v_rel_arr = v_cross_arr + v_core - vg

    #Caclculate relative terminal velocity b/t obj and core, assuming an encounter angle enc_ang
    # print v_cg_master

    #Calculate the particle's velocity relative to an intertial frame
    v_oc = v_p_master

    v_enc_arr =  np.zeros((len(a),int(pnts)))
    v_enc_arr_gas =  np.zeros((len(a),int(pnts)))
    #Determine velocity of object encountering core as the maxima of v_inf and the orbital velocity around the core
    # for i in range (0,len(a)):
    #     for j in range (0,int(pnts)):
    #         if v_oc[i][j] > v_cross_arr[i][j]:
    #             v_enc_arr[i][j] = v_oc[i][j]
    #             v_enc_arr_gas[i][j] = v_pg_master[i][j]
    #         else:
    #             v_enc_arr[i][j] = v_cross_arr[i][j]
    #             v_enc_arr_gas[i][j] = v_cross_arr[i][j] + v_gas_tot


    #KE of an object in orbit about the core
    shear_arr = r_acc_arr*om
    v_entry = np.maximum(v_oc,shear_arr)
    # v_entry = v_oc
    ke_arr = .5*mass_objects*np.power(v_entry,2)

    v_kick_arr = fn.G*m_core/r_acc_arr/v_entry

    for i in range (0,len(a)):
        for j in range (0,int(pnts)):
            if v_entry[i][j] > v_cross_arr[i][j]:
                v_enc_arr_gas[i][j] = max(v_pg_master[i][j],v_kick_arr[i][j])
            else:
                v_enc_arr_gas[i][j] = max(v_pg_master[i][j],v_cross_arr[i][j])

    # for i in range (0,len(a)):
    #     for j in range (0,int(pnts)):
    #         if v_pg_master[i][j] > v_cross_arr[i][j] + v_gas_tot_arr[i]:
    #             v_enc_arr_gas[i][j] = v_pg_master[i][j]
    #         else:
    #             v_enc_arr_gas[i][j] = v_cross_arr[i][j] + v_gas_tot_arr[i]

    v_enc_gas_master.append(v_enc_arr_gas)
    #Use these velocities to calculate the drag force on the object
    reys_enc = fn.rey(r,v_enc_arr_gas,vth,mfp)
    drag_enc = fn.drag_c(reys_enc)
    force_enc_arr = np.zeros((len(a),int(pnts)))
    for i in range (0,len(a)):
        for j in range(0,int(pnts)):
            force_enc_arr[i][j]= fn.drag_force(r[j],v_enc_arr_gas[i][j],drag_enc[i][j],rho_g,mfp,vth)
    work_enc_arr = np.multiply(force_enc_arr,2*np.cos(enc_ang)*r_acc_arr) #Work takes into account enc_ang b/t particle and eddy
    #

    #
    # #Calculate the height of dust in the disk
    # h_disk_arr = np.true_divide(v_p_turb_master,om)

    h_turb_arr = np.zeros((len(a),len(r)))
    for i,alpha in enumerate(a):
        for j,st in enumerate(stl_master[i]):
            h_turb_arr[i][j] = min(np.sqrt(alpha/st)*h,h)

    #Height from KH Instability
    kh_height = np.zeros(np.shape(h_turb_arr))
    kh_height.fill(fn.acc_height(h,a_core))

    h_disk_arr = np.maximum(h_turb_arr,kh_height)
    #
    # #Determine height of accrection rectangle as the minimum of the stable radius and the dust height
    height_arr = np.minimum(h_disk_arr,r_acc_arr)
    #
    # #Calculate the area of the accrection rectangle
    area_arr = 4*r_acc_arr*height_arr
    #
    # #Core growth time
    times = fn.growth_time(m_core,h_disk_arr,sig*.01,area_arr,v_entry)*fn.sec_to_years

    # for i in range(0,5):
    #     print "h = %s" %height_arr[i][2500]
    #     print "r = %s" %r_stab_arr[i][2500]
    #     print "v_oc = %s" %v_oc[i][2500]
    #     print "\n"

    focus_time = fn.focus_time(r_core,m_core,h_r,sig*.01,om)*fn.sec_to_years

    if verbose:
        print focus_time
    focus_arr = np.zeros(pnts)
    focus_arr.fill(focus_time)

    #Delete times where acc criterion are not met
    for i in range (0,len(a)):
        for j in range(0,int(pnts)):
            if en_regime[i][j] and ke_arr[i][j] > work_enc_arr[i][j]:
                times[i][j] = 0
            elif not(en_regime[i][j]) and ke_arr[i][j] < work_enc_arr[i][j]:
                times[i][j] = 0
            elif focus_time < times[i][j]:
                times[i][j] = focus_time

    return r,times

def TvsR_sng(alph = 1e-100,a_au=1,m_suns=1,m_earths=1,verbose=0,r=1e1,out='time',gas_dep=1.,sol_gas_ratio=0.01,sig_p_in=0\
             ,temp_in=0,focus_max=0,alpha_z=0,h_mod=1,shear_off=0,lam_vel=0,b_shear_off=0,extend_rh=1):
    if verbose:
        print "a = %.3g AU" %a_au
        print "m_star = %.3g m_sun" %m_suns
        print "m_core = %.3g" %m_earths



    a_core = fn.au*a_au
    m_star = fn.m_sun*m_suns
    r_core = fn.r_earth*((m_earths)**(1./3.))
    m_core = fn.m_earth*m_earths

    rho_obj = 2.0

    #Return derived parameters
    sig = fn.surf_dens(a_core)/gas_dep
    if sig_p_in:
        sig_p = sig_p_in
    else:
        sig_p = sig*sol_gas_ratio

    if temp_in:
        t = temp_in
    else:
        t =  fn.temp(a_core)

    om = fn.omega(m_star,a_core)
    cs = fn.sound_speed(t)
    h = fn.scale_height(cs,om)
    rho_g = fn.gas_density(sig,h)
    mfp = fn.mean_free_path(fn.mu,rho_g,fn.cross)
    vth = fn.therm_vel(cs)
    v_core = fn.vkep(m_star,a_core)
    vg = fn.v_gas(v_core,cs) #FULL gas velocity (not relative to Keplerian). There's no reason we need this. Just use eta*v_k in future.
    vkep = v_core
    v_cg = np.absolute(v_core - vg)
    vrel_i = v_cg
    v_g_phi = v_cg
    eta = fn.eta(cs,vkep)

    #Build array of Hill Radii -- not an array anymore
    h_r = fn.hill_rad(m_core,a_core,m_star)
    # h_r_arr.fill(fn.hill_rad(m_core,a_core,m_star))

    #Build array of Bondi Radii -- not an array anymore
    # b_rad_arr = np.zeros(pnts)
    b_r = fn.bondi_rad(m_core,cs)
    # b_rad_arr.fill(fn.bondi_rad(m_core,cs))
    b_rad = fn.bondi_rad(m_core,cs)

    #Don't let Bondi radius shrink below size of the core
    rho_core = 2.0
    rad_core = (3./4./np.pi*m_core/rho_core)**(1./3.)
    b_r = max(b_r,rad_core)

    if verbose:
        print "sig = %s" %sig
        print "T = %s" %t
        print "om = %s" %om
        print "cs = %s" %cs
        print "H = %s" %h
        print "rho_g = %s" %rho_g
        print "mfp = %s" %mfp
        print "vth = %s" %vth
        print "v_cg = %s" %v_cg

    #Plotting Arrays -- not used
    time_master = []
    time_off_master = []
    stl_master = []
    ke_master = []
    work_master = []


    # per = 2*np.pi/om

    time_arr = []
    time_off_arr = []
    stl_arr = []
    ke_arr = []
    work_arr = []

    rho_obj = 2.0 #Why is this defined twice?
    m_obj = (4./3.*np.pi*r**3)*rho_obj

    if verbose:
        print "\nr = %.7g\n" %r

    #Parameters for caclculating the turbulent gas velocity, see e.g. Ormel and Cuzzi (2007).
    re_f = fn.re_f(alph,cs,h,vth,mfp)
    v_turb = fn.turb_vel(alph,cs)
    v_cg = np.sqrt(v_turb**2 + (vg-v_core)**2)
    v_gas_tot = np.sqrt(v_turb**2 + (vg-v_core)**2)

    #Calculate stopping time of particle. Solve iteratively if we're in the fluid regime.
    if r > 9.*mfp/4:
                    #Calculate terminal velocity and stopping time by iterating over force law. I've ignored angle here...
                    delta = 1
                    v_i = vrel_i
                    while np.abs(delta) > .001:
                        re = fn.rey(r,v_i,vth,mfp)
                        mobj = fn.obj_mass(r,rho_obj)
                        dc = fn.drag_c(re)
                        fd = fn.stokes_ram(r,dc,rho_g,v_i)
                        t_s = mobj*v_i/fd
                        # st = fn.stl(t_s,per)
                        v_L_r = fn.v_new_r(eta,vkep,t_s,om)
                        v_L_phi = fn.v_new_phi(eta,vkep,t_s,om)
                        v_L = np.sqrt(v_L_r**2 + v_L_phi**2)
                        t_eddy = (om*(1 + (v_L/v_turb)**2)**.5)**-1
                        stl = t_s/t_eddy
                        if stl > 10:
                            v_new_turb = v_turb*np.sqrt(1-(1 + stl)**-1)
                        else:
                            v_new_turb = fn.v_pg(v_turb,stl,re_f)
                        v_new = np.sqrt(v_L**2 + v_new_turb**2)
                        delta = v_new - v_i #Should really be in terms of fractional change, i.e. delta = (v_new - v_i)/v_i. Shouldn't make a huge different though.
                        # print delta
                        v_i = v_new
    else:
        t_s = fn.ts_eps(rho_obj,rho_g,r,vth)
        # print "yes"

    # print "t_s = %.7g" %t_s
    # print v_L

    #Velocities calculated from stopping time.
    v_L_r = fn.v_new_r(eta,vkep,t_s,om)
    v_L_phi = fn.v_new_phi(eta,vkep,t_s,om)
    v_L_tot = np.sqrt(v_L_r**2 + v_L_phi**2)

    t_eddy = (om*(1 + (v_L_tot/v_turb)**2)**.5)**-1 #Eddy crossing effects
    stl = t_s/t_eddy

    tau_s = t_s*om
    stl_arr.append(tau_s)

    if stl > 10: #Don't use OC07 expressions for large stopping time, they don't converge to the correct value.
        v_pg_turb = v_turb*np.sqrt(1-(1 + stl)**-1)
    else:
        v_pg_turb = fn.v_pg(v_turb,stl,re_f)

    v_pg = np.sqrt(v_L_tot**2 + v_pg_turb**2)



    re = fn.rey(r,v_pg,vth,mfp)
    dc = fn.drag_c(re)
    forces_obj_cap = fn.drag_force(r,v_pg,dc,rho_g,mfp,vth)

    #Calculate R_WS. Using orbit capture radius is antiquated, and it is set to "infinity" after the calculation.

    # print "st_L = %.7g" %(t_s*om)
    # print "v_pg = %.7g" %v_pg
    # print "v_pg_turb = %.7g" %v_pg_turb
    # print "re = %.7g" %re
    # print "dc = %.7g" %dc
    # print "force_obj_cap = %.7g" %forces_obj_cap

    v_cap = v_gas_tot #Set relevant velocity for orbit maintain
    re_m = fn.rey(r,v_cap,vth,mfp)
    dc_m = fn.drag_c(re_m)
    forces_obj_main = fn.drag_force(r,v_cap,dc_m,rho_g,mfp,vth)

    if out == 'vary_sig':
        st_r_ws = m_obj*v_cap/forces_obj_main
    # print "force_obj_main = %.7g" %forces_obj_main

    forces_core = fn.drag_force(r_core,v_cg,fn.drag_c(fn.rey(r_core,v_cg,vth,mfp)),rho_g,mfp,vth)
    # print "force_core = %.7g" %forces_core

    #Calculate differential acceleration between the core and object for both orbit capture and orbit maintain
    delta_a_cap = np.abs(forces_obj_cap/m_obj - forces_core/m_core) #Is this dividing the forces by mass_objs correctly?
    delta_a_main = np.abs(forces_obj_main/m_obj - forces_core/m_core) # Same as above. It is, because these aren't arrays anymore.

    # print "del_a_cap = %.7g" %delta_a_cap
    # print "del_a_main = %.7g" %delta_a_main

    r_ws_cap = fn.wish_radius(delta_a_cap,m_core,m_obj)
    r_ws_main = fn.wish_radius(delta_a_main,m_core,m_obj)

    r_ws_cap = 1e100

    b_shear = 3**(1./3.)*h_r*tau_s**(1./3.)
    if b_shear_off:
        b_shear = 1e100

    # print "r_ws_cap = %.7g" %r_ws_cap
    # print "r_ws_main = %.7g" %r_ws_main

    min_1 = np.minimum(r_ws_cap,r_ws_main) #Always returns r_ws_main
    min_2 = np.minimum(min_1,b_shear)
    r_stab = np.minimum(min_2,h_r)


    # print "r_stab = %.7g" %r_stab

    r_acc = np.maximum(b_r,r_stab)

    v_cross = fn.vkep(m_core,r_acc) #Orbit velocity about core

    v_obj_phi = np.abs(v_L_phi + vg) #Velocities relative to Keplerian
    v_L_iner = np.sqrt((v_obj_phi - v_core)**2 + v_L_r**2)
    v_p_turb = np.sqrt(v_turb**2 - v_pg_turb**2)
    v_p = np.sqrt(v_p_turb**2 + v_L_iner**2)


    if lam_vel:
        v_p = v_L_iner
        v_pg = v_L_tot


    # if v_oc > v_cross:
    #     v_enc = v_oc
    #     v_enc_gas = v_pg
    # else:
    #     v_enc = v_cross
    #     v_enc_gas = v_cross + v_cg
    #
    # v_enc_gas = np.maximum(v_pg,v_cross + v_cg)

    v_oc = v_p #Duplicate variables...need to fix

    shear_vel = r_acc*om #Shear velocity

    if shear_off:
        shear_vel = h_r*om

    v_entry = np.maximum(v_oc,shear_vel) #Set v_infty = max(v_pk,v_shear)

    v_kick = fn.G*m_core/r_acc/v_entry #Calculate v_enc

    if v_entry > v_cross:
        v_enc_gas = max(v_kick,v_pg)
        v_grav = v_kick
    else:
        v_enc_gas = max(v_cross,v_pg)
        v_grav = v_cross



    # print "r_cross = %.7g" %r_cross
    # print "v_cross = %.7g" %v_cross
    # print "v_p_turb = %.7g" %v_p_turb
    # print "v_oc = %.7g" %v_oc
    # print "v_enc = %.7g" %v_enc
    # print "v_enc_gas = %.7g" %v_enc_gas
    # print "v_pg = %.7g" %v_pg
    # print "alpha = %.7g\n" %alph
    # print "v_entry = %.7g" %v_entry

    reys_enc = fn.rey(r,v_enc_gas,vth,mfp)
    drag_enc = fn.drag_c(reys_enc)
    force_enc = fn.drag_force(r,v_enc_gas,drag_enc,rho_g,mfp,vth)
    work_enc = force_enc*2*r_acc
    ke = .5*m_obj*v_entry**2

    if (r_acc == h_r and v_entry==shear_vel and extend_rh):
        # r_acc = h_r*min(work_enc/ke,1)
        prob = min(work_enc/ke,1.)
    else:
        prob = 1.

    # print "work_enc = %.7g" %work_enc
    # print "ke = %.7g" %ke
    #
    # h_turb = v_p_turb/om

    if not(alpha_z): #Calculate particle scale height
        alpha_z = alph
    h_turb = min(np.sqrt(alpha_z/tau_s)*h,h)

    kh_height = fn.acc_height(h,a_core)*h_mod*min(1.,1/np.sqrt(tau_s))
    h_disk = np.maximum(h_turb,kh_height)

    acc_height = np.minimum(h_disk,r_acc)
    # print "h_turb = %.7g" %h_turb
    # print "h = %.7g" %h
    #
    area = 4*r_acc*acc_height
    # print "area = %.7g" %area

    time = fn.growth_time(m_core,h_disk,sig_p,area,v_entry)*fn.sec_to_years/prob #Modified to account for collision probability

    # print "time = %.7g" %time
    focus_time = fn.focus_time(r_core,m_core,h_r,sig_p,om)*fn.sec_to_years #Needs to be updated

    # if out == 'rate':
    #     r_GF = np.sqrt(r_core*h_r)
    #     r_acc_GF = np.maximum(r_acc,r_GF)
    #     area_GF = 4*r_acc_GF*acc_height
    #     focus_time = fn.growth_time(m_core,h_disk,sig_p,area_GF,v_entry)*fn.sec_to_years
    #     time = min(time,focus_time)
    # else:
    #     time = min(time,focus_time)

    if focus_max:
        time = min(time,focus_time)
    if (((r_stab > b_r and work_enc < ke) or (r_stab < b_r and work_enc > ke)) and not(r_stab == h_r and\
                                                                                                 v_entry==shear_vel and extend_rh and work_enc<ke)):
        time = 0
        # print "Particle Cannot Accrete"

    if verbose:
        print "time = %.7g" %time

    if out=='time':
        return time,focus_time
    elif out=='rate':
        if time==0:
            return 0
        else:
            return m_core/time
    elif out=='st':
        return time,tau_s
    elif out=='vary_sig':
        return time,tau_s,st_r_ws
    elif out=='vel':
        return v_p,shear_vel
    elif out=='en_param':
        return v_p,shear_vel,v_grav,v_pg,r_acc
    elif out=='len':
        return time,min_1,h_r,b_r,b_shear
    elif out=='en':
        return work_enc,ke
    elif out=='height':
        return h_turb,kh_height
    elif out=='time_param':
        return h_disk,sig_p,r_acc,acc_height,v_entry
    elif out=='pcol':
        return 2.*r_acc*v_entry*prob
    else:
        print "Error"
        return

def ts_sto_ram(r,params=[]):
    vth,mfp,rho_g,v_pg_tot,rho_s,tmp = params

    re = 4*r*v_pg_tot/vth/mfp
    cd = fn.drag_c(re)
    fd = 1./2.*cd*np.pi*r**2*rho_g*v_pg_tot**2

    t_s = 4./3.*np.pi*r**3*rho_s*v_pg_tot/fd

    return t_s

def st_solver(st=1,alph = 1e-100,a_au=1,m_suns=1,m_earths=1,verbose=0,smooth=1,gas_dep=1.,sig_in = 0, temp_in = 0):

    if verbose:
        print "St = %.7g" %st
        print "a = %.3g AU" %a_au
        print "m_star = %.3g m_sun" %m_suns
        print "m_core = %.3g m_earths" %m_earths

    a_core = fn.au*a_au
    m_star = fn.m_sun*m_suns
    r_core = fn.r_earth*((m_earths)**(1./3.))
    m_core = fn.m_earth*m_earths

    rho_obj = 2.0

    if sig_in:
        sig = sig_in
    else:
        sig = fn.surf_dens(a_core)

    if temp_in:
        t = temp_in
    else:
        t =  fn.temp(a_core)

    #Return derived parameters
    # sig = fn.surf_dens(a_core)/gas_dep
    # t =  fn.temp(a_core)
    om = fn.omega(m_star,a_core)
    cs = fn.sound_speed(t)
    h = fn.scale_height(cs,om)
    rho_g = fn.gas_density(sig,h)
    mfp = fn.mean_free_path(fn.mu,rho_g,fn.cross)
    vth = fn.therm_vel(cs)
    v_core = fn.vkep(m_star,a_core)
    vg = fn.v_gas(v_core,cs)
    vkep = v_core
    v_cg = np.absolute(v_core - vg)
    vrel_i = v_cg
    v_g_phi = v_cg
    eta = fn.eta(cs,vkep)

    #Build array of Hill Radii
    h_r = fn.hill_rad(m_core,a_core,m_star)
    # h_r_arr.fill(fn.hill_rad(m_core,a_core,m_star))

    #Build array of Bondi Radii
    # b_rad_arr = np.zeros(pnts)
    b_r = fn.bondi_rad(m_core,cs)
    # b_rad_arr.fill(fn.bondi_rad(m_core,cs))
    b_rad = fn.bondi_rad(m_core,cs)

    if verbose:
        print "sig = %s" %sig
        print "T = %s" %t
        print "om = %s" %om
        print "cs = %s" %cs
        print "H = %s" %h
        print "rho_g = %s" %rho_g
        print "mfp = %s" %mfp
        print "vth = %s" %vth
        print "v_cg = %s\n" %v_cg

    r_eps = st*rho_g/rho_obj*vth/om

    if verbose:
        print "r_eps = %.5g" %r_eps

    if r_eps < 9.*mfp/4.:
        if verbose:
            print "Epstein Regime"
        return r_eps
    elif verbose:
        print "Not in Epstein"


    #Directly calculate v_rel from given St
    t_s = st/om
    if verbose:
        print "t_s = %.7g" %t_s

    # v_pg_L = np.sqrt(5)/2.*eta*vkep

    re_f = fn.re_f(alph,cs,h,vth,mfp)
    v_turb = np.sqrt(alph)*cs
    v_cg = np.sqrt(v_turb**2 + (vg-v_core)**2)
    v_gas_tot = np.sqrt(v_turb**2 + (vg-v_core)**2)

    # v_pg_L = eta*vkep*np.sqrt(1+4*st**2)/(1+st**2)
    v_pg_L = eta*vkep*st*np.sqrt(4.+st**2.)/(1+st**2.)
    t_eddy = om**(-1)/np.sqrt(1+v_pg_L**2/v_turb**2)

    stl = t_s/t_eddy
    v_pg_turb = fn.v_pg(v_turb,stl,re_f)

    v_pg_tot = np.sqrt(v_pg_turb**2 + v_pg_L**2)

    if verbose:
        print "Re_t = %.7g" %re_f
        print "t_eddy = %.7g" %(t_eddy*om)
        print "st_L = %.7g" %stl
        print "v_turb = %.7g" %v_turb
        print "v_gas_tot = %.7g" %v_gas_tot
        print "v_pg_L = %.7g" %v_pg_L
        print "v_pg_turb = %.7g" %v_pg_turb
        print "v_pg_tot = %.7g" %v_pg_tot

    # if smooth:
    def ts_zero(r,params):
        t_s = params[-1]
        return ts_sto_ram(r,params)-t_s

    params = [vth,mfp,rho_g,v_pg_tot,rho_obj,t_s]
    r_sto_ram = fsolve(ts_zero,9.*mfp/4.,args=params)[0]
    return r_sto_ram
    # else:
    #     re_sto = 6.*v_pg_tot*np.sqrt(st*rho_g/vth/mfp/rho_obj/om)
    #     re_RAM = 3.*st*v_pg_tot**2*rho_g/2./vth/mfp/rho_obj/om
    #
    #     if verbose:
    #         print "re_Sto = %.7g" %re_sto
    #         print "re_RAM = %.7g" %re_RAM

def st_rad(rad=1e0,alph = 1e-100,a_au=1,m_suns=1,m_earths=1,verbose=0,smooth=1,sig_in = 0,temp_in = 0,no_ram=0):

    if verbose:
        print "rad = %.7g" %rad
        print "a = %.3g AU" %a_au
        print "m_star = %.3g m_sun" %m_suns
        print "m_core = %.3g m_earths" %m_earths

    a_core = fn.au*a_au
    m_star = fn.m_sun*m_suns
    r_core = fn.r_earth*((m_earths)**(1./3.))
    m_core = fn.m_earth*m_earths

    rho_obj = 2.0
    m_obj = 4./3.*np.pi*rad**3*rho_obj

    #Return derived parameters
    if sig_in:
        sig = sig_in
    else:
        sig = fn.surf_dens(a_core)

    if temp_in:
        t = temp_in
    else:
        t =  fn.temp(a_core)

    om = fn.omega(m_star,a_core)
    cs = fn.sound_speed(t)
    h = fn.scale_height(cs,om)
    rho_g = fn.gas_density(sig,h)
    mfp = fn.mean_free_path(fn.mu,rho_g,fn.cross)
    vth = fn.therm_vel(cs)
    v_core = fn.vkep(m_star,a_core)
    vg = fn.v_gas(v_core,cs)
    vkep = v_core
    v_cg = np.absolute(v_core - vg)
    vrel_i = v_cg
    v_g_phi = v_cg
    eta = fn.eta(cs,vkep)

    #Build array of Hill Radii
    h_r = fn.hill_rad(m_core,a_core,m_star)
    # h_r_arr.fill(fn.hill_rad(m_core,a_core,m_star))

    #Build array of Bondi Radii
    # b_rad_arr = np.zeros(pnts)
    b_r = fn.bondi_rad(m_core,cs)
    # b_rad_arr.fill(fn.bondi_rad(m_core,cs))
    b_rad = fn.bondi_rad(m_core,cs)

    if verbose:
        print "sig = %s" %sig
        print "T = %s" %t
        print "om = %s" %om
        print "cs = %s" %cs
        print "H = %s" %h
        print "rho_g = %s" %rho_g
        print "mfp = %s" %mfp
        print "vth = %s" %vth
        print "v_cg = %s\n" %v_cg

    if rad < 9.*mfp/4.:
        t_s = fn.ts_eps(rho_obj,rho_g,rad,vth)
        st = t_s*om
        if verbose:
            print "Epstein Regime"
            print "t_s = %.5g" %t_s
            print "St = %.5g" %st
        return t_s*om
    else:
        if verbose:
            print "Fluid Regime"

        if no_ram:
            t_s = 4.*rho_obj*rad**2./(9.*rho_g*vth*mfp)
        else:

            re_f = fn.re_f(alph,cs,h,vth,mfp)
            v_turb = np.sqrt(alph)*cs
            v_gas_tot = np.sqrt(v_turb**2 + (vg-v_core)**2)

            delta = 1
            vrel = v_gas_tot
            i = 0
            while delta > 1e-6:
                re = 4*vrel*rad/vth/mfp
                dc = fn.drag_c(re)
                fd = fn.stokes_ram(rad,dc,rho_g,vrel)
                t_s = m_obj*vrel/fd
                st = t_s*om
                # v_pg_L = eta*vkep*np.sqrt(1+4*st**2)/(1+st**2)
                v_pg_L = eta*vkep*st*np.sqrt(4.+st**2.)/(1+st**2.)
                t_eddy = om**(-1)/np.sqrt(1+v_pg_L**2/v_turb**2)
                stl = t_s/t_eddy
                v_pg_turb = fn.v_pg(v_turb,stl,re_f)
                v_pg_tot = np.sqrt(v_pg_L**2 + v_pg_turb**2)
                delta = (v_pg_tot - vrel)/vrel
                vrel = v_pg_tot

        return t_s*om

    #         print delta
    #         i+=1
    #         if i>1000:
    #             break
    #
    #
    #
    # v_pg_tot = np.sqrt(v_pg_turb**2 + v_pg_L**2)
    #
    # if verbose:
    #     print "Re_t = %.7g" %re_f
    #     print "t_eddy = %.7g" %(t_eddy*om)
    #     print "st_L = %.7g" %stl
    #     print "v_turb = %.7g" %v_turb
    #     print "v_gas_tot = %.7g" %v_gas_tot
    #     print "v_pg_L = %.7g" %v_pg_L
    #     print "v_pg_turb = %.7g" %v_pg_turb
    #     print "v_pg_tot = %.7g" %v_pg_tot
    #
    # # if smooth:
    # def ts_zero(r,params):
    #     t_s = params[-1]
    #     return ts_sto_ram(r,params)-t_s
    #
    # params = [vth,mfp,rho_g,v_pg_tot,rho_obj,t_s]
    # r_sto_ram = fsolve(ts_zero,9.*mfp/4.,args=params)[0]
    # return r_sto_ram
    # else:
    #     re_sto = 6.*v_pg_tot*np.sqrt(st*rho_g/vth/mfp/rho_obj/om)
    #     re_RAM = 3.*st*v_pg_tot**2*rho_g/2./vth/mfp/rho_obj/om
    #
    #     if verbose:
    #         print "re_Sto = %.7g" %re_sto
    #         print "re_RAM = %.7g" %re_RAM


def min_mass(alph = 1e-100,a_au=1,m_suns=1,st=1e-1,verbose=0,focus=False,t_disp=2.5e6,gas_dep=1.,sol_gas_ratio=0.01,m_extend=0,m_guess=0):

    pnts = 150
    pnts_2 = 150
    m_range = [-5,1]

    print 1*m_guess

    if m_guess:
        m_arr = np.logspace(max(np.log10(0.5*m_guess),-5),np.log10(1.5*m_guess))
    else:
        m_arr = np.logspace(m_range[0],m_range[1],num=pnts)

    time_arr = np.zeros(len(m_arr))

    a_core_CGS = a_au*fn.au
    m_star_CGS = m_suns*fn.m_sun
    sig = fn.surf_dens(a_core_CGS)/gas_dep
    sig_p = sig*sol_gas_ratio
    om = fn.omega(m_star_CGS,a_core_CGS)

    if verbose:
        print "a = %.7g AU" %a_au
        print "M* = %.7g M_sun" %m_suns
        print "sig = %.7g" %sig
        print "om = %s" %om

    if m_extend:
        m_arr = np.logspace(-6,7,num=pnts)

    for i,M in enumerate(m_arr):
        # rad = st_solver(st=st,alph=1e-100,a_au=a_au,m_suns=m_suns,m_earths=M,verbose=0,gas_dep=1)
        if focus:
            rho_s = 2.0
            m_core_CGS = M*fn.m_earth
            # r_core_CGS = fn.r_earth*((M)**(1./3.))
            r_core_CGS = (3.0*m_core_CGS/4./np.pi/rho_s)**(1./3.)
            h_r = fn.hill_rad(m_core_CGS,a_core_CGS,m_star_CGS)
            print r_core_CGS,h_r,m_core_CGS,sig_p,om
            time_arr[i] = fn.focus_time(r_core_CGS,m_core_CGS,h_r,sig_p,om)*fn.sec_to_years

            if np.where(time_arr > 2.5e6)[0].size == pnts:
                mass_ind = 0
            else:
                mass_ind = np.where(time_arr > 2.5e6)[0][0]-1 if np.where(time_arr > 2.5e6)[0].size else -1
        else:
            time_arr[i] = rate_total(alph = alph,a_au=a_au,m_suns=m_suns,m_earths=M,verbose=0,gas_dep=gas_dep,\
                                     sol_gas_ratio=sol_gas_ratio,out='time',int_pnts=300)
            # if np.where(time_arr==0)[0].size != 0:
            #     ind = np.where(time_arr==0)[0][0]
            #     time_arr_tmp = time_arr[:ind]
            # mass_ind = np.where(time_arr_tmp > 2.5e6)[0][-1]+1 if np.where(time_arr_tmp > 2.5e6)[0].size else 0

            mass_ind = -1

            for i,t in enumerate(time_arr):
                if (t>0 and t<2.5e6):
                    mass_ind = i
                    break

    m_new = m_arr[mass_ind]
    m_arr_2 = np.logspace(max(np.log10(0.4*m_new),-5),np.log10(1.6*m_new),num=pnts_2)
    # m_arr = np.linspace(0.5*m_new,1.5*m_new,num=pnts_2)
    time_arr_2 = np.zeros(len(m_arr_2))

    for i,M in enumerate(m_arr_2):
        # rad = st_solver(st=st,alph=1e-100,a_au=a_au,m_suns=m_suns,m_earths=M,verbose=0,gas_dep=1)
        if focus:
            m_core_CGS = M*fn.m_earth
            r_core_CGS = fn.r_earth*((M)**(1./3.))
            h_r = fn.hill_rad(m_core_CGS,a_core_CGS,m_star_CGS)
            time_arr_2[i] = fn.focus_time(r_core_CGS,m_core_CGS,h_r,sig*sol_gas_ratio,om)*fn.sec_to_years

            if np.where(time_arr_2 > 2.5e6)[0].size == pnts_2:
                mass_ind = 0
            else:
                mass_ind = np.where(time_arr_2 > 2.5e6)[0][0]-1 if np.where(time_arr_2 > 2.5e6)[0].size else -1
        else:
            time_arr_2[i] = rate_total(alph = alph,a_au=a_au,m_suns=m_suns,m_earths=M,verbose=0,gas_dep=gas_dep,\
                                     sol_gas_ratio=sol_gas_ratio,out='time',int_pnts=100)
            # if np.where(time_arr==0)[0].size != 0:
            #     ind = np.where(time_arr==0)[0][0]
            #     time_arr_tmp = time_arr[:ind]
            # mass_ind = np.where(time_arr_tmp > 2.5e6)[0][-1]+1 if np.where(time_arr_tmp > 2.5e6)[0].size else 0

            mass_ind = -1

            for i,t in enumerate(time_arr):
                if (t>0 and t<2.5e6):
                    mass_ind = i
                    break


            # time_arr[i] = TvsR_sng(alph,a_au,m_suns,M,verbose=0,r=rad,gas_dep=gas_dep)


    # else:
    #     time_arr_tmp = time_arr
    # if np.where(time_arr_tmp > 2.5e6)[0].size == pnts:
    #     mass_ind = -1
    # else:


    # print mass_ind,time_arr_tmp

    # for i,ind in enumerate(np.where(time_arr>0)[0]):
    #     if (i != ind) or time_arr[i]>2.5e6:
    #         break

    # if time_arr[0] == 0:
    #     j = np.where(time_arr>0)[0][0]
    # else:
    #     for i,ind in enumerate(np.where(time_arr>0)[0]):
    #         if (i != ind):
    #                 j = i-1
    #                 break
    #         elif time_arr[i]>2.5e6:
    #             if ind != 0:
    #                 j = ind - 1
    #                 break
    #             else:
    #                 j=ind
    #                 break
    #         else:
    #             j=-1

    return time_arr,m_arr,m_arr_2[mass_ind]

def rate_total(alph = 1e-100,a_au=1,m_suns=1,m_earths=1,verbose=0,gas_dep=1.,sol_gas_ratio=0.01,st_min=1e-4,st_max=1e0,out='rate',int_pnts = 1000,\
               rs_max_in=0,rs_min_in=0):

    if verbose:
        print "a = %.3g AU" %a_au
        print "m_star = %.3g m_sun" %m_suns
        print "m_core = %.3g" %m_earths

    a_core = fn.au*a_au
    m_star = fn.m_sun*m_suns

    # if m_core_cgs:
    #     m_earths = m_earths/fn.m_earth
    #     # r_core = fn.r_earth*((m_earths/fn.m_earth)**(1./3.))
    # else:
    r_core = fn.r_earth*((m_earths)**(1./3.))
    m_core = fn.m_earth*m_earths

    rho_obj = fn.rho_obj

    #Gas density
    sig = fn.surf_dens(a_core)/gas_dep
    sig_p = sig*sol_gas_ratio

    #Power law exponent for size distribution
    q = 3.5

    #Minimum and maximum sizes
    if rs_min_in:
        rs_min = rs_min_in
    else:
        rs_min = st_solver(st=st_min,a_au=a_au,m_suns=m_suns,m_earths=m_earths,gas_dep=gas_dep)

    if rs_max_in:
        rs_max = rs_max_in
    else:
        rs_max = st_solver(st=st_max,a_au=a_au,m_suns=m_suns,m_earths=m_earths,gas_dep=gas_dep)

    r_arr = np.logspace(np.log10(1.01*rs_min),np.log10(rs_max),num=int_pnts)

    sum = 0
    for rad in r_arr:
        sum += rad**(-q+4) - rs_min**(-q+4)

    #Normalization constant for size distribution
    norm = sig_p/sum

    surf_dens_arr = norm*(r_arr**(-q+4)-rs_min**(-q+4))

    dM_dt = 0

    rate_arr = np.zeros(len(r_arr))

    for i,sig_p in enumerate(surf_dens_arr):
        dM_dt += TvsR_sng(alph = alph,a_au=a_au,m_suns=m_suns,m_earths=m_earths,r=r_arr[i],out='rate',gas_dep=gas_dep,\
                 sol_gas_ratio=sol_gas_ratio,sig_p_in=sig_p)
        rate_arr[i] = TvsR_sng(alph = alph,a_au=a_au,m_suns=m_suns,m_earths=m_earths,r=r_arr[i],out='rate',gas_dep=gas_dep,\
                 sol_gas_ratio=sol_gas_ratio,sig_p_in=sig_p)

    # if dM_dt:
    #     return m_core/dM_dt
    # else:
    #     return 0
    if out=='rate':
        return dM_dt/fn.m_earth
    elif out=='calc':
        t = m_core/dM_dt if dM_dt else 1e100
        return t
    else:
        t = m_core/dM_dt if dM_dt else 0
        return t

def rate_odeint(m,t,p):

    alph,a_au,m_suns,verbose = p
    rate = rate_total(alph = alph,a_au=a_au,m_suns=m_suns,m_earths=m)
    if verbose:
        print rate
    return rate

def MvsT(a_au=1e0,m0=1e-3,m_suns=1e0,alpha=1e-100,stop_time=2.5e6,pnts=1e4,verbose=0,out='times'):

    # times = np.linspace(0,stop_time,num=pnts)
    times = np.concatenate(([0], np.logspace(1,np.log10(stop_time),num=pnts)))

    print times
    p = [alpha,a_au,m_suns,verbose]

    wsol = odeint(rate_odeint,m0,times,args=(p,))

    if out=='times':
        return wsol,times
    else:
        return wsol




