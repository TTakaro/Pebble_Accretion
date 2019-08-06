import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

#Define constants
G = 6.67259e-8
au = 1.49597871e13
k = 1.381e-16
mu = 3.93e-24
cross = 1e-15
sec_to_years = 1./31556926.
m_earth = 5.976e27
m_sun = 1.9891e33
r_earth = 6.378e8
r_sun = 6.957e10
Z = 0.01
rho_obj = 2.0


def obj_mass(r_obj,rho_obj):
    m_obj = (4./3.)*np.pi*(r_obj**3)*rho_obj
    return m_obj

def surf_dens(a): #Disk surface density
    sig = 5.0e2*(a/au)**-1
    return sig

def temp(a): #Disk temperature profile
    t = 200.*(a/au)**(-3./7.)
    return t

def omega(m_star,a): #Keplerian orbital frequency
    om = (G*m_star/a**3.)**.5
    return om

def sound_speed(T): #Isothermal sound speed
    cs=(k*T/mu)**.5
    return cs

def scale_height(cs,om): #This function takes in a,b and returns a/b. Just...why?
    h = cs/om
    return h

def gas_density(sig,h): #Volumetric density.
    rho_g = sig/(2.*h)
    return rho_g

def mean_free_path(mu,rho_g,cross): #Mean free path of gas molecules
    mfp = mu/(rho_g*cross)
    return mfp

def therm_vel(cs): #Average thermal velocity
    vth = ((8./np.pi)**.5)*cs
    return vth

def hill_rad(m_core, a_core, m_star):
    h_rad = a_core*(m_core/(3.*m_star))**(1./3.)
    return h_rad

def bondi_rad(m_core,cs):
    b_r = G*m_core/cs**2
    return b_r

def rey(robj,vrel,vth,mfp):  #Reynolds number of gas flow
    num = 4.*robj*vrel/(vth*mfp)
    return num

def drag_c(renum): #Drag coefficient from smoothed drag law
    dc = 24./renum*(1+.27*renum)**.43+.47*(1-np.exp(-.04*renum**.38))
    return dc

def stokes_ram(robj,dc,rhog,vrel): #Drag force in smoothed stokes-ram drag law (fluid regime).
    fd = .5*dc*np.pi*(robj**2)*rhog*(vrel**2)
    return fd

def ram(robj,rhog,vrel):
    fd = .5*np.pi*(robj**2)*rhog*(vrel**2)
    return fd

def stokes(robj,rhog,vrel,vth,mfp):
    fd = 6*np.pi*rhog*vth*mfp*robj*vrel
    return fd


def epstein_drag(robj,rhog,vrel,vth): #Drag force in Epstein regime (diffuse regime)
    fd = (4./3.)*np.pi*rhog*vth*vrel*(robj**2)
    return fd

# def drag_force(robj,vrel_i,re,rho_g,mfp,vth):
#     if robj > 9.*mfp/4:
#         if re > 1:
#             fd = ram(robj,rho_g,vrel_i)
#         else:
#             fd = stokes(robj,rho_g,vrel_i,vth,mfp)
#     else:
#         fd = epstein_drag(robj,rho_g,vrel_i,vth)
#     return fd

def drag_force(robj,vrel_i,dc,rho_g,mfp,vth):
    if robj > 9.*mfp/4.:
        fd = stokes_ram(robj,dc,rho_g,vrel_i)
    else:
        fd = epstein_drag(robj,rho_g,vrel_i,vth)
    return fd

def vkep(m_star,a): #Keplerian orbital velocity
    vk = (G*m_star/a)**.5
    return vk

def v_gas(v,cs): #Returns total gas orbital velocity, NOT relative to Keplerian. Don't really need this.
    vg = v - cs**2/(2*v)
    return vg

def eta(cs,vk):
    eta = .5*(cs**2./vk**2.)
    return eta


def v_new_r(eta,vk,ts,om): #v_r_lam from Nakagawa (1986)
    v_n_r = -2.*eta*vk*(ts*om/(1.+(ts*om)**2.))
    return v_n_r

def v_new_phi(eta,vk,ts,om): #v_phi_lam from Nakagawa (1986)
    v_n_ph = -eta*vk*(1./(1.+(ts*om)**2.)-1.)
    return v_n_ph


def wish_radius(a_ws, m_core, m_obj):
    r_ws = (G*(m_core + m_obj)/a_ws)**.5
    return r_ws

def acc_height(h,a_core):
    h_a = (h/a_core)**2*a_core
    return h_a

# Why was there a 1/2 in the above equation?

def growth_time(m_core,h,sig,A,v_oc):
    t = (m_core*2*h)/(sig*A*v_oc) # Factor of 2 in num because rho = sig/(2H)
    # print m_core,h,sig,A,v_oc
    return t

#New functions for turbulence
def kin_visc(a,cs,H):
    kv = a*cs*H
    return kv

def turb_vel(alph,cs):
    turb_v = np.sqrt(alph)*cs
    return turb_v

def orb_per(om):
    T = 2*np.pi/om
    return T

def stl(ts,orb_per):
    stl = ts/orb_per
    return stl

def re_f(a,cs,H,vth,mfp):
    re = a*cs*H/vth/mfp
    return re

def v_pg(vg,stl,re): #Velocity from turublence from OC07. Returns 0 for low reynolds numbers to avoid error.
    if re > 1:
        vpg = np.sqrt(vg**2*(stl**2*(np.sqrt(re)-1)/(stl+1)/(stl*np.sqrt(re)+1)))
        return vpg
    else:
        return 0

def vrel_sto_ram(rho_g,robj,mobj,v_i,v_turb,orb_per,re_f,vth,mfp):
    # i = 0
    delta = 1
    while np.abs(delta) > .001:
        re = rey(robj,v_i,vth,mfp)
        dc = drag_c(re)
        fd = stokes_ram(robj,dc,rho_g,v_i)
        ts = mobj*v_i/fd
        st = stl(ts,orb_per)
        v_new = v_pg(v_turb,st,re_f)
        delta = v_new - v_i
        # print delta
        v_i = v_new
        # i = i+1
        # print i
    return v_i, ts

def ts_stokes(rho_obj,rho_g,r_obj,kv): #Stopping time in Stokes regime
    ts = 2./9.*rho_obj/rho_g*r_obj**2/kv
    return ts

def ts_eps(rho_obj,rho_g,r_obj,vth): #Stopping time in Epstein regime
    ts = rho_obj/rho_g*r_obj/vth
    return ts

def vrel_RAM(rho_g,robj,mobj,v_i,v_turb,orb_per,re_f):
    # i = 0
    delta = 1
    while np.abs(delta) > .001:
        fd = ram(robj,rho_g,v_i)
        ts = mobj*v_i/fd
        st = stl(ts,orb_per)
        v_new = v_pg(v_turb,st,re_f)
        delta = v_new - v_i
        # print delta
        v_i = v_new
        # i = i+1
        # print i
    return v_i, ts

def max_boltz(x,a):
    f = np.sqrt(2./np.pi)*x**2*np.exp(-x**2/2./a**2)/a**3 #Parameter a = (k*T/m)^1/2
    return f

def T_eff(stl,re,vt,m):
    T = m*(np.sqrt(re)-1)*stl**2*vt**2/(3*k*(stl + 1)*(np.sqrt(re)*stl+1))
    return T

def dohnyani(norm,r):
    frac = norm*r**(-3.5)
    return frac

def v_new_r(eta,vk,ts,om):
    v_n_r = -2.*eta*vk*(ts*om/(1.+(ts*om)**2.))
    return v_n_r

def v_new_phi(eta,vk,ts,om):
    v_n_ph = -eta*vk*(1./(1.+(ts*om)**2.)-1.)
    return v_n_ph

def eta(cs,vk):
    eta = .5*(cs**2./vk**2.)
    return eta

def focus_time(r_core,m_core,h_r,sig,om,raf_fast=0):
    v_esc = np.sqrt(2*G*m_core/r_core)
    # print v_esc
    v_hill = np.sqrt(G*m_core/h_r)
    # print v_hill
    b = r_core*np.sqrt(1 + v_esc**2/v_hill**2)
    # print b
    focus_area = np.pi*b**2
    focusing_time = 2*m_core/focus_area/(sig)/om #Factor of two from n = sig/2/H/m
    if raf_fast:
        focusing_time = focusing_time*(r_core/h_r)**(0.5)
    # print focusing_time
    return focusing_time

def print_sci(r,arr,val, name = ""):
    f = interp1d(r,arr[0][:])
    print(name + "%.7g" %f(val))

def t_min(m_core=1,sig_p=20,h_r=r_earth,om=2*np.pi/3.15e7):
    return 2*m_core/(sig_p)/(2*h_r)**2/om*sec_to_years

