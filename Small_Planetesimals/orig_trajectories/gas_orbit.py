__author__ = 'michaelrosenthal'

from scipy.integrate import odeint
import drag_functions_turb as fn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import orbits_diffeqs_linear as eqns
from matplotlib.patches import Circle

def int_orbit(a_core,m_star,m_core,r_s,init_vals,flags):

    #Enter input parameters
    # a_core = a_star
    # m_suns = m_star
    # m_earths = m_core

    #Convert input values to CGS
    # a_core = fn.au*a_au
    # m_star = fn.m_sun*m_suns
    r_core = fn.r_earth*((m_core/fn.m_earth)**(1./3.))
    # m_core = fn.m_earth*m_earths

    r_obj = r_s

    #Hill radius
    h_r = fn.hill_rad(m_core,a_core,m_star)

    #Unpack initial values
    x, vx, y, vy = init_vals

    #Impact parameter
    # b = h_r/pos

    #Initial velocity, particle comes in from the right with inital velocity in the x-direction
    # v_init = 265500.30233116221*1.05
    # v_init = np.sqrt(fn.G*m_core/b)/vel

    #GF cross section
    # v_esc = np.sqrt(2*fn.G*m_core/r_core)
    # GF_rad = r_core*np.sqrt(1 + v_esc**2/v_init**2)

    #Initial value
    # w0 = [b/np.sqrt(2),v_init,-b/np.sqrt(2),0]
    w0 = [x,vx,y,vy]
    p = [a_core, m_star, m_core, r_obj, flags]

    # #Orbital period
    # omega = np.sqrt(fn.G*m_core/())

    #Time array
    om = np.sqrt(fn.G*m_star/a_core**3.)
    stop_time = (1e0)*(2*np.pi/om)
    # print stop_time
    t = np.linspace(0,stop_time,num=1e5)

    #Calculate mass of object with given density (assumed spherical)
    rho_obj = 2.0
    m_obj = 4./3.*np.pi*r_obj**3*rho_obj


    wsol = odeint(eqns.force_field,w0,t,args=(p,))

    return wsol



