__author__ = 'michaelrosenthal'

import drag_functions_turb as fn
import numpy as np

def force_field(w, t, p):

    #Unpack positions and velocities, as well as parameters for the orbit
    x, vx, y, vy = w
    a_core, m_star, m_core, r_s, flags = p

    verbose, two_body, gas_only, gas_off = flags

    #Produce distance from core and angle
    r = np.sqrt(x**2 + y**2)
    th = np.arctan2(y,x)

    r_core = fn.r_earth*((m_core/fn.m_earth)**(1./3.))
    # print "r_core = %.3g" %r_core

    if r < r_core:
        f = [0,0,0,0]
        # print "Yes"
        return f

    #Disk parameters
    sig = fn.surf_dens(a_core)
    T =  fn.temp(a_core)
    om = fn.omega(m_star,a_core)
    cs = fn.sound_speed(T)
    h = fn.scale_height(cs,om)
    rho_g = fn.gas_density(sig,h)
    mfp = fn.mean_free_path(fn.mu,rho_g,fn.cross)
    vth = fn.therm_vel(cs)
    v_kep = fn.vkep(m_star,a_core)
    eta = cs**2./2./v_kep**2.
    # vg = fn.v_gas(v_core,cs)
    # v_cg = np.absolute(v_core - vg)

    #Calculate gas drag force

    #Begin by getting velocity of the particle relative to the gas, assume distance from star is small enough that
    # gas always moves in the -y_hat direction
    gas_vec = np.array([0,-eta*v_kep-3./2.*om*x,0]) #Get gas velocity from eta*v_k, include Keplerian Shear (still need to add turb)
    vel_vec = np.array([vx,vy,0])
    v_rel = vel_vec - gas_vec
    v_hat = v_rel/np.linalg.norm(v_rel)
    v_rel_mag = np.linalg.norm(v_rel)

    #Calculate magnitude of gas drag force
    re = fn.rey(r_s,v_rel_mag,vth,mfp)
    dc = fn.drag_c(re)

    #Calculate mass of object with given density (assumed spherical)
    rho_obj = 2.0
    m_obj = 4./3.*np.pi*r_s**3*rho_obj

    f_drag_mag = fn.drag_force(r_s,v_rel_mag,dc,rho_g,mfp,vth)/m_obj #Calculate force per unit mass due to gas drag
    f_drag = np.multiply(-f_drag_mag,v_hat)

    # print f_drag_vec

    f_core = -fn.G*m_core/r**2

    #Force from central star, including non-intertial terms (c.f. OK10, Eqn. 7)
    f_star = [2.*om*vy + 3.*om**2.*x,-2.*om*vx,0]

    #print(f_star)
    if two_body:
        # f_star = [0,0,0]
        # f_cent = [0,0,0]
        # f_cor = [0,0,0]
        f_core = 0
        # f_drag = [0,0,0]
    elif gas_only:
        f_core = 0
        f_star = [0,0,0]
        f_cent = [0,0,0]
        f_cor = [0,0,0]
    elif gas_off:
        f_drag = [0,0,0]
        #f_core = 0



    if verbose:
        print("f_star = %.5g" %(np.linalg.norm(f_star)))
        print("f_core = %.5g" %f_core)
        print("f_drag = %.5g" %(np.linalg.norm(f_drag)))
        print("v_rel = %.3g" %v_rel_mag)
        print("v_x = %.3g" %vx)
        print("v_y = %.3g" %vy)
        print("t = %.3g\n" %t)


    #Define and return vector of derivatives
    f = [vx,
         f_core*np.cos(th) + f_star[0] + f_drag[0],
         vy,
         f_core*np.sin(th) + f_star[1] + f_drag[1]
         ]

    return f

def orbit_params(w0,M):
    #Unpack intial values
    x, vx, y, vy = w0

    # print x,vx,y,vy

    #Convert to polar
    r = np.sqrt(x**2 + y**2)
    u = r**(-1)
    v = np.sqrt(vx**2 + vy**2)

    # print u

    th = np.arctan2(y,x)
    v_r = v*np.cos(th)
    th_dot = (x*vy - y*vx)/r**2

    # print th, v_r, th_dot

    #Get constants of integration, k is from Goldstein's notation
    E = 0.5*v**2 - fn.G*M/r
    L = r**2*th_dot
    k = fn.G*M

    # print E

    #Calculate eccentricity, semi-major axis
    ecc = np.sqrt(1 + 2*E*L**2/k**2)
    a = -k/2/E

    # print a,e
    print("Omega = %.5g" %(2*np.pi/ (np.sqrt(fn.G * M / a**3) ) ))

    #Calculate argument of periastron (I think that's what it is) - what Goldstein calls theta'
    # if 0:
    #     arg_peri = 0
    # else:
    arg_peri = th - np.arccos((a*(1 - ecc**2) - r)/ecc/r)
    # arg_peri = th + np.arccos( ((L**2*u/k) - 1)/np.sqrt(1 + 2*E*L**2/k**2))

    print(arg_peri)

    #Produce array of angles and then find r from the equation of an ellipse
    thetas = np.linspace(0,2*np.pi,num=1e5)
    rad = a*(1 - ecc**2)/(1 + ecc*np.cos(thetas - arg_peri))
    # rad = (k/L**2*(1 + np.sqrt(1 + 2*E*L**2/k**2)*np.cos(thetas-arg_peri)))**(-1)

    return thetas,rad