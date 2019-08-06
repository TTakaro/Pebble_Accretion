from scipy.integrate import odeint
from time_TvsR import st_solver
from time_TvsR import st_rad
import drag_functions_turb as fn
import numpy as np
import scipy.optimize as opt

import warnings
warnings.filterwarnings('error')

#Flags
verbose = 0 #Print forces at each time step
two_body = 0 #Include only gravitational force from star, also plot analytic expression for orbit
gas_only = 0 #Consider only the force from gas drag
gas_off = 0 #Turn off gas drag
use_st = 0 #Pass a particle Stokes number instead of radius
flags = [verbose, two_body, gas_only, gas_off]

def int_orbit(a_core,m_star,m_core,r_s,init_vals,flags):
    # Enter input parameters
    r_obj = r_s

    # Hill radius
    h_r = fn.hill_rad(m_core,a_core,m_star)

    # Unpack initial values
    x, vx, y, vy, x_c, vx_c, y_c, vy_c, x_s, vx_s, y_s, vy_s = init_vals

    # Initial value
    w0 = [x, vx, y, vy, x_c, vx_c, y_c, vy_c, x_s, vx_s, y_s, vy_s]
    p = [a_core, m_star, m_core, r_obj, flags]

    # Calculate mass of object with given density (assumed spherical)
    rho_obj = 2.0
    m_obj = 4./3.*np.pi*r_obj**3*rho_obj

    wsol = odeint(force_field,w0,t,args=(p,))

    return wsol


def force_field(w, t, p):
    # Unpack positions and velocities, as well as parameters for the orbit
    x, vx, y, vy, x_c, vx_c, y_c, vy_c, x_s, vx_s, y_s, vy_s = w
    a_core, m_star, m_core, r_s, flags = p

    verbose, two_body, gas_only, gas_off = flags
    global Accreted

    # Produce distance from core and angle
    a = np.sqrt(np.power(x - x_s,2) + np.power(y - y_s,2))
    r = np.sqrt(np.power(x - x_c,2) + np.power(y - y_c,2))
    r_core_obj = np.array([x - x_c, y - y_c, 0])
    r_star_obj = np.array([x - x_s, y - y_s, 0])
    r_star_core = np.array([x_c - x_s, y_c - y_s])

    r_core = fn.r_earth*((m_core/fn.m_earth)**(1./3.))

    if r < r_core:
        f = [0,0,0,0,0,0,0,0]
        #print("Should have found it.")
        Accreted = True
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

    # Accretion check, to halt integration early:
    h_r = fn.hill_rad(m_core,a_core,m_star) # Hill radius
    v_gas = 0.5*cs**2./v_kep # gas velocity
    re = fn.rey(r_s,v_gas,vth,mfp) # reynold's number
    dc = fn.drag_c(re) # drag coefficient
    rho_obj = 2.0
    m_obj = 4./3.*np.pi*r_s**3*rho_obj # Calculate mass of object with given density (assumed spherical)
    f_drag_WS = fn.drag_force(r_s,v_gas,dc,rho_g,mfp,vth)/m_obj # Drag force on particle
    r_WS = fn.wish_radius(f_drag_WS,m_core,m_obj) # WISH radius
    r_stab = min(r_WS,h_r) # Stability radius
    #if (r < h_r/20):
        #print("Accreted!")
    #    Accreted = True
    #    f = [0,0,0,0,0,0,0,0]
    #    return f

    # Calculate gas drag force

    # Begin by getting velocity of the particle relative to the gas, assume distance from star is small enough that
    # gas always moves in the -y_hat direction
        #Begin by getting velocity of the particle relative to the gas
    r_star_obj = [x - x_s, y - y_s,0] #Position vector relative to star
    r_star_obj_hat = r_star_obj/np.linalg.norm(r_star_obj)
    gas_dir = np.cross(r_star_obj_hat, [0,0,-1]) #Direction of gas velocity
    v_kep = fn.vkep(m_star, np.linalg.norm(r_star_obj))
    try:
        gas_vec = np.multiply(np.sqrt(np.power(v_kep,2) - np.power(cs,2)), gas_dir) #Get gas velocity from eta*v_k (still need to add turb)
    except Warning:
        gas_vec = np.multiply(np.sqrt(np.power(fn.vkep(m_star, a_core),2) - np.power(cs,2)), gas_dir)
    vel_vec = [vx - vx_s, vy - vy_s,0]
    v_rel = vel_vec - gas_vec
    v_hat = v_rel/np.linalg.norm(v_rel)
    v_rel_mag = np.linalg.norm(v_rel)

    r_star_core = [x_c - x_s, y_c - y_s, 0]
    gas_dir_core = np.cross(r_star_core/np.linalg.norm(r_star_core), [0,0,-1])
    v_kep_core = fn.vkep(m_star,a)
    try:
        gas_vec_core = np.multiply(np.sqrt(np.power(v_kep_core,2) - np.power(cs,2)), gas_dir_core) #np.array([0,-eta*v_kep-3./2.*om*x_c,0])
    except Warning:
        gas_vec = np.multiply(np.sqrt(np.power(fn.vkep(m_star, a_core),2) - np.power(cs,2)), gas_dir_core)
    vel_vec_core = np.array([vx_c - vx_s, vy_c - vy_s,0])
    v_rel_core = vel_vec_core - gas_vec_core
    v_rel_core_mag = np.linalg.norm(v_rel_core)
    v_hat_core = v_rel_core/v_rel_core_mag

    #Calculate magnitude of gas drag force
    f_drag_mag = fn.drag_force(r_s,v_rel_mag,dc,rho_g,mfp,vth)/m_obj # Calculate force per unit mass due to gas drag
    f_drag = np.multiply(-f_drag_mag,v_hat)
    re_core = fn.rey(r_core, v_rel_core_mag, vth, mfp)
    dc_core = fn.drag_c(re_core)
    f_drag_core_mag = fn.drag_force(r_core,v_rel_core_mag,dc_core,rho_g,mfp,vth)/m_core # Calculate force per unit mass due to gas drag on core
    f_drag_core = np.multiply(-f_drag_core_mag,v_hat_core)
    re_star = 0#fn.rey(r_star, v_rel_star_mag, vth, mfp)
    dc_star = 0#fn.drag_c(re_star)
    f_drag_star_mag = 0 #fn.drag_force(r_star,v_rel_star_mag,dc_star,rho_g,mfp,vth)/m_star # Calculate force per unit mass due to gas drag on core
    f_drag_star = np.multiply(-f_drag_star_mag,v_hat_core)

    #print(vel_vec, f_drag)

    f_core = -fn.G*m_core/r**3 # Acceleration of particle from core
    f_star = -fn.G*m_star/a**3 # Acceleration of particle from star
    f_obj = -fn.G*m_obj/np.power(np.power(x_c - x,2) + np.power(y_c - y,2), 3./2.) # Acceleration of core from particle
    f_star_core = -fn.G*m_star/np.power(np.power(x_c - x_s,2) + np.power(y_c - y_s,2), 3./2.) # Acceleration of core from star
    f_core_star = -fn.G*m_core/np.power(np.power(x_s - x_c,2) + np.power(y_s - y_c,2), 3./2.) # Acceleration of star from core
    f_obj_star = -fn.G*m_obj/np.power(np.power(x_s - x,2) + np.power(y_s - y,2), 3./2.) # Acceleration of star from particle

    if two_body:
        f_core = [0,0,0]
        f_obj = [0,0,0]
    elif gas_only:
        f_core = [0,0,0]
        f_obj = [0,0,0]
        f_star = [0,0,0]
        f_star_core = [0,0,0]
        f_core_star = [0,0,0]
        f_obj_star = [0,0,0]
    elif gas_off:
        f_drag = [0,0,0]
        f_drag_core = [0,0,0]
        f_drag_star = [0,0,0]

    if verbose:
        print("f_star = %.5g")%(np.linalg.norm(f_star))
        print("f_core = %.5g")%f_core
        print("f_drag = %.5g")%(np.linalg.norm(f_drag))
        print("v_rel = %.3g")%v_rel_mag
        print("v_x = %.3g")%vx
        print("v_y = %.3g")%vy
        print("t = %.3g\n")%t

    #Define and return vector of derivatives
    f = [vx, (x - x_c)*f_core + (x - x_s)*f_star + f_drag[0],
         vy, (y - y_c)*f_core + (y - y_s)*f_star + f_drag[1],
         vx_c, (x_c - x)*f_obj + (x_c - x_s)*f_star_core + f_drag_core[0],
         vy_c, (y_c - y)*f_obj + (y_c - y_s)*f_star_core + f_drag_core[1],
         vx_s, (x_s - x_c)*f_core_star + (x_s - x)*f_obj_star + f_drag_star[0],
         vy_s, (y_s - y_c)*f_core_star + (y_s - y)*f_obj_star + f_drag_star[1]]

    return f


def test_impact_params(size, a_AU=1., m_st=1., m_c=1e-3):
    #Input parameters to integrator
    a_core_AU = a_AU #In AU
    m_star_suns = m_st #In Solar masses
    m_core_earths = m_c #In Earth masses

    #Convert input values to CGS
    a_core = fn.au*a_core_AU
    m_star = fn.m_sun*m_star_suns
    r_core = fn.r_earth*((m_core_earths)**(1./3.))
    m_core = fn.m_earth*m_core_earths

    if use_st:
        r_s = st_solver(st=size,a_au=a_core_AU)
        st = size
        st_core = st_rad(rad=r_core, a_au=a_core_AU)
    else:
        r_s = size
        st = st_rad(rad=size,a_au=a_core_AU)
        st_core = st_rad(rad=r_core, a_au=a_core_AU)

    sig = fn.surf_dens(a_core) # surface density
    T =  fn.temp(a_core) # temperature
    om = fn.omega(m_star,a_core) # orbital frequency
    cs = fn.sound_speed(T) # sound speed
    h = fn.scale_height(cs,om) # scale height
    rho_g = fn.gas_density(sig,h) # gas mass density
    mfp = fn.mean_free_path(fn.mu,rho_g,fn.cross) # mean free path
    vth = fn.therm_vel(cs) # thermal velocity
    v_kep = fn.vkep(m_star,a_core) # keplerian velocity

    # Calculate mass of object with given density (assumed spherical)
    rho_obj = 2.0
    m_obj = 4./3.*np.pi*r_s**3*rho_obj

    # Hill radius
    h_r = fn.hill_rad(m_core,a_core,m_star)

    # WISH Radius
    v_gas = 0.5*cs**2./v_kep # gas velocity
    re = fn.rey(r_s,v_gas,vth,mfp) # reynold's number
    dc = fn.drag_c(re) # drag coefficient

    #OK10 zeta_w
    zeta_w = v_gas/(h_r*om) # dimensionless gas velocity

    f_drag_WS = fn.drag_force(r_s,v_gas,dc,rho_g,mfp,vth)/m_obj # drag force on particle
    r_WS = fn.wish_radius(f_drag_WS,m_core,m_obj)
    r_stab = min(r_WS,h_r)


    fracs = []
    for j in range(401):
        fracs.append(2 - j/100)
    b_arr = np.multiply(h_r,fracs)
    steps = 1e5

    x_arr = np.zeros((len(fracs),int(steps)))
    y_arr = np.zeros((len(fracs),int(steps)))
    v_x_arr = np.zeros((len(fracs),int(steps)))
    v_y_arr = np.zeros((len(fracs),int(steps)))
    x_c_arr = np.zeros((len(fracs),int(steps)))
    y_c_arr = np.zeros((len(fracs),int(steps)))
    v_x_c_arr = np.zeros((len(fracs),int(steps)))
    v_y_c_arr = np.zeros((len(fracs),int(steps)))
    x_s_arr = np.zeros((len(fracs),int(steps)))
    y_s_arr = np.zeros((len(fracs),int(steps)))
    v_x_s_arr = np.zeros((len(fracs),int(steps)))
    v_y_s_arr = np.zeros((len(fracs),int(steps)))

    b_final = 0
    for i,b in enumerate(b_arr):
        #Use OK10 Expressions to send particle in with impact parameter b
        y_init = 3.*h_r

        #A,B coefficients from OK10, c.f. Equation 20.
        A = 3.*(1.+st**2.)/(8.*st*zeta_w)/h_r
        B = (2.*st)**(-1.)
        C = -A*b**2. - B*b

        x_eqn = lambda x: A*x**2. + B*x + C - y_init
        x_init = opt.fsolve(x_eqn,b)[0] + a_core
        theta_0 = np.arctan2(y_init, x_init)
        v_x_init = -2.*v_gas*st/(1 + st**2.) - fn.vkep(m_star, np.sqrt(x_init**2 + y_init**2)) * np.sin(theta_0)
        v_y_init = -v_gas/(1. + st**2.) + fn.vkep(m_star, np.sqrt(x_init**2 + y_init**2)) * np.cos(theta_0)

        # Set core initial values:
        x_c_init = a_core
        y_c_init = 0
        v_x_c_init = -2.*v_gas*st_core/(1 + st_core**2.)
        v_y_c_init = -v_gas/(1. + st_core**2.) - 3./2.*(x_c_init - a_core)*om + om*a_core
        # Set star initial values:
        x_s_init = 0
        y_s_init = 0
        v_x_s_init = -(m_obj * v_x_init + m_core * v_x_c_init)/m_star
        v_y_s_init = -(m_obj * v_y_init + m_core * v_y_c_init)/m_star

        init_values = [x_init, v_x_init, y_init, v_y_init, x_c_init, v_x_c_init, y_c_init, v_y_c_init, x_s_init, v_x_s_init, y_s_init, v_y_s_init]
        wsol = int_orbit(a_core,m_star,m_core,r_s,init_values,flags)

        x = wsol[:,0]
        y = wsol[:,2]
        v_x_arr[i] = wsol[:,1]
        v_y_arr[i] = wsol[:,3]
        x_c = wsol[:,4]
        y_c = wsol[:,6]
        v_x_c_arr[i] = wsol[:,5]
        v_y_c_arr[i] = wsol[:,7]
        x_s = wsol[:,8]
        y_s = wsol[:,10]
        v_x_s_arr[i] = wsol[:,9]
        v_y_s_arr[i] = wsol[:,11]

        x_arr[i] = x
        y_arr[i] = y
        x_c_arr[i] = x_c
        y_c_arr[i] = y_c
        x_s_arr[i] = x_s
        y_s_arr[i] = y_s
        global Accreted
        if Accreted:
            #print("Found it!")
            # Adjust to get perpendicular impact parameter, using OK10 fig. 3, and the caption
            b_final = b * ((2 * A * b + B)/np.sqrt(B**2 + 4 * A**2 * b**2 + 4 * A * B * b + 1))
            break

    #v_x_gas = v_x_arr
    #v_y_gas = v_y_arr-(-v_gas-3./2.*om*x_arr)

    with open("Impact_Param_tests.txt", "a") as text_file:
        text_file.write("Orbital Separation = {a}, Stellar Mass = {m_s}, Core mass = {m_c}, Object radius = {r_o}\n".format(
            a=a_core_AU, m_s=m_star_suns, m_c=m_core_earths, r_o=r_s))
        text_file.write("Maximum accretion radius = {b}, Hill radius = {hr}, WISH radius = {wr}\n".format(
            b=b_final, hr=h_r, wr=r_WS))
    return b_final

Accreted = False
size_array = np.logspace(-5,0,10)
b_array = np.zeros(size_array.shape)
a_core_AU = 1
m_star_suns = 1
a_core = fn.au*a_core_AU
m_star = fn.m_sun*m_star_suns
# Time array
om = np.sqrt(fn.G*m_star/a_core**3.)
stop_time = (5e0)*(2*np.pi/om)
t = np.linspace(0,stop_time,num=1e5)

for i,size in enumerate(size_array):
    Accreted = False
    b_array[i] = test_impact_params(size, a_AU=a_core_AU, m_st=m_star_suns, m_c=1e-4)

np.save("Impact_Param_arrays.npy", np.array([size_array, b_array]))