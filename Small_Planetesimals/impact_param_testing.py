from scipy.integrate import odeint
from time_TvsR import st_solver
from time_TvsR import st_rad
import drag_functions_turb as fn
import numpy as np
import scipy.optimize as opt

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
    x, vx, y, vy, x_c, vx_c, y_c, vy_c = init_vals

    # Initial value
    w0 = [x,vx,y,vy,x_c,vx_c,y_c,vy_c]
    p = [a_core, m_star, m_core, r_obj, flags]

    # Time array
    om = np.sqrt(fn.G*m_star/a_core**3.)
    stop_time = (5e2)*(2*np.pi/om)
    t = np.linspace(0,stop_time,num=1e5)

    # Calculate mass of object with given density (assumed spherical)
    rho_obj = 2.0
    m_obj = 4./3.*np.pi*r_obj**3*rho_obj

    wsol = odeint(force_field,w0,t,args=(p,))

    return wsol


def force_field(w, t, p):
    # Unpack positions and velocities, as well as parameters for the orbit
    x, vx, y, vy, x_c, vx_c, y_c, vy_c = w
    a_core, m_star, m_core, r_s, flags = p

    verbose, two_body, gas_only, gas_off = flags

    # Produce distance from core and angle
    r = np.sqrt((x - x_c)**2 + (y - y_c)**2) #np.sqrt(x**2 + y**2)
    r_vec = np.array([x - x_c, y - y_c, 0])
    th = np.arctan2(r_vec[1],r_vec[0])

    r_core = fn.r_earth*((m_core/fn.m_earth)**(1./3.))

    if r < r_core:
        f = [0,0,0,0,0,0,0,0]
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

    # Calculate gas drag force

    # Begin by getting velocity of the particle relative to the gas, assume distance from star is small enough that
    # gas always moves in the -y_hat direction
    gas_vec = np.array([0,-eta*v_kep-3./2.*om*x,0]) # Get gas velocity from eta*v_k, include Keplerian Shear (still need to add turb)
    vel_vec = np.array([vx,vy,0])
    v_rel = vel_vec - gas_vec
    v_hat = v_rel/np.linalg.norm(v_rel)
    v_rel_mag = np.linalg.norm(v_rel)

    gas_vec_core = np.array([0,-eta*v_kep-3./2.*om*x_c,0])
    vel_vec_core = np.array([vx_c,vy_c,0])
    v_rel_core = vel_vec_core - gas_vec_core
    v_hat_core = v_rel_core/np.linalg.norm(v_rel_core)
    v_rel_mag_core = np.linalg.norm(v_rel_core)

    #Calculate magnitude of gas drag force
    re = fn.rey(r_s, v_rel_mag, vth, mfp)
    dc = fn.drag_c(re)

    re_core = fn.rey(r_core, v_rel_mag_core, vth, mfp)
    dc_core = fn.drag_c(re_core)

    #Calculate mass of object with given density (assumed spherical)
    rho_obj = 2.0
    m_obj = 4./3.*np.pi*r_s**3*rho_obj

    f_drag_mag = fn.drag_force(r_s,v_rel_mag,dc,rho_g,mfp,vth)/m_obj #Calculate force per unit mass due to gas drag
    f_drag = np.multiply(-f_drag_mag,v_hat)

    f_drag_mag_core = fn.drag_force(r_core, v_rel_mag_core, dc_core, rho_g, mfp, vth)/m_core
    f_drag_core = np.multiply(-f_drag_mag_core, v_hat_core)

    # print f_drag_vec
    f_core = -fn.G*m_core/np.power(r,3) * r_vec
    f_obj = fn.G*m_obj/np.power(r,3) * r_vec

    #Force from central star, including non-intertial terms (c.f. OK10, Eqn. 7)
    f_star = [2.*om*vy + 3.*om**2.*x, -2.*om*vx, 0]
    f_star_core = [2.*om*vy_c + 3.*om**2.*x_c, -2.*om*vx_c, 0]

    if two_body:
        f_core = [0,0,0]
        f_obj = [0,0,0]
    elif gas_only:
        f_core = [0,0,0]
        f_obj = [0,0,0]
        f_star = [0,0,0]
        f_star_core = [0,0,0]
        f_cent = [0,0,0]
        f_cor = [0,0,0]
    elif gas_off:
        f_drag = [0,0,0]

    if verbose:
        print("f_star = %.5g")%(np.linalg.norm(f_star))
        print("f_core = %.5g")%f_core
        print("f_drag = %.5g")%(np.linalg.norm(f_drag))
        print("v_rel = %.3g")%v_rel_mag
        print("v_x = %.3g")%vx
        print("v_y = %.3g")%vy
        print("t = %.3g\n")%t

    #Define and return vector of derivatives
    f = [vx, f_core[0] + f_star[0] + f_drag[0],
         vy, f_core[1] + f_star[1] + f_drag[1],
         vx_c, f_obj[0] + f_star_core[0] + f_drag_core[0],
         vy_c, f_obj[1] + f_star_core[1] + f_drag_core[1]]

    return f

#Input parameters to integrator
a_core_AU = 10. #In AU
m_star_suns = 1 #In Solar masses
m_core_earths = 1e-3 #In Earth masses

#Convert input values to CGS
a_core = fn.au*a_core_AU
m_star = fn.m_sun*m_star_suns
r_core = fn.r_earth*((m_core_earths)**(1./3.))
m_core = fn.m_earth*m_core_earths

size = 100 # #size of particle as either radius or Stokes number depending on use_st flag
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

#Hill radius
h_r = fn.hill_rad(m_core,a_core,m_star)

#WISH Radius
v_gas = 0.5*cs**2./v_kep # gas velocity
re = fn.rey(r_s,v_gas,vth,mfp) # reynold's number
dc = fn.drag_c(re) # drag coefficient

#OK10 zeta_w
zeta_w = v_gas/(h_r*om) # dimensionless gas velocity

#Calculate mass of object with given density (assumed spherical)
rho_obj = 2.0
m_obj = 4./3.*np.pi*r_s**3*rho_obj

f_drag_WS = fn.drag_force(r_s,v_gas,dc,rho_g,mfp,vth)/m_obj # drag force on particle
r_WS = fn.wish_radius(f_drag_WS,m_core,m_obj)
r_stab = min(r_WS,h_r)

# Desired Impact Parameter
b_eqn = lambda x: x**3. + 2.*zeta_w/3.*x**2. - 8*st
# b_sol = opt.fsolve(b_eqn,zeta_w)[0]
b_sol = opt.brentq(b_eqn,0,5)
b_set = b_sol*h_r
len_scale = [b_set,'$b_{\rm{set}}$']
print('b_sol = {b}'.format(b=b_sol))
print('zeta_w = {z}'.format(z=zeta_w))
print('st = {s}'.format(s=st))
print(b_eqn(b_sol))
print(b_eqn(3.))
fracs = []
for j in range(21):
	fracs.append(2 - j/10)
b_arr = np.multiply(b_set,fracs)

x_arr = np.zeros((len(fracs),int(1e5)))
y_arr = np.zeros((len(fracs),int(1e5)))
v_x_arr = np.zeros((len(fracs),int(1e5)))
v_y_arr = np.zeros((len(fracs),int(1e5)))
x_c_arr = np.zeros((len(fracs),int(1e5)))
y_c_arr = np.zeros((len(fracs),int(1e5)))
v_x_c_arr = np.zeros((len(fracs),int(1e5)))
v_y_c_arr = np.zeros((len(fracs),int(1e5)))

Accreted = False
for i,b in enumerate(b_arr):
    #Use OK10 Expressions to send particle in with impact parameter b
    y_init = 3.*h_r

    #A,B coefficients from OK10, c.f. Equation 20.
    A = 3.*(1.+st**2.)/(8.*st*zeta_w)/h_r
    B = (2.*st)**(-1.)
    C = -A*b**2. - B*b

    x_eqn = lambda x: A*x**2. + B*x + C - y_init
    x_init = opt.fsolve(x_eqn,b)[0]
    v_x_init = -2.*v_gas*st/(1 + st**2.)
    v_y_init = -v_gas/(1. + st**2.) - 3./2.*x_init*om

    # Set core initial values:
    x_c_init = 0
    y_c_init = 0
    v_x_c_init = -2.*v_gas*st_core/(1 + st_core**2.)
    v_y_c_init = -v_gas/(1. + st_core**2.) - 3./2.*x_c_init*om

    init_values = [x_init, v_x_init, y_init, v_y_init, x_c_init, v_x_c_init, y_c_init, v_y_c_init]
    wsol = int_orbit(a_core,m_star,m_core,r_s,init_values,flags)

    x = wsol[:,0]
    y = wsol[:,2]
    v_x_arr[i] = wsol[:,1]
    v_y_arr[i] = wsol[:,3]
    x_c = wsol[:,4]
    y_c = wsol[:,6]
    v_x_c_arr[i] = wsol[:,5]
    v_y_c_arr[i] = wsol[:,7]

    x_arr[i] = x
    y_arr[i] = y
    x_c_arr[i] = x_c
    y_c_arr[i] = y_c
    if Accreted:
    	print("Found it!")
    	b_final = b
    	break

v_x_gas = v_x_arr
v_y_gas = v_y_arr-(-v_gas-3./2.*om*x_arr)

stop_time = (5e2)*(2*np.pi/om)
t = np.linspace(0,stop_time,num=1e5)

with open("Impact_Param_tests.txt", "w") as text_file:
    text_file.write("Orbital Separation = {a}, Stellar Mass = {m_s}, Core mass = {m_c}, Object radius = {r_o}\n".format(
    	a=a_core_AU, m_s=m_star_suns, m_c=m_core_earths, r_o=r_s))
    text_file.write("Maximum accretion radius = {b}, Hill radius = {hr}, WISH radius = {wr}, Settling radius = {sr}\n".format(
    	b=b_final, hr=h_r, wr=r_WS, sr=b_set))