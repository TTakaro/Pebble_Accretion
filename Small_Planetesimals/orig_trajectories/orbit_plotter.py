__author__ = 'michaelrosenthal'

from scipy.integrate import odeint
from time_TvsR import st_solver
from time_TvsR import st_rad
import drag_functions_turb as fn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import orbits_diffeqs as eqns
from gas_orbit import int_orbit
from matplotlib.patches import Circle
import scipy.optimize as opt

plt.rc('lines', linewidth=2)
plt.rcParams['font.family'] = 'serif'
colors = ['#9cb3c3','#566871','#c48388','#9fb18e','#b04e3b','#c37028','#0392cf','#78282C','#40124C','#42C14D','#4F5350']
mpl.rcParams['axes.color_cycle'] = colors


#Flags
verbose = 0 #Print forces at each time step
two_body = 0 #Include only gravitational force from star, also plot analytic expression for orbit
gas_only = 0 #Consider only the force from gas drag
gas_off = 0 #Turn off gas drag
use_st = 1 #Pass a particle Stokes number instead of radius
save_fig = 0
flags = [verbose, two_body, gas_only, gas_off]

#Input parameters to integrator
a_core_AU = 1. #In AU
m_star_suns = 1 #In Solar masses
m_core_earths = 1e-4 #In Earth masses

size = 1.0e-1 #size of particle as either radius or Stokes number depending on use_st flag
if use_st:
    r_s = st_solver(st=size,a_au=a_core_AU)
    st = size
else:
    r_s = size
    st = st_rad(rad=size,a_au=a_core_AU)

#Convert input values to CGS
a_core = fn.au*a_core_AU
m_star = fn.m_sun*m_star_suns
r_core = fn.r_earth*((m_core_earths)**(1./3.))
m_core = fn.m_earth*m_core_earths

sig = fn.surf_dens(a_core)
T =  fn.temp(a_core)
om = fn.omega(m_star,a_core)
cs = fn.sound_speed(T)
h = fn.scale_height(cs,om)
rho_g = fn.gas_density(sig,h)
mfp = fn.mean_free_path(fn.mu,rho_g,fn.cross)
vth = fn.therm_vel(cs)
v_kep = fn.vkep(m_star,a_core)

#Hill radius
h_r = fn.hill_rad(m_core,a_core,m_star)

#WISH Radius
v_gas = 0.5*cs**2./v_kep
re = fn.rey(r_s,v_gas,vth,mfp)
dc = fn.drag_c(re)

#OK10 zeta_w
zeta_w = v_gas/(h_r*om)
# print zeta_w

#Calculate mass of object with given density (assumed spherical)
rho_obj = 2.0
m_obj = 4./3.*np.pi*r_s**3*rho_obj

f_drag_WS = fn.drag_force(r_s,v_gas,dc,rho_g,mfp,vth)/m_obj
r_WS = fn.wish_radius(f_drag_WS,m_core,m_obj)
r_stab = min(r_WS,h_r)

# Desired Impact Parameter
b_eqn = lambda x: x**3. + 2.*zeta_w/3.*x**2. - 8*st
# b_sol = opt.fsolve(b_eqn,zeta_w)[0]
b_sol = opt.brentq(b_eqn,0,5)
b_set = b_sol*h_r
len_scale = [b_set,'$b_{\rm{set}}$']
print('b_sol = %.3g' %b_sol)
print('zeta_w = %.3g' %zeta_w)
print('st = %.3g' %st)
print(b_eqn(b_sol))
print(b_eqn(3.))
# fracs = [0.1,0.5,1.,1.1,1.25,1.5,2.]
# fracs = [0.3,0.5,0.9,1.0,1.1,1.2,1.3]
#b_arr = np.multiply(b_set,fracs)
fracs = [0.78] #fracs = [1.,1.1,1.11,1.12,1.13,1.14,1.15,1.2]
b_arr = np.multiply(h_r,fracs)

x_arr = np.zeros((len(fracs),int(1e5)))
y_arr = np.zeros((len(fracs),int(1e5)))
v_x_arr = np.zeros((len(fracs),int(1e5)))
v_y_arr = np.zeros((len(fracs),int(1e5)))

zoom = 6e0
fig = plt.figure(1,figsize=(9,9))

stop_time = (1e0)*(2*np.pi/om)
t = np.linspace(0,stop_time,num=1e5)

file = open('non-inertial_forces.txt', 'w')
file.close()

for i,b in enumerate(b_arr):
    #Use OK10 Expressions to send particle in with impact parameter b
    y_init = 3.*h_r


    #A,B coefficients from OK10, c.f. Equation 20.
    A = 3.*(1.+st**2.)/(8.*st*zeta_w)/h_r
    B = (2.*st)**(-1.)

    C = -A*b**2. - B*b

    x_eqn = lambda x: A*x**2. + B*x + C - y_init
    x_init = opt.fsolve(x_eqn,b)[0]

    # x_vert = -B/2./A

    # y_init = A*x_init**2. + B*x_init + C

    #
    v_x_init = -2.*v_gas*st/(1 + st**2.)
    v_y_init = -v_gas/(1. + st**2.) - 3./2.*x_init*om
    #x_init = h_r
    #y_init = 0
    #v_x_init = 0#-fn.vkep(m_star,np.sqrt((x_init+a_core)**2 + y_init**2)) * np.sin(np.arctan2(y_init,x_init+a_core))
    #v_y_init = fn.vkep(m_star,np.sqrt((x_init+a_core)**2 + y_init**2)) - om*(x_init + a_core)

    #print(v_x_init, v_y_init)
    wsol = int_orbit(a_core,m_star,m_core,r_s,[x_init,v_x_init,y_init,v_y_init],flags)

    x = wsol[:,0]
    y = wsol[:,2]
    v_x_arr[i] = wsol[:,1]
    v_y_arr[i] = wsol[:,3]

    in_window = np.where(y<zoom*h_r)

    #print(x[0], y[0], x_c_analytic[0], y_c_analytic[0])
    #j = 1000
    #plt.plot(x[j] + x_c_analytic[j], y[j] + y_c_analytic[j], 'ro')
    plt.plot(x[in_window],y[in_window],label=r'$b = %.2g b_{\rm{set}}$' %(b/b_set))
    #plt.plot(pos_vec_rotated[:,0], pos_vec_rotated[:,1])
    #plt.plot(x_c_analytic, y_c_analytic, 'y--')
    #plt.plot(x + x_c_analytic,y + y_c_analytic,label=r'$b = %.2g b_{\rm{set}}$' %(b/b_set))

    x_arr[i] = x
    y_arr[i] = y

v_x_gas = v_x_arr
v_y_gas = v_y_arr-(-v_gas-3./2.*om*x_arr)



# x = wsol[:,0]
# y = wsol[:,2]

# plt.plot(x,y)

# v_x_b = -2.*v_gas*st/(1. + st**2.)
# v_y_b = -v_gas/(1. + st**2.) - 3./2.*x_b*om
#
# wsol = int_orbit(a_core,m_star,m_core,r_s,[x_b,v_x_b,y_init,v_y_b],flags)
# x = wsol[:,0]
# y = wsol[:,2]

# plt.plot(x,y)
# ax = plt.plot(x,y,label=r'$v_0 = v_{circ}/%.4f$' %frac)

#Use Ruth's ICs
# rad_2 = 4.999995379435623*fn.au
# theta_2 = 6.320078083447993e-7
# r_2 = [rad_2*np.cos(theta_2)-5*fn.au,rad_2*np.sin(theta_2)] #Position of incoming particle, core is at [5 AU,0]
# v_2 = []

# rad_2 = (1-1e-4)*fn.au
# theta_2 = -0.01
# r_2 = [rad_2*np.cos(theta_2)-a_core_AU*fn.au,rad_2*np.sin(theta_2)]
# v_2 = [0,np.sqrt(fn.G*fn.m_sun/(a_core_AU*fn.au)**3.)]
#
# wsol = int_orbit(a_core,m_star,m_core,r_s,[r_2[0],v_2[0],r_2[1],v_2[1]],flags)
# x = wsol[:,0]
# y = wsol[:,2]
#
# plt.plot(x,y)

circle1 = Circle((0, 0), 3.*r_core,color='k',label='Core')
# circle2 = Circle((0, b), r_core,color='r',label='Starting Position')
circle3 = Circle((0, 0), h_r ,fill=False,color='#0392cf',linewidth=3,linestyle='dotted',label=r'$R_H$')
circle4 = Circle((0, 0), r_WS ,fill=False,color='#9fb18e',linewidth=4,linestyle='dotted',label=r'$R_{WS}$')
circle5 = Circle((0, 0), b_set ,fill=False,color='#4D4D4D',linewidth=4,linestyle='dotted',label=r'$b_{set}$')
# circle5 = Circle((0, 0), r_WS_circ ,fill=False,color='#9cb3c3',linewidth=3,linestyle='dashed',label=r'$R_{WS,\circ}$')

ax = plt.gca()
ax.add_patch(circle1)
# ax.add_patch(circle2)
ax.add_patch(circle3)
ax.add_patch(circle4)
ax.add_patch(circle5)

# if two_body:
#
#     thetas,rad = eqns.orbit_params(w0,m_core)
#     plt.plot(rad*np.cos(thetas),rad*np.sin(thetas),color='#BF3EFF',linestyle='dashed',linewidth=4)
#     vel_vec = plt.arrow(w0[0],w0[2],2e4*w0[1],2e4*w0[3],head_width=2.5e9, head_length=2.5e9,fc='k',ec='k')
#     # vel_vec = plt.arrow(w0[0],w0[2],1e5*w0[1],1e5*w0[3],arrowstyle=)
#     # circle4 = Circle((0, 0),GF_rad ,fill=False,color='#566871',linewidth=3,linestyle='dotted')
#
#     ax.add_patch(vel_vec)
#     # ax.add_patch(circle4)

plt.axis(np.multiply([-h_r,h_r,-h_r,h_r],zoom))
#plt.axis([-a_core*1.1,a_core*1.1,-1.1*a_core,1.1*a_core])
plt.gca().legend(loc='best')
# plt.suptitle(r'$v_{0} \, = \, %.3g$' %v_init,fontsize=20)
param_str = r'$a = %.3f$ AU, $M_* = %.3fM_{\odot}$, $M_{core} = %.3fM_{\oplus}$, $r_{s} = %.3f$ cm, $\zeta_w = %.3g$ St=%.3g' \
            % (a_core/fn.au,m_star/fn.m_sun,m_core/fn.m_earth,r_s,zeta_w,st)
plt.title(param_str)
plt.ylabel('y (cm)',fontsize=14)
plt.xlabel('x (cm)',fontsize=14)
plt.show()

# v_orb = np.sqrt(0.5)*np.sqrt(fn.G*M/r)

# w0 = [r,-v_orb,r,0]
#
# thetas,rad = eqns.orbit_params(w0,M)
# x = rad*np.cos(thetas)
# y = rad*np.sin(thetas)
#
# plt.figure(1,figsize=(9,9))
# plt.plot(x,y)
#j
# circle1 = Circle((0, 0), 1.5e12,color='k',label='Core')
# ax = plt.gca()
# ax.add_patch(circle1)
# plt.axes().set_aspect('equal', 'datalim')
# plt.show()

if save_fig:
    fig.savefig('../tst.pdf')


plt.figure(2)
for i in range(len(b_arr)):
    plt.plot(y_arr[i],v_x_gas[i])
    plt.plot(y_arr[i],v_y_gas[i])
#     plt.semilogx(t,v_x_gas[i])
#     plt.semilogx(t,v_y_gas[i])