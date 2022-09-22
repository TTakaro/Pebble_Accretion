# This  defines all the relevant functions to calculate the disk surface density
import numpy as np
import drag_functions_turb as fn
from time_TvsR import *
boltzmann = 1.380658e-16 #cgs
mH = 1.6733000e-24 #grams
G = 6.67259e-8 #cgs
SB = 5.6704e-5
Ldot = 3.839e33
rsun = 7e10

def y_str_eqn(y_str,st):
    """Equation 21d from OC07 with term on RHS subtracted over. Ignores V_0 term following text,
    which we may want to include later."""
    return 2./3. * y_str * (y_str - 1.)**2. - (1 + y_str)**(-1.) + (1 + st**(-1.))**(-1.)

def st_str(st):
    """Function to solve for t_* following OC07 21d. Ignores V_0 term. """
    if st>=1:
        return 1. #Don't allow t_* > t_L
    else:
        return opt.fsolve(y_str_eqn,st,args=(st)) [0]*st
    
#def d_V_12(t_1=1.,t_2=1.,t_L=1,Re=1e8):
#    """Function for reproducing Eqn. (16) in OC07. Uses the full solution"""
#    t_eta = Re**(-0.5)*t_L
#    t_1_str = st_str(t_1)
#    t_2_str = st_str(t_2)
#    t_12_str = max(t_1_str,t_2_str) 
#    t_12_str = max(t_12_str,t_eta)

#    term_1 = (t_12_str + t_1**2./(t_1 + t_12_str) - (t_eta + t_1**2./(t_1 + t_eta))) +\
#                ( (t_2 - t_1)/(t_1 + t_2) * (t_1**2./(t_1 + t_L) - t_1**2./(t_1 + t_12_str)) )

#    term_2 = (t_12_str + t_2**2./(t_2 + t_12_str) - (t_eta + t_2**2./(t_2 + t_eta))) +\
#                ( (t_1 - t_2)/(t_2 + t_1) * (t_2**2./(t_2 + t_L) - t_2**2./(t_2 + t_12_str)) )

#    return np.sqrt(term_1 + term_2)

def d_V_12(t_1=1.,t_2=1.,t_L=1,Re=1e8):
    """Function for reproducing Eqn. (16) in OC07. Uses a piecewise formulation"""
    t_eta = Re**(-0.5)*t_L
    if (t_1 <= t_eta) and (t_2 <= t_eta):
        return np.sqrt(t_L / t_eta * (t_1 - t_2)**2)
    elif (t_1 > t_eta) and (t_1 < t_L):
        y_a = 1.6
        eps = t_1 / t_2
        #print("This is the intermediate regime. Don't use it!")
        return np.sqrt(2 * y_a - (1 + eps) + 2/(1 + eps) * (1/(1 + y_a) + eps**3/(y_a + eps))) * np.sqrt(t_1)
    elif (t_1 >= t_L):
        return np.sqrt(1/(1 + t_1) + 1/(1 + t_2))
    else:
        print("Something's broken. Probably an issue with St_core.", t_1, t_eta)#, t_eta, t_1, t_2)
        return np.sqrt(1/(1 + t_1) + 1/(1 + t_2))

def sig_g(a_arr, sig_0, params):
    """Gaseous surface density."""
    T_0, m_star, sigma_0, r_crit, rho_int, alpha, f_d, eps_g, eps_d, delt, Lamb = params
    gamma = 1
    return sig_0 * ((a_arr/r_crit)**(-gamma)) * np.exp(-(a_arr/r_crit)**(2-gamma))

def t_grow(s, a, f_d, params):
    """Growth time for a particle, from m/m_dot."""
    T_0, m_star, sigma_0, r_crit, rho_int, alpha, f_d, eps_g, eps_d, delt, Lamb = params
    
    T = T_0 * (a**(-3/7))
    c_s = np.sqrt((boltzmann * T)/(2.35*mH))
    Om = np.sqrt((fn.G * m_star * fn.m_sun)/(a * fn.au)**3.)
    v_kep = Om *(a * fn.au)
    eta = (c_s)**2/(2*(v_kep)**2)
    H  = c_s/Om
    sigma = sig_g(a, sigma_0, params)
    rhoGas = fn.gas_density(sigma, H)
    
    M = 1 # Assumed that M_core = 1 earth mass
    st = st_rad(rad=s,alph=alpha,a_au=a,m_suns=m_star,m_earths=M,sig_in=sigma,temp_in=T)
    m = 4/3 * np.pi * rho_int * s**3
    rho = 1e-2 * sigma/H #f_d * sigma/H
    sig = np.pi * s**2
    v0 = eta*v_kep
    vgas = np.sqrt(v0**2 + alpha*c_s**2.)
    lambda_mfp = fn.mean_free_path(fn.mu,rhoGas,fn.cross) #1./((rhoGas/(2.3*mH))*10.**(-15.))
    nu = lambda_mfp*np.sqrt(8/np.pi)*c_s
    Re = (alpha*c_s**2)/(nu*Om) #fn.rey(s, v0, fn.therm_vel(c_s), lambda_mfp)
    v = v0 * d_V_12(t_1=st, t_2=0.5*st, t_L=1, Re=Re)
    
    return m/(eps_g * rho * sig * v)

def dust_to_gas(disk_age, a_PF, a_arr, params):    
    """Calculates the solid surface density (as dust-to-gas ratio) in the Powell et al. 2017 regime.
    Takes an array of orbital distances, at a single disk time."""
    T_0, m_star, sigma_0, r_crit, rho_int, alpha, f_d, eps_g, eps_d, delt, Lamb = params
    
    T = T_0 * (a_arr**(-3/7))
    mstar = m_star * fn.m_sun
    time_grow = disk_age
    sigma = sig_g(a_arr, sigma_0, params)
        
    c_s = np.sqrt((boltzmann*T)/(2.35*mH))
    Om = np.sqrt((G*mstar)/(a_arr*fn.au)**3.)
    H = c_s/Om
    rhoGas = fn.gas_density(sigma, H)
    v_kep = Om * (a_arr*fn.au)
    eta = (c_s)**2/(2*(v_kep)**2)
    v0 = eta*v_kep
    
    St_max = (a_arr*fn.au)/(2 * v0 * time_grow)
    # Assumed that M_core = 1 earth mass
    s_max = np.zeros(St_max.shape)
    for i, St in enumerate(St_max):
        s_max[i] = st_solver(st=St, alph=alpha, a_au=a_arr[i], m_suns=m_star, m_earths=1, temp_in=T[i], sig_in=sigma[i])
    
    H_d = H * np.sqrt(alpha/(alpha + St_max))
    
    vgas = np.sqrt(v0**2 + alpha*c_s**2.)
    lambda_mfp = fn.mean_free_path(fn.mu,rhoGas,fn.cross) #1./((rhoGas/(2.3*mH))*10.**(-15.))
    nu = lambda_mfp*c_s
    Re = (alpha*c_s**2)/(nu*Om) # Why is this v_turb**2/v_th**2?
    deltaV = np.zeros(St_max.size)
    for idx,t in enumerate(St_max):
        deltaV[idx] = vgas[idx] * d_V_12(t,0.5*t,1,Re[idx])

    f = 0.55
    d_t_g = (8 * rho_int * s_max * H_d)/(3 * sigma * time_grow * deltaV * f)
    
    t_g_arr = np.zeros(a_arr.shape) # Growth times for largest particle
    for i, a_au in enumerate(a_arr):
        if a_au > a_PF:
            d_t_g[i] = 1e-2
    return d_t_g
    
def sig_p(f_d, a_PF, a_arr, t, params):
    """Calculates the solid surface density in the Lambrechts and Johansen 2014 regime.
    Takes an array of orbital distances and a single disk time."""
    T_0, m_star, sigma_0, r_crit, rho_int, alpha, f_d, eps_g, eps_d, delt, Lamb = params
    
    ### Sets up constants and orbital variables ###
    eps_p = eps_d
    Sig_g = sig_g(a_arr, sigma_0, params)
    Sig_p = f_d * Sig_g
    
    T = T_0 * (a_arr**(-3/7))
    c_s = np.sqrt((boltzmann*T)/(2.35*mH))
    om = np.sqrt(fn.G * m_star * fn.m_sun/(a_arr*fn.au)**3)
    v_kep = om *(a_arr * fn.au)
    eta = (c_s)**2/(2*(v_kep)**2)
    i_PF = np.argmin(abs(a_arr - a_PF))
    Sig_p0_PF = f_d*Sig_g[i_PF]
    M_dot = (4*np.pi)/(3) * (a_PF*fn.au)**2/t * Sig_p0_PF
    lambda_mfp = fn.mean_free_path(fn.mu, Sig_g/(2*c_s/om), fn.cross)
    Re = (eta**2 * v_kep**2 + alpha * c_s**2)/(lambda_mfp * np.sqrt(8/np.pi) * c_s * om)
    
    ### Checks for the turbulence level of the disk, using the orbital variables ###
    turbulent = False
    for i, a_au in enumerate(a_arr):
        if turbulent or (alpha * c_s[i]**2) > (eta[i]**2 * v_kep[i]**2): #Re[i]**(1/4) > 4 * eta[i] * v_kep[i]/(np.sqrt(alpha) * c_s[i]):
            turbulent = True
    
    ### Find the transition location between Epstein and Stokes ###
    
    # Attempts to numerically solve for the transition location
    s_Ep = np.zeros(a_arr.shape)
    s_St = np.zeros(a_arr.shape)
    if turbulent:
        for i, a_ind in enumerate(a_arr[:i_PF+1]):
            s_Ep[i] = ((3 * Lamb * eps_p)/(8 * delt**2))**(2/5) * (2 * np.pi**7)**(-1/10) * (Sig_g[i]**(3/5) * (M_dot * om[i])**(2/5)/rho_int) * (v_kep[i]/c_s[i]**2)**(4/5) #((3 * Lamb * eps_p * f_d)/(8 * delt))**(2/3) * (2/np.pi)**(1/6) * (v_kep/c_s)**(4/3) * Sig/rho_int
            s_St[i] = 3/(2*np.sqrt(2) * (2*np.pi)**(1/4) * (8*np.pi)**(1/16)) * (Lamb * eps_p)**(1/4)/np.sqrt(delt) * np.sqrt(v_kep[i])/(c_s[i]**(7/4)) * ((om[i] * M_dot * Sig_g[i]**(3/2) * lambda_mfp[i]**(3/2))/(rho_int**(7/2) * alpha))**(1/4) #3/(2 * np.sqrt(2) * (8 * np.pi)**(1/8)) * np.sqrt(Lamb * eps_p * f_d/delt) * v_kep/(c_s**(3/2)) * ((Sig**3 * lambda_mfp)/(rho_int**3))**(1/4)
        j_minimum = np.argmin(abs(s_Ep - s_St)) # Index with the location of drag law transition
        #print(s_Ep,s_St, 9*lambda_mfp/4)
        a_transition = a_arr[:i_PF+1][j_minimum-1]
    else:
        j_minimum = 0
        #print("GIVING BROKEN RESULTS")
        a_transition = a_arr[0] # Will need to replace this once I have a good method for s_St (laminar)
        
    ### Sets the solid surface densities ###
    for i, a_au in enumerate(a_arr):
        lambda_mfp = fn.mean_free_path(fn.mu, Sig_g[i]*om[i]/(2*c_s[i]), fn.cross)
        if turbulent:
            # Assumes Turbulence dominated relative velocity
            Sig_p_Ep = ((8 * delt * Sig_g[i])/(3 * Lamb * eps_p * c_s[i] * v_kep[i]**2))**(2/5) * (np.pi/2)**(1/10) * (M_dot * om[i]/delt)**(3/5) * 2**(-3/10) * np.pi**(-9/10)
            Sig_p_St = (8*np.pi)**(1/8) * (8/np.pi)**(1/4)/np.sqrt(2*np.pi) * np.sqrt((om[i] * M_dot)/(Lamb * eps_p * c_s[i] * v_kep[i]**2)) * (rho_int * lambda_mfp * Sig_g[i])
        else:
            # Assumes Laminar dominated instead
            Sig_p_Ep = (8/(3 * Lamb))**2 * delt/(np.pi**2) * (alpha * M_dot * om[i])/(eps_p**2 * f_d**2 * v_kep[i]**2)
            Sig_p_St = 4 * (8/np.pi)**(1/4) * np.sqrt(Sig_g[i] * lambda_mfp * rho_int * alpha * eta[i])/eps_p

        Passed = False
        if (a_au > a_transition) or (Passed):
            Sig_p[i] = Sig_p_Ep
            Passed = True
        else:
            Sig_p[i] = Sig_p_Ep #Sig_p_St
    
    return Sig_p

def LJ_sig_p(f_d, a_PF, a_arr, t, params):
    """Calculates the solid surface density in Lambrechts and Johansen 2014 regime, using their formulae."""
    T_0, m_star, sigma_0, r_crit, rho_int, alpha, f_d, eps_g, eps_d, delt, Lamb = params
    
    Sig_g = sig_g(a_arr, sigma_0, params)
    om = np.sqrt(fn.G * m_star * fn.m_sun/(a_arr*fn.au)**3)
    return 2**(5/6) * 3**(-7/12) * eps_g**(1/3)/eps_p**(1/2) * f_d**(5/6) * om**(-1/6) * t**(-1/6) * Sig_g

def surface_density(a_arr, t_disk, params):
    """Calculates the gaseous and solid surface density in a protoplanetary disk. This function operates to 
    switch regimes between the Lambrechts and Johansen 2014 and Powell 2017 regimes. It operates on an array
    of orbital distances, and an array of disk ages."""
    T_0, m_star, sigma_0, r_crit, rho_int, alpha, f_d, eps_g, eps_d, delt, Lamb = params
    
    ### Sets up constants and orbital variables ###
    eps_p = eps_g # This might be wrong
    
    sigma_g = sig_g(a_arr, sigma_0, params)    
    T = T_0 * (a_arr**(-3/7))
    c_s = np.sqrt((boltzmann*T)/(2.35*mH))
    Om = np.sqrt(fn.G * m_star*fn.m_sun/(a_arr * fn.au)**3)
    v_kep = np.sqrt(fn.G * m_star*fn.m_sun/(a_arr * fn.au))
    eta = (c_s)**2/(2*(v_kep)**2)
    lambda_mfp = fn.mean_free_path(fn.mu, sigma_g/(2*c_s/Om), fn.cross)
    Re = (eta**2 * v_kep**2 + alpha * c_s**2)/(lambda_mfp * np.sqrt(8/np.pi) * c_s * Om)
    
    ### Checks for the turbulence level of the disk, using the orbital variables ###
    turbulent = False
    for i, a_au in enumerate(a_arr):
        if turbulent or (alpha * c_s[i]**2) > (eta[i]**2 * v_kep[i]**2): #Re[i]**(1/4) > 4 * eta[i] * v_kep[i]/(np.sqrt(alpha) * c_s[i]):
            turbulent = True
    
    ### Find pebble front location  ###
    a_PF = (3/16)**(1/3) * (fn.G * m_star * fn.m_sun)**(1/3) * (eps_d * f_d)**(2/3) * t_disk**(2/3) / fn.au
    
    ### Solving for Solid Surface Density ###
    a_interior = np.zeros(t_disk.size)
    sigma_d = np.zeros([t_disk.size, a_arr.size])
    # Attempts to numerically solve for the transition location
    s_Ep = np.zeros([t_disk.size, a_arr.size])
    s_St = np.zeros([t_disk.size, a_arr.size])
    
    for i,t in enumerate(t_disk):        
        # Check how much of a_arr is inside of pebble front
        a_applicable = np.trim_zeros(np.where(a_arr < a_PF[i], a_arr, 0))
        Sig = sig_g(a_applicable, sigma_0, params)
        # Set solid surface density inside of pebble front using our version of LJ'14 solution
        sigma_d[i,:a_applicable.size] = sig_p(f_d, a_PF[i], a_applicable, t, params)
        sigma_d[i,a_applicable.size:] = f_d * sigma_g[a_applicable.size:] # Outside, set to initial f_d times Sig_g
        
        # Solve for M_dot from drift at the pebble front
        j_PF = np.argmin(abs(a_applicable - a_PF[i]))
        Sig_p0_PF = f_d*Sig[j_PF]
        M_dot = (4*np.pi)/(3) * (a_PF[i]*fn.au)**2/t * Sig_p0_PF
        
        # Set max particle size
        if turbulent:
            for j, a_ind in enumerate(a_arr):
                if a_ind < a_PF[i]:
                    s_Ep[i,j] = ((3 * Lamb * eps_p)/(8 * delt**2))**(2/5) * (2 * np.pi**7)**(-1/10) * (sigma_g[j]**(3/5) * (M_dot * Om[j])**(2/5)/rho_int) * (v_kep[j]/c_s[j]**2)**(4/5) #((3 * Lamb * eps_p * f_d)/(8 * delt))**(2/3) * (2/np.pi)**(1/6) * (v_kep/c_s)**(4/3) * Sig/rho_int
                    s_St[i,j] = 3/(2*np.sqrt(2) * (2*np.pi)**(1/4) * (8*np.pi)**(1/16)) * (Lamb * eps_p)**(1/4)/np.sqrt(delt)
        
        # Set orbital variables just inside the pebble front
        T = T_0 * (a_applicable**(-3/7))
        cs = np.sqrt((boltzmann*T)/(2.35*mH))
        v_k = np.sqrt(fn.G * m_star*fn.m_sun/(a_applicable * fn.au))
        v0 = (cs)**2/(2*(v_k)**2) * v_k
        lambda_mfp = fn.mean_free_path(fn.mu, Sig/(2*cs/(v_k/(a_applicable * fn.au))), fn.cross)
        
        # Calculate the M_dot per area that we want to compare, between global drift and local growth
        St_max = (a_applicable*fn.au)/(2 * v0 * t)
        s_max = np.zeros(St_max.shape)
        for j, St in enumerate(St_max):
            s_max[j] = st_solver(st=St, alph=alpha, a_au=a_applicable[j], m_suns=m_star, m_earths=1, temp_in=T[j], sig_in=Sig[j])
        M_dot_loc_specific = dust_to_gas(t, a_PF[i], a_applicable, params)*Sig/t
        M_dot_glo_specific = M_dot/(2*np.pi * (a_applicable*fn.au)**2)
        
        # Previously, if there's a location where t_drift is longer than the disk age, switch regimes
        # Now, compare the local (growth) M_dot to the global (drift) M_dot, calculated at the pebble front
        if np.amax(M_dot_loc_specific - M_dot_glo_specific) > 0: #np.amax(t_grow_arr) > t: #np.amax(t_drift) > t:
            print("Transitioned!")
            # Local growth sets the size of particles, so we're in Powell et al. 2017 regime
            # Find location in disk where local M_dot is greatest 
            j = np.argmax(M_dot_loc_specific - M_dot_glo_specific) #np.argmax(t_grow_arr)
            
            # Define disk edge
            if a_interior[i-1] != 0: # If this isn't the first time step where we're calculating an inner pebble front
                a = a_interior[i-1] # Set a to be the last inner a_PF
                j = np.argmin(abs(a - a_applicable)) # Pull the index of this inner a_PF
            else:
                # Sets the first inner a_PF to be the location of max local growth inside of a_PF
                # This is the location at which the M_dot transition occurs
                a = a_applicable[j]
                st_edge = St_max[j] # The maximum St at the edge of the disk, when the pebble front hits the disk edge
                t_edge = 0.99 * t # Modified to give non-zero time to drift the 1st time
            
            # Calculate location to which largest particles have drifted
            v_drift = - 2 * v0[j] * (st_edge/(1 + st_edge**2))
            t_old = max(t_edge, t_disk[i-1])
            
            # Set this location to which the largest particle has drifted. Doesn't use 0, bc of earlier check
            a_interior[i] = np.maximum(a - abs((t - t_old) * v_drift)/fn.au, 0.0001)
            
            # Find the index in a_applicable closest to a_interior[i], then only update from k on
            k = np.argmin(abs(a_interior[i] - a_applicable)) # Pull the index of this innermost a_PF
            # Apply Powell regime from k on, leaving LJ'14 regime from 0 to k
            sigma_d[i,k:a_applicable.size] = dust_to_gas(t, a_PF[i], a_applicable[k:], params) * Sig[k:]
            # Apply Powell regime for the particle size as well
            s_Ep[i,k:a_applicable.size] = s_max[k:a_applicable.size]
    return [sigma_g, sigma_d, s_Ep]