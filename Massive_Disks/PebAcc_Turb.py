import drag_functions_turb as fn
import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import odeint

def d_V_12(t_1=1.,t_2=1.,t_L=1,Re=1e8):
    """Function for reproducing Eqn. (16) in OC07. Might need to add more input parameters"""
    t_eta = Re**(-0.5)*t_L
    if (t_1 <= t_eta) and (t_2 <= t_eta):
        return np.sqrt(t_L / t_eta * (t_1 - t_2)**2)
    elif (t_1 > t_eta) and (t_1 < t_L):
        y_a = 1.6
        eps = t_1 / t_2
        return np.sqrt(2 * y_a - (1 + eps) + 2/(1 + eps) * (1/(1 + y_a) + eps**3/(y_a + eps))) * np.sqrt(t_1)
    elif (t_1 >= t_L):
        return np.sqrt(1/(1 + t_1) + 1/(1 + t_2))
    else:
        print(t_eta, t_1, t_2)
        return np.sqrt(1/(1 + t_1) + 1/(1 + t_2))
    #else:
    #    return "Error! No regime found."

class Disk:
    """ This class contains all of the properties of the protoplanetary disk. """
    def __init__(self, alpha=1e-100, a_core_au=1, m_star_solarmasses=1, gas_dep=1., sol_gas_ratio=0.01,
                 sig_p_in=0, sig_g_in=0, temp_in=0):
        """ List of old names -> new names:
            alph -> alpha
            m_suns -> m_star_solarmasses
            a_au -> a_core_au
            sig -> sig_gas
            sig_p -> sig_solid
            t -> T
            h -> H
            rho_g -> rho_gas
            vth -> v_th
            vg -> v_gas_lam
            v_cg -> v_core_gas_lam, v_core_gas (after turbulence is added in)
            vrel_i -> v_rel_i
            v_turb -> v_gas_turb """
        self.m_star = m_star_solarmasses * fn.m_sun # Defines mass of star
        self.a_core = a_core_au * fn.au # Converts semi-major axis into cgs units

        ### Set disk parameters ###
        if sig_g_in: self.sig_gas = sig_g_in
        else: self.sig_gas = fn.surf_dens(self.a_core)/gas_dep # Calculates gas surface density
        if sig_p_in: self.sig_solid = sig_p_in # Calculates solid surface density
        else: self.sig_solid = self.sig_gas * sol_gas_ratio
        
        # Calculates temperature of disk
        if temp_in: self.T = temp_in
        else: self.T = fn.temp(self.a_core)

        self.alpha = alpha
        self.om = fn.omega(self.m_star, self.a_core) # Orbital frequency
        self.cs = fn.sound_speed(self.T) # Sound speed
        self.H = fn.scale_height(self.cs, self.om) # Scale height (OF GAS?)
        self.rho_gas = fn.gas_density(self.sig_gas, self.H) # Density of gas
        self.mfp = fn.mean_free_path(fn.mu, self.rho_gas, fn.cross) # Mean free path
        self.v_th = fn.therm_vel(self.cs) # Thermal velocity
        self.v_kep = fn.vkep(self.m_star, self.a_core) # Keplerian velocity of core
        self.eta = fn.eta(self.cs, self.v_kep) # η, a measure of the local gas pressure support
        #self.v_core = fn.vkep(self.m_star, self.a_core) # Sets velocity of core to the keplerian velocity [WILL CHANGE]
        #self.v_gas_lam = fn.v_gas(self.v_core, self.cs) # Absolute laminar gas velocity. There's no reason we need this. Just use eta*v_k in future.
        #self.v_core_gas_lam = np.absolute(self.v_core - self.v_gas_lam) # Core velocity relative to the gas (ignoring turbulence)
        self.v_rel_i = self.v_kep # Sets initial velocity of core relative to gas

        # Turbulent gas parameters, see e.g. Ormel and Cuzzi (2007)
        self.re_f = fn.re_f(alpha, self.cs, self.H, self.v_th, self.mfp) # Reynolds number, MAYBE?
        self.v_gas_turb = fn.turb_vel(alpha, self.cs) # Turbulent velocity of gas
        #self.v_core_gas = np.sqrt(self.v_gas_turb**2 + (self.v_gas_lam - self.v_core)**2) # Core velocity relative to gas
        self.v_gas_tot = np.sqrt(self.v_gas_turb**2 + (self.eta * self.v_kep)**2) # Total velocity of gas

class Core(Disk):
    """ This class contains all of the properties of an accreting core. """
    def __init__(self, a_core_au=1, m_core_earthmasses=1, alpha=1e-100, m_star_solarmasses=1, gas_dep=1.,
                 sol_gas_ratio=0.01, rho_core=2., sig_p_in=0, sig_g_in=0, temp_in=0, r_shear_off=0, extend_rh=1, alpha_z=0, h_mod=1):
        """ List of old names -> new names:
            m_earths -> m_core_earthmasses
            h_r -> r_hill
            b_r -> r_bondi
            r -> s
            mobj -> m_obj
            fd -> f_d
            v_L_r -> v_r_lam
            v_L_phi -> v_phi_lam
            v_L -> v_lam
            forces_obj -> f_drag_obj
            forces_core -> f_drag_core
            b_shear -> r_sh (function to calculate is r_shear)
            fd -> f_drag
            b_ram -> r_shear_ram
            b_an -> r_shear_an
            v_pg_turb -> v_obj_gas_turb
            v_pg -> v_obj_gas
            v_L_iner -> v_lam_iner
            v_p_turb -> v_obj_turb
            v_p -> v_obj
            v_oc -> v_obj_core
            shear_vel -> v_shear
            v_entry -> v_inf
            v_kick -> v_enc
            v_enc_gas -> v_gas_enc
            reys_enc -> re_enc
            drag_enc -> dc_enc
            force_enc -> f_drag_enc
            h_turb -> H_turb
            kh_height -> H_KH
            h_disk -> H_disk
            acc_height -> H_acc
            area -> area_acc
            time -> t_acc """
        super().__init__(alpha, a_core_au, m_star_solarmasses, gas_dep, sol_gas_ratio, sig_p_in, sig_g_in, temp_in)
        self.r_shear_off = r_shear_off
        self.extend_rh = extend_rh
        self.alpha_z = alpha_z
        self.h_mod = h_mod

        self.m_core = m_core_earthmasses * fn.m_earth # Converts core mass into cgs units
        self.r_core = ((3 * self.m_core)/(4 * np.pi * rho_core))**(1/3) # Calculates radius of core
        # Used to be self.r_core = (m_core_earthmasses**(1./3)) * fn.r_earth. MAKE SURE STILL RIGHT
        
        ### Set core parameters ###
        self.r_hill = fn.hill_rad(self.m_core, self.a_core, self.m_star) # Hill radius
        self.r_bondi = fn.bondi_rad(self.m_core, self.cs) # Bondi radius
        self.r_bondi = max(self.r_bondi, self.r_core) # Prevents Bondi radius from shrinking below core radius
        self.rho_core = rho_core


    def t_stop(self, s, rho_obj=2.):
        """ Calculates stopping time, given particle size s. """
        if s > 9./4 * self.mfp: # Solve iteratively if we're in the fluid regime
            # Calculate terminal velocity and stopping time by iterating over force law. We've ignored angle here.
            delta = 1 # Used to check if we've converged
            v_i = self.v_rel_i
            while np.abs(delta) > .001:
                re = fn.rey(s, v_i, self.v_th, self.mfp) # Reynolds number
                m_obj = fn.obj_mass(s, rho_obj) # Mass of accreting objects
                dc = fn.drag_c(re) # Drag coefficient
                f_d = fn.stokes_ram(s, dc, self.rho_gas, v_i) # Drag force (ram pressure regime)
                t_s = m_obj*v_i/f_d # Stopping time
                # st = fn.stl(t_s, per)
                v_r_lam = fn.v_new_r(self.eta, self.v_kep, t_s, self.om) # Radial component of laminar velocity (particle rel. to gas)
                v_phi_lam = fn.v_new_phi(self.eta, self.v_kep, t_s, self.om) # Phi component of laminar velocity
                v_lam = np.sqrt(v_r_lam**2 + v_phi_lam**2) # Total laminar velocity
                t_eddy = (self.om * (1 + (v_lam/self.v_gas_turb)**2)**.5)**-1 # Eddy crossing time
                stl = t_s/t_eddy # Stokes number
                if stl > 10:
                    v_new_turb = self.v_gas_turb * np.sqrt(1 - (1 + stl)**-1)
                else:
                    v_new_turb = fn.v_pg(self.v_gas_turb, stl, self.re_f)
                v_new = np.sqrt(v_lam**2 + v_new_turb**2) # Calculates total velocity, adding laminar and turbulent
                delta = (v_new - v_i)/v_i # This was changed from v_new - v_i. MAKE SURE STILL WORKS
                v_i = v_new
            self.t_s = t_s
        else: # Applies Epstein drag law if in diffuse regime
            self.t_s = fn.ts_eps(rho_obj, self.rho_gas, s, self.v_th)
        
        # Calculate laminar velocities from converged stopping time.
        self.v_r_lam = fn.v_new_r(self.eta, self.v_kep, self.t_s, self.om) # Radial component
        self.v_phi_lam = fn.v_new_phi(self.eta, self.v_kep, self.t_s, self.om) # Phi component
        self.v_lam = np.sqrt(self.v_r_lam**2 + self.v_phi_lam**2) # Total velocity

        self.t_eddy = (self.om * (1 + (self.v_lam/self.v_gas_turb)**2)**.5)**-1 # Eddy crossing time
        self.stl = self.t_s/self.t_eddy
        self.tau_s = self.t_s * self.om # Dimensionless stopping time

        if self.stl > 10: # Avoids using OC07 expressions for large stopping time
            self.v_obj_gas_turb = self.v_gas_turb * np.sqrt(1 - (1 + self.stl)**-1)
        else:
            self.v_obj_gas_turb = fn.v_pg(self.v_gas_turb, self.stl, self.re_f)
        self.v_obj_gas = np.sqrt(self.v_lam**2 + self.v_obj_gas_turb**2)

        return self.t_s


    def t_stop_core(self):
        """ Calculatues stopping time for the core. """
        if self.r_core > 9./4 * self.mfp: # Solve iteratively if we're in the fluid regime
            # Calculate terminal velocity and stopping time by iterating over force law. We've ignored angle here.
            delta = 1 # Used to check if we've converged
            v_i = self.v_rel_i
            while np.abs(delta) > .001:
                re = fn.rey(self.r_core, v_i, self.v_th, self.mfp) # Reynolds number
                dc = fn.drag_c(re) # Drag coefficient
                f_d = fn.stokes_ram(self.r_core, dc, self.rho_gas, v_i) # Drag force (ram pressure regime)
                t_s = self.m_core*v_i/f_d # Stopping time
                # st = fn.stl(t_s, per)
                v_r_lam = fn.v_new_r(self.eta, self.v_kep, t_s, self.om) # Radial component of laminar velocity
                v_phi_lam = fn.v_new_phi(self.eta, self.v_kep, t_s, self.om) # Phi component of laminar velocity
                v_lam = np.sqrt(v_r_lam**2 + v_phi_lam**2) # Total laminar velocity
                t_eddy = (self.om * (1 + (v_lam/self.v_gas_turb)**2)**.5)**-1 # Eddy crossing time
                stl = t_s/t_eddy # Stokes number
                if stl > 10:
                    v_new_turb = self.v_gas_turb * np.sqrt(1 - (1 + stl)**-1)
                else:
                    v_new_turb = fn.v_pg(self.v_gas_turb, stl, self.re_f)
                v_new = np.sqrt(v_lam**2 + v_new_turb**2) # Calculates total velocity, adding laminar and turbulent
                delta = (v_new - v_i)/v_i
                v_i = v_new
            self.t_s_core = t_s
        else: # Applies Epstein drag law if in diffuse regime
            self.t_s_core = fn.ts_eps(self.rho_core, self.rho_gas, self.r_core, self.v_th)

        # Calculate laminar velocities from converged stopping time.
        self.v_r_lam_core = fn.v_new_r(self.eta, self.v_kep, self.t_s_core, self.om) # Radial component
        self.v_phi_lam_core = fn.v_new_phi(self.eta, self.v_kep, self.t_s_core, self.om) # Phi component
        self.v_lam_core = np.sqrt(self.v_r_lam_core**2 + self.v_phi_lam_core**2) # Total velocity

        self.t_eddy_core = (self.om * (1 + (self.v_lam_core/self.v_gas_turb)**2)**.5)**-1 # Eddy crossing time
        self.stl_core = self.t_s_core/self.t_eddy_core
        self.tau_s_core = self.t_s_core * self.om # Dimensionless stopping time

        if self.stl_core > 10: # Avoids using OC07 expressions for large stopping time
            self.v_core_gas_turb = self.v_gas_turb * np.sqrt(1 - (1 + self.stl_core)**-1)
        else:
            self.v_core_gas_turb = fn.v_pg(self.v_gas_turb, self.stl_core, self.re_f)
        self.v_core_gas = np.sqrt(self.v_lam_core**2 + self.v_core_gas_turb**2)

        return self.t_s_core


    def r_wish(self, s, rho_obj=2.):
        """ Calculates wind-shearing radius, given particle size s. """
        v_cap = self.v_gas_tot # Set relevant velocity for orbit capture
        self.re = fn.rey(s, v_cap, self.v_th, self.mfp) # Reynolds number
        self.dc = fn.drag_c(self.re) # Drag coefficient
        self.f_drag_obj = fn.drag_force(s, v_cap, self.dc, self.rho_gas, self.mfp, self.v_th)

        # Drag force on core
        self.f_drag_core = fn.drag_force(self.r_core, self.v_core_gas, fn.drag_c(fn.rey(self.r_core, self.v_core_gas,
                                         self.v_th, self.mfp)), self.rho_gas, self.mfp, self.v_th)

        self.m_obj = (4./3.*np.pi*s**3)*rho_obj # Calculates mass of accreted object
        self.delta_a = np.abs(self.f_drag_obj/self.m_obj - self.f_drag_core/self.m_core) # Differential acceleration between core and object

        self.r_ws = fn.wish_radius(self.delta_a, self.m_core, self.m_obj) # Wind shearing radius
        return self.r_ws


    def r_shear(self, s, rho_obj=2.):
        """ Calculates shearing radius, given particle size s. """
        self.m_obj = (4./3.*np.pi*s**3)*rho_obj # Calculates mass of accreted object

        if self.r_shear_off: r_shear = 1e100 # Checks for flag
        elif s > 9./4 * self.mfp:
            def r_shear_solver(r_shear): # Function to pass to f_solve to determine r_shear
                v_rel = r_shear * self.om
                re = fn.rey(s, v_rel, self.v_th, self.mfp)
                dc = fn.drag_c(re)
                f_drag = fn.drag_force(s, v_rel, dc, self.rho_gas, self.mfp, self.v_th)
                return r_shear - np.sqrt(fn.G * self.m_core * self.m_obj/f_drag)

            # Solution for r_shear in the ram regime
            self.r_shear_ram = (fn.G * self.m_core * self.m_obj/
                                (.5 * .47 * self.rho_gas * np.pi * s**2. * self.om**2))**(1./4)
            self.r_shear_an = (3. * self.tau_s)**(1./3) * self.r_hill
            # Guess is the minimum of the analtyic solution in a linear regime and the solution in Ram
            r_shear_guess = min(self.r_shear_ram, self.r_shear_an)
            self.r_sh = fsolve(r_shear_solver, r_shear_guess)[0]
        else:
            self.r_sh = 3.**(1./3) * self.r_hill * self.tau_s**(1./3)
        return self.r_sh


    def r_accretion(self):
        """ Calculates accretion radius. """
        min_1 = np.minimum(self.r_ws, self.r_sh)
        self.r_stab = np.minimum(min_1, self.r_hill)

        self.r_atm = np.minimum(self.r_bondi, self.r_hill) # Calculates atmospheric radius as minimum of shearing and hill radii. Mickey currently has this coded as maximum, but should be minimum
        self.r_acc = np.maximum(self.r_atm, self.r_stab) # Calculates Accretion radius as maximum of bondi radius and stability radius
        return self.r_acc


    def set_velocities(self):
        """ Calculates a bunch of velocities, given particle size s. """
        self.v_cross = fn.vkep(self.m_core, self.r_acc) # Orbit velocity about core

        self.v_obj_phi = -self.eta * self.v_kep * (1/(1 + self.tau_s**2)) # Object velocity relative to Keplerian #np.abs(self.v_phi_lam + self.v_gas_lam)
        self.v_core_phi = -self.eta * self.v_kep * (1/(1 + self.tau_s_core**2)) # Core velocity relative to Keplerian #np.abs(self.v_phi_lam_core + self.v_gas_lam)

        self.v_lam_iner = np.sqrt((self.v_obj_phi - self.v_core_phi)**2 + self.v_r_lam**2) # Laminar, relative to inertial frame
        #self.v_obj_turb = np.sqrt(self.v_gas_turb**2 - self.v_obj_gas_turb**2) # WHICH FRAME IS THIS?
        #self.v_obj = np.sqrt(self.v_obj_turb**2 + self.v_lam_iner**2) # Total body velocity relative to Keplerian

        self.v_obj_core_lam = np.sqrt((self.v_obj_phi - self.v_core_phi)**2 + (self.v_r_lam - self.v_r_lam_core)**2) # Velocity of object relative to core
        self.v_obj_core_turb = self.v_gas_turb * d_V_12(t_1=self.tau_s_core,t_2=self.tau_s,t_L=1,Re=self.re_f) # Use expresioons from Ormel Cuzzi 2007 to get relative turbulent particle velocity from stopping time of core #np.sqrt(self.v_gas_turb**2 - self.v_core_gas_turb**2)
        self.v_obj_core = np.sqrt(self.v_obj_core_lam**2 + self.v_obj_core_turb**2)
        self.v_shear = self.r_acc * self.om # Shear velocity

        self.v_inf = np.maximum(self.v_obj_core, self.v_shear) # Sets v_infinity
        self.v_enc = fn.G * self.m_core/self.r_acc/self.v_inf # Applies impulse approx to calculate encounter velocity

        if self.v_inf > self.v_cross: # Checks if impulse approximation is OK
            self.v_gas_enc = max(self.v_enc, self.v_obj_gas)
            self.v_grav = self.v_enc
        else:
            self.v_gas_enc = max(self.v_cross, self.v_obj_gas)
            self.v_grav = self.v_cross


    def encounter(self, s, rho_obj=2.):
        """ Calculate drag force, work, and kinetic energy during encounter, given particle size s. 
            Also calculates the accretion probability. """
        self.re_enc = fn.rey(s, self.v_gas_enc, self.v_th, self.mfp) # Reynolds number during encounter
        self.dc_enc = fn.drag_c(self.re_enc) # Drag coefficient during encounter
        self.f_drag_enc = fn.drag_force(s, self.v_gas_enc, self.dc_enc, self.rho_gas, self.mfp, self.v_th) # Drag force
        self.work_enc = 2 * self.f_drag_enc * self.r_acc # Work done by drag over course of encounter
        self.ke = .5 * self.m_obj * self.v_inf**2 # Kinetic energy of object during encounter

        # Modify growth time by the ratio of the kinetic energy to work done over one orbit
        if (self.r_acc == self.r_hill and self.v_inf == self.v_shear and self.extend_rh):
            self.prob = min(self.work_enc/self.ke, 1.)
        else:
            self.prob = 1.


    def scale_heights(self):
        """ Calculate the turbulent scale height, Kelvin-Helmholtz scale height, disk scale height,
            accretion height, and accretion area. """
        if not(self.alpha_z): # Checks if vertical turbulence is different from other directions
            self.alpha_z = self.alpha
        self.H_turb = min(np.sqrt(self.alpha_z/self.tau_s) * self.H, self.H) # Turbulent scale height
        # Kelvin-Helmholtz scale height
        self.H_KH = fn.acc_height(self.H, self.a_core) * self.h_mod * min(1., 1/np.sqrt(self.tau_s))
        self.H_disk = np.maximum(self.H_turb, self.H_KH) # Scale height of disk, also called h_p

        self.H_acc = np.minimum(self.H_disk, self.r_acc) # Accretion height
        self.area_acc = 4 * self.r_acc * self.H_acc # Area over which objects are accreted


    def t_accretion(self):
        """ Calculate the accretion/growth time for an object of size s. """
        self.t_acc = (fn.growth_time(self.m_core, self.H_disk, self.sig_solid, self.area_acc, self.v_inf)
                      * fn.sec_to_years/self.prob)

        # Check energy criterion for accretion
        if (((self.r_stab > self.r_bondi and self.work_enc < self.ke) or (self.r_stab < self.r_bondi and
              self.work_enc > self.ke)) and not
           (self.r_stab == self.r_hill and self.v_inf == self.v_shear and self.extend_rh and self.work_enc < self.ke)):
            self.t_acc = 0 # Really should be infinity, but set to 0 for easy plotting
        return self.t_acc

    def main(self, s, rho_obj=2.):
        """ Runs each method defined for this class, in order to calculate and set all of the attributes
            of the object. """
        t_s = self.t_stop(s, rho_obj)
        t_s_core = self.t_stop_core()
        r_ws = self.r_wish(s, rho_obj)
        r_sh = self.r_shear(s, rho_obj)
        r_acc = self.r_accretion()
        self.set_velocities()
        self.encounter(s, rho_obj)
        self.scale_heights()
        t_acc = self.t_accretion()

# Cut flags: verbose=0, focus_max=0, shear_off=0, lam_vel=0):