import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import rebound
from matplotlib.ticker import FormatStrFormatter
import time
from operator import attrgetter
import scipy.optimize as op
from matplotlib.patches import Ellipse

plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['legend.borderpad'] = 0.5
plt.rcParams['legend.labelspacing'] = 0.1
plt.rcParams['legend.handletextpad'] = 0.1
plt.rcParams['font.family'] = 'stixgeneral'
plt.rcParams['font.size'] = 18
mpl.rcParams['legend.numpoints'] = 1
plt.rc('lines', linewidth=1.0)
colors = ['#4D4D4D','#5DA5DA', '#FAA43A', '#60BD68', '#F17CB0','#B2912F','#B276B2','#DECF3F','#F15854']
#  (blue)
# (orange)
# (green)
# (pink)
#  (brown)
# (purple)
#  (yellow)
# (red)
# ']
mpl.rcParams['axes.color_cycle'] = colors

def ls_sums(f,times,obs,w_i):
    """Intermediate function for calculating the sums in the LS periodogram. See Zechmeister and Kurster (2009) for
    details"""

    omega = 2*np.pi*f
    C = np.sum(w_i*np.cos(omega*times))
    S = np.sum(w_i*np.sin(omega*times))
    YC_hat = np.sum(w_i*obs*np.cos(omega*times))
    YS_hat = np.sum(w_i*obs*np.sin(omega*times))
    CC_hat = np.sum(w_i*np.cos(omega*times)**2.)
    SS_hat = np.sum(w_i*np.sin(omega*times)**2.)
    CS_hat = np.sum(w_i*np.cos(omega*times)*np.sin(omega*times))

    return [C,S,YC_hat,YS_hat,CC_hat,SS_hat,CS_hat]

def ls_period(freqs,times,obs,errs):
    """Return the LS periodogram for a given set of data, given as (times,obs), with errors errs. The power is calculated
    at the frequencies that are given to the function. Uses a 'floating mean,' and weights the data points by the square
    of the error bars. See Zechmeister and Kurster (2009) for the formuals as well as a more in depth explanation"""

    W = np.sum(1/errs**2.)
    w_i = 1./(W*errs**2.)
    Y = np.sum(w_i*obs)
    YY_hat = np.sum(w_i*obs*obs)
    YY = YY_hat - Y*Y

    powers = np.zeros(len(freqs))
    phases = np.zeros(len(freqs))

    for i,f in enumerate(freqs):
        [C,S,YC_hat,YS_hat,CC_hat,SS_hat,CS_hat] = ls_sums(f,times,obs,w_i)
        YC = YC_hat - Y*C
        YS = YS_hat - Y*S
        CC = CC_hat - C*C
        SS = SS_hat - S*S
        CS = CS_hat - C*S
        D = CC*SS-CS**2.

        a = (YC*SS - YS*CS)/D
        b = (YS*CC - YC*CS)/D
        phases[i] = np.arctan2(b,a)
        powers[i] = (YY*D)**(-1.)*(SS*YC**2. + CC*YS**2. - 2*CS*YC*YS)

    return powers,phases

class RVPlanet:

    """Class creating planets, used to add planets to RVSystem"""
    
    def __init__(self, per=365.25, mass=1, M=0, e=0, pomega=0, i=90.,Omega=0):
        M_J = 9.5458e-4
        
        self.per = per
        self.mass = mass*M_J
        self.M = M
        self.e = e
        self.pomega = pomega
        self.i = i
        self.Omega = Omega
        self.l = M + pomega


class RVSystem(RVPlanet):

    """Main class for RV Simulations"""
    
    def __init__(self,mstar=1.0):
        
        self.mstar = mstar #Mass of central star
        self.planets = np.array([]) #Array containing RVPlanets class
        self.RV_files = [] #Array of RV velocities, assumed to be of the form: JD, RV, error
        self.offsets = [] #Array of constant velocity offsets for each data set
        self.path_to_data = "" #Optional prefix that points to location of datasets
        self.RV_data=[]
    
    def sort_data(self):    
        JDs = []
        vels = []
        errs = []
        dataset=[]

        for i,fname in enumerate(self.RV_files):
            tmp_arr = np.loadtxt(self.path_to_data + fname)
            JDs = np.concatenate((JDs,tmp_arr[:,0]))
            vels = np.concatenate((vels,(tmp_arr[:,1]-self.offsets[i])))
            errs = np.concatenate((errs,tmp_arr[:,2]))
            dataset=np.concatenate((dataset,i*np.ones(len(tmp_arr[:,0]))))

        #There might be a better way to do this -- these commands sort the data by time so that we can integrate
        #up to each time
        sort_arr = [JDs,vels,errs,dataset]
        sort_arr = np.transpose(sort_arr)
        self.RV_data = sort_arr[np.argsort(sort_arr[:,0])]
    
    def add_planet(self,per=365.25, mass=1, M=0, e=0, pomega=0, i=90.,Omega=0):
        """Add planet to RV simulation. Angles are in degrees, planet mass is in Jupiter masses"""
        self.planets = np.append(self.planets,RVPlanet(per,mass,M,e,pomega,i,Omega))
        # self.planets.append(RVPlanet(per,mass,M,e,pomega,i,Omega))
        
    def semi_maj(self):

        """List semi-major axes of all planets in simulation"""

        G = 6.674e-8
        JD_sec = 86400.0
        msun_g = 1.989e33
        AU = 1.496e13
            
        for i,planet in enumerate(self.planets):
            r = (G * (self.mstar)*msun_g * (planet.per*JD_sec)**2/(4.*np.pi**2))**(1./3.)
            r_AU = r/AU
            print( "a_%i = %.3f AU" %(i,r_AU))

    def plot_RV(self,epoch=2450000,save_plot=0,filename="tmp.pdf",plot_theo=1,plot_data=1,save_RV=0,jacobi=0):

        """Make a plot of the RV time series with data and integrated curve"""

        JDs = []
        vels = []
        errs = []

        for i,fname in enumerate(self.RV_files): #Read in RV data
            tmp_arr = np.loadtxt(self.path_to_data + fname)
            JDs.append(tmp_arr[:,0])
            vels.append(tmp_arr[:,1]-self.offsets[i])
            errs.append(tmp_arr[:,2])

        #Intialize Rebound simulation
        deg2rad = np.pi/180.
        sim = rebound.Simulation()
        sim.units = ('day', 'AU', 'Msun')
        sim.t = epoch #Epoch is the starting time of simulation
        sim.add(m=self.mstar,hash='star')


        min_per = np.inf
        if jacobi:
            per_arr = np.array([planet.per for planet in self.planets])
            for planet in self.planets[np.argsort(per_arr)]:
                sim.add(m=planet.mass,P=planet.per,M=planet.M*deg2rad,e=planet.e,pomega=planet.pomega*deg2rad,
                    inc=planet.i*deg2rad,Omega=planet.Omega*deg2rad)
                min_per = min(min_per,planet.per) #Minimum period, used for plotting purposes
        else:
            for planet in self.planets: #Add planets in self.planets to Rebound simulation
                sim.add(primary=sim.particles[0],m=planet.mass,P=planet.per,M=planet.M*deg2rad,e=planet.e,pomega=planet.pomega*deg2rad,
                    inc=planet.i*deg2rad,Omega=planet.Omega*deg2rad)
                min_per = min(min_per,planet.per) #Minimum period, used for plotting purposes

        
        JD_max = max(np.amax(JDs[i]) for i in range(len(self.RV_files)))
        JD_min = min(np.amin(JDs[i]) for i in range(len(self.RV_files)))
        
        
        Noutputs = int((JD_max-JD_min)/min_per*100.)
        
        sim.move_to_com()
        ps = sim.particles

        times = np.linspace(JD_min, JD_max, Noutputs)
        AU_day_to_m_s = 1.731456e6 #Conversion factor from Rebound units to m/s

        rad_vels = np.zeros(Noutputs)

        for i,t in enumerate(times): #Perform integration
            sim.integrate(t)
            rad_vels[i] = -ps['star'].vz * AU_day_to_m_s

        fig = plt.figure(1,figsize=(11,6)) #Plot RV

        if plot_theo:
            plt.plot(times,rad_vels)

        if plot_data:
            for i in range(len(self.RV_files)):
                plt.errorbar(JDs[i],vels[i],yerr = errs[i],fmt='o')

        plt.xlabel("Time [JD]")
        plt.ylabel("RV [m/s]")
        ax = plt.gca()
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

        plt.show()
        
        if save_plot:
            fig.savefig(filename,format='pdf')
            
        if save_RV:
            np.savetxt('tmp.txt',[times,rad_vels])

    def calc_chi2(self,epoch=2450000,jacobi=0):

        """Calculate the chi^2 value of the RV time series for the planets currently in the system"""

       

        deg2rad = np.pi/180.
        sim = rebound.Simulation()
        sim.units = ('day', 'AU', 'Msun')
        sim.t = epoch
        sim.add(m=self.mstar,hash='star')


        if jacobi:
            per_arr = np.array([planet.per for planet in self.planets])
            for planet in self.planets[np.argsort(per_arr)]:
                sim.add(m=planet.mass,P=planet.per,M=planet.M*deg2rad,e=planet.e,pomega=planet.pomega*deg2rad,
                    inc=planet.i*deg2rad,Omega=planet.Omega*deg2rad)
        else:
            for planet in self.planets: #Add planets in self.planets to Rebound simulation
                sim.add(primary=sim.particles[0],m=planet.mass,P=planet.per,M=planet.M*deg2rad,e=planet.e,pomega=planet.pomega*deg2rad,
                    inc=planet.i*deg2rad,Omega=planet.Omega*deg2rad)

        sim.move_to_com()
        ps = sim.particles

        times = self.RV_data[:,0] #Times to integrate to are just the times for each data point, no need to integrate
        #between data points

        AU_day_to_m_s = 1.731456e6
        rad_vels = np.zeros(len(times))

        for i,t in enumerate(times):
            sim.integrate(t)
            rad_vels[i] = -ps['star'].vz * AU_day_to_m_s

        chi_2 = 0

        for i,vel_theory in enumerate(rad_vels):
                chi_2 += (self.RV_data[i,1]-vel_theory)**2/self.RV_data[i,2]**2

        return chi_2

    def log_like(self,epoch=2450000,jitter=0,jacobi=0):

        """Calculate the log likelihood for MCMC"""

        deg2rad = np.pi/180.
        sim = rebound.Simulation()
        sim.units = ('day', 'AU', 'Msun')
        sim.t = epoch
        sim.add(m=self.mstar,hash='star')

        if jacobi:
            per_arr = np.array([planet.per for planet in self.planets])
            for planet in self.planets[np.argsort(per_arr)]:
                sim.add(m=planet.mass,P=planet.per,M=planet.M*deg2rad,e=planet.e,pomega=planet.pomega*deg2rad,
                    inc=planet.i*deg2rad,Omega=planet.Omega*deg2rad)
        else:
            for planet in self.planets: #Add planets in self.planets to Rebound simulation
                sim.add(primary=sim.particles[0],m=planet.mass,P=planet.per,M=planet.M*deg2rad,e=planet.e,pomega=planet.pomega*deg2rad,
                    inc=planet.i*deg2rad,Omega=planet.Omega*deg2rad)

        sim.move_to_com()
        ps = sim.particles

        times = self.RV_data[:,0] #Times to integrate to are just the times for each data point, no need to integrate
        #between data points

        AU_day_to_m_s = 1.731456e6
        rad_vels = np.zeros(len(times))

        for i,t in enumerate(times):
            sim.integrate(t)
            rad_vels[i] = -ps['star'].vz * AU_day_to_m_s
            
        return -np.sum(0.5*(self.RV_data[:,1]-rad_vels)**2/(self.RV_data[:,2]**2+jitter**2) + np.log(np.sqrt(2*np.pi*(self.RV_data[:,2]**2+jitter**2))))

    def rem_planet(self,i=0):
        del self.planets[i]

    def clear_planets(self):
        self.planets = []

    def orbit_stab(self,periods=1e4,pnts_per_period=5,outputs_per_period=1,verbose=0,integrator='whfast',safe=1,
                   timing=0,save_output=0,plot=0,energy_err=0,jacobi=0):


        deg2rad = np.pi/180.
        sim = rebound.Simulation()
        sim.integrator = integrator
        exact = 1
        if integrator != 'ias15':
            exact = 0
        sim.units = ('day', 'AU', 'Msun')
        # sim.t = epoch #Epoch is the starting time of simulation
        sim.add(m=self.mstar,hash='star')

      
        min_per = min(self.planets,key=attrgetter('per')).per
        max_per = max(self.planets,key=attrgetter('per')).per

        if jacobi:
            per_arr = np.array([planet.per for planet in self.planets])
            for planet in self.planets[np.argsort(per_arr)]:
                sim.add(m=planet.mass,P=planet.per,M=planet.M*deg2rad,e=planet.e,pomega=planet.pomega*deg2rad,
                    inc=planet.i*deg2rad,Omega=planet.Omega*deg2rad)
        else:
            for planet in self.planets: #Add planets in self.planets to Rebound simulation
                sim.add(primary=sim.particles[0],m=planet.mass,P=planet.per,M=planet.M*deg2rad,e=planet.e,pomega=planet.pomega*deg2rad,
                    inc=planet.i*deg2rad,Omega=planet.Omega*deg2rad)
          
      
        
        t_max = max_per*periods
        Noutputs = int(t_max/min_per*outputs_per_period)
        times = np.linspace(0,t_max, Noutputs)

        sim.move_to_com()
        sim.dt = min_per/pnts_per_period
        ps = sim.particles[1:]

        if plot:
            semi_major_arr = np.zeros((len(ps),Noutputs))

            print (Noutputs)

        if timing:
            start_time = time.time()

        if energy_err:
            E0 = sim.calculate_energy()

        # if not(safe):
        #     sim.ri_whfast.safe_mode = 0

        a0 = [planet.a for planet in ps]

        stable = 1

        for i,t in enumerate(times): #Perform integration
            sim.integrate(t,exact_finish_time = exact)
            for k,planet in enumerate(ps):
                if (np.abs((a0[k]-planet.a)/a0[k])>10) or planet.a < 0.1:
                    stable = 0
                if plot:
                    semi_major_arr[k,i] = planet.a
            if verbose and (i % (Noutputs/10) == 0):
                print ("%3i %%" %(float(i+1)/float(Noutputs)*100.))
                print ("%2i %%" %(100*i/Noutputs))

            if stable == 0:
                break

        if timing:
            print ("Integration took %.5f seconds" %(time.time() - start_time))

        if energy_err:
            Ef = sim.calculate_energy()
            print( "Energy Error is %.3f%% " %(np.abs((Ef-E0)/E0*100)))


        if plot:
            plt.figure(1,figsize=(11,6))

            for i in range(len(ps)):
                plt.semilogx(times/365.25,semi_major_arr[i])

            plt.xlabel("Time [Years]")
            plt.ylabel("a [AU]")
           

        return stable

    def stab_logprob(self,epoch=2450000):
        stable = self.orbit_stab(periods=1e4,pnts_per_period=10,outputs_per_period=1)
        if stable:
            return self.log_like(epoch=epoch)
        else:
            return -np.inf
        
    def plot_phi(self,p=2.,q=1.,pert_ind=0,test_ind=1,periods=1e2,pnts_per_period=100.,
                outputs_per_period=20.,verbose=0,log_t = 0, integrator='whfast',plot=1):

        deg2rad = np.pi/180.
        sim = rebound.Simulation()
        sim.integrator = integrator
        exact = 1
        if integrator != 'ias15':
            exact = 0
        sim.units = ('day', 'AU', 'Msun')
        # sim.t = epoch #Epoch is the starting time of simulation
        sim.add(m=self.mstar,hash='star')

        min_per = np.inf
        max_per = 0

        res_inds = [pert_ind,test_ind]

        per_max = 0
        per_min = np.inf

        for i,planet in enumerate(self.planets): #Add planets in self.planets to Rebound simulation
            sim.add(m=planet.mass,P=planet.per,M=planet.M*deg2rad,e=planet.e,pomega=planet.pomega*deg2rad,
                    inc=planet.i*deg2rad,Omega=planet.Omega*deg2rad)
            if i in res_inds:
                if planet.per > per_max:
                    outer = i
                    per_max = planet.per
                if planet.per < per_min:
                    inner = i
                    per_min = planet.per


            min_per = min(min_per,planet.per) #Minimum period
            max_per = max(max_per,planet.per)

        t_max = max_per*periods
        Noutputs = int(t_max/min_per*outputs_per_period)
        times = np.linspace(0,t_max, Noutputs)

        sim.move_to_com()
        sim.dt = min_per/pnts_per_period
        ps = sim.particles[1:]

        pert = ps[pert_ind]
        test = ps[test_ind]
        outer = ps[outer]
        inner = ps[inner]

        phi_arr = np.zeros(Noutputs)

        for i,t in enumerate(times): #Perform integration
            sim.integrate(t,exact_finish_time = exact)
            phi_arr[i] = ((p+q)*outer.l - p*inner.l - q*test.pomega)%(2*np.pi)

        angle_fixed = lambda phi: phi-2*np.pi if phi>np.pi else phi
        phi_arr = [angle_fixed(phi) for phi in phi_arr]

        if plot:
            plt.figure(1,figsize=(11,6))

            if log_t:
                plt.semilogx(times/365.25,phi_arr)
            else:
                plt.plot(times/365.25,phi_arr)

            plt.xlabel("Time [Years]")
            plt.ylabel(r"$\phi$ [deg]")

        return times, phi_arr

        print (inner,outer)

    def save_params(self,fname):

        param_arr = np.zeros((len(self.planets),7))

        M_J = 9.5458e-4

        for i,planet in enumerate(self.planets):
            arr_tmp = [planet.per, planet.mass/M_J, planet.M, planet.e, planet.pomega, planet.i, planet.Omega]
            param_arr[i] = arr_tmp

        np.savetxt(fname,param_arr)
        np.savetxt(fname + "_offsets",self.offsets)

    def plot_planet_RV(self,epoch=2450000,jacobi=0):

        """Make a plot of the RV time series for the star and planets"""

        #Intialize Rebound simulation
        deg2rad = np.pi/180.
        sim = rebound.Simulation()
        sim.units = ('day', 'AU', 'Msun')
        sim.t = epoch #Epoch is the starting time of simulation
        sim.add(m=self.mstar,hash='star')


        min_per = np.inf
        max_per = 0

        if jacobi:
            per_arr = np.array([planet.per for planet in self.planets])
            for planet in self.planets[np.argsort(per_arr)]:
                sim.add(m=planet.mass,P=planet.per,M=planet.M*deg2rad,e=planet.e,pomega=planet.pomega*deg2rad,
                    inc=planet.i*deg2rad,Omega=planet.Omega*deg2rad)
        else:
            for planet in self.planets: #Add planets in self.planets to Rebound simulation
                sim.add(primary=sim.particles[0],m=planet.mass,P=planet.per,M=planet.M*deg2rad,e=planet.e,pomega=planet.pomega*deg2rad,
                    inc=planet.i*deg2rad,Omega=planet.Omega*deg2rad)
            min_per = min(min_per,planet.per) #Minimum period
            max_per = max(max_per,planet.per)

        sim.move_to_com()
        ps = sim.particles

        Noutputs = 1000
        times = np.linspace(0,10*max_per,Noutputs)

        AU_day_to_m_s = 1.731456e6 #Conversion factor from Rebound units to m/s

        rad_vels = np.zeros((len(sim.particles),Noutputs))


        for i,t in enumerate(times): #Perform integration
            sim.integrate(t)
            rad_vels[0,i] = -ps['star'].vz * AU_day_to_m_s
            for j,plan in enumerate(ps[1:]):
                rad_vels[j+1,i] = plan.vz * AU_day_to_m_s * (self.planets[j].mass/self.mstar)
            print (i)

        fig = plt.figure(1,figsize=(11,6)) #Plot RV

        plt.plot(times,rad_vels[0])

        for i in range(len(ps[1:])):
            plt.plot(times,rad_vels[i+1],linestyle='dashed')

        plt.xlabel("Time [JD]")
        plt.ylabel("RV [m/s]")
        ax = plt.gca()
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

        plt.show()


    def plot_RV_kep(self,epoch=2450000,save_plot=0,filename="",plot_data=1,save_RV=0,t_p=0):

        """Make a plot of the RV time series with data and integrated curve"""

       

        min_per = np.inf
        for planet in self.planets:
            min_per = min(min_per,planet.per) #Minimum period


        Noutputs = int((self.RV_data[-1,0]-self.RV_data[0,0])/min_per*100.)

        #times = np.linspace(self.RV_data[0,0], self.RV_data[-1,0], Noutputs)
        times = np.linspace(0,self.RV_data[-1,0]-self.RV_data[0,0], Noutputs)
        rad_vels = np.zeros(Noutputs)

        deg2rad = np.pi/180.
        if t_p:
            t_p_arr = [t_peri-epoch for t_peri in t_p]
            #for i,planet in enumerate(self.planets):
                #planet.M = 2*np.pi/(planet.P)*(epoch-t_p[i])
        else:
            t_p_arr = [-planet.M*deg2rad/2./np.pi*planet.per + planet.per for planet in self.planets]

        def kep_sol(t=0,t_p=0,e=0.1,per=100.):
            n = 2*np.pi/per
            M = n*(t-t_p)

            kep = lambda E: M - E + e*np.sin(E)

            E = op.fsolve(kep,M)

            return 2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E/2))%(2*np.pi)

        def RV_amp(m_star = 1.0, m_p = 9.5458e-4, omega = 0., i = np.pi/2., per = 365.25, f = 0., e = 0.0):
            omega = omega*deg2rad
            G = 6.67259e-11
            m_sun = 1.988435e30
            JD_sec = 86400.0

            m_star_mks = m_star*m_sun
            m_p_mks = m_p*m_sun

            per_mks = per*JD_sec
            n = 2.*np.pi/per_mks
            a = (G*(m_star_mks + m_p_mks)/n**2.)**(1./3.)

            return np.sqrt(G/(m_star_mks + m_p_mks)/a/(1-e**2.))*(m_p_mks*np.sin(i))*(np.cos(omega+f)+e*np.cos(omega))


        for i,t in enumerate(times):
            rv = 0
            for j,planet in enumerate(self.planets):
                f = kep_sol(t=t,t_p=t_p_arr[j],e=planet.e,per=planet.per)
                rv += RV_amp(m_star = self.mstar, m_p = planet.mass, per = planet.per, f = f, e = planet.e,
                             omega=planet.pomega)
            rad_vels[i] = rv


        fig = plt.figure(1,figsize=(11,6)) #Plot RV

        plt.plot(times+epoch,rad_vels)
        
        JDs = []
        vels = []
        errs = []

        for i,fname in enumerate(self.RV_files): #Read in RV data
            tmp_arr = np.loadtxt(self.path_to_data + fname)
            JDs.append(tmp_arr[:,0])
            vels.append(tmp_arr[:,1]-self.offsets[i])
            errs.append(tmp_arr[:,2])
        
        
        if plot_data:
            for i in range(len(self.RV_files)):
                plt.errorbar(JDs[i],vels[i],yerr = errs[i],fmt='o')

        plt.xlabel("Time [JD]")
        plt.ylabel("RV [m/s]")
        ax = plt.gca()
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))

        plt.show()

        if save_plot:
            fig.savefig(filename)
            print( "Saved")
            
        if save_RV:
            np.savetxt('tmp.txt',[times+epoch,rad_vels])

    def calc_chi2_kep(self,epoch=2450000):

        """Calculate the chi^2 value of the RV time series for the planets currently in the system"""

        

        times = self.data[:,0]-epoch #Times to integrate to are just the times for each data point, no need to integrate
        #between data points
        rad_vels = np.zeros(len(times))

        min_per = np.inf
        for planet in self.planets:
            min_per = min(min_per,planet.per) #Minimum period

        deg2rad = np.pi/180.
        t_p_arr = [-planet.M*deg2rad/2./np.pi*planet.per + planet.per for planet in self.planets]

        def kep_sol(t=0,t_p=0,e=0.1,per=100.): #Solve Kepler's Equation for f eccentric anomaly, true anomaly
            n = 2*np.pi/per
            M = n*(t-t_p)

            kep = lambda E: M - E + e*np.sin(E)

            E = op.fsolve(kep,M)

            return 2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E/2))%(2*np.pi)

        def RV_amp(m_star = 1.0, m_p = 9.5458e-4, omega = 0., i = np.pi/2., per = 365.25, f = 0., e = 0.0):
            omega = omega*deg2rad
            G = 6.67259e-11
            m_sun = 1.988435e30
            JD_sec = 86400.0

            m_star_mks = m_star*m_sun
            m_p_mks = m_p*m_sun

            per_mks = per*JD_sec
            n = 2.*np.pi/per_mks
            a = (G*(m_star_mks + m_p_mks)/n**2.)**(1./3.)

            return np.sqrt(G/(m_star_mks + m_p_mks)/a/(1-e**2.))*(m_p_mks*np.sin(i))*(np.cos(omega+f)+e*np.cos(omega))


        for i,t in enumerate(times):
            rv = 0
            for j,planet in enumerate(self.planets):
                f = kep_sol(t=t,t_p=t_p_arr[j],e=planet.e,per=planet.per)
                rv += RV_amp(m_star = self.mstar, m_p = planet.mass, per = planet.per, f = f, e = planet.e,
                             omega=planet.pomega)
            rad_vels[i] = rv

        chi_2 = 0

        for i,vel_theory in enumerate(rad_vels):
                chi_2 += (self.data[i,1]-vel_theory)**2/self.data[i,2]**2

        return chi_2

    def periodogram(self,epoch=2450000,pnts=int(1e5),plot_per=1,plot_range=0,save_per=0):
        """Make periodograms to analyze RV data. Currently returns an LS periodogram of the data,the "window function,"
        which is essentially the Fourier transform of the measurement times, and the periodogram of the residuals"""
        if not(plot_range): #Flag to specify plotted range as an input, otherwise defaults are below
            plot_range = [-0.3,4] if plot_per else [1e-3,4.0]
        if plot_per: #Flag to plot period instead of frequency
            periods = np.logspace(plot_range[0],plot_range[1],num=int(pnts))
            freqs = periods**(-1.)
            log_per = np.log10(periods)
            x_data = log_per
            x_lab = r'$\log P$ [d]'
        else:
            freqs = np.linspace(plot_range[0],plot_range[1],num=int(pnts))
            x_data = freqs
            x_lab = 'Freq [1/day]'


        times = self.RV_data[:,0]
        obs = self.RV_data[:,1]
        errs = self.RV_data[:,2]

        powers,phases = ls_period(freqs,times,obs,errs) #LS periodogram for the data. Formulas are from Zechmeister and
                                                        #Kurster (2009)

        #Calculate the window function. See Dawson and Fabrycky (2010). Currently done in a non-FFT way, there's probably
        #a faster way to do it
        window = np.zeros(len(freqs),dtype=np.complex)
        N = len(times)

        for i,f in enumerate(freqs):
            # f = 1./p
            w = 0
            for t in times:
                w+=np.exp(-2*np.pi*1j*f*t)
            window[i] = w/N

        win_powers = np.abs(window)
        # win_phases = np.arctan2(window.imag,window.real)
        win_phases = np.angle(window)

        #Calculate the RVs of the planets in order to calculate the residuals
        deg2rad = np.pi/180.
        sim = rebound.Simulation()
        sim.units = ('day', 'AU', 'Msun')
        sim.t = epoch
        sim.add(m=self.mstar,hash='star')

        for planet in self.planets:
            sim.add(m=planet.mass,P=planet.per,M=planet.M*deg2rad,e=planet.e,pomega=planet.pomega*deg2rad,
                    inc=planet.i*deg2rad,Omega=planet.Omega*deg2rad)

        sim.move_to_com()
        ps = sim.particles

        AU_day_to_m_s = 1.731456e6
        rad_vels = np.zeros(len(times))

        for i,t in enumerate(times):
            sim.integrate(t)
            rad_vels[i] = -ps['star'].vz * AU_day_to_m_s

        #Periodogram of the residuals
        res = obs - rad_vels
        res_powers,res_phases = ls_period(freqs,times,res,errs)


        #Make the LS periodogram
        axis_format = '%.2f'

        fig = plt.figure(1,figsize=(6,6))
        plt.title('Periodogram')

        plt.plot(x_data,powers)
        plt.axis([min(x_data),max(x_data),0,max(powers)*1.2])
        plt.xlabel(x_lab)
        plt.ylabel('Power')

        #Everything here is for plotting the "dials," which indicate the phase at the peaks. This code seems way too
        #complicated. If someone can pretty it up that'd be much appreciated.
        ax = plt.gca()
        ax.xaxis.set_major_formatter(FormatStrFormatter(axis_format))
        dx = ax.get_xlim()[1] - ax.get_xlim()[0]
        dy = ax.get_ylim()[1] - ax.get_ylim()[0]

        grad = np.gradient(powers,x_data[1]-x_data[0])
        asign = np.sign(grad)
        signchange = ((np.roll(asign, 1) - asign) == 2).astype(int) #Calculate where the gradient switches from + to -
                                                                    #to find local maxima

        power_ind = np.array([[0,0],[0,0],[0,0]],dtype='float64')

        max_inds = []
        search_per = 0.01 #How far from the maximum do we search?
        search_len = int(len(powers)*search_per)

        #Search near local max for actual highest value
        for ind in np.where(signchange==1)[0]:
            lower = np.maximum(ind-search_len,0)
            upper = np.minimum(ind+search_len,len(powers))

            max_ind = 0
            Max = 0
            for i in range(lower,upper):
                if powers[i] > Max:
                        Max = powers[i]
                        max_ind = i
            max_inds.append(max_ind)


        #Find the 3 peaks in the data to make dials on. I'm sure this can be coded up more elegantly
        for ind in max_inds:
            if powers[ind] > power_ind[0,1]:
                power_ind[2] = power_ind[1]
                power_ind[1] = power_ind[0]
                power_ind[0] = [ind,powers[ind]]
            elif powers[ind] > power_ind[1,1]:
                power_ind[2]= power_ind[1]
                power_ind[1] = [ind,powers[ind]]
            elif powers[ind] > power_ind[2,1]:
                power_ind[2] = [ind,powers[ind]]


        for i in range(len(power_ind)):
            ind = int(power_ind[i,0])

        #     cen = (freqs**(-1.))[ind]
            cen = x_data[ind]
            peak = powers[ind]
            phase = phases[ind]
            rad = 0.05
            w = rad*dx
            h = rad*dy
            y0 = dy*0.075+peak

            #Plot an ellipse that will look like a circle with the correct phase when viewed on the plot
            phase_plt = np.arctan2(dy*np.sin(phase),dx*np.cos(phase))
            ell = Ellipse((cen,y0), width=w, height = h,fill=False)
            r = 0.5*w*h/np.sqrt((h*np.cos(-phase_plt))**2.+(w*np.sin(-phase_plt))**2.)
            fs = np.linspace(cen,cen+r*np.cos(phase_plt),num=100)
            plt.plot(fs,y0+np.tan(phase_plt)*(fs-cen),linewidth=1.5,zorder=1,color=colors[1])

            ax.add_artist(ell)

        if save_per:
            fig.savefig('tmp.png',dpi=500)

        #Plot the window function. Code is essentially the same as above. This stuff should really all be combined into
        #one function, which is called to make the 3 plots.
        plt.figure(2,figsize=(6,6))
        plt.title('Window Function')
        plt.axis([min(x_data),max(x_data),0,max(win_powers)*1.2])
        plt.plot(x_data,win_powers)
        plt.xlabel(x_lab)
        plt.ylabel('Power')

        ax = plt.gca()
        ax.xaxis.set_major_formatter(FormatStrFormatter(axis_format))
        dx = ax.get_xlim()[1] - ax.get_xlim()[0]
        dy = ax.get_ylim()[1] - ax.get_ylim()[0]

        grad = np.gradient(win_powers,x_data[1]-x_data[0])
        asign = np.sign(grad)
        signchange = ((np.roll(asign, 1) - asign) == 2).astype(int)

        max_inds = []
        search_per = 0.01
        search_len = int(len(win_powers)*search_per)

        for ind in np.where(signchange==1)[0]:
            lower = np.maximum(ind-search_len,0)
            upper = np.minimum(ind+search_len,len(win_powers))

            max_ind = 0
            Max = 0
            for i in range(lower,upper):
                if win_powers[i] > Max:
                        Max = win_powers[i]
                        max_ind = i
            max_inds.append(max_ind)

        power_ind = np.array([[0,0],[0,0],[0,0]],dtype='float64')
        for ind in max_inds:
            if win_powers[ind] > power_ind[0,1]:
                power_ind[2] = power_ind[1]
                power_ind[1] = power_ind[0]
                power_ind[0] = [ind,win_powers[ind]]
            elif win_powers[ind] > power_ind[1,1]:
                power_ind[2]= power_ind[1]
                power_ind[1] = [ind,win_powers[ind]]
            elif win_powers[ind] > power_ind[2,1]:
                power_ind[2] = [ind,win_powers[ind]]


        for i in range(len(power_ind)):
            ind = int(power_ind[i,0])

        #     cen = (freqs**(-1.))[ind]
            cen = x_data[ind]
            peak = win_powers[ind]
            phase = win_phases[ind]
            rad = 0.05
            w = rad*dx
            h = rad*dy
            y0 = dy*0.075+peak


            phase_plt = np.arctan2(dy*np.sin(phase),dx*np.cos(phase))
            ell = Ellipse((cen,y0), width=w, height = h,fill=False)
            r = 0.5*w*h/np.sqrt((h*np.cos(-phase_plt))**2.+(w*np.sin(-phase_plt))**2.)
            fs = np.linspace(cen,cen+r*np.cos(phase_plt),num=100)
            plt.plot(fs,y0+np.tan(phase_plt)*(fs-cen),linewidth=1.5,zorder=1,color=colors[1])

            ax.add_artist(ell)

        #Plot the periodogram of the residuals. Same as the two above.
        plt.figure(3,figsize=(6,6))
        plt.title('Periodogram of Residuals')
        plt.axis([min(x_data),max(x_data),0,max(res_powers)*1.2])
        plt.plot(x_data,res_powers)
        plt.xlabel(x_lab)
        plt.ylabel('Power')

        ax = plt.gca()
        ax.xaxis.set_major_formatter(FormatStrFormatter(axis_format))
        dx = ax.get_xlim()[1] - ax.get_xlim()[0]
        dy = ax.get_ylim()[1] - ax.get_ylim()[0]

        grad = np.gradient(res_powers,x_data[1]-x_data[0])
        asign = np.sign(grad)
        signchange = ((np.roll(asign, 1) - asign) == 2).astype(int)

        power_ind = np.array([[0,0],[0,0],[0,0]],dtype='float64')

        max_inds = []
        search_per = 0.01
        search_len = int(len(res_powers)*search_per)

        for ind in np.where(signchange==1)[0]:
            lower = np.maximum(ind-search_len,0)
            upper = np.minimum(ind+search_len,len(res_powers))

            max_ind = 0
            Max = 0
            for i in range(lower,upper):
                if powers[i] > Max:
                        Max = res_powers[i]
                        max_ind = i
            max_inds.append(max_ind)

        for ind in max_inds:
            if res_powers[ind] > power_ind[0,1]:
                power_ind[2] = power_ind[1]
                power_ind[1] = power_ind[0]
                power_ind[0] = [ind,res_powers[ind]]
            elif res_powers[ind] > power_ind[1,1]:
                power_ind[2]= power_ind[1]
                power_ind[1] = [ind,res_powers[ind]]
            elif res_powers[ind] > power_ind[2,1]:
                power_ind[2] = [ind,res_powers[ind]]


        for i in range(len(power_ind)):
            ind = int(power_ind[i,0])

        #     cen = (freqs**(-1.))[ind]
            cen = x_data[ind]
            peak = res_powers[ind]
            phase = res_phases[ind]
            rad = 0.05
            w = rad*dx
            h = rad*dy
            y0 = dy*0.075+peak


            phase_plt = np.arctan2(dy*np.sin(phase),dx*np.cos(phase))
            ell = Ellipse((cen,y0), width=w, height = h,fill=False)
            r = 0.5*w*h/np.sqrt((h*np.cos(-phase_plt))**2.+(w*np.sin(-phase_plt))**2.)
            fs = np.linspace(cen,cen+r*np.cos(phase_plt),num=100)
            plt.plot(fs,y0+np.tan(phase_plt)*(fs-cen),linewidth=1.5,zorder=1,color=colors[1])

            ax.add_artist(ell)

        return [x_data,[powers,phases],[win_powers,win_phases],[res_powers,res_phases]]



# def like_wrap(params_opt,params_fixed,RVsys):
#
# #
# # def opt_params(RVsys,planet_num = 0,min_pars="p_m_e",replace_planet=1):
# #     if min_pars == "p_m_e":
# #         guesses = [RVsys.planets[planet_num].per,RVsys.planets[planet_num].mass,RVsys.planets[planet_num].e]
# #     elif min_pars == 'p_m'
# #         guesses = [RVsys.planets[planet_num].per,RVsys.planets[planet_num].mass,RVsys.planets[planet_num].e]
