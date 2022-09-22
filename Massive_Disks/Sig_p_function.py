def sig_p(Sig_g, M_star, f_d, a_PF, a, t):
    eps_p = 0.5
    eps_d = 0.5/10 #Growth efficiency, eps_d = eps_g,d/xi from eqn ~9 of LJ14
    om = np.sqrt(fn.G * M_star * fn.m_sun/(a*fn.au)**3)
    a_applicable = np.trim_zeros(np.where(a < a_PF, a, 0))
    Sig_p = 2**(5/6) * 3**(-7/12) * eps_d**(1/3)/eps_p**(1/2) * f_d**(5/6) * om**(-1/6) * t**(-1/6) * Sig_g
    
    T = T_0 * (a**(-3/7))
    c_s = np.sqrt((boltzmann*T)/(2.35*mH))
    v_kep = om *(a * fn.au) #np.sqrt(fn.G * m_star*fn.m_sun/(a * fn.au))
    eta = (c_s)**2/(2*(v_kep)**2)
    i_PF = np.argmin(abs(a - a_PF))
    Sig_p0_PF = f_d*Sig_g[i_PF]
    
    for i, a_au in enumerate(a):
        s_old = np.sqrt(3)/8 * Sig_p[i]/(eta[i]*rho_int)
        St = st_rad(rad=s_old,alph=alpha,a_au=a_au,m_suns=M_star,m_earths=1,sig_in=Sig_g[i],temp_in=T[i])
        Sig_p[i] = (3/16)**(2/3) * ((fn.G * M_star*fn.m_sun)**(2/3) * (eps_d * f_d)**(4/3) * (Sig_p0_PF) * t**(1/3))/(3 * St * eta[i] * v_kep[i] * a_au * fn.au)
        
        j = 0
        delta = 1
        while abs(delta) > 0.01:
            if j != 0: 
                s_old = s
            s = 0.34*s_old + 0.66*np.sqrt(3)/8 * Sig_p[i]/(eta[i]*rho_int)
            St = st_rad(rad=s,alph=alpha,a_au=a_au,m_suns=M_star,m_earths=1,sig_in=Sig_g[i],temp_in=T[i])
            Sig_p[i] = (3/16)**(2/3) * ((fn.G * M_star*fn.m_sun)**(2/3) * (eps_d * f_d)**(4/3) * (Sig_p0_PF) * t**(1/3))/(3 * St * eta[i] * v_kep[i] * a_au * fn.au)
            delta = (s - s_old)/s
            j = j + 1
    
    Sig_p[a_applicable.size:] = 1e-2 * Sig_g[a_applicable.size:]
    return Sig_p