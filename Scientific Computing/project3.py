"""Scientific Computation Project 3
02045099
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
#use scipy as needed

#===== Code for Part 1=====#

def plot_field(lat,lon,u,time,levels=20):
    """
    Generate contour plot of u at particular time
    Use if/as needed
    Input:
    lat,lon: latitude and longitude arrays
    u: full array of wind speed data
    time: time at which wind speed will be plotted (index between 0 and 364)
    levels: number of contour levels in plot
    """
    


    plt.figure()
    plt.contourf(lon,lat,u[time,:,:],levels)
    plt.axis('equal')
    plt.grid()
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    
    return None


def part1():#add input if needed
    """
    Code for part 1
    """ 

    #--- load data ---#
    d = np.load('data1.npz')
    lat = d['lat'];lon = d['lon'];u=d['u']
    #-------------------------------------#

    #Add code here 
    #Do PCA for fluctuations in space
    X1 = u.reshape(365, -1).T
    A1 = ((X1 - np.mean(X1, axis = 0))/np.std(X1, axis = 0)).T
    U1, S1, WT1 = np.linalg.svd(A1)
    T1 = U1.T
    Anew1 = np.matmul(T1, A1)
    explained_variance_1 = S1**2 / np.sum(S1**2)


    #Do PCA in space for fluctuations in time
    X2 = u.reshape(365, -1)
    A2 = ((X2 - np.mean(X2, axis = 0))/np.std(X2, axis = 0)).T
    U2, S2, WT2 = np.linalg.svd(A2)
    T2 = U2.T
    Anew2 = np.matmul(T2, A2)
    explained_variance_2 = S2**2 / np.sum(S2**2)
    n = np.argmax(np.cumsum(explained_variance_2) > 0.95)
    Areduced2 = np.matmul(U2[:, n:], Anew2[n:, :]).reshape(np.shape(u))

    min_value1 = min(min(Anew1[0,:]), min(Anew1[1,:]), min(Anew1[2,:]))
    max_value1 = max(max(Anew1[0,:]), max(Anew1[1,:]), max(Anew1[2,:]))
    min_value2 = min(-abs(min_value1), -abs(max_value1))
    max_value2 = max(abs(min_value1), abs(max_value1))
    norm = Normalize(vmin=min_value2, vmax=max_value2)

    umean = np.mean(u, axis=0)

    y = Anew2[0, :]
    c = np.fft.fft(y)
    c = np.fft.fftshift(c)/365
    k = np.arange(-182, 183)  
    
    # Create plots of first three principal components
    fig0, ax0 = plt.subplots(figsize=(10, 2))
    fig1, ax1 = plt.subplots(3, 1, figsize=(10, 6))
    fig1.tight_layout(h_pad=3.0)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    fig3, ax3 = plt.subplots(1, 2, figsize=(10, 4))
    fig4, ax4 = plt.subplots(1, 3, figsize=(10, 2))
    fig4.tight_layout(w_pad=4.0)
    fig5, ax5 = plt.subplots(figsize=(10, 6))

    con0 = ax0.contourf(lon, lat, umean, levels=100)
    ax0.set_xticks([0, 60, 120, 180, 240, 300, 357.5])
    ax0.set_yticks([-45, -52.5, -60, -67.5, -75, -82.5])
    ax0.set_xlabel('Longitude')
    ax0.set_ylabel('Latitude')
    ax0.set_title('Mean Wind Speed')
    fig1.colorbar(con0, ax=ax0, label='Mean Wind Speed')

    con2 = ax1[0].contourf(lon, lat, Anew1[0, :].reshape(u.shape[1],-1), levels=100, cmap='RdYlGn', norm=norm)
    con3 = ax1[1].contourf(lon, lat, Anew1[1, :].reshape(u.shape[1],-1), levels=100, cmap='RdYlGn', norm=norm)
    con4 = ax1[2].contourf(lon, lat, Anew1[2, :].reshape(u.shape[1],-1), levels=100, cmap='RdYlGn', norm=norm)

    for i in range(3):
        ax1[i].set_xticks([0, 60, 120, 180, 240, 300, 357.5])
        ax1[i].set_yticks([-45, -52.5, -60, -67.5, -75, -82.5])
        ax1[i].set_xlabel('Longitude')
        ax1[i].set_ylabel('Latitude')
        ax1[i].set_title(f'Principal Component {i+1}')
    fig1.colorbar(con2, ax=ax1, label='Value for Principle Component', norm=norm, aspect=60)

    ax2.plot(np.cumsum(explained_variance_1), marker = 'o')
    ax2.set_title('Cumulative Explained Variance Ratio for PCA on space')
    ax2.set_xlabel('Number of Principal Components')
    ax2.set_ylabel('Cumulative Explained Variance Ratio')
    ax2.set_ylim(0, 1.1)

    ax3[0].plot(Anew2[0, :])
    ax3[0].set_xticks(np.linspace(0, 364, 5))
    ax3[0].set_xlabel('Time')
    ax3[0].set_ylabel('Principal Component 1 Weight')
    ax3[0].set_title('Time Series')

    ax3[1].plot(k, np.abs(c), 'x')
    ax3[1].set_xlabel('Mode')
    ax3[1].set_title('Fourier Coefficients')

    ax4[0].plot(Anew2[0, :-1], Anew2[0, 1:], 'x', ms=4)
    ax4[0].set_xlabel('$PC1_N$')
    ax4[0].set_ylabel('$PC1_{N+1}$')
    ax4[0].set_title('Day N vs Day N+1')

    ax4[1].plot(Anew2[0, :-2], Anew2[0, 2:], 'x', ms=4)
    ax4[1].set_xlabel('$PC1_N$')
    ax4[1].set_ylabel('$PC1_{N+2}$')
    ax4[1].set_title('Day N vs Day N+2')

    ax4[2].plot(Anew2[0, :-20], Anew2[0, 20:], 'x', ms=4)
    ax4[2].set_xlabel('$PC1_N$')
    ax4[2].set_ylabel('$PC1_{N+20}$')
    ax4[2].set_title('Day N vs Day N+20')

    ax5.plot(np.cumsum(explained_variance_2), marker = 'o')
    ax5.set_title('Cumulative Explained Variance Ratio for PCA on time')
    ax5.set_xlabel('Number of Principal Components')
    ax5.set_ylabel('Cumulative Explained Variance Ratio')
    ax5.set_ylim(0, 1.1)

    return None #modify if needed



#===== Code for Part 2=====#
def part2(f,method=2):
    """
    Question 2.1 i)
    Input:
        f: m x n array
        method: 1 or 2, interpolation method to use
    Output:
        fI: interpolated data (using method)
    """
    import scipy as sp
    m,n = f.shape
    fI = np.zeros((m-1,n)) #use/modify as needed

    if method==1:
        fI = 0.5*(f[:-1,:]+f[1:,:])
    else:
        #Coefficients for method 2
        alpha = 0.3
        a = 1.5
        b = 0.1
        
        #coefficients for near-boundary points
        a_bc,b_bc,c_bc,d_bc = (5/16,15/16,-5/16,1/16)

        #add code here
        #using linear system representation of interpolation, form LHS and RHS coefficient matrices
        #first make LHS coefficients including adjustments for i = 1/2 and i = m - 3/2
        
        LHS = np.zeros((3, m-1))
        LHS[1, :] = 1
        LHS[2, :-2] = alpha
        LHS[0, 2:] = alpha


        #now make RHS coefficients, again taking care of i = 1/2 and i = m - 3/2 cases
        boundary_conds = [a_bc - a/2, b_bc - a/2, c_bc - b/2, d_bc, a_bc - a/2, b_bc - a/2, c_bc - b/2, d_bc]
        bound_rows = [0, 0, 0, 0, m - 2, m - 2, m - 2, m - 2]
        bound_cols = [0, 1, 2, 3, m - 1, m - 2, m - 3, m - 4]
        RHS_diags = sp.sparse.diags([b/2, a/2, a/2, b/2], [-1, 0, 1, 2], (m-1, m))
        boundaries = sp.sparse.coo_array((boundary_conds, (bound_rows, bound_cols)), shape = (m-1, m))
        adjusted_RHS = RHS_diags + boundaries

        #evaluate the RHS
        RHS = adjusted_RHS @ f

        #solve system of form AX = B where A, B, X all matrices
        fI = sp.linalg.solve_banded((1, 1), LHS, RHS)

    return fI #modify as needed

def part2_analyze():
    """
    Add input/output as needed
    """

    #----- Code for generating grid, use/modify/discard as needed ----#
    n,m = 50,40 #arbitrary grid sizes
    x = np.linspace(0,1,n)
    y = np.linspace(0,1,m)
    xg,yg = np.meshgrid(x,y)
    dy = y[1]-y[0]
    yI = y[:-1]+dy/2 #grid for interpolated data
    #--------------------------------------------#

    #add code here
    import time
    #Testing function (Franke's Function)
    def frankes_function(x, y):
        return (0.75 * np.exp(-((9 * x - 2) ** 2) / 4 - (9 * y - 2) ** 2 / 4) 
                + 0.75 * np.exp(-((9 * x + 1) ** 2) / 49 - (9 * y + 1) / 10)
                + 0.5 * np.exp(-((9 * x - 7) ** 2) / 4 - (9 * y - 3) ** 2 / 4)
                - 0.2 * np.exp(-((9 * x - 4) ** 2) - (9 * y - 7) ** 2))

    # Numerical test with method 1
    f = frankes_function(xg, yg)
    t1 = time.time()
    fI1 = part2(f, method=1)
    t2 = time.time()

    # Numerical test with method 2
    fI2 = part2(f, method=2)
    t3 = time.time()

    fI_true = frankes_function(xg[:-1, :], yI[:, None])
    # Plot the results
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.title("Franke's Function")
    plt.pcolormesh(xg[:-1, :], yI, fI_true, shading='auto')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.title("Method 1 Absolute Error")
    plt.pcolormesh(xg[:-1, :], yI, np.abs(fI1 - fI_true), shading='auto')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.title("Method 2 Absolute Error")
    plt.pcolormesh(xg[:-1, :], yI, np.abs(fI2 - fI_true), shading='auto')
    plt.colorbar()

    plt.tight_layout()
    plt.show()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(xg, yg, frankes_function(xg, yg), cmap='viridis')
    ax.set_title("Franke's Function")

    return np.max(np.abs(fI1 - fI_true)), np.max(np.abs(fI2 - fI_true)), 1-np.mean(np.abs(fI1 - fI_true)/fI_true), np.mean(np.abs(fI2 - fI_true)/fI_true), t2-t1, t3-t2, np.sqrt(np.mean((fI1 - fI_true) ** 2)), np.sqrt(np.mean((fI2 - fI_true) ** 2)) #modify as needed




#===== Code for Part 3=====#
def part3q1(y0,alpha,beta,b,c,tf=200,Nt=800,err=1e-6,method="RK45"):
    """
    Part 3 question 1
    Simulate system of 2n nonlinear ODEs

    Input:
    y0: Initial condition, size 2*n array
    alpha,beta,b,c: model parameters
    tf,Nt: Solutions are computed at Nt time steps from t=0 to t=tf (see code below)

    Output:
    tarray: size Nt+1 array
    yarray: Nt+1 x 2*n array containing y at
            each time step including the initial condition.
    """
    
    #Set up parameters, arrays

    n = y0.size//2
    tarray = np.linspace(0,tf,Nt+1)
    yarray = np.zeros((Nt+1,2*n))
    yarray[0,:] = y0


    def RHS(t,y):
        """
        Compute RHS of model
        """
        #add code here
        u = y[:n];v=y[n:]
        r2 = u**2+v**2
        nu = r2*u
        nv = r2*v
        cu = np.roll(u,1)+np.roll(u,-1)
        cv = np.roll(v,1)+np.roll(v,-1)

        dydt = alpha*y
        dydt[:n] += beta*(cu-b*cv)-nu+c*nv+b*(1-alpha)*v
        dydt[n:] += beta*(cv+b*cu)-nv-c*nu-b*(1-alpha)*u

        return dydt


    sol = solve_ivp(RHS, (tarray[0],tarray[-1]), y0, t_eval=tarray, method=method,atol=err,rtol=err)
    yarray = sol.y.T 
    return tarray,yarray


def part3_analyze():#add/remove input variables if needed
    """
    Part 3 question 1: Analyze dynamics generated by system of ODEs
    """
    import scipy as sp
    #Set parameters
    beta = (25/np.pi)**2
    alpha = 1-2*beta
    b =-1.5
    n = 4000

    #Set initial conidition
    L = (n-1)/np.sqrt(beta)
    k = 40*np.pi/L
    a0 = np.linspace(0,L,n)
    A0 = np.sqrt(1-k**2)*np.exp(1j*a0)
    y0 = np.zeros(2*n)
    y0[:n]=1+0.2*np.cos(4*k*a0)+0.3*np.sin(7*k*a0)+0.1*A0.real

    #---Example code for computing solution, use/modify/discard as needed---#
    c_vals = [0.5, 1.0, 1.3, 1.4]
    epsilons = np.logspace(-1, 2.5, 100)
    a_vals = np.zeros(4)
    indices = [40, 70]
    frequencies = np.zeros(4)
    var_ratios = np.zeros(4)

    fig1, ax1 = plt.subplots(2, 2)
    fig1.tight_layout(w_pad=4.8, h_pad=3.2)
    fig2, ax2 = plt.subplots(2, 2)
    fig2.tight_layout(w_pad=4.8, h_pad=3.2)
    fig3, ax3 = plt.subplots(2, 2)
    fig3.tight_layout(w_pad=4.8, h_pad=3.2)
    fig4, ax4 = plt.subplots(2, 2)
    fig4.tight_layout(w_pad=4.8, h_pad=3.2)
    fig5, ax5 = plt.subplots(2, 2)
    fig5.tight_layout(w_pad=4.8, h_pad=3.2)
    fig6, ax6 = plt.subplots(2, 2)
    fig6.tight_layout(w_pad=4.8, h_pad=3.2)

    for i, c_val in enumerate(c_vals):
        y0 = np.zeros(2*n)
        y0[:n]=1+0.2*np.cos(4*k*a0)+0.3*np.sin(7*k*a0)+0.1*A0.real
        t,y = part3q1(y0,alpha,beta,b,c_val,tf=20,Nt=2,method='RK45') #for transient, modify tf and other parameters as needed
        y0 = y[-1,:]
        t,y = part3q1(y0,alpha,beta,b,c_val,method='RK45',err=1e-6)
        u,v = y[:,:n],y[:,n:]
        u_clean = u[:, 100:-100]
        n1 = len(u_clean)
        D = sp.spatial.distance.pdist(u_clean)
        C = np.array([D[D < eps].size for eps in epsilons]) * 2/(n1*(n1-1))
        plot = np.unravel_index(i, (2, 2))
        ax1[plot].loglog(epsilons, C, marker='x')
        ax1[plot].set_xlabel('$\epsilon$')
        ax1[plot].set_ylabel('$C(\epsilon)$')
        ax1[plot].set_title(f'log-log plot for c = {c_val}')
        m, c = np.polyfit(np.log(epsilons[indices[0]:indices[1]]), np.log(C[indices[0]:indices[1]]), 1)
        ax1[plot].plot(epsilons, np.exp(c) * epsilons**m, linestyle='dashed', label = f'slope = {"%.2f" % round(m, 2)}')
        ax1[plot].legend(loc = 'upper left')
        a_vals[i] = m

        A = ((u_clean - np.mean(u_clean, axis = 0))/np.std(u_clean, axis = 0)).T
        U, S, WT = np.linalg.svd(A)
        T = U.T
        Anew = np.matmul(T, A)
        plot = np.unravel_index(i, (2, 2))
        explained_variance = S**2 / np.sum(S**2)
        var_ratios[i] = explained_variance[0]
        ax2[plot].plot(np.cumsum(explained_variance), marker = 'o', markersize = 3.0)
        ax2[plot].set_xlabel('Number of PCs')
        ax2[plot].set_ylabel('Cumulative Explained Variance')
        ax2[plot].set_title(f'c = {c_val}')

        ax3[plot].set_xlabel('t')
        ax3[plot].set_title(f'First PC against time')
        ax3[plot].plot(Anew[0, :])

        fB, PxxB = sp.signal.welch(Anew[0, :])
        ax4[plot].set_xlabel('Frequency')
        ax4[plot].set_ylabel('Power Spectral Density')
        ax4[plot].set_title(f"Welch's plot for c = {c_val}")
        ax4[plot].semilogy(fB, PxxB)

        ax5[plot].set_title(f'Contour plot of solution for c = {c_val}')
        ax5[plot].contourf(np.arange(n),t,u,20)

        frequencies[i] = fB[np.argmax(PxxB)]
        dt = t[1]
        tau = 1/(20*frequencies[i])
        Del = int(tau/dt)
        x = Anew[0, :]
        v1 = np.vstack([x[:-2*Del],x[Del:-Del],x[2*Del:]])
        ax6[plot].set_xlabel('$PC1_t$')
        ax6[plot].set_ylabel('$PC1_{t+ \tau }$')
        ax6[plot].set_title(f'Solution recreation for PC1, c = {c_val}')
        ax6[plot].plot(v1[0],v1[1])

    return a_vals, frequencies #modify if needed


def part3q2(x,c=1.0):
    """
    Code for part 3, question 2
    """
    #Set parameters
    beta = (25/np.pi)**2
    alpha = 1-2*beta
    b =-1.5
    n = 4000

    #Set initial conidition
    L = (n-1)/np.sqrt(beta)
    k = 40*np.pi/L
    a0 = np.linspace(0,L,n)
    A0 = np.sqrt(1-k**2)*np.exp(1j*a0)
    y0[:n]=1+0.2*np.cos(4*k*a0)+0.3*np.sin(7*k*a0)+0.1*A0.real

    #Compute solution
    t,y = part3q1(y0,alpha,beta,b,c,tf=20,Nt=2,method='RK45') #for transient, modify tf and other parameters as needed
    y0 = y[-1,:]
    t,y = part3q1(y0,alpha,beta,b,c,method='RK45',err=1e-6)
    A = y[:,:n]

    #Analyze code here
    l1,v1 = np.linalg.eigh(A.T.dot(A))
    v2 = A.dot(v1)
    A2 = (v2[:,:x]).dot((v1[:,:x]).T)
    e = np.sum((A2.real-A)**2)

    return A2.real,e, np.shape(A)


if __name__=='__main__':
    x=None #Included so file can be imported
    #Add code here to call functions above if needed
