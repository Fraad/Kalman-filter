# -*- coding: utf-8 -*-
"""
Created on Mon Jun  9 13:55:22 2025

@author: frede
"""

import autograd.numpy as np  # Thinly-wrapped numpy
from autograd import jacobian
from autograd import hessian
import pandas as pd
from scipy.optimize import minimize, BFGS
import scipy.linalg as linalg
import warnings
import time
import matplotlib.pyplot as plt



######################################################
#---------------- Infomation matrix -----------------#
######################################################

def info_matrix(parameters, data, futures_TTM, model):

    
    trace = np.zeros((len(parameters), len(parameters)))
    
    n_params = len(parameters)
    
    n_data = len(data)
    
    #Run model with parameters
    KF_opt = model(parameters, data, futures_TTM)
    model_output_opt = KF_opt.main(data)
    
    F_t_opt = model_output_opt['list_F_t']
    invF_t_opt = model_output_opt['list_invF_t']
    
    
    # List of matrixes, each matrix list is deriv. vrt to parameter i/j
    F_t_deriv_i = [[] for _ in range(n_params)]
    
    F_t_deriv_j = [[] for _ in range(n_params)]
    
    
    
    coef = 0.2
    
    print("coef:", coef)
    # Calculate derivaties for i
    for i in range(n_params):
        
        #Get steps for i part
        one_step_params_i = np.copy(parameters)
        one_step_params_i_plus = np.copy(parameters)
        
        
        two_step_params_i = np.copy(parameters)
        
        '''
        if i == 0:
            one_step_params_i[i] = one_step_params_i[i] - 4
            one_step_params_i_plus[i] = one_step_params_i[i] + 4
            
            two_step_params_i[i] = two_step_params_i[i] - 2 * 4
        '''
        
        
        step_i =  coef * one_step_params_i[i]
        #step_i =  0.1 
        
        one_step_params_i[i] -= step_i
        one_step_params_i_plus[i] +=  step_i
        
        two_step_params_i[i] = two_step_params_i[i] - 2 * step_i
            

        #Run model with changed param
        
        KF_one_step_i = model(one_step_params_i, data, futures_TTM)
        KF_one_step_i_plus = model(one_step_params_i_plus, data, futures_TTM)
        #KF_one_step.initialize_matrices()
        
        
        KF_two_step_i = model(two_step_params_i, data, futures_TTM)
        #KF_two_step.initialize_matrices()
        
        F_t_one_step_i = (KF_one_step_i.main(data)['list_F_t']) #F_t_step is 8x8x687
        F_t_one_step_i_plus = (KF_one_step_i_plus.main(data)['list_F_t'])
        
        F_t_two_step_i = (KF_two_step_i.main(data)['list_F_t']) #F_t_step is 8x8x687
        
        
        
        for k in range(n_data):
            #Approx derivative with three-point forward difference
            #result =  (-F_t_two_step_i[k] + 4 * F_t_one_step_i[k] - 3 * F_t_opt[k] ) / (2*step_i)
            result =  (F_t_one_step_i_plus[k] -  F_t_one_step_i[k] ) / (2*step_i)
            #if k == (n_data -1):  print("i_deriv:",result[0,0])
            
            F_t_deriv_i[i].append(  result ) 
        

    #Calculate trace using the pre calculated derivaties
    for i in range(n_params):
        for j in range(n_params):
            for k in range(n_data):
                # The core calculation: 0.5 * trace(invF_t_opt[k] @ dF_t/dpi[k] @ invF_t_opt[k] @ dF_t/dpj[k])
                term = 0.5 * np.trace(invF_t_opt[k] @ F_t_deriv_i[i][k] @ invF_t_opt[k] @ F_t_deriv_i[j][k])
                trace[i][j] += term


    return trace



#Get loglike


def info_matrix_loglike(parameters, data, futures_TTM, model):
    n_params = len(parameters)
    info = np.zeros((n_params, n_params))

    # Store original log-likelihood
    KF_opt = model(parameters, data, futures_TTM)
    model_output_opt = KF_opt.main(data)
    loglike_opt = model_output_opt['log_likelihood']

    og_parameters = np.copy(parameters)
    coef = 0.1

    for i in range(n_params):
        step_i = coef * og_parameters[i]

        for j in range(n_params):
            step_j = coef * og_parameters[j]

            # Fresh copies for each term
            pp = np.copy(og_parameters)
            pm = np.copy(og_parameters)
            mp = np.copy(og_parameters)
            mm = np.copy(og_parameters)

            # Apply perturbations
            pp[i] += step_i; pp[j] += step_j
            pm[i] += step_i; pm[j] -= step_j
            mp[i] -= step_i; mp[j] += step_j
            mm[i] -= step_i; mm[j] -= step_j

            # Evaluate log-likelihoods
            loglike_pp = model(pp, data, futures_TTM).main(data)['log_likelihood']
            loglike_pm = model(pm, data, futures_TTM).main(data)['log_likelihood']
            loglike_mp = model(mp, data, futures_TTM).main(data)['log_likelihood']
            loglike_mm = model(mm, data, futures_TTM).main(data)['log_likelihood']

            # Mixed partial derivative
            if i == j:
                info[i, j] = (loglike_pp - 2 * loglike_opt + loglike_mm) / (step_i ** 2)
            else:
                info[i, j] = (loglike_pp - loglike_pm - loglike_mp + loglike_mm) / (4 * step_i * step_j)

            
    return info



######################################################
#------------ Tests and price functions -------------#
######################################################

def RMSE(est_var, obsv_var):
    
    RMSE_cubed = 0
    n_vars = est_var.shape[0]
    rmse = np.zeros(n_vars)

    for i in range(n_vars):
        mask = est_var[i, :] != 0 
        if np.any(mask):
            squared_errors = (obsv_var[i, mask] - est_var[i, mask]) ** 2
            rmse[i] = np.sqrt(np.mean(squared_errors))
        else:
            rmse[i] = np.nan
    
    return rmse
    
    

#Mean error of Time series
def mean_error(est_var, obsv_var):
    ME = 0

    n_vars = est_var.shape[0]
    mae = np.zeros(n_vars)

    for i in range(n_vars):
          mask = est_var[i, :] != 0
          if np.any(mask):
              errors = np.abs(obsv_var[i, mask] - est_var[i, mask])
              mae[i] = np.mean(errors)
          else:
              mae[i] = np.nan  

    return mae



def Time_series(data, out_of_sample_length, Filter, optimizer, parameters, TTM, bounds):
    
    
    
    error = []
    
    for t in range(out_of_sample_length):
        data_t = data[:,0: len(data[0]) -out_of_sample_length + t ]
        
        optimal_params = optimizer(data_t, parameters, TTM, bounds)

        #Initial parameters should be optimised 
        KF_t = Filter(optimal_params['optimal parameters'], data_t, TTM)
        
        
        #Get state
        state_t = np.array(KF_t.main(data_t)['State_t_t']).reshape(-1, int(len(KF_t.main(data_t)['State_t_t'])))

        #Calculate the predicted price at t+1
        
        #Fix state dimension
        state_fix = np.expand_dims(state_t[:,-1], axis = 1)
        
        # Get predicted state at t + 1
        state_t_t1 = KF_t.T @ state_fix + KF_t.c
        
        #Find error
        error.append( np.expand_dims(data[:,out_of_sample_length -t + 1], axis=1) - (KF_t.Z @ state_t_t1 + KF_t.d) )
        
        print(state_t_t1)
    

    return error

def price_function_m23(Filter, state,  TTM):
        
    prices = []
    # Final result (variance-like term)
    for i in range(len(state[0])):
        v = state[2,i]
        delta = state[1,i]
        x = state[0,i]


        prices.append( Filter.num_A + Filter.num_B*v + Filter.D_func(TTM)*delta + x)

    return np.squeeze(prices)

def price_function_m1(Filter,state, TTM):

    prices = []
    
    for i in range(len(state[0])):
        delta = state[1,i]
        x = state[0,i]

        prices.append( np.array([Filter.d]) + np.array([Filter.Z[:,1]]).T * delta + x )
        
    return np.squeeze(prices)


def est_vol(Filter, state, TTM):
    
    obs_vol = []
    
    for i in range(len(state[0])):
        v = state[2,i]

        obs_vol.append( Filter.C_func(TTM) * v )
    
    return np.squeeze(obs_vol)


######################################################
#--------------Model 1 implementation ---------------#
######################################################

class Kalman_filter_m1:
    def __init__(self, params,data, TTM):
        self.mu, self.beta, self.alpha, self.sigma1, self.sigma2, self.rho, self.lambda_, self.e1,= params
        #self.e2, self.e3, self.e4
        self.dim_y = int(data.shape[0])  # Measurement equation
        self.dim_x = 2  # State dimension

        # Define global variables
        self.delta_t = 1/52.1775
        self.futures_TTM = TTM
        self.r = 0.08
        #self.mu = 0.08
        

        #Define alpha hat
        self.alpha_hat = self.alpha - self.lambda_/self.beta

        # Transition matrix
        self.T = np.array([[1, - self.delta_t],
                          [0, 1 - self. beta * self.delta_t]])
        #print("T (Transition Matrix):", self.T)

        # Transition intercept
        self.c = np.array([[(self.mu - 0.5 * self.sigma1 ** 2) * self.delta_t],
                          [self. beta * self.alpha * self.delta_t]])
        # print("c (Transition Intercept):", self.c)

        # Transition covariance | I changed element 2x2 to sigma2^2 instead of sigma1^2 in Schwartz
        self.Q = np.array([[(self.sigma1**2) * self.delta_t, self.rho * self.sigma1 * self.sigma2 * self.delta_t],
                           [ self.rho * self.sigma1 * self.sigma2 * self.delta_t, (self.sigma1 ** 2) * self.delta_t]])
        # print("Q (Transition Covariance):", self.Q)


        # Measurement matrix
        self.Z = np.hstack((np.ones((self.dim_y,1)), - (1 - np.exp(-self. beta * self.futures_TTM))/ (self. beta)))
        # print("Z (Measurement Matrix):", self.Z)

        # Measurement intercept (Fixed)
        self.d = np.array(
            (self.r - self.alpha_hat +
                   1/2 * (self.sigma2 ** 2) / (self.beta ** 2) - (self.sigma1 * self.sigma2 * self.rho)/(self.beta) ) * self.futures_TTM 
                + 1/4 * (self.sigma2 ** 2) * (1 - np.exp(- 2 * self.beta * self.futures_TTM)) / (self.beta ** 3)
                + ( self.alpha_hat * self.beta + self.sigma1 * self.sigma2 * self.rho - ((self.sigma2 ** 2) / self.beta))
                * (1 - np.exp(- self.beta * self.futures_TTM) / (self.beta ** 2)))
        # print("d (Measurement Intercept):", self.d)
            

        # Measurement covariance | Assuming no covariance between futures | And same variance for all!!
        #self.H = np.array(np.diag([self.e1, self.e2, self.e3, self.e4, self.e5, self.e6,
        #self.e7, self.e8]))

        # Measurement covariance | Assuming no covariance between futures & vol_obs | And same variance for all!!
        self.H = np.eye( int(self.dim_y ) ) * self.e1  
        # Inverse of H, avoiding division by zero
        self.invH = np.diag(np.where(np.diag(self.H) > 0, 1/np.diag(self.H), 0))
        # print("H (Measurement Covariance):", self.H)

    def main(self, y):
        T, c, Q, Z, d, H, invH = self.T, self.c, self.Q, self.Z, self.d, self.H, self.invH
        n_obs = y.shape[1]

        # Values of interest

        log_likelihood = 0
        RMSE_cubed = 0
        Error = []
        State_t_t1 = []
        State_t_t = []
        list_F_t = []
        list_invF_t = [] 

        # Initialize state mean and covariance
        x_t_t = np.array([[4.5],[0.1]])
        P_t_t = np.eye(2)

        for t in range(n_obs):
            # Prediction step
            x_t_t1 = T @ x_t_t + c
            P_t_t1 = T @ P_t_t @ T.T + Q

     
            #Set Matrixes to full again
            Z = self.Z
            d = self.d
            H = self.H
            invH = self.invH

            # Transition covariance
            Q = self.Q
            

            # Prediction step
            x_t_t1 = T @ x_t_t + c
            P_t_t1 = T @ P_t_t @ T.T + Q


            # Check for missing values
            y_t = y[:,t].reshape(self.dim_y,1)
            
            # True / False list of Nans | Nx1 array
            Nans = np.squeeze(np.isnan(y_t))
            
            # Remove nans from observations
            y_t = np.delete(y_t, Nans, axis = 0)
            
            # Remove Nans-rows from Z and d
            Z = np.delete(Z, Nans, axis = 0)
            d = np.delete(d, Nans, axis = 0)
            
            # Remove Nans rows and col from H and invH
            H = np.delete(np.delete(H, Nans, 0), Nans, 1)
            invH = np.delete(np.delete(invH, Nans, 0), Nans, 1)
            
            
            # Measurement update step (Fixed indexing)
            v_t = y_t - (Z @ x_t_t1 + d)
            

            PreInv = linalg.inv(P_t_t1) + Z.T @ invH @ Z

            try:
                invF_t = invH - invH @ Z @ linalg.solve(PreInv, Z.T @ invH, assume_a="sym")
            except linalg.LinAlgError:
                warnings.warn("Singular matrix encountered in inversion, skipping iteration.")
                continue

            # Stable log determinant calculation
            signH, logdetH = np.linalg.slogdet(H)
            signP, logdetP = np.linalg.slogdet(P_t_t1)
            signPreInv, logdetPreInv = np.linalg.slogdet(PreInv)
            
            if signH <= 0 or signP <= 0 or signPreInv <= 0:
                warnings.warn("Negative determinant encountered, skipping iteration.")
                continue
            
            detF_t = linalg.det(H) * linalg.det(P_t_t1) * linalg.det(linalg.inv(P_t_t1) + Z.T @ invH @ Z)

            # Calculate log likelihood
            log_likelihood += -0.5 * (self.dim_y * np.log(2 * np.pi) + logdetH + logdetP + logdetPreInv + v_t.T @ invF_t @ v_t)

            # Calculate  RMSE, model prices, error
            RMSE_cubed += (np.exp(y_t) - np.exp(Z @ x_t_t1 + d) ) ** 2 / n_obs

            # Error
            Error.append(v_t)
            
            #F_t
            F_t = (Z @ P_t_t1 @ Z.T + H)
            list_F_t.append(F_t)
            
            # Inverse F_t
            list_invF_t.append(invF_t)
            
            # Kalman Gain MISSING T infront!!
            K_t = P_t_t1 @ Z.T @ invF_t

            # Update step
            x_t_t = x_t_t1 + K_t @ v_t
            P_t_t = P_t_t1 - K_t @ Z @ P_t_t1
            
            #State_t_t1 
            State_t_t1.append(x_t_t1)
            
            #State_t_t
            State_t_t.append(x_t_t)

        return {'log_likelihood': log_likelihood, 'RMSE_cubed': RMSE_cubed,
        'Error': Error, 'State_t_t1': State_t_t1,
        'State_t_t': State_t_t,
        'list_invF_t': list_invF_t, 'list_F_t': list_F_t}

#Function to estimate hyperparameters


def estimate_hyperparameters_m1(y, params, TTM, bounds):
    def objective(params):
        kf = Kalman_filter_m1(params, y, TTM)
        return -kf.main(y)['log_likelihood']
    
    # Use 'L-BFGS-B' instead of 'Nelder-Mead' to support bounds
    result = minimize(objective, x0= params, method='L-BFGS-B', bounds = bounds, options = {'maxiter': 20})
    return {'optimal parameters': result.x.tolist(), 
            'hessian': result.hess_inv.todense()}


######################################################
#--------------Model 2 implementation ---------------#
######################################################



class Kalman_filter_m2:
    def __init__(self, params, data, TTM):
        self.mu, self.alpha, self.beta, self.theta, self.kappa, self.sigma_delta, self.sigma_v, self.lambda_S, self.lambda_delta, self.lambda_v, self.rho_12, self.rho_13, self.rho_23, self.e1,= params
        self.dim_y = int(data.shape[0]) # Measurement equation
        self.dim_x = 3  # State dimension
        # self.e3, self.e4, self.e5, self.e6, self.e7, self.e8,

        # Define global variables
        self.delta_t = 1/52.1775
        self.futures_TTM = TTM
        
        self.r = 0.008
        
        # Measurement covariance | Assuming no covariance between futures & vol_obs | And same variance for all!!
        #self.H = np.array(np.diag([self.e1, self.e1, self.e1, self.e1, self.e2, self.e2, self.e2, self.e2]))
        #self.H = np.array(np.diag([self.e1, self.e2, self.e3, self.e4, self.e5, self.e6, self.e7]))

        self.H = np.eye( int(self.dim_y ) ) * self.e1       

        # Inverse of H, avoiding division by zero
        
        self.invH = np.diag(np.where(np.diag(self.H) > 0, 1/np.diag(self.H), 0))
        # print("H (Measurement Covariance):", self.H)
        
        # Autograd: True, Num_Jacob: False
        self.use_autograd = True
        
        #Initial values: B(T,T) = 0
        self.y0 = np.zeros((int(self.dim_y ) ,1))
        
        
    # State transition function
        self.T = np.array([[ 1, - self.delta_t ,  -0.5 * self.delta_t],
                            [0, 1 - self.beta * self.delta_t,  0],
                            [0, 0, 1  -self.kappa * self.delta_t]])

    # Transition intercept
        self.c = np.array([[self.mu * self.delta_t],
                      [self.alpha * self.beta * self.delta_t],
                      [self.kappa * self.theta * self.delta_t]])    
    

        self.num_B = self.num_integrate(self.B_diff, self.futures_TTM ,self.y0)

    # Measurement function
        #Log Future price measurement function
        Upper = np.hstack((np.ones((int(self.dim_y ) ,1)), self.D_func(self.futures_TTM), self.num_B ))
        #Vol measurement function
        Lower = np.hstack((np.zeros((int(self.dim_y ) ,2)), self.C_func(self.futures_TTM)))
        #Combinded
        
        self.Z = np.hstack((np.ones((int(self.dim_y ) ,1)), self.D_func(self.futures_TTM), self.num_B ))

    

    # Measurement intercept
        self.num_A = self.num_integrate(self.A_diff, self.futures_TTM, self.y0)
        
    
        self.d = self.num_A
   
    # Transition covariance function
    def Q_func(self, state):
        result = np.array([[1, self.rho_12 * self.sigma_delta, self.rho_13 *  self.sigma_v],
                         [self.rho_12  * self.sigma_delta, self.sigma_delta ** 2, self.rho_23  * self.sigma_v * self.sigma_delta ],
                         [self.rho_13  * self.sigma_v, self.rho_23  * self.sigma_v * self.sigma_delta,  self.sigma_v ** 2 ]]) * self.delta_t * (state[2])
        return np.squeeze(result)

        return result
    '''
    def Q_func(self, state):
        result = np.array([[state[2], self.rho_12 * state[2] * self.sigma_delta, self.rho_13 * state[2] * self.sigma_v],
                         [self.rho_12 * state[2] * self.sigma_delta, state[2] * self.sigma_delta ** 2, self.rho_23 * state[2] * self.sigma_v * self.sigma_delta ],
                         [self.rho_13 * state[2] * self.sigma_v, self.rho_23 * state[2] * self.sigma_v * self.sigma_delta, state[2] * self.sigma_v ** 2 ]]) * self.delta_t
        return np.squeeze(result, axis = 2)

        return result
    '''
    # Price intercept function
    def A_diff(self, t, y0):
        
        result = + self.r + self.beta * self.alpha * self.D_func(t)+ self.kappa * self.theta * self.B_diff(t, y0)
        #result = + self.r * self.futures_TTM + self.beta * self.alpha * ( - (self.futures_TTM / self.beta +  np.exp(-self.beta * self.futures_TTM) -1) / self.beta**2) + self.kappa * self.theta * self.num_B
        return result


    # Price function for vol
    def B_diff(self, t, y0):
        """ dy/dt=f(y,t) """
      
        #Define D-function
        
        
        
        First_term =  1/2 * self.sigma_delta ** 2 * (self.D_func(t)) ** 2 - self.lambda_S - self.lambda_delta * self.sigma_delta * self.D_func(t) + self.rho_12 * self.sigma_delta * self.D_func(t)    
     
        Second_term = 1/2 * (self.sigma_v ** 2) * y0 ** 2
        
        Third_term = (- self.lambda_v * self.sigma_v + self.rho_13 * self.sigma_v + self.rho_23 * self.sigma_delta * self.sigma_v * self.D_func(t) - self.kappa ) * y0
        
        return (First_term + Second_term + Third_term)

    #

    # Price function for conv. yield |
    def D_func(self, t):
        result = - 1/self.beta * (1 - np.exp(-self.beta * t))
        
        return result
    
    # Vol measurement function
    def C_func(self, t):
        result =    (1 - self.kappa * t)
        # self.psi *( np.exp(- self.gamma * t))
        return result

    # Vol intercept function
    def E_func(self, t):
        #result = ( self.theta - self.lambda_v / self.kappa)* (1 - np.exp(- self.kappa * t))
        result = 0 * t
        return result
    
        

    
    def num_integrate(self, func, t,  y0):
       
       """
       Initial state B(T,T) = 0
       """
        
       """
       The maximum time step is 5 days (0.01369 yrs), ie. the h for F8 is ~220. 
       Every future should have 220 steps.
       The short futures will have a better estimation.
           
      
       """
       x0 = np.zeros((len(t),1))
    
       for j in range(len(x0)):
           N = 320
           h = t[j] / N
           
           for i in range(N):
                
                k1= h * func(x0[j], y0[j])
                k2= h * func(x0[j] + 0.5 * h , y0[j] + 0.5*k1)
                k3= h * func(x0[j] + 0.5 * h, y0[j] + 0.5*k2)
                k4= h * func(x0[j] + h, y0[j] + k3)
                 
                x0[j] = x0[j] + h
                y0[j,0] = y0[j,0] + (k1 + 2*(k2 + k3) + k4)/6
                 
       return y0
        

    
    def numerical_jacobian(self, func, x, epsilon=1e-5):
        x = x.reshape(-1, 1)  # Ensure x is column vector
        n = x.shape[0]        # Input dimension
        fx = func(x)
        m = fx.shape[0]       # Output dimension
    
        J = np.zeros((m, n))  # Jacobian: (output_dim, input_dim)
    
        for i in range(n):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i, 0] += epsilon
            x_minus[i, 0] -= epsilon
            f_plus = func(x_plus)
            f_minus = func(x_minus)
            J[:, i] = ((f_plus - f_minus) / (2 * epsilon)).flatten()
    
        return J
      
    
    def compute_jacobian(self, func, x):
        if self.use_autograd:
            grad_J = jacobian(func)
            J = grad_J(x)
        else:
            J = self.numerical_jacobian(func, x)
        return J
    
    def main(self,y):
        T, c, Q, Z, d, H, invH = self.T, self.c, self.Q_func, self.Z, self.d, self.H, self.invH

        n_obs = y.shape[1]

        log_likelihood = 0
        RMSE_cubed = 0
        Error = []
        State_t_t1 = []
        State_t_t = []
        list_F_t = []
        list_invF_t = [] 
        
        

        # Initialize state mean and covariance
        x_t_t = np.array([[4.5],[0.5],[0.5]])
        P_t_t = np.eye(3) * 1 

        for i in range(n_obs):

        # Update time dependent functions and matrixes
        
            #Set Matrixes to full again
            Z = self.Z
            d = self.d
            H = self.H
            invH = self.invH

            # Transition covariance
            Q = self.Q_func(x_t_t)
            

            # Prediction step
            x_t_t1 = T @ x_t_t + c
            P_t_t1 = T @ P_t_t @ T.T + Q


            # Check for missing values
            y_t = y[:,i].reshape(self.dim_y,1)
            
            # True / False list of Nans | Nx1 array
            Nans = np.squeeze(np.isnan(y_t))
            
            # Remove nans from observations
            y_t = np.delete(y_t, Nans, axis = 0)
            
            # Remove Nans-rows from Z and d
            Z = np.delete(Z, Nans, axis = 0)
            d = np.delete(d, Nans, axis = 0)
            
            # Remove Nans rows and col from H and invH
            H = np.delete(np.delete(H, Nans, 0), Nans, 1)
            invH = np.delete(np.delete(invH, Nans, 0), Nans, 1)
            
            
            # Measurement update step (Fixed indexing)
            v_t = y_t - (Z @ x_t_t1 + d)
            
                
            # Check if P_t_t1 is singular or ill-conditioned:
            reg_attempts = 0
            while True:
                try:
                    InvP_t_t1 = linalg.inv(P_t_t1)
                    break
                except:
                    if reg_attempts >= 7:
                        raise ValueError("P_t_t1 remains singular after multiple regularization attempts.")
                    P_t_t1 += 1e-1 * np.eye(P_t_t1.shape[0])
                    reg_attempts += 1
            

            PreInv = InvP_t_t1 + Z.T @ invH @ Z 
            
            try:
                invF_t = invH - invH @ Z @ linalg.solve(PreInv, Z.T @ invH, assume_a= 'gen') 
            except:
                warnings.warn("Singular matrix encountered in inversion, skipping iteration.")
                continue

            cond = np.linalg.cond(invF_t)            

            # Stable log determinant calculation
            signH, logdetH = np.linalg.slogdet(H)
            signP, logdetP = np.linalg.slogdet(P_t_t1)
            signPreInv, logdetPreInv = np.linalg.slogdet(PreInv)
            
            #if signH <= 0 or signP <= 0 or signPreInv <= 0:
            #    warnings.warn("Negative determinant encountered, skipping iteration.")
            #    continue
            
            #detF_t = linalg.det(H) * linalg.det(P_t_t1) * linalg.det(linalg.inv(P_t_t1) + Z.T @ invH @ Z)

            # Calculate log likelihood
            log_likelihood += -0.5 * (self.dim_y * np.log(2 * np.pi) + logdetH + logdetP + logdetPreInv + v_t.T @ invF_t @ v_t)

            # Error
            Error.append(v_t)
            
            #F_t
            F_t = (Z @ P_t_t1 @ Z.T + H)
            list_F_t.append(F_t)
            
            # Inverse F_t
            list_invF_t.append(invF_t)
            
            # Kalman Gain MISSING T infront!!
            K_t = P_t_t1 @ Z.T @ invF_t

            # Update step
            x_t_t = x_t_t1 + K_t @ v_t
            P_t_t = P_t_t1 - K_t @ Z @ P_t_t1
            
            #State_t_t1 
            State_t_t1.append(x_t_t1)
            
            #State_t_t
            State_t_t.append(x_t_t)
            
            
        return {'log_likelihood': log_likelihood, 'RMSE_cubed': RMSE_cubed,
        'Error': Error, 'State_t_t1': State_t_t1,
        'State_t_t': State_t_t,
        'list_invF_t': list_invF_t, 'list_F_t': list_F_t}



def estimate_hyperparameters_m2(y , params, TTM, bounds):
    def objective(params):
        ekf = Kalman_filter_m2(params, y, TTM)
        return -ekf.main(y)['log_likelihood']

    # Use 'L-BFGS-B' instead of 'Nelder-Mead' to support bounds
    result = minimize(objective, x0= params, method='L-BFGS-B', bounds = bounds, options = {'maxiter': 1, 'maxls':30})
    return {'optimal parameters': result.x.tolist(), 
            'hessian': result.hess_inv.todense()}


######################################################
#------------- Model 3 implementation ---------------#
######################################################



class ExtendedKalmanFilter:
    def __init__(self, params, data, TTM):
        self.mu, self.alpha, self.beta, self.theta, self.kappa, self.sigma_delta, self.sigma_v, self.lambda_S, self.lambda_delta, self.lambda_v, self.rho_12, self.rho_13, self.rho_23, self.e1, self.e2,= params
        self.dim_y = int(data.shape[0]) # Measurement equation
        self.dim_x = 3  # State dimension
        # self.e3, self.e4, self.e5, self.e6, self.e7, self.e8,
        
        print(params)
        
        # Define global variables
        self.delta_t = 1/52.1775
        self.futures_TTM = TTM
        
        self.r = 0.008
        
        # Measurement covariance | Assuming no covariance between futures & vol_obs | And same variance for all!!
        p_error = np.vstack( ( np.eye( int(self.dim_y / 2) ) * self.e1, np.zeros( ( int(self.dim_y / 2), int(self.dim_y / 2) ) )  )  )
        v_error = np.vstack( ( np.zeros( ( int(self.dim_y / 2), int(self.dim_y / 2) ) )   , np.eye( int(self.dim_y / 2))  * self.e2  ) )

        #self.H = np.array(np.diag([self.e1, self.e1, self.e1, self.e1, self.e2, self.e2, self.e2, self.e2]))
        #self.H = np.array(np.diag([self.e1, self.e2, self.e3, self.e4, self.e5, self.e6, self.e7]))

        self.H = np.hstack( (p_error, v_error) )        

        # Inverse of H, avoiding division by zero
        
        self.invH = np.diag(np.where(np.diag(self.H) > 0, 1/np.diag(self.H), 0))
        # print("H (Measurement Covariance):", self.H)
        
        # Autograd: True, Num_Jacob: False
        self.use_autograd = True
        
        #Initial values: B(T,T) = 0
        self.y0 = np.zeros((int(self.dim_y / 2) ,1))
        
        
    # State transition function
        self.T = np.array([[ 1, - self.delta_t ,  -0.5 * self.delta_t],
                            [0, 1 - self.beta * self.delta_t,  0],
                            [0, 0, 1  -self.kappa * self.delta_t]])
        
    # Transition intercept 

        self.c = np.array([[self.mu * self.delta_t ], 
                      [ self.alpha * self.beta * self.delta_t],
                      [ self.kappa * self.theta * self.delta_t]])  


        self.num_B = self.num_integrate(self.B_diff, self.futures_TTM ,self.y0)

    # Measurement function
        #Log Future price measurement function
        Upper = np.hstack((np.ones((int(self.dim_y / 2) ,1)), self.D_func(self.futures_TTM), self.num_B ))
        #Vol measurement function
        Lower = np.hstack((np.zeros((int(self.dim_y / 2) ,2)), self.C_func(self.futures_TTM)))
        #Combinded
        
        self.Z = np.vstack((Upper, Lower))

    
    # Measurement intercept
        self.num_A = self.num_integrate(self.A_diff, self.futures_TTM, self.y0)
        
        self.d =  np.vstack((self.num_A, self.E_func(self.futures_TTM) ))
    
    # Transition covariance function
    def Q_func(self, state):
        result = np.array([[1, self.rho_12 * self.sigma_delta, self.rho_13 *  self.sigma_v],
                         [self.rho_12  * self.sigma_delta, self.sigma_delta ** 2, self.rho_23  * self.sigma_v * self.sigma_delta ],
                         [self.rho_13  * self.sigma_v, self.rho_23  * self.sigma_v * self.sigma_delta,  self.sigma_v ** 2 ]]) * self.delta_t * (state[2])
        return np.squeeze(result)

        return result

    # Price intercept function
    def A_diff(self, t, y0):
        
        result = + self.r + self.beta * self.alpha * self.D_func(t)+ self.kappa * self.theta * self.B_diff(t, y0)
        #result = + self.r * self.futures_TTM + self.beta * self.alpha * ( - (self.futures_TTM / self.beta +  np.exp(-self.beta * self.futures_TTM) -1) / self.beta**2) + self.kappa * self.theta * self.num_B
        return result


    # Price function for vol
    def B_diff(self, t, y0):
        """ dy/dt=f(y,t) """
      
        #Define D-function
        
        
        
        First_term =  1/2 * self.sigma_delta ** 2 * (self.D_func(t)) ** 2 - self.lambda_S - self.lambda_delta * self.sigma_delta * self.D_func(t) + self.rho_12 * self.sigma_delta * self.D_func(t)    
     
        Second_term = 1/2 * (self.sigma_v ** 2) * y0 ** 2
        
        Third_term = (- self.lambda_v * self.sigma_v + self.rho_13 * self.sigma_v + self.rho_23 * self.sigma_delta * self.sigma_v * self.D_func(t) - self.kappa ) * y0
        
        return (First_term + Second_term + Third_term)

    #

    # Price function for conv. yield |
    def D_func(self, t):
        result = - 1/self.beta * (1 - np.exp(-self.beta * t))
        
        return result
    
    # Vol measurement function
    def C_func(self, t):
        result =    (1 - self.kappa * t)
        # self.psi *( np.exp(- self.gamma * t))
        return result

    # Vol intercept function
    def E_func(self, t):
        #result = ( self.theta - self.lambda_v / self.kappa)* (1 - np.exp(- self.kappa * t))
        result = 0 * t
        return result
    
        

    
    def num_integrate(self, func, t,  y):
       
       """
       Initial state B(T,T) = 0
       """
        
       """
       The maximum time step is 5 days (0.01369 yrs), ie. the h for F8 is ~220. 
       Every future should have 220 steps.
       The short futures will have a better estimation.
           
      
       """
       x0 = np.zeros((len(t), 1))
       y0 = np.copy(y)
       results = np.copy(y0)  # Initialize results array

       for j in range(len(x0)):
            N = 320
            h = t[j] / N

            current_y = np.copy(y0[j])  # Work with a copy for each integration

            for i in range(N):
                k1 = h * func(x0[j], current_y)
                k2 = h * func(x0[j] + 0.5 * h, current_y + 0.5 * k1)
                k3 = h * func(x0[j] + 0.5 * h, current_y + 0.5 * k2)
                k4 = h * func(x0[j] + h, current_y + k3)

                x0[j] = x0[j] + h
                current_y = current_y + (k1 + 2 * (k2 + k3) + k4) / 6

            results[j, 0] = current_y

       return results

    
    def numerical_jacobian(self, func, x, epsilon=1e-5):
        x = x.reshape(-1, 1)  # Ensure x is column vector
        n = x.shape[0]        # Input dimension
        fx = func(x)
        m = fx.shape[0]       # Output dimension
    
        J = np.zeros((m, n))  # Jacobian: (output_dim, input_dim)
    
        for i in range(n):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i, 0] += epsilon
            x_minus[i, 0] -= epsilon
            f_plus = func(x_plus)
            f_minus = func(x_minus)
            J[:, i] = ((f_plus - f_minus) / (2 * epsilon)).flatten()
    
        return J
      
    
    def compute_jacobian(self, func, x):
        if self.use_autograd:
            grad_J = jacobian(func)
            J = grad_J(x)
        else:
            J = self.numerical_jacobian(func, x)
        return J
    
    def main(self,y):
        T, c, Q, Z, d, H, invH = self.T, self.c, self.Q_func, self.Z, self.d, self.H, self.invH

        n_obs = y.shape[1]

        log_likelihood = 0
        RMSE_cubed = 0
        Error = []
        State_t_t1 = []
        State_t_t = []
        list_F_t = []
        list_invF_t = [] 
        
        

        # Initialize state mean and covariance
        x_t_t = np.array([[4.5],[0.1],[0.4]])
        P_t_t = np.eye(3) * 1 

        for i in range(n_obs):

        # Update time dependent functions and matrixes
        
            #Set Matrixes to full again
            
            Z = self.Z
            d = self.d
            H = self.H
            invH = self.invH


            # Transition covariance
            Q = self.Q_func(x_t_t)
            
            

            # Prediction step
            x_t_t1 = T @ x_t_t + c
            P_t_t1 = T @ P_t_t @ T.T + Q

        
            
            # Check for missing values
            #y_t = y[:,i].reshape(self.dim_y,1)
            
            # True / False list of Nans | Nx1 array
            # Nans = np.squeeze(np.isnan(y_t))
            
            # Remove nans from observations
            #y_t = np.delete(y_t, Nans, axis = 0)
            
            # Remove Nans-rows from Z and d
            #Z = np.delete(Z, Nans, axis = 0)
            #d = np.delete(d, Nans, axis = 0)
            
            # Remove Nans rows and col from H and invH
            #H = np.delete(np.delete(H, Nans, 0), Nans, 1)
            #invH = np.delete(np.delete(invH, Nans, 0), Nans, 1)
            
            # Masked measurement update:
            y_t = y[:, i].reshape(self.dim_y, 1)
            nan_mask = np.isnan(y_t).flatten()
            valid_indices = ~nan_mask
        
            y_t = y_t[valid_indices]
            Z = Z[valid_indices]
            d = d[valid_indices]
            H = H[np.ix_(valid_indices, valid_indices)]
            invH = invH[np.ix_(valid_indices, valid_indices)]
        
            v_t = y_t - (Z @ x_t_t1 + d)
            # Measurement update step (Fixed indexing)
            #v_t = y_t - (Z @ x_t_t1 + d)
            
            
            # Check if P_t_t1 is singular or ill-conditioned:
            reg_attempts = 0
            while True:
                try:
                    InvP_t_t1 = linalg.inv(P_t_t1)
                    break
                except linalg.LinAlgError:
                    if reg_attempts >= 7:
                        raise ValueError("P_t_t1 remains singular after multiple regularization attempts.")
                    P_t_t1 += 1e-4 * np.eye(P_t_t1.shape[0])
                    reg_attempts += 1
            

            PreInv = InvP_t_t1 + Z.T @ invH @ Z 
            
            try:
                invF_t = invH - invH @ Z @ linalg.solve(PreInv, Z.T @ invH, assume_a= 'gen') 
            except:
                warnings.warn("Singular matrix encountered in inversion, skipping iteration.")
                continue

            cond = np.linalg.cond(invF_t)            

            # Stable log determinant calculation
            signH, logdetH = np.linalg.slogdet(H)
            signP, logdetP = np.linalg.slogdet(P_t_t1)
            signPreInv, logdetPreInv = np.linalg.slogdet(PreInv)
            
            #if signH <= 0 or signP <= 0 or signPreInv <= 0:
            #    warnings.warn("Negative determinant encountered, skipping iteration.")
            #    continue
            
            #detF_t = linalg.det(H) * linalg.det(P_t_t1) * linalg.det(linalg.inv(P_t_t1) + Z.T @ invH @ Z)

            # Calculate log likelihood
            log_likelihood += -0.5 * (self.dim_y * np.log(2 * np.pi) + logdetH + logdetP + logdetPreInv + v_t.T @ invF_t @ v_t)

            # Error
            Error.append(v_t)
            
            #F_t
            F_t = (Z @ P_t_t1 @ Z.T + H)
            list_F_t.append(F_t)
            
            # Inverse F_t
            list_invF_t.append(invF_t)
            
            # Kalman Gain MISSING T infront!!
            K_t = P_t_t1 @ Z.T @ invF_t

            # Update step
            x_t_t = x_t_t1 + K_t @ v_t
            P_t_t = P_t_t1 - K_t @ Z @ P_t_t1
            
            #State_t_t1 
            State_t_t1.append(x_t_t1)
            
            #State_t_t
            State_t_t.append(x_t_t)
            
            
        return {'log_likelihood': log_likelihood, 'RMSE_cubed': RMSE_cubed,
        'Error': Error, 'State_t_t1': State_t_t1,
        'State_t_t': State_t_t,
        'list_invF_t': list_invF_t, 'list_F_t': list_F_t}



def estimate_hyperparameters_m3(y , params, TTM, bounds):
    def objective(params):
        ekf = ExtendedKalmanFilter(params, y, TTM)
        return -ekf.main(y)['log_likelihood']

    
    # Use 'L-BFGS-B' instead of 'Nelder-Mead' to support bounds
    result = minimize(objective, x0= params, method='L-BFGS-B', bounds = bounds, options = {'maxiter': 20})
    return {'optimal parameters': result.x.tolist(), 
            'hessian': result.hess_inv.todense()}




######################################################
#-------------- Global parameters -------------------#
######################################################

futures_TTM = np.array([[30/365],
                             [60/365],
                             [90/365],
                             [180/365],
                             [270/365],
                             [365/365]])



# Bounds
m3_bounds = [(1e-4, 4),  (1e-4, 4), (1e-4, 4), (1e-4,  4),(1e-4, 4), (1e-5,4), (1e-5,4), (-4,4), (-4,4), (-4,4), (-1,1), (-1,1), (-1,1)] + [(1e-7, 4)] * 2  # Fixed bounds

m2_bounds = [(1e-4, 5),  (1e-4, 5), (1e-4, 5), (1e-4,  5),(1e-4, 5), (1e-5, 5), (1e-5, 5), (-5, 5), (-5, 5), (-5, 5), (-1,1), (-1,1), (-1,1)] + [(1e-7, 5)] * 1  # Fixed bounds

m1_bounds = [(1e-4, 4), (1e-4, 4), (-4, 4),(1e-4, 4), (1e-4, 4), (-1,1), (-4, 4)] + [(1e-5, 4)] * 1


######################################################
#------------------- Load Data ----------------------#
######################################################



#Importer data | 31/01/2007 - 24/9/2019
file_path = 'FinalData.csv'
log_df = pd.read_csv(file_path)



#Convert Data to numpy
ren_data = np.transpose(log_df.drop(log_df.columns[[0]], axis=1))
#print(ren_data)


# Model 3 data:
#Only keep v60 due to missing data!
m3_data_array_full = np.array(ren_data)
m3_data_array_first = np.array(ren_data)[:,0:330]
m3_data_array_second = np.array(ren_data)[:,330:661]


# Model 2 data:
m2_data_array_full = np.array(ren_data)[0:6, :]
m2_data_array_first = np.array(ren_data)[0:6, 0:330]
m2_data_array_second = np.array(ren_data)[0:6, 330:661]

    

# Model 1 data:
m1_data_array_full = np.array(ren_data)[0:6, :]
m1_data_array_first = np.array(ren_data)[0:6, 0:330]
m1_data_array_second = np.array(ren_data)[0:6, 330:661]




######################################################
#-------------- Optimal parameters ------------------#
######################################################

# ----------------- Model 1 -----------------

initial_params_m1 = [0.435 , 0.337, 0.518, 0.269, 0.231, 0.488, 0.173, 0.17]

m1_optimal_params_full = [0.17849227898125014, 0.000657794099057409, 0.1346069222493534, 0.23822844897193443, 0.00010025536072665744, 0.16785437327315875, 7.707703939644629e-05, 0.00016073535550350357]


m1_optimal_params_full = estimate_hyperparameters_m1(m1_data_array_full, initial_params_m1, futures_TTM, m1_bounds)

m1_optimal_params_first = estimate_hyperparameters_m1(m1_data_array_first, initial_params_m1, futures_TTM, m1_bounds)

m1_optimal_params_second = estimate_hyperparameters_m1(m1_data_array_second, initial_params_m1, futures_TTM, m1_bounds)

# Rerun
#[0.17849227898125014, 0.000657794099057409,0.1346069222493534,0.23822844897193443,0.00010025536072665744,0.16785437327315875,7.707703939644629e-05,0.00016073535550350357]
m1_optimal_params_full = estimate_hyperparameters_m1(m1_data_array_full, m1_optimal_params_full['optimal parameters'], futures_TTM, m1_bounds)

m1_optimal_params_first = estimate_hyperparameters_m1(m1_data_array_first, m1_optimal_params_first['optimal parameters'], futures_TTM, m1_bounds)

m1_optimal_params_second = estimate_hyperparameters_m1(m1_data_array_second, m1_optimal_params_second['optimal parameters'], futures_TTM, m1_bounds)



# ----------------- Model 2 -----------------

#[0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1]
#[0.19420987747364793, 0.03918963351912097, 1.0312708315235384, 0.39063075854988233, 0.43829831948664316, 0.2314387402874212, 0.46766866560630305, -0.2017901477934989, 0.12118969803500919, 0.12043699608951561, 0.10914420160634943, -0.13043144445329105, 0.09212280990867878, 0.0001720385739828221]
m2_parameters_full =  [1.0849182360881828, 0.2841048272503885, 2.6275784309323416, 0.19085951481757069, 1.90402066501319736, 0.2613138387916598, 0.9494905040806007, -0.48904234322926193, 0.07841584273341712, -0.45431290574702304, 0.5201980829513129, -0.5983798735970491, -0.9506142618624227, 0.0007052850897346007]
m2_optimal_params_full = [1.0849182360881828, 0.2841048272503885, 2.6275784309323416, 0.19085951481757069, 1.90402066501319736, 0.2613138387916598, 0.9494905040806007, -0.48904234322926193, 0.07841584273341712, -0.45431290574702304, 0.5201980829513129, -0.5983798735970491, -0.9506142618624227, 0.0007052850897346007]
print("Done with M2 full")

m2_parameters_first = [2.084919585083964, 0.28413325360180414, 2.627577531716743, 0.19085557509299197, 1.9040209373271346, 0.261315199140686, 0.9494903414371904, -0.48903774423848145, 0.07841379587137932, -0.45430960649068414, 0.3201996650155888, -0.19838408619343406, -0.3506139073917787, 0.00019590358662117127]
m2_optimal_params_first = [2.084919585083964, 0.28413325360180414, 2.627577531716743, 0.19085557509299197, 1.9040209373271346, 0.261315199140686, 0.9494903414371904, -0.48903774423848145, 0.07841379587137932, -0.45430960649068414, 0.3201996650155888, -0.19838408619343406, -0.3506139073917787, 0.00019590358662117127]
print("Done with M2 first")

m2_parameters_second = [1.0930092761081243, 0.2648340920089138, 0.8215101491549283, 0.17768080483753934, 1.9040861948111847, 0.24401224151752637, 1.0001071172099074, -0.12251697457852263, 0.4070604980678292, -0.43584506970932996, 0.39398969270906686, -0.11828362276427971, -0.3939375645844859, 0.0002359018415207992]
m2_optimal_params_second =  [1.0930092761081243, 0.2648340920089138, 0.8215101491549283, 0.17768080483753934, 1.9040861948111847, 0.24401224151752637, 1.0001071172099074, -0.12251697457852263, 0.4070604980678292, -0.43584506970932996, 0.39398969270906686, -0.11828362276427971, -0.3939375645844859, 0.0002359018415207992]

m2_optimal_params_full = estimate_hyperparameters_m2(m2_data_array_full, m2_optimal_params_full, futures_TTM, m2_bounds)
print("Done with M2 full")
m2_optimal_params_first = estimate_hyperparameters_m2(m2_data_array_first, m2_optimal_params_first, futures_TTM, m2_bounds)
print("Done with M2 first")
m2_optimal_params_second = estimate_hyperparameters_m2(m2_data_array_second, m2_optimal_params_second, futures_TTM, m2_bounds)


# Rerun

m2_optimal_params_full = estimate_hyperparameters_m2(m2_data_array_full, m2_optimal_params_full['optimal parameters'], futures_TTM, m2_bounds)
print("Done with M2 full")
m2_optimal_params_first = estimate_hyperparameters_m2(m2_data_array_first, m2_optimal_params_first['optimal parameters'], futures_TTM, m2_bounds)
print("Done with M2 first")
m2_optimal_params_second = estimate_hyperparameters_m2(m2_data_array_second, m2_optimal_params_second['optimal parameters'], futures_TTM, m2_bounds)


 
# ----------------- Model 3 -----------------

initial_params_m3 = [0.339214828197962, 0.016997407831544718, 0.9569232133069763, 0.39767976403351374, 0.42905920216770027, 0.30694433482367095, 0.4523480996393099, -0.06255588996523798, 0.032131734373888526, 0.014112010855355274, 0.0977973944731917, -0.14074148561755362, 0.09049977571194741, 0.0001853385615369501, 0.0006771734450511502]
m3_optimal_params_full =  [0.339214828197962, 0.016997407831544718, 0.9569232133069763, 0.39767976403351374, 0.42905920216770027, 0.30694433482367095, 0.4523480996393099, -0.06255588996523798, 0.032131734373888526, 0.014112010855355274, 0.0977973944731917, -0.14074148561755362, 0.09049977571194741, 0.0001853385615369501, 0.0006771734450511502]
m3_optimal_params_full = estimate_hyperparameters_m3(m3_data_array_full, initial_params_m3, futures_TTM, m3_bounds)

m3_optimal_params_first = estimate_hyperparameters_m3(m3_data_array_first, initial_params_m3, futures_TTM, m3_bounds)

m3_optimal_params_second = estimate_hyperparameters_m3(m3_data_array_second, initial_params_m3, futures_TTM, m3_bounds)

# Rerun

m3_optimal_params_full = estimate_hyperparameters_m3(m3_data_array_full, m3_optimal_params_full['optimal parameters'], futures_TTM, m3_bounds)

m3_optimal_params_first = estimate_hyperparameters_m3(m3_data_array_first, m3_optimal_params_first['optimal parameters'], futures_TTM, m3_bounds)


m3_optimal_params_second = estimate_hyperparameters_m3(m3_data_array_second, m3_optimal_params_second['optimal parameters'], futures_TTM, m3_bounds)


print("Optimal parameters finished at:", time.ctime())

######################################################
#----------------- Standard errors -----------------#
######################################################

# ----------------- Model 1 -----------------

m1_info_matrix_full = info_matrix_loglike(m1_optimal_params_full['optimal parameters'], m1_data_array_full, futures_TTM, Kalman_filter_m1)

m1_info_matrix_first = info_matrix_loglike(m1_optimal_params_first['optimal parameters'], m1_data_array_first, futures_TTM, Kalman_filter_m1)

m1_info_matrix_second = info_matrix_loglike(m1_optimal_params_second['optimal parameters'], m1_data_array_second, futures_TTM, Kalman_filter_m1)


m1_SD_full = np.diagonal(np.sqrt(linalg.inv(-m1_info_matrix_full)))

m1_SD_first = np.diagonal(np.sqrt(linalg.inv(-m1_info_matrix_first)))

m1_SD_second = np.diagonal(np.sqrt(linalg.inv(-m1_info_matrix_second)))



# ----------------- Model 2 -----------------


m2_info_matrix_full = info_matrix_loglike(m2_optimal_params_full['optimal parameters'], m2_data_array_full, futures_TTM, Kalman_filter_m2)

m2_info_matrix_first = info_matrix_loglike(m2_optimal_params_first['optimal parameters'], m2_data_array_first, futures_TTM, Kalman_filter_m2)

m2_info_matrix_second = info_matrix_loglike(m2_optimal_params_second['optimal parameters'], m2_data_array_second, futures_TTM, Kalman_filter_m2)


m2_SD_full = np.diagonal(np.sqrt(linalg.inv(-m2_info_matrix_full)))

m2_SD_first = np.diagonal(np.sqrt(linalg.inv(-m2_info_matrix_first)))

m2_SD_second = np.diagonal(np.sqrt(linalg.inv(-m2_info_matrix_second)))


# ----------------- Model 3 -----------------

m3_info_matrix_full = info_matrix_loglike(m3_optimal_params_full['optimal parameters'], m3_data_array_full, futures_TTM, ExtendedKalmanFilter)

m3_info_matrix_first = info_matrix_loglike(m3_optimal_params_first['optimal parameters'], m3_data_array_first, futures_TTM, ExtendedKalmanFilter)

m3_info_matrix_second = info_matrix_loglike(m3_optimal_params_second['optimal parameters'], m3_data_array_second, futures_TTM, ExtendedKalmanFilter)


m3_SD_full =  np.diagonal(np.sqrt(linalg.inv(-m3_info_matrix_full)))

m3_SD_first =  np.diagonal(np.sqrt(linalg.inv(-m3_info_matrix_first)))

m3_SD_second =  np.diagonal(np.sqrt(linalg.inv(-m3_info_matrix_second)))


print("Standart deviation finished at:", time.ctime())


######################################################
#--------------- Log-Likelihood ---------------------#
######################################################

# ----------------- Model 1 -----------------


m1_kf_opt_full = Kalman_filter_m1(m1_optimal_params_full, m1_data_array_full, futures_TTM)

m1_kf_opt_first = Kalman_filter_m1(m1_optimal_params_first['optimal parameters'], m1_data_array_first, futures_TTM)

m1_kf_opt_second = Kalman_filter_m1(m1_optimal_params_second['optimal parameters'], m1_data_array_second, futures_TTM)

# Log Likelihood

m1_kf_opt_loglike_full = m1_kf_opt_full.main(m1_data_array_full)['log_likelihood']

m1_kf_opt_loglike_first = m1_kf_opt_full.main(m1_data_array_first)['log_likelihood']

m1_kf_opt_loglike_second = m1_kf_opt_full.main(m1_data_array_second)['log_likelihood']


# ----------------- Model 2 -----------------


m2_kf_opt_full = Kalman_filter_m2(m2_optimal_params_full, m2_data_array_full, futures_TTM)

m2_kf_opt_first = Kalman_filter_m2(m2_optimal_params_first, m2_data_array_first, futures_TTM)

m2_kf_opt_second = Kalman_filter_m2(m2_parameters_second, m2_data_array_second, futures_TTM)

# Log Likelihood

m2_kf_opt_loglike_full = m2_kf_opt_full.main(m2_data_array_full)['log_likelihood']

m2_kf_opt_loglike_first = m2_kf_opt_full.main(m2_data_array_first)['log_likelihood']

m2_kf_opt_loglike_second = m2_kf_opt_full.main(m2_data_array_second)['log_likelihood']


# ----------------- Model 3 -----------------


m3_kf_opt_full = ExtendedKalmanFilter(m3_optimal_params_full, m3_data_array_full, futures_TTM)

m3_kf_opt_first = ExtendedKalmanFilter(m3_optimal_params_first['optimal parameters'], m3_data_array_first, futures_TTM)

m3_kf_opt_second = ExtendedKalmanFilter(m3_optimal_params_second['optimal parameters'], m3_data_array_second, futures_TTM)

# Log Likelihood

m3_kf_opt_loglike_full = m3_kf_opt_full.main(m3_data_array_full)['log_likelihood']

m3_kf_opt_loglike_first = m3_kf_opt_full.main(m3_data_array_first)['log_likelihood']

m3_kf_opt_loglike_second = m3_kf_opt_full.main(m3_data_array_second)['log_likelihood']


print("Log-Likelihood finished at:", time.ctime())


######################################################
#----------------- Time series ----------------------#
######################################################

# ----------------- Model 1 -----------------

m1_time_series = Time_series(m1_data_array_full, 50, Kalman_filter_m1, estimate_hyperparameters_m1, m1_optimal_params_full['optimal parameters'], futures_TTM, m1_bounds)


#Get RMSE and ME of time_series
m1_time_series = np.squeeze(np.asarray(m1_time_series)).T

m1_time_series_clean = np.nan_to_num(np.copy(m1_time_series)) 
#RMSE:
m1_time_series_RMSE = RMSE(m1_time_series_clean, np.zeros((6,50)))

m1_time_series_ME = mean_error(m1_time_series_clean, np.zeros((6,50)))


# ----------------- Model 2 -----------------

m2_time_series = Time_series(m2_data_array_full, 50, Kalman_filter_m2, estimate_hyperparameters_m2,  m2_optimal_params_full['optimal parameters'], futures_TTM, m2_bounds)


#Get RMSE and ME of time_series
m2_time_series = np.squeeze(np.asarray(m2_time_series)).T

m2_time_series_clean = np.nan_to_num(np.copy(m2_time_series)) 
#RMSE:
m2_time_series_RMSE = RMSE(m2_time_series_clean, np.zeros((6,50)))

m2_time_series_ME = mean_error(m2_time_series_clean, np.zeros((6,50)))


# ----------------- Model 3 -----------------

m3_time_series = Time_series(m3_data_array_full, 50, ExtendedKalmanFilter, estimate_hyperparameters_m3, m3_optimal_params_full['optimal parameters'], futures_TTM, m3_bounds)


#Get RMSE and ME of time_series
m3_time_series = np.squeeze(np.asarray(m3_time_series)).T

m3_time_series_clean = np.nan_to_num(np.copy(m3_time_series)).T
#RMSE:
m3_time_series_RMSE = RMSE(m3_time_series_clean, np.zeros((12,50)))

m3_time_series_ME = mean_error(m3_time_series_clean, np.zeros((12,50)))


print("Time-series finished at:", time.ctime())



######################################################
#---------------- Cross section ---------------------#
######################################################

# Handle data

# M1
m1_in_sample_data = ren_data.iloc[[0,2,4]]
m1_in_sample_data = np.squeeze(np.asarray([m1_in_sample_data]))

m1_out_sample_data = ren_data.iloc[[1,3,5]]
m1_out_sample_data = np.squeeze(np.asarray([m1_out_sample_data]))


# M2
m2_in_sample_data = ren_data.iloc[[0,2,4]]
m2_in_sample_data = np.squeeze(np.asarray([m2_in_sample_data]))

m2_out_sample_data = ren_data.iloc[[1,3,5]]
m2_out_sample_data = np.squeeze(np.asarray([m2_out_sample_data]))

# M3
m3_in_sample_data = ren_data.iloc[[0,2,4,6,8,10]]
m3_in_sample_data = np.squeeze(np.asarray([m3_in_sample_data]))

m3_out_sample_data = ren_data.iloc[[1,3,5,7,9,11]]
m3_out_sample_data = np.squeeze(np.asarray([m3_out_sample_data]))



#Set Nan entrences to zero:
#M1
m1_nan_mask = np.isnan(m1_out_sample_data)

m1_out_sample_data_clean = np.copy( m1_out_sample_data )
m1_out_sample_data_clean[m1_nan_mask] = 0

#M2
m2_nan_mask = np.isnan(m2_out_sample_data)

m2_out_sample_data_clean = np.copy( m2_out_sample_data )
m2_out_sample_data_clean[m2_nan_mask] = 0

#M3
m3_nan_mask = np.isnan(m3_out_sample_data)

m3_out_sample_data_clean = np.copy( m3_out_sample_data )
m3_out_sample_data_clean[m3_nan_mask] = 0




futures_TTM_in =  np.array([[30/365],
                             [90/365],
                             [270/365]])

futures_TTM_out =  np.array([[60/365],
                             [180/365],
                             [365/365]])

futures_TTM = np.array([[30/365],
                        [60/365],
                        [90/365],
                        [180/365],
                        [270/365],
                        [365/365]])




#Define optimal filter: in sample

# ----------------- Model 1 -----------------
m1_cross_kf_in = Kalman_filter_m1(m1_optimal_params_full['optimal parameters'], m1_in_sample_data, futures_TTM_in)


m1_cross_state = np.array(m1_cross_kf_in.main(m1_in_sample_data)['State_t_t']).reshape(-1, 2).T


m1_pred_obsv = price_function_m1(m1_cross_kf_in, m1_cross_state, futures_TTM_out).T


m1_pred_obsv_clean = np.copy( m1_pred_obsv )
m1_pred_obsv_clean[m1_nan_mask] = 0

#RMSE of predicted error
m1_cross_RMSE = RMSE(m1_pred_obsv_clean, m1_out_sample_data_clean)


#ME of predicted error
m1_cross_ME = mean_error(m1_pred_obsv_clean, m1_out_sample_data_clean)

# ----------------- Model 2 -----------------
m2_cross_kf_in = Kalman_filter_m2(m2_optimal_params_full, m2_in_sample_data, futures_TTM_in)


m2_cross_state = np.array(m2_cross_kf_in.main(m2_in_sample_data)['State_t_t']).reshape(-1, 3).T


m2_pred_obsv = price_function_m23(m2_cross_kf_in, m2_cross_state, futures_TTM_out).T


m2_pred_obsv_clean = np.copy( m2_pred_obsv )
m2_pred_obsv_clean[m2_nan_mask] = 0

#RMSE of predicted error
m2_cross_RMSE = RMSE(m2_pred_obsv_clean, m2_out_sample_data_clean)


#ME of predicted error
m2_cross_ME = mean_error(m2_pred_obsv_clean, m2_out_sample_data_clean)


# ----------------- Model 3 -----------------
m3_cross_kf_in = ExtendedKalmanFilter(m3_optimal_params_full['optimal parameters'], m3_in_sample_data, futures_TTM_in)


m3_cross_state = np.array(m3_cross_kf_in.main(m3_in_sample_data)['State_t_t']).reshape(-1, 3).T


m3_pred_obsv = price_function_m23(m3_cross_kf_in, m3_cross_state, futures_TTM_out).T

m3_pred_prices = price_function_m23(m3_cross_kf_in, m3_cross_state, futures_TTM_out).T
m3_pred_vol = est_vol(m3_cross_kf_in, m3_cross_state, futures_TTM_out).T

m3_pred_obsv = np.vstack((m3_pred_prices, m3_pred_vol))



m3_pred_obsv_clean = np.copy( m3_pred_obsv )
m3_pred_obsv_clean[m3_nan_mask] = 0

#RMSE of predicted error
m3_cross_RMSE = RMSE(m3_pred_obsv_clean, m3_out_sample_data_clean)


#ME of predicted error
m3_cross_ME = mean_error(m3_pred_obsv_clean, m3_out_sample_data_clean)



print("Cross-section finished at:", time.ctime())



######################################################
#--------------------- Plots ------------------------#
######################################################

start_date = pd.to_datetime("2007-01-31")
end_date = pd.to_datetime("2019-09-24")
num_steps = m1_data_array_full.shape[1]  # number of time steps = number of columns

time_index = pd.date_range(start=start_date, end=end_date, periods=num_steps)



# Error plots for model 1, model 2 and model 3 for all six prices 
# Model 3 also get a error plot for vol obsv


# State of models


m1_state = np.array(m1_kf_opt_full.main(m1_data_array_full)['State_t_t']).reshape(-1, 2).T

m2_state = np.array(m2_kf_opt_full.main(m2_data_array_full)['State_t_t']).reshape(-1, 3).T

m3_state = np.array(m3_kf_opt_full.main(m3_data_array_full)['State_t_t']).reshape(-1, 3).T


# Convience yield plots

#Convience yield
conv_yield = np.array = np.vstack( ( np.vstack( (m1_state[1], m2_state[1] )  ), m3_state[1] ) )

#Price plots
fig, axes = plt.subplots(3,1, figsize=(12, 15))  
axes = axes.flatten()  # Flatten to easily index subplots

for i in range(3):  
    axes[i].plot(time_index, conv_yield[i, :], linestyle='-', color='b', label='Conv. yield')
    axes[i].set_xlabel('Time')
    axes[i].set_ylabel('State Value')
    axes[i].set_title(f'Model {i+1}: Convenience yield ')
    axes[i].legend()
    axes[i].grid(True)

plt.tight_layout()
plt.show


#Volatility plots
vol =  np.vstack( ( m2_state[2], m3_state[2] ) )


fig, axes = plt.subplots(2,1, figsize=(12, 15))  
axes = axes.flatten()  # Flatten to easily index subplots

for i in range(2):  
    axes[i].plot(time_index, vol[i, :], linestyle='-', color='g', label='Volatility')
    axes[i].set_xlabel('Time')
    axes[i].set_ylabel('State Value')
    axes[i].set_title(f'Model {i+2}: Volatility ')
    axes[i].legend()
    axes[i].grid(True)

plt.tight_layout()
plt.show

#Error plots
m1_error = np.array(m1_kf_opt_full.main(m1_data_array_full)['Error']).reshape(-1, 6).T

m2_error = np.array(m2_kf_opt_full.main(m2_data_array_full)['Error']).reshape(-1, 6).T



m3_error_kf = ExtendedKalmanFilter(m3_optimal_params_full['optimal parameters'], m3_data_array_full, futures_TTM)


m3_error_state = np.array(m3_error_kf.main(m3_data_array_full)['State_t_t1']).reshape(-1, 3).T


m3_pred_prices_error = price_function_m23(m3_error_kf, m3_error_state, futures_TTM).T


m3_pred_vol_error = est_vol(m3_error_kf, m3_error_state, futures_TTM).T

m3_pred_obsv = np.vstack((m3_pred_prices_error, m3_pred_vol_error))

m3_nan_mask_full = np.isnan(m3_data_array_full)

m3_data_array_clean = np.copy( m3_data_array_full )
m3_data_array_clean[m3_nan_mask_full] = 0
m3_pred_obsv[m3_nan_mask_full] = 0


m3_error = m3_data_array_clean - m3_pred_obsv

# Filter out arrays with 11 rows

plt.figure(figsize=(10, 5))
for i in range(6):  
    plt.plot(time_index, m1_error[i, :],marker = ".", linestyle='None', label=f'e{i+1}')
plt.xlabel('Time Step')
plt.ylabel('log price')
plt.legend()
plt.grid(True)
plt.show




plt.figure(figsize=(10, 5))
for i in range(6):  
    plt.plot(time_index, m2_error[i, :],marker = ".", linestyle='None', label=f'e{i+1}')
plt.xlabel('Time Step')
plt.ylabel('log price')
plt.legend()
plt.grid(True)
plt.show



plt.figure(figsize=(10, 5))
for i in range(6):  
    plt.plot(time_index, m3_error[i, :],marker = ".", linestyle='None', label=f'e{i+1}')
plt.xlabel('Time Step')
plt.ylabel('log price')
plt.legend()
plt.grid(True)
plt.show

#Vol error
plt.figure(figsize=(10, 5))
for i in range(6,12):  
    plt.plot(time_index, m3_error[i, :],marker = ".", linestyle='None', label=f'e{i+1}')
plt.xlabel('Time Step')
plt.ylabel('log price')
plt.legend()
plt.grid(True)
plt.show





