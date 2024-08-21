#!/usr/bin/env/python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 16:18:06 EDT 2024

@author: Andrius Burnelis
"""
import numpy as np, sys; sys.path.append('./')
from scipy.special import gamma
from base_model import BaseModel
from data_loader import DataLoader
from numpy import typing as npt

class BS_C(BaseModel):
    """
    This model is a simple model that includes information about the bound state, and the asymptotic normalization coefficients.
    This has a fixed error model.
    """
    def __init__(self, data : npt.ArrayLike, param_prior : npt.ArrayLike, norm_prior : npt.ArrayLike, use_theory_cov : bool):
        super().__init__(data, param_prior, norm_prior, use_theory_cov)

        # Set the Rutherford amplitude
        self.f_r = (-self.etavalue_cs / (2 * self.kvalue_cs)) * (
            1 / np.sin((self.theta_cs * (np.pi / 180)) / 2)**2)**(1 + (1.j) * self.etavalue_cs)

        ########################################
        # Precompute values used in cs_theory()
        ########################################
        # For r1plus
        self.r1plus_val1 = 2 * self.gamma1_plus**2 * gamma(2 + self.eta_B_plus)**2
        self.r1plus_val2 = 4 * self.kc_plus * self.H_plus
        self.r1plus_val3 = 2.j * self.kc_plus * self.eta_B_plus * (1 - self.eta_B_plus**2) * self.H_prime_plus
        # For r1minus
        self.r1minus_val1 = 2 * self.gamma1_minus**2 * gamma(2 + self.eta_B_minus)**2
        self.r1minus_val2 = 4 * self.kc_minus * self.H_minus
        self.r1minus_val3 = 2.j * self.kc_minus * self.eta_B_minus * (1 - self.eta_B_minus**2) * self.H_prime_minus
        # Legendre polynomials + derivatives
        self.LP1 = np.cos(self.theta_cs * (np.pi / 180))
        self.DLP1 = np.sin(self.theta_cs * (np.pi / 180))
        # Phase terms
        alpha_1 = np.arcsin((self.etavalue_cs / np.sqrt(1 + np.square(self.etavalue_cs))))
        self.phase_1 = np.cos(2 * alpha_1) + (1.j) * np.sin(2 * alpha_1)
        # ERE1
        self.ERE1_val1 = (self.kcvalue_cs**2 + self.kvalue_cs**2) * self.Hvalue_cs
        # f_c
        self.fc_val1 = self.kvalue_cs**2 * self.phase_1 * self.LP1
        # f_i
        self.fi_val1 = self.kvalue_cs**2 * self.phase_1 * self.DLP1

        # Set the total dimension
        self.total_dim -= 2  # Remove the truncation uncertainty parameters

        # Set up the covariance matrix
        cov_expt_matrix = np.diag(self.err_cs**2)
        cov_theory_matrix = self.cov_theory()
        self.thing = cov_theory_matrix
        if use_theory_cov:
            cov_matrix = cov_expt_matrix + cov_theory_matrix
        else:
            cov_matrix = cov_expt_matrix
        self.inv_cov_matrix = np.linalg.inv(cov_matrix)
        self.Q_rest = np.sum(np.log(1.0 / np.sqrt(2.0 * np.pi * np.diag(cov_matrix))))



    def cs_theory(self, params : npt.ArrayLike, order : int) -> npt.ArrayLike:
        """
        This method utilizes the existence of a bound state and also the asymptotic normalization coefficients
        to compute the theoretical cross section at the energies and angles where we have data.

        NOTE: C1+/- is really (C1+/-)^2 since we are sampling C1 squared
        """
        # Extract the parameters from params
        A0, r0, C1plus, P1plus, C1minus, P1minus = params

        # Compute r1plus and r1minus based on the relation
        r1plus = np.real(-self.r1plus_val1 / C1plus + (P1plus * self.gamma1_plus**2) + (self.r1plus_val2) + (self.r1plus_val3))
        r1minus = np.real(-self.r1minus_val1 / C1minus + (P1minus * self.gamma1_minus**2) + (self.r1minus_val2) + (self.r1minus_val3))

        # Compute A1plus and A1minus based on the bound state condition
        A1plus = - (r1plus * self.gamma1_plus**2) / 2 + (0.25 * P1plus * self.gamma1_plus**4) + (
            2 * self.kc_plus * (self.gamma1_plus**2 - self.kc_plus**2)) * self.H_plus
        A1minus = - (r1minus * self.gamma1_minus**2) / 2 + (0.25 * P1minus * self.gamma1_minus**4) + (
            2 * self.kc_minus * (self.gamma1_minus**2 - self.kc_minus**2)) * self.H_minus

       # Set the parameterization based on the order
        if order == 0:
            # All parameters switched off
            A0 = 0
            r0 = 0
            A1plus = 0
            r1plus = 0
            P1plus = 0
            A1minus = 0
            r1minus = 0
            P1minus = 0
        elif order == 1:
            # Only r0, r1+, P1+/- are non-zero
            A0 = 0
            A1plus = 0
            A1minus = 0
            r1minus = 0
        elif order == 2:
            # Include all parameters
            pass
        else:
            raise Exception('Order {} not implemented!'.format(order))

        K_0 = (1 / (2 * self.kcvalue_cs)) * (-A0 + 0.5 * r0 * self.kvalue_cs**2)
        K_1_plus = (1 / (2 * self.kcvalue_cs**3)) * (-A1plus + 
                0.5 * r1plus * self.kvalue_cs**2 + 
                    0.25 * P1plus * self.kvalue_cs**4)
        K_1_minus = (1 / (2 * self.kcvalue_cs**3)) * (-A1minus + 
                0.5 * r1minus * self.kvalue_cs**2 + 
                    0.25 * P1minus * self.kvalue_cs**4)
        
        ERE_0 = (2 * self.kcvalue_cs * (K_0 - self.Hvalue_cs)) / self.C0_2value_cs
        ERE_1_plus = (2 * self.kcvalue_cs / (9 * self.C1_2value_cs)) * (
            self.kcvalue_cs**2 * K_1_plus - self.ERE1_val1)
        ERE_1_minus = (2 * self.kcvalue_cs / (9 * self.C1_2value_cs)) * (
            self.kcvalue_cs**2 * K_1_minus - self.ERE1_val1)

        # Compute the amplitude of each of the components
        # Rutherford - precomputed in __init__()
        
        # Coulomb
        f_c = self.f_r + (1 / ERE_0) + self.fc_val1 * (
            (2 / ERE_1_plus) + (1 / ERE_1_minus))

        # Interaction
        f_i = self.fi_val1 * ((1 / ERE_1_minus) - (1 / ERE_1_plus))

        sigma = 10 * (np.abs(f_c)**2 + np.abs(f_i)**2)
        sigma_R = 10 * np.abs(self.f_r)**2
        sigma_ratio = sigma / sigma_R
        return sigma_ratio



    def get_cs_theory(self, params : npt.ArrayLike, theta_deg : npt.ArrayLike, E_lab : npt.ArrayLike) -> npt.ArrayLike:
        """
        This method computes the theoretical cross section at the energies and angles where we have data.
        """
        pass


    
    def cov_theory(self):
        """
        Computes the covariance theory matrix.
        """
        y0 = np.reshape(self.cs_LO_values, (1, len(self.cs_LO_values)))
        c_rms = 0.70 # Also just sqrt(c_bar_squared)
        Lambda = 200.0 / self.h_bar_c  # fm^-1

        # Convert theta to radians
        theta_rad = self.theta_cs * (np.pi / 180)

        # Get the momentum transfer
        q = 2.0 * self.kvalue_cs * np.sin(theta_rad / 2.0)
        Q = ((np.array([max(q[t], self.kvalue_cs[t])
                    for t in range(len(theta_rad))])) / Lambda)  # dimless
        Q = np.reshape(Q, (1, len(Q)))
        K = 2.0  # N3LO is taken as dominant theoretical error
        cov_theory_cs = (c_rms * y0 * (Q**(K + 1))).transpose() @ (
            c_rms * y0 * (Q**(K + 1))) / (1 - Q.transpose() @ Q)
        return cov_theory_cs
    


    def log_prior(self, parameters):
        """
        Computes the log-prior of a given set of parameters.
        """
        # Cast the parameters to an array
        parameters = np.array(parameters)

        # If all the parameters fall within the bounds of the prior, compute the prior
        if np.logical_and(self.prior_info[:, 0] <= parameters, parameters <= self.prior_info[:, 1]).all():
            log_p = 0.0

            # Sum over all the contributions to the prior
            for i in range(0, self.total_dim):
                mu, sigma = self.prior_info[i, 2:]
                log_p += self.lp_gauss(parameters[i], mu, sigma, self.prior_info[i, :2])

            return log_p
        else:
            return -np.inf
        

    
    def log_likelihood(self, parameters):
        """
        Determines the log-likelihood of a set of parameters.
        """
        # Cast the parameters to an array
        parameters = np.array(parameters)

        # Unpack the erps and the normalization parameters
        params = parameters[:self.erp_dim]
        params_f = parameters[self.erp_dim:]

        # If the parameters are within the bounds, compute the log-likelihood
        if np.logical_and(self.bounds[:, 0] <= parameters, parameters <= self.bounds[:, 1]).all():
            # Get the normalizations and theory
            norm = params_f[self.data[:, 4].astype(int)]
            theory = self.cs_theory(params, order = 2)

            # Compute the chi squared (data is already in chi_squared)
            chi2 = self.chi_squared(theory, norm)[0]

            # Compute and return the log-likelihood
            log_L = -0.5 * chi2 + self.Q_rest
            return log_L
        else:
            return -np.inf
        

    def chi_squared(self, theory, norm):
        """
        Returns the chi squared and the residual when given theory, data, and norm.
        """
        theory = np.asarray(theory)
        r = norm * theory - self.cs_data
        chi_2 = r.transpose() @ self.inv_cov_matrix @ r
        return chi_2, r










def main():
    # Load the data
    E_min = 0.676
    E_max = 2.624
    which_data = 'som'
    loader = DataLoader(E_min, E_max, which_data)
    data = loader.get_data()
    gauss_prior_f = loader.get_normalization_priors()

    # Set up the prior parameters
    param_bounds = np.array([[-0.02, 0.06], [-3, 3], [5.0, 25.0], [-6, 6], [5.0, 25.0], [-6, 6]])
    params_prior = np.array([[0.025, 0.015], [0.8, 0.4], [13.84, 1.63], [0.0, 1.6], [12.59, 1.85], [0.0, 1.6]]) # center, width
    gauss_prior_params = np.hstack([param_bounds, params_prior])

    # Define the model
    model = BS_C(data, gauss_prior_params, gauss_prior_f, True)

    # Generate the test parameters
    # test = np.concatenate([[1.0], [0.8], gauss_prior_params[:, 2], gauss_prior_f[:, 2]])
    
    test = np.concatenate([[1.0], [0.8], gauss_prior_params[:, 2], [0.97, 0.93, 1.07, 1.02, 1.04, 1.04, 1.05]])

    print(model.log_posterior(test))


if __name__ == "__main__":
    main()