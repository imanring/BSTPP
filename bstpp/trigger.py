from abc import ABC, abstractmethod 
import numpyro
from scipy.stats import lomax
import jax.numpy as jnp
import numpy as np
import jax


class Trigger(ABC):
    def __init__(self,prior):
        """
        Abstract Trigger class to be extented for Hawkes models. The trigger is assumed to be a pdf and the reproduction rate is coded separately. The required methods to implement are:
        
        - `compute_trigger`: compute the trigger function (pdf)
        
        - `compute_integral`: compute the integral of the trigger function given limits (cdf)
        
        - `get_par_names`: returns a list of the parameter names used in the trigger function
        
        `simulate_trigger` is used only if a user wishes to simulate from the trigger function.
        
        Parameters
        ----------
        prior: dict of numpyro distributions
            Used to sample parameters for trigger
        """
        self.prior = prior

    def sample_parameters(self):
        """
        Sample parameters using numpyro
        e.g. return {'beta': numpyro.sample('beta', self.prior['beta'])}
        
        Returns
        -------
        dict of a single sample of parameters
        """
        names = self.get_par_names()
        return {n:numpyro.sample(n,self.prior[n]) for n in names}
    
    @abstractmethod
    def simulate_trigger(self,pars):
        """
        Simulate a point from the trigger function (assuming the trigger is a pdf). Optional. Only necessay for data simulation.
        Parameters
        ----------
        pars: dict
            parameters for the trigger to generate point.
        Returns
        -------
            spatial triggers - np.array [2]
            temporal triggers - float
        """
        pass
    
    @abstractmethod
    def compute_trigger(self,pars,mat):
        """
        Compute the trigger function. Computes the trigger function for the [n,n] difference matrix of points.
        Parameters
        ----------
        pars: dict
            results from sample_parameters
        mat: jax numpy matrix
            Difference matrix, whose shape is different for each kind of trigger.
                 temporal triggers - [n, n]
                 spatial triggers - [2, n, n]
                 spatiotemporal triggers - [3, n, n]
        Returns
        -------
        jax numpy matrix [n,n]. Trigger function computed for each entry in the matrix
        """
        pass
    
    @abstractmethod
    def compute_integral(self,pars,limits):
        """
        Compute the integral of the trigger function from the given limits. For temporal triggers, the integral is computed from 0 to the upper bound. For spatial triggers, the integral is over the rectangle defined by [[x_max,x_min],[y_max,y_min]]
        Parameters
        -----------
        pars: dict
            results from sample_parameters
        limits: jax numpy matrix
            limits of integration with shape
                temporal - [n] compute integal from 0 to limit
                spatial - [2, 2, n] compute integral over rectangle defined by [[x_max,x_min],[y_max,y_min]]
                spatiotemporal - ([n], [2, 2, n]) combination of temporal limits and spatial limits
        Returns
        -------
        jax numpy [n]
        """
        pass

    @abstractmethod
    def get_par_names(self):
        """
        Get list of parameter names. Parameter names may not overlap with any other parameter in the model.
        Excluded names include ['alpha','a_0','b_0','f_xy','v_xy','f_t','v_t','w','mu_xyt','rate_t','z_spatial','z_temporal','rate_xy']. Each parameter named here must have a prior with the same name specified in the model.
        Returns
        -------
        list of names of parameters
        """
        pass

class Temporal_Power_Law(Trigger):

    def __init__(self,prior):
        r"""
        Power Law Temporal trigger. Lomax distribution given by,
    
        $$f(t;\beta,\gamma) = \beta \gamma^\beta (\gamma + t)^{-\beta - 1}$$

        """
        super().__init__(prior)
    
    def simulate_trigger(self,pars):
        return lomax.rvs(pars['beta'])*pars['gamma']

    def compute_trigger(self,pars,mat):
        #\beta * \gamma ^ \beta * (\gamma + t) ^ (- \beta - 1)
        return pars['beta']/pars['gamma'] * (1 + mat/pars['gamma']) ** (-pars['beta'] - 1)

    def compute_integral(self,pars,dif):
        return 1-(1+dif/pars['gamma'])**(-pars['beta'])

    def get_par_names(self):
        return ['beta','gamma']

    

class Temporal_Exponential(Trigger):
    r"""
    Temporal exponential trigger function given by,

    $$f(t;\beta) = \frac{1}{\beta} e^{-t/\beta}$$
    
    """
    
    def simulate_trigger(self, pars):
        return np.random.exponential(pars['beta'])
    
    def compute_trigger(self,pars,mat):
        return jnp.exp(-mat/pars['beta'])/pars['beta']
    
    def compute_integral(self,pars,dif):
        return 1-jnp.exp(-dif/pars['beta'])
    
    def get_par_names(self):
        return ['beta']


class Spatial_Symmetric_Gaussian(Trigger):
    r"""
    Single parameter symmetric spatial gaussian trigger given by,

    $$\varphi(\mathbf{x};\sigma_x^2) = \frac{1}{2 \pi \sigma_x} exp(-\frac{1}{2\sigma_x^2} \mathbf{x} \cdot \mathbf{x})$$
    
    """

    def simulate_trigger(self, pars):
        return np.random.normal(scale=pars['sigmax_2']**0.5,size=2)
    
    def compute_trigger(self,pars,mat):
        S_diff_sq=(mat[0]**2)/pars['sigmax_2']+(mat[1]**2)/pars['sigmax_2']
        return jnp.exp(-0.5*S_diff_sq)/(2*jnp.pi*pars['sigmax_2'])
    
    def compute_integral(self,pars,dif):
        gaussianpart1 = 0.5*jax.scipy.special.erf(dif[0,0]/jnp.sqrt(2*pars['sigmax_2']))+\
                    0.5*jax.scipy.special.erf(dif[0,1]/jnp.sqrt(2*pars['sigmax_2']))
        
        gaussianpart2 = 0.5*jax.scipy.special.erf(dif[1,0]/jnp.sqrt(2*pars['sigmax_2']))+\
                    0.5*jax.scipy.special.erf(dif[1,1]/jnp.sqrt(2*pars['sigmax_2']))
        return gaussianpart2*gaussianpart1

    def get_par_names(self):
        return ['sigmax_2']