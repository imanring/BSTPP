from abc import ABC, abstractmethod 
import numpyro
import jax.numpy as jnp
import jax


class Trigger(ABC):
    def __init__(self,prior):
        """
        Abstract Trigger class to be extented for Hawkes models.
        
        Parameters
        ----------
        prior: dict of numpyro distributions
            Used to sample parameters for 
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
        return dict(zip(names,[numpyro.sample(n,self.prior[n]) for n in names]))
    
    @abstractmethod
    def compute_trigger(self,pars,mat):
        """
        Compute the trigger function
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
        jax numpy matrix [n,n]
        """
        pass
    
    @abstractmethod
    def compute_integral(self,pars,dif):
        """
        Compute the integral of the trigger function
        Parameters
        -----------
        pars: dict
            results from sample_parameters
        dif: jax numpy matrix
            limits of integration with shape
                temporal - [n]
                spatial - [2, 2, n]
                spatiotemporal - ([n], [2, 2, n])
        Returns
        -------
        jax numpy [n]
        """
        pass

    @abstractmethod
    def get_par_names(self):
        """
        Returns
        -------
        list of names of parameters
        """
        pass

class Temporal_Power_Law(Trigger):
    """
    Power Law Temporal trigger. Lomax distribution given by,

    $$f(t;\beta,\gamma) = \beta \gamma ^\beta (\gamma + t)^{-\beta - 1}$$
    """

    def compute_trigger(self,pars,mat):
        #\beta * \gamma ^ \beta * (\gamma + t) ^ (- \beta - 1)
        return pars['beta']/pars['gamma'] * (1 + mat/pars['gamma']) ** (-pars['beta'] - 1)

    def compute_integral(self,pars,dif):
        return 1-(1+dif/pars['gamma'])**(-pars['beta'])

    def get_par_names(self):
        return ['beta','gamma']

class Temporal_Exponential(Trigger):
    """
    Temporal exponential trigger function given by,

    $$f(t;\beta) = \frac{1}{\beta} e^{-t/\beta}$$
    """
    
    def compute_trigger(self,pars,mat):
        return jnp.exp(-mat/pars['beta'])/pars['beta']
    
    def compute_integral(self,pars,dif):
        return 1-jnp.exp(-dif/pars['beta'])
    
    def get_par_names(self):
        return ['beta']


class Spatial_Symmetric_Gaussian(Trigger):
    """
    Single parameter symmetric spatial gaussian trigger given by,

    $$\varphi(<x,y>;\sigma_x^2) = \frac{1}{2 \pi \sigma_x} exp(-\frac{1}{2\sigma_x^2} (x^2 + y^2))$$
    """
    
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