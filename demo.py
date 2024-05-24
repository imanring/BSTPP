from bstpp.main import LGCP_Model, Hawkes_Model, load_Chicago_Shootings, load_Boko_Haram
import numpyro.distributions as dist
import numpy as np
import matplotlib.pyplot as plt
#set seed for reproducability
np.random.seed(16)

data = load_Chicago_Shootings()

column_names = ['UNEMP_DENS','MEDINC','MED_HV','assoc_plus','VACANT_DEN',
       'VAC_HU_pct','HCUND20K_L','POP_DENS','CT_SP_WCHI']

# Will produce a warning because we are using latitude, longitude coordinates instead 
# of geometrically projected coordinates. The area that we are looking at is small 
# enough that it doesn't matter if we are using the geometric projection though.

model = Hawkes_Model(data['events_2022'],#spatiotemporal points
                     data['boundaries'],#Chicago boundaries
                     365,#Time frame (1 yr)
                     True,#use Cox as background
                     spatial_cov=data['covariates'],#spatial covariate matrix
                     cov_names = column_names,#columns to use from covariates
                     a_0=dist.Normal(1,10), alpha = dist.Beta(20,60),#set priors
                     beta=dist.HalfNormal(2.0),sigmax_2=dist.HalfNormal(0.25)
                    )
# trains the model using Stochastic Variational Inference
# !! IMPORTANT !! run_svi first trains the variational distribution and then 
# samples the variational distribution to be consistent with mcmc methods.

model.run_svi(lr=0.02,num_steps=15000)
print("Log Expected Likelihood:",model.log_expected_likelihood(data['events_2023']))
print("Expected AIC:",model.expected_AIC())

model.plot_prop_excitation()
plt.show()

model.plot_trigger_posterior(trace=False)
plt.show()

model.plot_trigger_time_decay()
plt.show()

model.plot_spatial(include_cov=True)
plt.show()

model.cov_weight_post_summary(trace=True)
plt.show()




from bstpp.trigger import Trigger
import jax.numpy as jnp

# Defining a custom trigger function to illustrate the process

class spatial_double_exp(Trigger):
    def compute_trigger(self,pars,dif_mat):
         return jnp.exp(-jnp.abs(dif_mat).sum(axis=0)/pars['Lambda'])/(2*pars['Lambda'])**2
    
    def compute_integral(self,pars,limits):
        x_limits = limits[0] #shape [2,n]
        y_limits = limits[1] #shape [2,n]
        x_int = 1-0.5*jnp.exp(-jnp.abs(x_limits[0]/pars['Lambda'])) - \
            0.5*jnp.exp(-jnp.abs(x_limits[1]/pars['Lambda']))
        y_int = 1-0.5*jnp.exp(-jnp.abs(y_limits[0]/pars['Lambda'])) - \
            0.5*jnp.exp(-jnp.abs(y_limits[1]/pars['Lambda']))
        return x_int*y_int
    
    def simulate_trigger(self,pars):
        return np.random.laplace(size=2,scale=pars['Lambda'])
    
    def get_par_names(self):
        return ['Lambda']

# same as before except with new trigger!

model = Hawkes_Model(data['events_2022'],#spatiotemporal points
                     data['boundaries'],#Chicago boundaries
                     365,#Time frame (1 yr)
                     True,#use Cox as background
                     spatial_cov=data['covariates'],#spatial covariate matrix
                     cov_names = column_names,#columns to use from covariates
                     a_0=dist.Normal(1,10), alpha = dist.Beta(20,60),#set priors
                     beta=dist.HalfNormal(2.0),Lambda=dist.HalfNormal(0.5),
                     spatial_trig=spatial_double_exp
                    )
model.run_svi(lr=0.02,num_steps=15000)

lel = model.log_expected_likelihood(data['events_2023'])
print(f"Log Expected Likelihood: {lel}")
eaic = model.expected_AIC()
print(f"Expected AIC {eaic}")

model.plot_trigger_posterior(trace=True)
plt.show()

model.plot_trigger_time_decay()
plt.show()

model.plot_spatial(include_cov=True)
plt.show()

model.plot_temporal()
plt.show()

model.cov_weight_post_summary()
plt.show()