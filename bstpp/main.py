# general libraries
import os
from io import BytesIO
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from math import erf, ceil
import warnings 
import dill
import pickle
import pkgutil

# JAX
import jax.numpy as jnp
from jax import random
import numpyro.distributions as dist
from numpyro.diagnostics import hpdi
from jax.scipy.special import logsumexp
from numpyro.infer import log_likelihood

from .utils import * 
from .inference_functions import *
from .trigger import *
from .vae_functions import *


def load_Boko_Haram():
    """
    Load Boko Haram dataset
    Returns
    -------
    dict
        events: event dataset from https://ucdp.uu.se/downloads/
        covariates: covariates from PRIO-GRID (https://grid.prio.org/#/)
    """
    events = pd.read_csv(BytesIO(pkgutil.get_data(__name__, "data/BH_conflicts.csv")))
    cov = pd.read_csv(BytesIO(pkgutil.get_data(__name__, "data/BH_cov.csv")))
    boundaries = np.array([[3,15.5],[4,16.5]])
    return {"events":events, "covariates":cov,'boundaries':boundaries}


def load_Chicago_Shootings():
    """
    Load Chicago Shootings dataset
    Returns
    -------
    dict
        Shooting report data from:
            https://data.cityofchicago.org/Public-Safety/Chicago-Shootings/fsku-dr7m
        Community Area boundaries from:
            https://data.cityofchicago.org/Facilities-Geographic-Boundaries/Boundaries-Community-Areas-current-/cauq-8yn6
        Community Area Covariates from:
            https://datahub.cmap.illinois.gov/maps/2a0b0316dc2c4ecfa40a171c635503f8/about
    """
    events_2022 = pd.read_csv(BytesIO(pkgutil.get_data(__name__, "data/Chicago_2022_xyt.csv")))
    events_2023 = pd.read_csv(BytesIO(pkgutil.get_data(__name__, "data/Chicago_2023_xyt.csv")))
    cov = gpd.read_file(BytesIO(pkgutil.get_data(__name__, "data/Chicago_cov.zip")))
    boundaries = gpd.read_file(BytesIO(pkgutil.get_data(__name__, "data/Boundaries - Community Areas (current).zip")))
    return {"events_2022":events_2022, "events_2023":events_2023,
            "covariates":cov, "boundaries":boundaries}


class Point_Process_Model:
    def __init__(self,model,data,A,T,spatial_cov=None,cov_names=None,
                 cov_grid_size=None,standardize_cov=True,**priors):
        """
        Spatiotemporal Point Process Model.
        The data is rescaled to fit in a 1x1 spatial grid and a lenght 50 time window. Posterior samples must be interpreted with this in mind.

        Parameters
        ----------
        model: str
            one of ['cox_hawkes','lgcp','hawkes'].
        data: str or pd.DataFrame
            either file path or DataFrame containing spatiotemporal data. Columns must include 'X', 'Y', 'T'.
        A: np.array [2x2], GeoDataFram
            Spatial region of interest. If np.array first row is the x-range, second row is y-range.
        T: float
            Maximum time in region of interest. Time is assumed to spart at 0.
        spatial_cov: str,pd.DataFrame,gpd.GeoDataFrame
            Either file path (.csv or .shp), DataFrame, or GeoDataFrame containing spatial covariates. 
            Spatial covariates must cover all the points in data.
            If spatial_cov is a csv or pd.DataFrame, the first 2 columns must be 'X', 'Y' and cov_grid_size must be specified.
        cov_names: list
            List of covariate names. Must all be columns in spatial_cov.
        cov_grid_size: list-like
            Spatial covariate grid (width, height).
        standardize_cov: bool
            Standardize covariates
        priors: dict
            priors for parameters (a_0,w,alpha,beta,sigmax_2). Must be a numpyro distribution.
        """
        if type(data)==str:
            data = pd.read_csv(data)
        self.data = data
        
        args={}
        args['T']=50
        # Spatial grid is 1x1
        args['t_min']=0
        args['x_min']=0
        args['x_max']=1
        args['y_min']=0
        args['y_max']=1
        args['model']=model

        if type(A) is gpd.GeoDataFrame:
            A_ = np.stack((A.bounds.min(axis=0)[['minx','miny']],
                      A.bounds.max(axis=0)[['maxx','maxy']])).T
            #proportion of area of rectangle A_ covered by A. Used for Hawkes integral.
            args['A_area'] = A.area.sum()/((A_[0,1]-A_[0,0])*(A_[1,1]-A_[1,0]))
        else:# A is rectangle specified by np.array
            args['A_area'] = 1
            A_ = A
        args['A_'] = A_
        
        #create computational grid
        n_t=50
        x_t = jnp.arange(0, args['T'], args['T']/n_t)
        args["n_t"]=n_t
        args["x_t"]=x_t
        
        n_xy = 25
        args["n_xy"]= n_xy
        cols = np.arange(0, 1, 1/n_xy)
        polygons = []
        for y in cols:
            for x in cols:
                polygons.append(Polygon([(x,y), (x+1/n_xy, y), (x+1/n_xy, y+1/n_xy), (x, y+1/n_xy)]))
        comp_grid = gpd.GeoDataFrame({'geometry':polygons,'comp_grid_id':np.arange(n_xy**2)})
        comp_grid.geometry = comp_grid.geometry.scale(xfact=A_[0,1]-A_[0,0],yfact=A_[1,1]-A_[1,0],
                                                      origin=(0,0)).translate(A_[0,0],A_[1,0])
        
        if type(A) is gpd.GeoDataFrame:
            # find grid cells overlapping with A
            comp_grid.crs = A.crs
            args['spatial_grid_cells'] = np.unique(comp_grid.sjoin(A, how='inner')['comp_grid_id'])
            self.A = A
        else:
            self.A = comp_grid
            args['spatial_grid_cells'] = np.arange(25**2)
        
        self.comp_grid = comp_grid
        self.T = T
        
        args,points = self._scale_xyt(data,args,comp_grid)
        self.points = points
        
        if args['model'] in ['lgcp','cox_hawkes']:
            args["gp_kernel"]=exp_sq_kernel 
            
            # temporal VAE training arguments
            args["hidden_dim_temporal"]= 35
            args["z_dim_temporal"]= 11

            # spatial VAE training arguments
            args["hidden_dim1_spatial"]= 75
            args["hidden_dim2_spatial"]= 50
            args["z_dim_spatial"]=20
        
            decoder_params = pickle.loads(pkgutil.get_data(__name__, "decoders/decoder_1d_T50_fixed_ls"))
            args["decoder_params_temporal"] = decoder_params
            
            #Load 2d spatial trained decoder
            decoder_params = pickle.loads(pkgutil.get_data(__name__, "decoders/2d_decoder_15_5_large.pkl"))
            args["decoder_params_spatial"] = decoder_params
        
        if spatial_cov is not None:
            #convert input into geopandas dataframe.
            if type(spatial_cov)==str:
                if spatial_cov[-4:] == '.zip' or spatial_cov[-4:] == '.shp':
                    spatial_cov = gpd.read_file(spatial_cov)
                else:
                    spatial_cov = pd.read_csv(spatial_cov)
            if type(spatial_cov) == pd.DataFrame:
                polygons = []
                for i in spatial_cov.index:
                    polygons.append(Polygon([(spatial_cov.loc[i,'X']-cov_grid_size[0]/2,
                                              spatial_cov.loc[i,'Y']-cov_grid_size[1]/2), 
                                             (spatial_cov.loc[i,'X']+cov_grid_size[0]/2,
                                              spatial_cov.loc[i,'Y']-cov_grid_size[1]/2), 
                                             (spatial_cov.loc[i,'X']+cov_grid_size[0]/2,
                                              spatial_cov.loc[i,'Y']+cov_grid_size[1]/2), 
                                             (spatial_cov.loc[i,'X']-cov_grid_size[0]/2,
                                              spatial_cov.loc[i,'Y']+cov_grid_size[1]/2)]))
                spatial_cov = gpd.GeoDataFrame(data=spatial_cov,geometry=polygons)
                spatial_cov.crs = self.A.crs
            spatial_cov['cov_ind'] = np.arange(len(spatial_cov))
            #find covariate cell index for each point
            self.points.crs = spatial_cov.crs
            args['cov_ind'] = self.points.sjoin(spatial_cov).sort_values(by='point_id')['cov_ind'].values
            if len(args['cov_ind']) != len(self.points):
                raise Exception("Spatial covariates are not defined for all data points!")
            
            args['num_cov'] = len(cov_names)
            self.cov_names = cov_names
            self.spatial_cov = spatial_cov
            
            X_s = spatial_cov[self.cov_names].values
            # standardize covariates
            if standardize_cov:
                args['spatial_cov'] = (X_s-X_s.mean(axis=0))/(X_s.var(axis=0)**0.5)
            else:
                args['spatial_cov'] = X_s
            
            #Create Computational Grid GeoDataFrame
            if args['model'] in ['lgcp','cox_hawkes']:
                self.comp_grid.crs = spatial_cov.crs
                #find covariate cell intersection with computational grid cells area
                intersect = gpd.overlay(self.comp_grid, spatial_cov, how='intersection', keep_geom_type=True)
                intersect['area'] = intersect.area/((A_[0,1]-A_[0,0])*(A_[1,1]-A_[1,0]))
                intersect = intersect[intersect['area']>1e-10]
                args['int_df'] = intersect
                #find cells on the computational grid that are in the domain
                args['spatial_grid_cells'] = np.unique(self.comp_grid.sjoin(spatial_cov, how='inner')['comp_grid_id'])
            else:
                args['cov_area'] = (spatial_cov.area/((A_[0,1]-A_[0,0])*(A_[1,1]-A_[1,0]))).values

        #Set up parameter priors
        default_priors = {}
        if 'num_cov' in args:
            default_priors["w"] = dist.Normal(jnp.zeros(args['num_cov']),jnp.ones(args["num_cov"]))
        args['sp_var_mu'] = 2.0
        for par, prior in priors.items():
            if isinstance(prior,dist.Distribution):
                default_priors[par] = prior
            else:
                raise Exception(f"Unknown argument {par}. Prior distributions must be instances of numpyro Distribution.")
        args['priors'] = default_priors
        self.args = args

    def load_rslts(self,file_name):
        """
        Load previously computed results
        Parameters
        ----------
        file_name: string
            File where pickled results are held
        """
        with open(file_name, 'rb') as f:
            output = pickle.load(f)
        if 'svi_results' in output:
            self.svi_results = output['svi_results']
        if 'mcmc' in output:
            self.mcmc = output['mcmc']
        self.samples = output['samples']

    def save_rslts(self,file_name):
        """
        Save previously computed results
        Parameters
        ----------
        file_name: string
            File where to save results
        """
        output = dict()
        if 'svi_results' in dir(self):
            output['svi_results'] = self.svi_results
        if 'mcmc' in dir(self):
            output['mcmc'] = self.mcmc
        output['samples'] = self.samples
        with open(file_name, 'wb') as f:
            output = pickle.dump(output,f)
        
    
    def run_svi(self,num_steps,lr,num_samples=1000,resume=False,plot_loss=True,**kwargs):
        """
        Perform Stochastic Variational Inference on the model.
        Parameters
        ----------
        num_samples: int, default=1000
            Number of samples to generate after SVI.
        resume: bool, default=False
            Pick up where last SVI run was left off. Can only be true if model has previous run_svi call.
        lr: float, default=0.001
            learning rate for SVI
        num_steps: int, default=10000
            Number of interations for SVI to run.
        plot_loss: bool
            
        auto_guide: numpyro AutoGuide, default=AutoMultivariateNormal
            See numpyro AutoGuides for details.
        init_strategy: function, default=init_to_median
            See numpyro init strategy documentation
        """
        rng_key, rng_key_predict = random.split(random.PRNGKey(10))
        rng_key, rng_key_post, rng_key_pred = random.split(rng_key, 3)
        self.args["num_samples"] = num_samples
        sites = list(self.get_params().keys())+['loglik','Itot_excite','Itot_txy']
        if resume:
            kwargs['num_steps'] = num_steps
            kwargs['lr'] = lr
            optimizer = numpyro.optim.Adam(
                jax.example_libraries.optimizers.inverse_time_decay(kwargs['lr'],kwargs['num_steps'],4)
            )
            self.svi.optim = optimizer
            self.svi_results = self.svi.run(rng_key, kwargs['num_steps'], self.args,
                                            init_state=self.svi_results.state)
            self.samples = get_samples(rng_key,self.model,self.svi.guide,self.svi_results,self.args,sites)
        else:
            self.svi,self.svi_results,self.samples=run_SVI(rng_key, self.model, self.args, num_steps, lr, sites, **kwargs)
        loss = np.asarray(self.svi_results.losses)
        plt.plot(np.arange(int(.01*len(loss)),len(loss)),loss[int(.01*len(loss)):])
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.show()

    
    def run_mcmc(self,batch_size=1,num_warmup=500,num_samples=1000,
                 num_chains=1,thinning=1):
        """
        Run MCMC posterior sampling on model.
        
        Parameters
        ----------
        batch_size: int
            See numpyro documentation for description
        num_warmup: int
        num_samples: int
        num_chains: int
        thinning: int
        """
        self.args["batch_size"]= batch_size
        self.args["num_warmup"]= num_warmup
        self.args["num_samples"] = num_samples
        self.args["num_chains"] = num_chains
        self.args["thinning"] = thinning
        rng_key, rng_key_predict = random.split(random.PRNGKey(10))
        rng_key, rng_key_post, rng_key_pred = random.split(rng_key, 3)
        
        self.mcmc = run_mcmc(rng_key_post, self.model, self.args)
        self.samples=self.mcmc.get_samples()
        

    def _scale_xyt(self,data,args,comp_grid):
        #scale temporal events
        t_events_total=data['T'].values/self.T*50
        args["t_events"]=t_events_total
        args['indices_t']=np.searchsorted(args['x_t'], t_events_total, side='right')-1
        
        #scale spatial events
        x_range = args['A_'][0]
        x_events_total=(data['X']-x_range[0]).to_numpy()
        x_events_total/=(x_range[1]-x_range[0])
        y_range = args['A_'][1]
        y_events_total=(data['Y']-y_range[0]).to_numpy()
        y_events_total/=(y_range[1]-y_range[0])
        
        xy_events_total=np.array((x_events_total,y_events_total)).transpose()

        geometry = gpd.points_from_xy(data.X, data.Y,crs=comp_grid.crs)
        points = gpd.GeoDataFrame(data=data,geometry=geometry)
        points['point_id'] = np.arange(len(data))
        
        #find grid cells where points are located
        args['indices_xy'] = points.sjoin(comp_grid).sort_values(by='point_id')['comp_grid_id'].values
        
        if len(args['indices_xy']) != len(points):
            raise Exception("Computational grid does not encompass all data points!")
            
        args["xy_events"]=xy_events_total.transpose()
        return args,points

    def log_expected_likelihood(self,data):
        """
        Computes the log expected likelihood for test data.
        $$E_{\theta|X}[\ell] = log(\frac{1}{S}\sum_{s=1}{S}{p(X|\theta^s)})$$
        Parameters
        ----------
        data: pd.DataFrame or str
            test events in the same format as original event dataset.
        """
        #Based on https://programtalk.com/vs4/python/pyro-ppl/numpyro/examples/baseball.py/
        if type(data)==str:
            data = pd.read_csv(data)
        test_args,points = self._scale_xyt(data,self.args.copy(),self.comp_grid)
        if 'cov_ind' in self.args:
            test_args['cov_ind'] = points.sjoin(self.spatial_cov).sort_values(by='point_id')['cov_ind'].values

        post_loglik = log_likelihood(self.model, self.samples, test_args)["t_events"]
        
        # computes expected log likelihood over the posterior
        exp_log_density = logsumexp(post_loglik, axis=0) - jnp.log(
            jnp.shape(post_loglik)[0]
        )
        return exp_log_density.sum().item()
    
    def expected_AIC(self):
        r"""
        Calculate the expected AIC over the posterior distribution.
        For $k = $ number of model parameters, expected AIC is defined as,
        $$E_{\theta|X}[AIC] = \frac{-2}{S}\sum_{s=1}{S}{log(p(X|\theta^s))} + 2k$$
        """
        k = sum(self.get_params().values())
        return -2*self.samples['loglik'].mean().item() + 2*k

    
    def cov_weight_post_summary(self,trace=False):
        """
        Plot and summarize posteriors of weights and bias.
        Returns
        -------
        pd.DataFrame
            summary of weights and bias
        trace: bool
            plot trace or histogram of posteriors
        """
        if 'samples' not in dir(self):
            raise Exception("MCMC posterior sampling has not been performed yet.")
        if 'spatial_cov' not in self.args:
            raise Exception("Spatial covariates were not included in the model.")
        
        n = self.samples['w'].shape[1]+1
        
        if trace:
            r = ceil(n**0.75)
            c = ceil(n/r)
        else:
            c = ceil(n**0.5)
            r = ceil(n/c)
        fig, ax = plt.subplots(r,c,figsize=(10,10), sharex=False)
        fig.suptitle('Covariate Weights', fontsize=16)
        for i in range(n-1):
            if trace:
                ax[i//c,i%c].plot(self.samples['w'].T[i])
                ax[i//c,i%c].set_ylabel(self.cov_names[i])
            else:
                ax[i//c,i%c].hist(self.samples['w'].T[i])
                ax[i//c,i%c].set_xlabel(self.cov_names[i])
        if trace:
            ax[(n-1)//c,(n-1)%c].plot(self.samples['a_0'])
            ax[(n-1)//c,(n-1)%c].set_ylabel("$a_0$")
        else:
            ax[(n-1)//c,(n-1)%c].hist(self.samples['a_0'])
            ax[(n-1)//c,(n-1)%c].set_xlabel("$a_0$")
        #clear unused parts of the grid
        for i in range(n,r*c):
            ax[i//c, i%c].axis('off')
        
        w_samps = np.concatenate((self.samples['w'],self.samples['a_0'].reshape(-1,1)),axis=1)
        mean = w_samps.mean(axis=0)
        std = w_samps.var(axis=0)**0.5
        p = (w_samps>0).mean(axis=0)
        quantiles = np.quantile(w_samps,[0.025,0.975],axis=0)
        w_summary = pd.DataFrame({'Post Mean':mean,'Post Std':std,'P(w>0)':p,
                      '[0.025':quantiles[0],'0.975]':quantiles[1]},index=self.cov_names+['a_0'])
        return w_summary
    
    def plot_temporal(self,rescale=True):
        """
        Plot mean posterior temporal gaussian process.
        
        Parameters
        ----------
        rescale: bool
            Scale posteriors to original dimensions of the data.
        """
        if 'samples' not in dir(self):
            raise Exception("MCMC posterior sampling has not been performed yet.")
        if self.args['model'] not in ['cox_hawkes','lgcp']:
            raise Exception("Nothing to plot: temporal background in constant.")
        
        b_scale = 1.
        if rescale:
            b_scale = 50/self.T
        x_t = jnp.arange(0, self.args['T'], self.args['T']/self.args["n_t"])/b_scale
        f_t_post=self.samples["f_t"]
        f_t_hpdi = hpdi(self.samples["f_t"])
        f_t_post_mean=jnp.mean(f_t_post, axis=0)
        
        fig,ax=plt.subplots(1,1,figsize=(8,5))
        event_time_height = np.ones(len(self.args['t_events']))*f_t_hpdi.min()
        ax.plot(self.args['t_events']/b_scale, event_time_height,'+',color="red", 
                alpha=.15, label="observed times")
        ax.set_ylabel('$f_t$')
        ax.set_xlabel('t')
        ax.plot(x_t, f_t_post_mean, label="mean estimated $f_t$")
        ax.fill_between(x_t, f_t_hpdi[0], f_t_hpdi[1], alpha=0.4, color="palegoldenrod", label="90%CI rate")
        ax.legend()
        
    def plot_spatial(self,include_cov=False,**kwargs):
        """
        Plot mean posterior spatial intensity (ignoring self-excitation) with/without covariates
        
        Parameters
        ----------
        include_cov: bool
            Include effects of spatial covariates.
        kwargs: dict
            Plotting parameters for geopandas plot.
        """
        if 'samples' not in dir(self):
            raise Exception("MCMC posterior sampling has not been performed yet.")
        if self.args['model'] not in ['cox_hawkes','lgcp'] and not include_cov:
            raise Exception("Nothing to plot: spatial background is constant")
        if include_cov and 'spatial_cov' not in self.args:
            raise Exception("No spatial covariates are in the model and include_cov was set to True")

        if 'alpha' not in kwargs:
            kwargs['alpha'] = .1
        
        if self.args['model'] in ['cox_hawkes','lgcp'] and include_cov:
            self._plot_cov_comp_grid(**kwargs)
        elif include_cov:
            self._plot_cov(**kwargs)
        else:
            self._plot_grid(**kwargs)

    def _plot_grid(self,**kwargs):
        """
        Plot spatial for computational grid only
        """
        
        f_xy_post = self.samples["f_xy"]
        f_xy_post_mean=jnp.mean(f_xy_post, axis=0)
        self.comp_grid['post_mean'] = f_xy_post_mean
        intersect = gpd.overlay(self.comp_grid, self.A[['geometry']], how='intersection',keep_geom_type=True)
        fig, ax = plt.subplots(1,2, figsize=(10, 5))
        intersect.plot(column='post_mean',ax=ax[0])
        ax[0].set_title('Mean Posterior $f_s$')
        intersect.plot(column='post_mean',ax=ax[1])
        self.points.plot(ax=ax[1],color='red',marker='x',**kwargs)
        ax[1].set_title('Mean Posterior $f_s$ With Events')
        return fig
    
    def _plot_cov_comp_grid(self,**kwargs):
        """
        Plot spatial for computational grid and spatial covariates.
        """
        post_samples = (self.samples['b_0'][:,self.args['int_df']['cov_ind'].values] + 
                        self.samples["f_xy"][:,self.args['int_df']['comp_grid_id'].values])
        self.args['int_df']['post_mean'] = post_samples.mean(axis=0)
        fig, ax = plt.subplots(1,2, figsize=(10, 5))
        self.args['int_df'].plot(column='post_mean',ax=ax[0])
        ax[0].set_title('Mean Posterior $f_s + X(s)w$')
        self.args['int_df'].plot(column='post_mean',ax=ax[1])
        self.points.plot(ax=ax[1],color='red',marker='x',**kwargs)
        ax[1].set_title('Mean Posterior $f_s + X(s)w$ With Events')
        
    def _plot_cov(self,**kwargs):
        """
        Plot spatial for covariates only.
        """
        self.spatial_cov['post_mean'] = self.samples['b_0'].mean(axis=0)
        fig, ax = plt.subplots(1,2, figsize=(10, 5))
        self.spatial_cov.plot(column='post_mean',ax=ax[0])
        ax[0].set_title('Mean Posterior $X(s)w$')
        self.spatial_cov.plot(column='post_mean',ax=ax[1])
        ax[1].set_title('Mean Posterior $X(s)w$ With Events')
        self.points.plot(ax=ax[1],color='red',marker='x',**kwargs)
        ax[1].set_title('Mean Posterior $f_s + X(s)w$ With Events')
    
    def _sim_spatial(self, geo_df):
        lam = geo_df['area']*np.exp(geo_df['log_intensity'])
        num_samp = np.random.poisson(lam)
        mask_zero = num_samp!=0
        samples = geo_df[mask_zero].sample_points(size=num_samp[mask_zero])
        return samples.explode(index_parts=False)
    
    def _sim_cox(self,parameters):
        if 'spatial_cov' in self.args:
            geo_df = self.args['int_df']
            geo_df['spatial_log_intensity'] = (parameters['b_0'][geo_df['cov_ind'].values] + 
                                   parameters['f_xy'][geo_df['comp_grid_id'].values])
        else:
            geo_df = self.comp_grid
            geo_df['spatial_log_intensity'] = parameters["f_xy"]
            geo_df = geo_df.sjoin(self.A,how='inner')
            geo_df['area'] = 1/self.args['n_xy']**2
        f_t = parameters['f_t']
        a_0 = parameters['a_0']
        t_lat = np.arange(0,self.args['T'],self.args['T']/self.args['n_t'])
        sp_samp = list()
        t_samp = list()
        for i in range(self.args['T']):
            geo_df['log_intensity'] = geo_df['spatial_log_intensity']+a_0+f_t[i]
            sp_samp.append(self._sim_spatial(geo_df))
            t_samp.append(np.random.uniform(size=len(sp_samp[-1]))+t_lat[i])
        sp = np.hstack([(p.x,p.y) for p in sp_samp])
        bg = np.append(sp.T,np.hstack(t_samp).reshape(-1,1),1)
        return bg

class Hawkes_Model(Point_Process_Model):
    def __init__(self,data, A, T, cox_background='cox',temporal_trig=Temporal_Exponential,
                 spatial_trig=Spatial_Symmetric_Gaussian,**kwargs):
        r"""
        Spatiotemporal Point Process Model given by,
        
        $$\lambda(t,s) = \mu(s,t) + \sum_{i:t_i<t}{\alpha f(t-t_i;\beta) \varphi(s-s_i;\sigma^2)}$$

        where $f$ is defined by spatial_trig, $\\varphi$ is defined by spatial_trig. If cox_background is true, $\mu$ is given by
        
        $$\mu(s,t) = exp(a_0 + X(s)w + f_s(s) + f_t(t))$$

        where $X(s)$ is the spatial covariate matrix, $f_s$ and $f_t$ are Gaussian Processes. 
        Both $f_s$ and $f_t$ are simulated by a pretrained VAE. We used a squared exponential kernel with hyperparameters $l \sim InverseGamma(15,1)$ and $\sigma^2 \sim LogNormal(2,0.5)$ 

        Otherwise, the $\mu$ is given by

        $$\mu(s,t) = exp(a_0 + X(s)w)$$
        
        The data is rescaled to fit in a 1x1 spatial grid and a lenght 50 time window. Posterior samples must be interpreted with this in mind.

        Parameters
        ----------
        data: str or pd.DataFrame
            either file path or DataFrame containing spatiotemporal data. Columns must include 'X', 'Y', 'T'. The file must be sorted by 'T'.
        A: np.array [2x2], GeoDataFram
            Spatial region of interest. If np.array first row is the x-range, second row is y-range.
        T: float
            Maximum time in region of interest. Time is assumed to spart at 0.
        cox_background: bool
            use gaussian processes in background
        temporal_trig: class Trigger
            an implementation of Trigger to parameterize the temporal triggering mechanism.
        spatial_trig: class Trigger
            an implementation of Trigger to parameterize the spatial triggering mechanism.
        kwargs: dict
            parameters from Point_Process_Model
        """
        self.model = spatiotemporal_hawkes_model
        if cox_background:
            name = 'cox_hawkes'
        else:
            name = 'hawkes'
        super().__init__(name, data, A, T, **kwargs)
        
        self.args['t_trig'] = temporal_trig(self.args['priors'])
        self.args['sp_trig'] = spatial_trig(self.args['priors'])
    
    def get_params(self):
        """
        Returns
        -------
            dict of parameter names as keys and lengths as values
        """
        pars = {}
        pars['alpha'] = 1
        for n in self.args['t_trig'].get_par_names():
            pars[n] = 1
        for n in self.args['sp_trig'].get_par_names():
            pars[n] = 1
        
        if self.args['model'] == 'cox_hawkes':
            pars['z_spatial'] = self.args['z_dim_spatial']
            pars['z_temporal'] = self.args['z_dim_temporal']
            pars['f_xy'] = 0
            pars['f_t'] = 0
        pars['a_0'] = 1
        if 'spatial_cov' in self.args:
            pars['w'] = self.args['spatial_cov'].shape[1]
            pars['b_0'] = 0
        return pars
    
    def plot_prop_excitation(self):
        """
        Plots a histogram of the posterior distribution of the proportion of the intensity due to self-excitation.
        
        Returns
        -------
            float: posterior mean of proportion of intensity due to self-excitation
        """
        p = self.samples['Itot_excite']/self.samples['Itot_txy']
        plt.hist(p,density=True)
        plt.title("Proportion of Intensity Due to Self-Excitation")
        plt.xlabel("Proportion of Intensity Due to Self-Excitation")
        return p.mean().item()
    
    def plot_trigger_posterior(self,trace=False):
        """
        Plot histograms of posterior trigger parameters.
        Returns
        -------
        pd.DataFrame
            Summary of trigger parameters.
        trace: bool
            plot trace or histogram of parameters
        """
        if 'samples' not in dir(self):
            raise Exception("MCMC posterior sampling has not been performed yet.")
        par_names = self.args['t_trig'].get_par_names()+self.args['sp_trig'].get_par_names()
        if trace:
            fig, ax = plt.subplots(1+len(par_names),1,figsize=(5,8), sharex=True)
            plt.suptitle("Trace Plots for Trigger Parameter Posteriors")
            ax[0].plot(self.samples['alpha'])
            ax[0].set_ylabel(r"${\alpha} $")
            for i in range(len(par_names)):
                ax[i+1].plot(self.samples[par_names[i]])
                ax[i+1].set_ylabel(par_names[i])
        else:
            fig, ax = plt.subplots(1, 1+len(par_names),figsize=(8,4), sharex=False)
            plt.suptitle("Trigger Parameter Posteriors")
            ax[0].hist(self.samples['alpha'])
            ax[0].set_xlabel(r"${\alpha} $")
            for i in range(len(par_names)):
                ax[i+1].hist(self.samples[par_names[i]])
                ax[i+1].set_xlabel(par_names[i])

        trig_pos = np.stack([self.samples[name] for name in ['alpha']+par_names]).T
        mean = trig_pos.mean(axis=0)
        std = trig_pos.var(axis=0)**0.5
        p_val = [(self.samples[name]>0).mean() for name in ['alpha']+par_names]
        quantiles = np.quantile(trig_pos,[0.025,0.975],axis=0)
        trig_summary = pd.DataFrame({'Post Mean':mean,'Post Std':std,r'P(w>0)':p_val,
                      '[0.025':quantiles[0],'0.975]':quantiles[1]},index=['alpha']+par_names)
        return trig_summary
    
    def plot_trigger_time_decay(self,t_units='days'):
        """
        Plot temporal trigger kernel sample posterior.
        
        Parameters
        ----------
        t_units: str
            Time units of original data.
        """
        if 'samples' not in dir(self):
            raise Exception("MCMC posterior sampling has not been performed yet.")
        
        par_names = self.args['t_trig'].get_par_names()
        post_mean = {}
        for name in par_names:
            post_mean[name] = self.samples[name].mean().item()
        scale = 50/self.T
        #estimate a good maximum
        t = self.args['t_trig'].compute_integral(post_mean,self.args['T']/10)
        t = np.log(0.025)/np.log(1-t)*self.args['T']/10
        t = np.arange(0,t,t/250)
        fig, ax = plt.subplots(figsize=(7,7))
        for i in np.random.choice(np.arange(len(self.samples['alpha'])),100):
            pars = {}
            for name in par_names:
                pars[name] = self.samples[name][i].item()
            plt.plot(t/scale,self.args['t_trig'].compute_trigger(pars,t),color='black',alpha=0.1)
        fig.suptitle('Time Decay of Trigger Function')
        ax.set_ylabel('Trigger Intensity')
        ax.set_xlabel(t_units.capitalize()+' After First Event')
        ax.axhline(0,color='black',linestyle='--')
        ax.axvline(0,color='black',linestyle='--')
    
    def _sim_hawkes_bg(self,parameters):
        a_0 = parameters['a_0']
        if 'spatial_cov' in self.args:
            geo_df = self.spatial_cov
            geo_df['log_intensity'] = a_0 + np.log(self.args['T']) + parameters['b_0']
        else:
            geo_df = self.A
            geo_df['log_intensity'] = a_0 + np.log(self.args['T'])
        A_ = self.args['A_']
        geo_df['area'] = geo_df.area/((A_[0,1]-A_[0,0])*(A_[1,1]-A_[1,0]))
        sp = self._sim_spatial(geo_df)
        return np.stack((sp.x,sp.y,np.random.uniform(self.args['T'],size=len(sp)))).T
    
    def _sim_offspring(self,bg,par):
        i = 0
        while i < len(bg):
            for j in range(np.random.poisson(lam=par['alpha'])):
                #simulate trigger and rescale
                sp_dif = (self.args['A_'][:,1]-self.args['A_'][:,0])*\
                            self.args['sp_trig'].simulate_trigger(par)
                t_dif = [self.args['t_trig'].simulate_trigger(par)]
                bg = np.concatenate((bg,[bg[i]+np.append(sp_dif,t_dif)]))
            i += 1
        return bg

    def simulate(self,parameters=None):
        """
        Simulate data from mean posterior parameters.
        Parameters
        ----------
        parameters: dict
            Parameters to simulate from. If parameters is None, use mean of posterior samples. keys are string parameter names. values are np.array or float. Names must be same as those that appear in the sample from the model.
        Returns
        -------
            geopandas DataFrame: ['X','Y','T'] columns
                simulated data
        """
        if parameters is None:
            parameters = {k:np.array(v).mean(axis=0) for k,v in self.samples.items()}
        if 'f_t' not in parameters and 'z_temporal' in parameters:
            decoder_nn_temporal = vae_decoder_temporal(self.args["hidden_dim_temporal"], self.args["n_t"])  
            # Approximate Gaussian Process with VAE
            v_t = decoder_nn_temporal[1](self.args["decoder_params_temporal"], parameters['z_temporal'])
            parameters['f_t'] = v_t[0:self.args["n_t"]]
        if 'f_xy' not in parameters and 'z_spatial' in parameters:
            decoder_nn = vae_decoder_spatial(self.args["hidden_dim2_spatial"], self.args["hidden_dim1_spatial"], self.args["n_xy"])  
            decoder_params = self.args["decoder_params_spatial"]
            # Generate Gaussian Process from VAE
            parameters['f_xy'] = jnp.exp(self.args['sp_var_mu']) * decoder_nn[1](decoder_params, parameters['z_spatial'])
        if 'w' in parameters and 'b_0' not in parameters:
            parameters['b_0'] = self.args['spatial_cov'] @ parameters['w']
        
        if self.args['model'] == 'cox_hawkes':
            bg = self._sim_cox(parameters)
        else:
            bg = self._sim_hawkes_bg(parameters)
        sample = self._sim_offspring(bg,parameters)
        #filter out offspring after cutoff
        sample = sample[sample.T[2]<self.args['T']]
        geometry = gpd.points_from_xy(sample.T[0], sample.T[1],crs=self.A.crs)
        points = gpd.GeoDataFrame(data=sample,geometry=geometry,columns=['X','Y','T'])
        #filter to time window
        points['T'] = (points['T']*self.T/self.args['T'])
        #filter to spatial window
        return points.sjoin(self.A[['geometry']])[['X','Y','T','geometry']]

class LGCP_Model(Point_Process_Model):
    def __init__(self,*args,**kwargs):
        r"""
        Spatiotemporal LGCP Model given by,
        
        $$\lambda(t,s) = exp(a_0 + X(s)w + f_s(s) + f_t(t))$$

        where $X(s)$ is the spatial covariate matrix, $f_s$ and $f_t$ are Gaussian Processes. 
        Both $f_s$ and $f_t$ are simulated by a pretrained VAE. We used a squared exponential kernel with hyperparameters $l \sim InverseGamma(15,1)$ and $\sigma^2 \sim LogNormal(2,0.5)$ 

        The data is rescaled to fit in a 1x1 spatial grid and a lenght 50 time window. Posterior samples must be interpreted with this in mind.

        Parameters
        ----------
        args: list
            Parameters from Point_Process_Model
        kwargs: dict
            Parameters from Point_Process_Model
        """
        name = 'lgcp'
        self.model = spatiotemporal_LGCP_model
        super().__init__(name,*args,**kwargs)
        
    def get_params(self):
        """
        Returns
        -------
            dict of parameter names as keys and lengths as values
        """
        pars = {}
        pars['z_spatial'] = self.args['z_dim_spatial']
        pars['z_temporal'] = self.args['z_dim_temporal']
        pars['f_xy'] = 0
        pars['f_t'] = 0
        pars['a_0'] = 1
        if 'spatial_cov' in self.args:
            pars['w'] = self.args['spatial_cov'].shape[1]
            pars['b_0'] = 0
        return pars
    
    def simulate(self,parameters=None):
        """
        Simulate data from mean posterior parameters. Requires model inference.
        Parameters
        ----------
        parameters: dict
            Parameters to simulate from. If parameters is None, use mean of posterior samples. keys are string parameter names. values are np.array or float. Names must be same as those that appear in the sample from the model.
        Returns
        -------
            geopandas DataFrame: ['X','Y','T'] columns
                simulated data
        """
        if parameters is None:
            parameters = {k:np.array(v).mean(axis=0) for k,v in self.samples.items()}
        sample = self._sim_cox(parameters)
        geometry = gpd.points_from_xy(sample.T[0], sample.T[1],crs=self.A.crs)
        points = gpd.GeoDataFrame(data=sample,geometry=geometry,columns=['X','Y','T'])
        points['T'] = (points['T']*self.T/self.args['T'])
        return points