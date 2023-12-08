# general libraries
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from math import erf, ceil
import warnings 
import dill
import pickle

# JAX
import jax.numpy as jnp
from jax import random
import numpyro.distributions as dist
from numpyro.diagnostics import hpdi

from .utils import * 
from .inference_functions import *
import pkgutil

class Point_Process_Model:
    def __init__(self,data,A,model='cox_hawkes',spatial_cov=None,cov_names=None,cov_grid_size=None,**priors):
        """
        Spatiotemporal Point Process Model given by,
        
        $$\lambda(t,s) = \mu(s,t) + \sum_{i:t_i<t}{\\alpha f(t-t_i;\\beta) \\varphi(s-s_i;\\sigma^2)}$$

        where $f$ is the exponential pdf, $\\varphi$ is the gaussian pdf, and $\mu$ is given by
        
        $$\mu(s,t) = exp(a_0 + X(s)w + f_s(s) + f_t(t))$$

        where $X(s)$ is the spatial covariate matrix, $f_s$ and $f_t$ are Gaussian Processes. 
        Both $f_s$ and $f_t$ are simulated by a pretrained VAE. We used a squared exponential kernel with hyperparameters $l \sim InverseGamma(10,1)$ and $\sigma^2 \sim LogNormal(2,0.5)$ 


        Parameters
        ----------
        data: str or pd.DataFrame
            either file path or DataFrame containing spatiotemporal data. Columns must include 'X', 'Y', 'T'.
        A: np.array [2x2], GeoDataFram
            Spatial region of interest. If np.array first row is the x-range, second row is y-range.
        model: str
            one of ['cox_hawkes','lgcp','hawkes'].
        spatial_cov: str,pd.DataFrame,gpd.GeoDataFrame
            Either file path (.csv or .shp), DataFrame, or GeoDataFrame containing spatial covariates. Spatial covariates must cover all the points in data.
            If spatial_cov is a csv or pd.DataFrame, the first 2 columns must be 'X', 'Y' and cov_grid_size must be specified.
        cov_names: list
            List of covariate names. Must all be columns in spatial_cov.
        cov_grid_size: list-like
            Spatial covariate grid (width, height).
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

        #scale temporal events
        t_events_total=((data['T']-data['T'].min())).to_numpy()
        t_events_total/=t_events_total.max()
        t_events_total*=50

        if type(A) is gpd.GeoDataFrame:
            A_ = np.stack((A.bounds.min(axis=0)[['minx','miny']],
                      A.bounds.max(axis=0)[['maxx','maxy']])).T
            #proportion of area of rectangle A_ covered by A. Used for Hawkes integral.
            args['A_area'] = A.area.sum()/((A_[0,1]-A_[0,0])*(A_[1,1]-A_[1,0]))
        else:# A is rectangle specified by np.array
            args['A_area'] = 1
            A_ = A
        
        #scale spatial events
        x_range = A_[0]
        x_events_total=(data['X']-x_range[0]).to_numpy()
        x_events_total/=(x_range[1]-x_range[0])
        y_range = A_[1]
        y_events_total=(data['Y']-y_range[0]).to_numpy()
        y_events_total/=(y_range[1]-y_range[0])
        
        xy_events_total=np.array((x_events_total,y_events_total)).transpose()
        args["t_events"]=t_events_total
        args["xy_events"]=xy_events_total.transpose()
        
        #create computational grid
        n_t=50
        T=50
        x_t = jnp.arange(0, T, T/n_t)
        args["n_t"]=n_t
        args["x_t"]=x_t
        
        n_xy = 25
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
            args['spatial_grid_cells'] = np.unique(comp_grid.sjoin(A, how='inner')['comp_grid_id'])
            self.A = A
            comp_grid.set_crs(crs=A.crs)
        else:
            self.A = comp_grid
            args['spatial_grid_cells'] = np.arange(25**2)
        
        self.comp_grid = comp_grid
        geometry = gpd.points_from_xy(data.X, data.Y,crs=comp_grid.crs)
        self.points = gpd.GeoDataFrame(data=data,geometry=geometry)
        
        #find grid cells where points are located
        args['indices_xy'] = self.points.sjoin(comp_grid)['comp_grid_id'].values
        if len(args['indices_xy']) != len(self.points):
            raise Exception("Computational grid does not encompass all data points!")
        args["n_xy"]= n_xy
        
        args["gp_kernel"]=exp_sq_kernel
        
        args['indices_t']=np.searchsorted(x_t, t_events_total, side='right')-1
        
        
        # temporal VAE training arguments
        args["hidden_dim_temporal"]= 35
        args["z_dim_temporal"]= 11
        args["T"]=T
        # spatial VAE training arguments
        args["hidden_dim1_spatial"]= 35
        args["hidden_dim2_spatial"]= 30
        args["z_dim_spatial"]=15
        
        if args['model'] in ['lgcp','cox_hawkes']:
            decoder_params = pickle.loads(pkgutil.get_data(__name__, "decoders/decoder_1d_T50_fixed_ls"))
            args["decoder_params_temporal"] = decoder_params
            
            #Load 2d spatial trained decoder
            #decoder_params = pickle.loads(pkgutil.get_data(__name__, "decoders/decoder_2d_n25_infer_hyperpars"))
            decoder_params = pickle.loads(pkgutil.get_data(__name__, "decoders/2d_decoder_10_5.pkl"))
            args["decoder_params_spatial"] = decoder_params
        
        if spatial_cov is not None:
            #convert input into geopandas dataframe.
            if type(spatial_cov)==str:
                if spatial_cov[-4:] == '.shp':
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
            args['cov_ind'] = self.points.sjoin(spatial_cov)['cov_ind'].values
            if len(args['cov_ind']) != len(self.points):
                raise Exception("Spatial covariates are not defined for all data points!")
            
            args['num_cov'] = len(cov_names)
            self.cov_names = cov_names
            self.spatial_cov = spatial_cov
            
            X_s = spatial_cov[self.cov_names].values
            # standardize covariates
            args['spatial_cov'] = (X_s-X_s.mean(axis=0))/(X_s.var(axis=0)**0.5)
            
            #Create Computational Grid GeoDataFrame
            if args['model'] in ['lgcp','cox_hawkes']:
                comp_grid.crs = spatial_cov.crs
                #find covariate cell intersection with computational grid cells area
                intersect = gpd.overlay(comp_grid, spatial_cov, how='intersection')
                intersect['area'] = intersect.area/((A_[0,1]-A_[0,0])*(A_[1,1]-A_[1,0]))
                intersect = intersect[intersect['area']>1e-10]
                args['int_df'] = intersect
                #find cells on the computational grid that are in the domain
                args['spatial_grid_cells'] = np.unique(comp_grid.sjoin(spatial_cov, how='inner')['comp_grid_id'])
            else:
                args['cov_area'] = (spatial_cov.area/((A_[0,1]-A_[0,0])*(A_[1,1]-A_[1,0]))).values

        #Set up parameter priors
        default_priors = {"a_0": dist.Normal(0,3),
                          "alpha": dist.Beta(1,1),
                          "beta": dist.HalfNormal(4.0),
                          "sigmax_2": dist.HalfNormal(0.25),
                         }
        if 'num_cov' in args:
            default_priors["w"] = dist.Normal(jnp.zeros(args['num_cov']),jnp.ones(args["num_cov"]))
        args['sp_var_mu'] = 3.0
        for par, prior in priors.items():
            if par in default_priors:
                default_priors[par] = prior
            else:
                warnings.warn(f'\"{par}\" prior is not being used. There is no such parameter in the model.') 
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
        self.mcmc = output['mcmc']
        self.mcmc_samples = output['samples']

    def save_rslts(self,file_name):
        """
        Save previously computed results
        Parameters
        ----------
        file_name: string
            File where to save results
        """
        output['mcmc'] = self.mcmc
        output['samples'] = self.mcmc_samples
        with open(file_name, 'wb') as f:
            output = pickle.dump(output,f)
        
    
    def run_svi(self,num_samples=1000,output_file=None,resume=False,**kwargs):
        """
        Perform Stochastic Variational Inference on the model.
        Parameters
        ----------
        num_samples: int, default=1000
            Number of samples to generate after SVI.
        output_file: string, default=None
            File name to save results.
        resume: bool, default=False
            Pick up where last SVI run was left off. Can only be true if model has previous run_svi call.
        lr: float, default=0.001
            learning rate for SVI
        num_steps: int, default=10000
            Number of interations for SVI to run.
        auto_guide: numpyro AutoGuide, default=AutoMultivariateNormal
            See numpyro AutoGuides for details.
        init_strategy: function, default=init_to_median
            See numpyro init strategy documentation
        """
        rng_key, rng_key_predict = random.split(random.PRNGKey(10))
        rng_key, rng_key_post, rng_key_pred = random.split(rng_key, 3)
        self.args["num_samples"] = num_samples
        if self.args['model'] in ['cox_hawkes','hawkes']:
            model = spatiotemporal_hawkes_model
        else:
            model = spatiotemporal_LGCP_model
        if resume:
            if 'num_steps' not in kwargs:
                kwargs['num_steps'] = 10000
            if 'lr' not in kwargs:
                kwargs['lr'] = 0.001
            optimizer = numpyro.optim.Adam(
                jax.example_libraries.optimizers.inverse_time_decay(kwargs['lr'],kwargs['num_steps'],4)
            )
            self.svi.optim = optimizer
            self.svi_results = self.svi.run(rng_key, kwargs['num_steps'], self.args,
                                            init_state=self.svi_results.state)
            self.mcmc_samples = get_samples(rng_key,model,self.svi.guide,self.svi_results,self.args)
        else:
            self.svi,self.svi_results,self.mcmc_samples=run_SVI(rng_key,model,self.args,**kwargs)
        loss = np.asarray(self.svi_results.losses)
        plt.plot(np.arange(int(.1*len(loss)),len(loss)),loss[int(.1*len(loss)):])
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.show()

    
    def run_mcmc(self,batch_size=1,num_warmup=500,num_samples=1000,num_chains=1,thinning=1,output_file=None):
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
        output_file: str
            File to save output to.
        """
        self.args["batch_size"]= batch_size
        self.args["num_warmup"]= num_warmup
        self.args["num_samples"] = num_samples
        self.args["num_chains"] = num_chains
        self.args["thinning"] = thinning
        rng_key, rng_key_predict = random.split(random.PRNGKey(10))
        rng_key, rng_key_post, rng_key_pred = random.split(rng_key, 3)
        
        output_dict = {}
        if self.args['model'] in ['cox_hawkes','hawkes']:
            self.mcmc = run_mcmc(rng_key_post, spatiotemporal_hawkes_model, self.args)
        elif self.args['model'] == 'lgcp':
            self.mcmc = run_mcmc(rng_key_post, spatiotemporal_LGCP_model, self.args)
        self.mcmc_samples=self.mcmc.get_samples()
        
        output_dict['samples']=self.mcmc_samples
        output_dict['mcmc']=self.mcmc
        if output_file is not None:
            with open(output_file, 'wb') as handle:
                dill.dump(output_dict, handle)

    def plot_trigger_posterior(self,output_file=None):
        """
        Plot histograms of posterior trigger parameters.
        
        Parameters
        ----------
        output_file: str
            Path in which to save plot.
        Returns
        -------
        pd.DataFrame
            Summary of trigger parameters.
        """
        if 'mcmc_samples' not in dir(self):
            raise Exception("MCMC posterior sampling has not been performed yet.")
        if self.args['model'] not in ['hawkes','cox_hawkes']:
            raise Exception("This is not a Hawkes Model. Cannot plot trigger parameters.")
        fig, ax = plt.subplots(1, 3,figsize=(8,4), sharex=False)
        plt.suptitle("Trigger Parameter Posteriors")
        ax[0].hist(self.mcmc_samples['alpha'])
        ax[0].set_xlabel(r"${\alpha} $")
        ax[1].hist(self.mcmc_samples['beta'])
        ax[1].set_xlabel(r"$\beta$")
        ax[2].hist(self.mcmc_samples['sigmax_2']**0.5)
        ax[2].set_xlabel('$\sigma$')
        if output_file is not None:
            plt.savefig(output_file)
        plt.show()

        trig_pos = np.concatenate((self.mcmc_samples['alpha'].reshape(-1,1),
                                  self.mcmc_samples['beta'].reshape(-1,1),
                                  self.mcmc_samples['sigmax_2'].reshape(-1,1)**0.5),axis=1)
        mean = trig_pos.mean(axis=0)
        std = trig_pos.var(axis=0)**0.5
        z_score = np.asarray(mean/std)
        p_val = 1-np.vectorize(erf)(abs(z_score)/2**0.5)
        quantiles = np.quantile(trig_pos,[0.025,0.975],axis=0)
        trig_summary = pd.DataFrame({'Post Mean':mean,'Post Std':std,'z':z_score,'P>|z|':p_val,
                      '[0.025':quantiles[0],'0.975]':quantiles[1]},index=['alpha','beta','sigma'])
        return trig_summary
    
    def plot_trigger_time_decay(self,output_file=None,t_units='days'):
        """
        Plot temporal trigger kernel sample posterior.
        
        Parameters
        ----------
        output_file: str
            Path in which to save plot.
        t_units: str
            Time units of original data.
        """
        if 'mcmc_samples' not in dir(self):
            raise Exception("MCMC posterior sampling has not been performed yet.")
        if self.args['model'] not in ['hawkes','cox_hawkes']:
            raise Exception("This is not a Hawkes Model. Cannot plot trigger parameters.")
        
        scale = 50/self.data['T'].max()
        post_mean = float(self.mcmc_samples['beta'].mean())
        x = np.arange(0,3.5*post_mean/scale,3.5*post_mean/scale/250)
        fig, ax = plt.subplots(figsize=(7,7))
        for b in np.random.choice(self.mcmc_samples['beta'],100):
            plt.plot(x,np.exp(-scale/b*x)/b,color='black',alpha=0.1)
        fig.suptitle('Time Decay of Trigger Function')
        ax.set_ylabel('Trigger Intensity')
        ax.set_xlabel(t_units.capitalize()+' After First Event')
        ax.axhline(0,color='black',linestyle='--')
        ax.axvline(0,color='black',linestyle='--')
        if output_file is not None:
            plt.savefig(output_file)
        plt.show()
        print(f'Mean trigger time: {round(post_mean/scale,2)} '+t_units)

    def cov_weight_post_summary(self,plot_file=None,summary_file=None):
        """
        Plot posteriors of weights and bias and save summary of posteriors.
        
        Parameters
        ----------
        plot_file: str
            Path in which to save plot.
        summary_file: str
            Path in which to save summary
        Returns
        -------
        pd.DataFrame
            summary of weights and bias
        """
        if 'mcmc_samples' not in dir(self):
            raise Exception("MCMC posterior sampling has not been performed yet.")
        if 'spatial_cov' not in self.args:
            raise Exception("Spatial covariates were not included in the model.")
        
        n = self.mcmc_samples['w'].shape[1]+1
        c = ceil(n**0.5)
        r = ceil(n/c)
        fig, ax = plt.subplots(r,c,figsize=(10,10), sharex=False)
        fig.suptitle('Covariate Weights', fontsize=16)
        for i in range(n-1):
            ax[i//c,i%c].hist(self.mcmc_samples['w'].T[i])
            ax[i//c,i%c].set_xlabel(self.cov_names[i])
        ax[(n-1)//c,(n-1)%c].hist(self.mcmc_samples['a_0'])
        ax[(n-1)//c,(n-1)%c].set_xlabel("$a_0$")
        if plot_file is not None:
            plt.savefig(plot_file)
        plt.show()

        
        w_samps = np.concatenate((self.mcmc_samples['w'],self.mcmc_samples['a_0'].reshape(-1,1)),axis=1)
        mean = w_samps.mean(axis=0)
        std = w_samps.var(axis=0)**0.5
        z_score = np.asarray(mean/std)
        p_val = 1-np.vectorize(erf)(abs(z_score)/2**0.5)
        quantiles = np.quantile(w_samps,[0.025,0.975],axis=0)
        w_summary = pd.DataFrame({'Post Mean':mean,'Post Std':std,'z':z_score,'P>|z|':p_val,
                      '[0.025':quantiles[0],'0.975]':quantiles[1]},index=self.cov_names+['a_0'])
        if summary_file is not None:
            w_summary.to_csv(summary_file)
        return w_summary
    
    def plot_temporal_background(self,output_file=None):
        """
        Plot mean posterior temporal gaussian process.
        
        Parameters
        ----------
        plot_file: str
            Path in which to save plot.
        """
        if 'mcmc_samples' not in dir(self):
            raise Exception("MCMC posterior sampling has not been performed yet.")
        if self.args['model'] not in ['cox_hawkes','lgcp']:
            raise Exception("Nothing to plot: temporal background in constant.")
        
        x_t = jnp.arange(0, self.args['T'], self.args['T']/self.args["n_t"])
        f_t_post=self.mcmc_samples["f_t"]
        f_t_hpdi = hpdi(self.mcmc_samples["f_t"])
        f_t_post_mean=jnp.mean(f_t_post, axis=0)
        
        fig,ax=plt.subplots(1,1,figsize=(8,5))
        event_time_height = np.ones(len(self.args['t_events']))*f_t_hpdi.min()
        #np.ones(len(self.args['t_events']))*(f_t_post_mean.min()-f_t_post_mean.var()**0.5/4)
        ax.plot(self.args['t_events'], event_time_height,'+',color="red", 
                alpha=.15, label="observed times")
        ax.set_ylabel('$f_t$')
        ax.set_xlabel('t')
        ax.plot(x_t, f_t_post_mean, label="mean estimated $f_t$")
        ax.fill_between(x_t, f_t_hpdi[0], f_t_hpdi[1], alpha=0.4, color="palegoldenrod", label="90%CI rate")
        ax.legend()
        if output_file is not None:
            plt.savefig(output_file)
        plt.show()
        
    def plot_spatial_background(self,output_file=None,include_cov=False,**kwargs):
        """
        Plot mean posterior spatial background with/without covariates
        
        Parameters
        ----------
        output_file: str
            Path in which to save plot.
        include_cov: bool
            Include effects of spatial covariates.
        kwargs: dict
            Plotting parameters for geopandas plot.
        """
        if 'mcmc_samples' not in dir(self):
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
        
        if output_file is not None:
            plt.savefig(output_file)
        plt.show()

    def _plot_grid(self,**kwargs):
        """
        Plot spatial for computational grid only
        """
        
        f_xy_post = self.mcmc_samples["f_xy"]
        f_xy_post_mean=jnp.mean(f_xy_post, axis=0)
        self.comp_grid['f_xy_post_mean'] = f_xy_post_mean
        intersect = gpd.overlay(self.comp_grid, self.A[['geometry']], how='intersection')
        fig, ax = plt.subplots(1,2, figsize=(10, 5))
        intersect.plot(column='f_xy_post_mean',ax=ax[0])
        ax[0].set_title('Mean Posterior $f_s$')
        intersect.plot(column='f_xy_post_mean',ax=ax[1])
        self.points.plot(ax=ax[1],color='red',marker='x',**kwargs)
        ax[1].set_title('Mean Posterior $f_s$ With Events')
        return fig
    
    def _plot_cov_comp_grid(self,**kwargs):
        """
        Plot spatial for computational grid and spatial covariates.
        """
        post_samples = (self.mcmc_samples['b_0'][:,self.args['int_df']['cov_ind'].values] + 
                        self.mcmc_samples["f_xy"][:,self.args['int_df']['comp_grid_id'].values])
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
        self.spatial_cov['post_mean'] = self.mcmc_samples['b_0'].mean(axis=0)
        fig, ax = plt.subplots(1,2, figsize=(10, 5))
        self.spatial_cov.plot(column='post_mean',ax=ax[0])
        ax[0].set_title('Mean Posterior $X(s)w$')
        self.spatial_cov.plot(column='post_mean',ax=ax[1])
        ax[1].set_title('Mean Posterior $X(s)w$ With Events')
        self.points.plot(ax=ax[1],color='red',marker='x',**kwargs)
        ax[1].set_title('Mean Posterior $f_s + X(s)w$ With Events')