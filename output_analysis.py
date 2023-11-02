import pandas as pd
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pickle
from math import erf, ceil

#load output
with open('output.pkl','rb') as f:
    output = pickle.load(f)
samples = output['samples']

df = pd.read_csv (r'data/pp_data.csv')
T = 50
t_events_total=((df['T']-df['T'].min())).to_numpy()
t_events_total/=t_events_total.max()
t_events_total*=T

x_range = (3,15.5)
x_events_total=(df['X']-x_range[0]).to_numpy()
x_events_total/=(x_range[1]-x_range[0])

y_range = (4,16.5)
y_events_total=(df['Y']-y_range[0]).to_numpy()
y_events_total/=(y_range[1]-y_range[0])

xy_events_total=np.array((x_events_total,y_events_total)).transpose()
n_xy = 25



#trace plots
fig, ax = plt.subplots(3, 1,figsize=(8,5), sharex=True)
ax[0].plot(samples['alpha'])
ax[0].set_ylabel(r"${\alpha} $")
ax[1].plot(samples['beta'])
ax[1].set_ylabel(r"$\beta$")
ax[2].plot(samples['sigmax_2'])
ax[2].set_ylabel('$\sigma ^2$')
ax[2].set_xlabel("MCMC iterations")
plt.savefig("output/trig_trace.png")

#trigger parameter posterior histograms
fig, ax = plt.subplots(1, 3,figsize=(8,3), sharex=False)
plt.suptitle("Trigger Parameter Posteriors")
ax[0].hist(samples['alpha'])
ax[0].set_xlabel(r"${\alpha} $")
ax[1].hist(samples['beta'])
ax[1].set_xlabel(r"$\beta$")
ax[2].hist(samples['sigmax_2']**0.5)
ax[2].set_xlabel('$\sigma$')
plt.savefig("output/trig_post.png")
plt.show()

# Trigger time decay posterior
scale = 50/df['T'].max()
x = np.arange(300)
fig, ax = plt.subplots(figsize=(7,7))
for b in np.random.choice(samples['beta'],100):
    plt.plot(x,b*np.exp(-b*scale*x),color='black',alpha=0.1)
fig.suptitle('Time Decay of Trigger Function')
ax.set_ylabel('Trigger Intensity')
ax.set_xlabel('Days After First Event')
ax.axhline(0,color='black',linestyle='--')
ax.axvline(0,color='black',linestyle='--')
plt.savefig("output/trig_post_time_decay.png")
b = float(samples['beta'].mean())
print(f'Mean trigger time: {round(1/(b*scale),2)} days')


#Posterior weights for covariates
l = ceil((samples['w'].shape[1]+1)/3)
fig, ax = plt.subplots(3,l,figsize=(10,10), sharex=False)
fig.suptitle('Covariate Weights', fontsize=16)
cov_names = ['droughtstart_speibase','urban_ih_log','droughtyr_speigdm','herb_gc','capdist',
           'grass_ih_log','nlights_sd_log','water_gc_log','pop_gpw_sd_log','pasture_ih']
for i in range(samples['w'].shape[1]):
    ax[i//l,i%l].hist(samples['w'].T[i])
    ax[i//l,i%l].set_xlabel(cov_names[i])
ax[2,(samples['w'].shape[1])%l].hist(samples['a_0'])
ax[2,(samples['w'].shape[1])%l].set_xlabel("$a_0$")
plt.savefig("output/cov_post_hist.png")



np.set_printoptions(precision=3)
pd.options.display.float_format = '{:.3f}'.format

w_samps = np.concatenate((samples['w'],samples['a_0'].reshape(-1,1)),axis=1)
mean = w_samps.mean(axis=0)
std = w_samps.var(axis=0)**0.5
z_score = np.asarray(mean/std)
p_val = 1-np.vectorize(erf)(abs(z_score)/2**0.5)
quantiles = np.quantile(w_samps,[0.025,0.975],axis=0)
w_summary = pd.DataFrame({'Post Mean':mean,'Post Std':std,'z':z_score,'P>|z|':p_val,
              '[0.025':quantiles[0],'0.975]':quantiles[1]},index=cov_names+['a_0'])
w_summary.to_csv("output/cov_par_summary.csv",index=False)
print(w_summary.to_string())



n_t=50
x_t = jnp.arange(0, T, T/50)

f_t_post=samples["f_t"]
f_t_post_mean=jnp.mean(f_t_post, axis=0)


fig,ax=plt.subplots(1,1,figsize=(8,5))
event_time_height = np.ones(len(t_events_total))*(f_t_post_mean.min()-f_t_post_mean.var()**0.5/4)
ax.plot(t_events_total, event_time_height, 'b+',color="red", label="observed times")
ax.set_ylabel('$f_t$')
ax.set_xlabel('t')
ax.plot(x_t, f_t_post_mean, label="mean estimated $f_t$")
ax.legend()
plt.savefig("output/temporal_gp_post_mean.png")



def plot_background(f_xy_post,f_xy_post_mean,file_name,fig_desc):
    fig, ax = plt.subplots(1,2, figsize=(10, 5))
    _min, _max = np.amin(f_xy_post), np.amax(f_xy_post)
    im = ax[0].imshow(f_xy_post_mean.reshape(n_xy,n_xy), cmap='viridis', interpolation='none', extent=[0,1,0,1], origin='lower',vmin=_min, vmax=_max)
    ax[0].title.set_text('Estimated '+fig_desc)
    
    im2 = ax[1].imshow(f_xy_post_mean.reshape(n_xy,n_xy), cmap='viridis', interpolation='none', extent=[0,1,0,1], origin='lower',vmin=_min, vmax=_max)
    ax[1].plot(xy_events_total[:,0],xy_events_total[:,1],'x', alpha=.25,color='red',label='true event locations')
    ax[1].title.set_text(f'Estimated {fig_desc} with true locations')
    for i in range(2):
        ax[i].set_xlabel('x')
        ax[i].set_ylabel('y')
        for item in ([ ax[i].xaxis.label, ax[i].yaxis.label, ax[i].title] ):
            item.set_fontsize(15)
    
    fig.subplots_adjust(right=0.8)
    cax = fig.add_axes([ax[1].get_position().x1+0.03,ax[1].get_position().y0,0.02,ax[1].get_position().height])
    fig.colorbar(im, cax=cax)
    plt.savefig('output/'+file_name)


f_xy_post=samples["f_xy"]
f_xy_post_mean=jnp.mean(f_xy_post, axis=0)
plot_background(f_xy_post,f_xy_post_mean,"spatial_gp_post_mean.png","$f_s$")

f_xy_post=samples["f_xy"] + samples['b_0']
f_xy_post_mean=jnp.mean(f_xy_post, axis=0)
plot_background(f_xy_post,f_xy_post_mean,"spatial_gp_cov_post_mean.png","$f_s$ + X(s)w")