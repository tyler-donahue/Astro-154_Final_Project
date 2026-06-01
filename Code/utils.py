#imports
import matplotlib.pyplot as plt
import numpy as np
import george
import scipy.optimize as op
import emcee

#constants
a_min = 0.1
a_max = 100
b_min = 0.1
b_max = 0.01
y_min = 0.1
y_max = 300
t_min = -50
t_max = 300
f_min = 1
f_max = 300
r_min = 0.01
r_max = 50
n_params = 6
n_walkers = 100
n_steps = 10000

#functions
def plot_curve_fit(time, mag_flipped, x_fit, pred, objectid, curve_type, peak, rise_time, fall_time):

    #initialize plotting figure
    fig, ax1 = plt.subplots()
    
    #plot raw lightcurve
    ax1.scatter(time, mag_flipped, marker='o')

    #split axis
    ax2 = ax1.twinx()

    #plot george
    ax2.plot(x_fit, pred, color='r')

    plt.title(f"Object: {objectid}; Type: SN_{curve_type}; Peak: {peak}; Rise: {rise_time}; Fall: {fall_time}")

    plt.show()
    plt.close()

def decompose_curve(light_curve, fid=2, mag_boundry=22):
    
    #initialize lists
    time = []
    mag = []
    error = []

    #check characteristics and store data
    for obs in light_curve:
        if obs["fid"] == fid:
            if obs["mag"] < mag_boundry:
                time.append(obs["mjd"])
                mag.append(obs["mag"])
                error.append(obs["e_mag"])

    #convert mag to psuedo flux
    mag = -1 * np.array(mag)
    mag_flipped = np.exp(mag)

    #convert errror to psuedo flux
    error = np.array(error)
    error_flipped = np.exp(-1 * error)

    return time, mag_flipped, error_flipped

def fit_curve(time, mag, error, mode="george"):

    match mode:
        case "george":
            return fit_george(time, mag, error)
        case "mcmc":
            return fit_custom(time, mag, error)

def fit_custom(time, mag, dmag):

    time = time / np.max(time)
    
    #initialize data range
    x_fit = np.linspace(np.min(time) - 50, np.max(time) + 100, 1000)

    rng = np.random.default_rng()
    a_random =  (rng.random(n_walkers) * a_max)
    b_random =  (rng.random(n_walkers) * b_max)
    y_random =  (rng.random(n_walkers) * y_max)
    t_random =  (rng.random(n_walkers) * t_max)
    f_random =  (rng.random(n_walkers) * f_max)
    r_random =  (rng.random(n_walkers) * r_max)
    
    initial_guesses = np.array([a_random, b_random, y_random, t_random, f_random, r_random])

    # initialize the sampler
    sampler = emcee.EnsembleSampler(n_walkers, n_params, log_posterior, args=[time, mag, dmag])

    # run!
    sampler.run_mcmc(initial_guesses.T, n_steps)

    flat = sampler.get_chain(discard=100, flat=True)

    theta = [np.mean(flat[:, 0]), np.mean(flat[:, 1]), np.mean(flat[:, 2]), np.mean(flat[:, 3]), np.mean(flat[:, 4]), np.mean(flat[:, 5])]
    
    return (custom_model(theta, x_fit), 0, x_fit)

def log_posterior(theta, time, mag, dmag):
    return log_likelihood(theta, time, mag, dmag) + log_prior(theta)

def log_prior(theta):

    a, b, y, t, f, r = theta

    a_cond = (a > a_min) & (a < a_max)
    a_cond = a_cond / (a_max - a_min)

    b_cond = (b > b_min) & (b < b_max)
    b_cond = b_cond / (b_max - b_min)

    y_cond = (y > y_min) & (y < y_max)
    y_cond = y_cond / (y_max - y_min)

    t_cond = (t > t_min) & (t < t_max)
    t_cond = t_cond / (t_max - t_min)

    f_cond = (f > f_min) & (f < f_max)
    f_cond = f_cond / (f_max - f_min)

    r_cond = (r > r_min) & (r < r_max)
    r_cond = r_cond / (r_max - r_min)

    return np.log(a_cond * b_cond * y_cond * t_cond * f_cond * r_cond)

def log_likelihood(theta, time, mag, dmag):

    likelihood = -0.5 * np.sum(((mag - (custom_model(theta, time))) / dmag)**2, axis=-1)

    return likelihood
    
def custom_model(theta, time):

    a, b, y, t, f, r = theta
    t = time - t
    numerator = a * (1 - (b * np.minimum(t, y))) * np.exp(-1 * ((np.maximum(t, y)) - y) / f)
    denominator = 1 + np.exp(-1 * (t / r))
    return numerator / denominator

def fit_george(time, mag_flipped, error):

    #initialize data range
    x_fit = np.linspace(np.min(time) - 5, np.max(time) + 100, 1000)

    #initialize george
    #kernel =  np.var(mag_flipped) * george.kernels.ExpSquaredKernel(100)
    kernel = np.var(mag_flipped) * george.kernels.Matern32Kernel(metric=100)
    gp = george.GP(kernel, solver=george.HODLRSolver)

    #define george optimization
    def nll(p):
        gp.set_parameter_vector(p)
        ll = gp.log_likelihood(mag_flipped, quiet=True)
        return -ll if np.isfinite(ll) else 1e25
    def grad_nll(p):
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(mag_flipped, quiet=True)

    #optimize george
    gp.compute(time, error)
    p0 = gp.get_parameter_vector()
    results = op.minimize(nll, p0, jac=grad_nll, method="L-BFGS-B")

    #update george
    gp.set_parameter_vector(results.x)
    
    #use george
    gp.compute(time, error)
    #print(mag_flipped.shape, x_fit.shape)
    pred, pred_var = gp.predict(mag_flipped, x_fit, return_var=True)

    return pred, pred_var, x_fit

def get_rise_time(pred, x_fit, peak, mode="basic"):

    match mode:
        case "slope":
            return rise_time_slope(pred, x_fit, peak)
        case _:
            return rise_time_basic(pred, x_fit, peak)

def rise_time_basic(pred, x_fit, peak):
    
    #get peak index
    peak_index = np.argmax(pred)

    return x_fit[peak_index] - x_fit[0]

def rise_time_slope(pred, x_fit, peak):
    #get peak index
    peak_index = np.argmax(pred)

    #get gradient
    slopes = np.gradient(pred, x_fit)

    #calculate rise time
    rise_days = 0
    for point in x_fit[:peak_index]:
        if slopes[np.where(x_fit == point)] < 1:
            rise_days = point
    rise_time = rise_days - x_fit[0]

    return rise_time

def get_fall_time(pred, x_fit, peak, mode="basic"):
    
    match mode:
        case "slope":
            return fall_time_slope(pred, x_fit, peak)
        case _:
            return fall_time_basic(pred, x_fit, peak)
    
def fall_time_basic(pred, x_fit, peak):
    
    #get peak index
    peak_index = np.argmax(pred)

    return x_fit[-1] - x_fit[peak_index]

def fall_time_slope(pred, x_fit, peak):
    #get peak index
    peak_index = np.argmax(pred)

    #get gradient
    slopes = np.gradient(pred, x_fit)
    
    #calculate fall time
    fall_days = 0
    for point in x_fit[peak_index:]:
        if slopes[np.where(x_fit == point)] > -1:
            fall_days = point
    fall_time = fall_days - x_fit[peak_index]

    return fall_time