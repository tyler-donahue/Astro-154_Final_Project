#imports
import matplotlib.pyplot as plt
import numpy as np
import george
import scipy.optimize as op
import emcee

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

def fit_custom(time, mag, error):
    return None


def fit_george(time, mag_flipped, error):

    #initialize data range
    x_fit = np.linspace(np.min(time) - 50, np.max(time) + 100, 1000)

    #get george off
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