
from pathlib import Path
import os
import zipfile
import sqlite3
import csv
import datetime
import pandas as pd
from help_functions import str_to_date, create_select_list
import numpy as np
import time
import matplotlib.pyplot as plt
# plt.ioff()
import pickle
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.signal import butter, filtfilt, blackman
from scipy.fftpack import fft, ifft, fftshift, ifftshift, fftfreq, rfft, rfftfreq
from mpl_toolkits.axes_grid1 import make_axes_locatable

from aemo import Database

cm_per_inch = 2.5


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y



def spectrum_func(x, p, cutoff=1 / 10):
    """ Evaluate linear fit in log-log space for given frequency. Value is capped at the frequency cutoff
    Used to compute spectral density of noise in case the linear fit option is chosen
    """
    A = np.exp(p[1])
    b = p[0]
    if abs(x) < cutoff:
        return A * np.power(cutoff, b)
    else:
        return A * np.power(np.abs(x), b)


def eval_spline(x, bins, fit):
    """ Evaluate spline function, for quantile regression with spline functions
    x - points for evaluation
    fit - panda series with spline coefficients
    bins - bins defining range of splines
    """
    nbins = bins.__len__() - 1
    res = np.zeros(x.__len__())
    x_dig = np.digitize(x, bins=bins, right=False)

    for i, (val, bin) in enumerate(zip(x, x_dig)):
        if bin > nbins:
            bin = nbins
        res[i] = fit['intercept'] \
                 + fit[f'b{bin}3'] * val ** 3 \
                 + fit[f'b{bin}2'] * val ** 2 \
                 + fit[f'b{bin}1'] * val
        if bin < nbins:
            res[i] += fit[f'b{bin}0']

    return res, x_dig


def eval_spline2(x,bins,fit):
    """ Evaluate spline with separate intercepts """
    nbins = bins.__len__() - 1
    res = np.zeros(x.__len__())
    x_dig = np.digitize(x, bins=bins, right=False)

    for i, (val, bin) in enumerate(zip(x, x_dig)):
        if bin > nbins:
            bin = nbins
        res[i] =   fit[f'b{bin}3'] * val ** 3 \
                   + fit[f'b{bin}2'] * val ** 2 \
                   + fit[f'b{bin}1'] * val \
                   + fit[f'b{bin}0']

    return res, x_dig


def eval_splines_at_x(x,bins,fit):
    """ Evaluate all quantile splines at single forecast point. Also add extreme quantiles 0 and 1"""

    quant = np.zeros(fit.__len__() + 2)
    quant[0] = x - 1 # highest possible underprediction (negative)
    quant[-1] = x # highest possible overprediction

    s = np.zeros(quant.__len__())
    s[0] = 0
    s[-1] = 1

    x_bin = 1
    while bins[x_bin] < x:
        x_bin += 1

    for i in fit.index:
        # assumes quantiles are stored in increasing order
        quant[i+1] = fit.at[i,f'b{x_bin}3'] * x ** 3 \
                     + fit.at[i,f'b{x_bin}2'] * x ** 2 \
                     + fit.at[i,f'b{x_bin}1'] * x \
                     + fit.at[i,f'b{x_bin}0']

        s[i+1] = fit.at[i,'q']

    return quant,s


def eval_splines_2d(x, bins, fit, cut_extreme=False):
    """ Evaluate all quantile splines at range of forecasts (x). Also add extreme quantiles 0 and 1

    cut_extreme: assume first and last quantile curves are quantiles for 0 and 1, to cut away part of pdf
    outside the given quantile curves
    """
    N = x.__len__()
    if cut_extreme:
        Nq = fit.__len__()
    else:
        Nq = fit.__len__() + 2

    quant = np.zeros((N,Nq))
    s = np.zeros(Nq)

    if not cut_extreme:
        # put extreme values
        for i in range(N):
            quant[i][0] = x[i] - 1  # highest possible underprediction (negative)
            quant[i][-1] = x[i]  # highest possible overprediction
    s[0] = 0
    s[-1] = 1

    x_bin = np.ones(N,dtype=int)
    for i in range(N):
        while bins[x_bin[i]] < x[i] and x_bin[i] < bins.__len__()-1:
            x_bin[i] += 1

    for i in fit.index:
        for ii in range(N):
            # assumes quantiles are stored in increasing order
            if not cut_extreme:
                quant[ii][i + 1] = fit.at[i, f'b{x_bin[ii]}3'] * x[ii] ** 3 \
                                   + fit.at[i, f'b{x_bin[ii]}2'] * x[ii] ** 2 \
                                   + fit.at[i, f'b{x_bin[ii]}1'] * x[ii] \
                                   + fit.at[i, f'b{x_bin[ii]}0']
                s[i + 1] = fit.at[i, 'q']
            else:
                quant[ii][i] = fit.at[i, f'b{x_bin[ii]}3'] * x[ii] ** 3 \
                               + fit.at[i, f'b{x_bin[ii]}2'] * x[ii] ** 2 \
                               + fit.at[i, f'b{x_bin[ii]}1'] * x[ii] \
                               + fit.at[i, f'b{x_bin[ii]}0']
                if i > 0 and i < Nq -1:
                    s[i] = fit.at[i,'q']

    return quant, s


def get_pdf_2d(bins,fit,fc=np.linspace(0,1,51),er=np.linspace(-1,1,101),cut_extreme=False):
    """
    Evaluate 2d pdf matrix, given spline fits of quantiles

    :param bins: division for splines, in fc range
    :param fit: dataframe with spline coefficients
    :param fc: points in forecast space (0,1) for evaluation
    :param er: points in error space (0,1) for evaluation
    :return: pdf[ner,nfc]: ner decreasing, nfc increasing
    """

    # err_vals = np.linspace(-1,1,100)
    Ner = er.__len__()

    # q = quant2d[0]
    quant2d,s = eval_splines_2d(fc,bins,fit,cut_extreme=cut_extreme)
    Nfc = quant2d.shape[0]

    qidx = np.zeros((Ner,Nfc),dtype=int)
    pdf = np.zeros(qidx.shape)
    for iq in range(Nfc):
        q = quant2d[iq]
        for i,e in enumerate(er):
            eidx = Ner-i-1
            if e <= q[0] or e >= q[q.__len__()-1]:
                pdf[eidx,iq] = 0
            else:
                while q[qidx[eidx,iq]] < e:
                    qidx[eidx,iq] += 1
                pdf[eidx,iq] = (s[qidx[eidx,iq]]-s[qidx[eidx,iq]-1]) / (q[qidx[eidx,iq]] - q[qidx[eidx,iq]-1])

    return pdf,quant2d

def cdf2uniform(fc,er,bins,fit):
    """
    Map error values to corresponding uniformly distributed variables, using the cumulative density function described
    by the cubic splines in fit at the given forecast value. The cdf is assumed to change linearly between
    the quantile curves given in fit
    Note: fc and er must be of equal length
    :param fc:
    :param er:
    :param bins:
    :param fit:
    :return:
    """
    uni = np.zeros(fc.__len__(),dtype=float) # univerate variable

    # get quantiles at given forecast values
    qmat,s = eval_splines_2d(x=fc,bins=bins,fit=fit)

    for i,(f,e) in enumerate(zip(fc,er)):
        # for ie,e in enumerate(er):
        q = qmat[i]
        # q: increasing quantiles

        if e <= q[0]:
            uni[i] = 0
        elif e >= q[-1]:
            uni[i] = 1
        else:
            qidx = 0
            while q[qidx] <= e:
                qidx += 1
            # interpolate uniform value between q[idx-1] and q[idx]
            uni[i] = s[qidx-1] + (e-q[qidx-1])*(s[qidx]-s[qidx-1])/(q[qidx]-q[qidx-1])

    return uni


def uniform2cdf(fc,un,bins,fit,cut_extreme=False):

    err = np.zeros(fc.__len__(),dtype=float)
    qmat,s = eval_splines_2d(x=fc,bins=bins,fit=fit,cut_extreme=cut_extreme)

    for i,(f,u) in enumerate(zip(fc,un)):

        q = qmat[i]

        emax = f # maxumum error in positive direction
        emin = f-1 # maximum error in negative direction

        if u == 0:
            err[i] = emin
        elif u == 1:
            err[i] = emax
        else:
            sidx = 0
            while s[sidx] <= u:
                sidx += 1
            # interpolate uniform value between q[idx-1] and q[idx]
            err[i] = q[sidx-1] + (u-s[sidx-1])*(q[sidx]-q[sidx-1])/(s[sidx]-s[sidx-1])

    return err


def make_5min_forecast(wfc,lp_cutoff,fs,lp_order):

    start = wfc.index[0]
    end = wfc.index[-1]

    wfc_5min = pd.DataFrame(index=pd.date_range(start,end,freq='5min'), columns=wfc.columns, dtype=float)

    # %%
    # note: 1st value is at 0 min or 30 min, then next value from wfc will be for next 6 consequtive 5-min instances
    wfc_5min.iloc[0, :] = wfc.iloc[0, :]
    tidx = 1
    vals = np.array(wfc.iloc[tidx, :])
    for t in wfc_5min.index:
        if t > wfc.index[tidx] and tidx < wfc.__len__() - 1:
            tidx += 1
            vals = np.array(wfc.iloc[tidx, :])
        wfc_5min.loc[t, :] = vals

    # %%
    # wfc_5min_lp = pd.DataFrame(index=err_idx, columns=m.units, dtype=float)
    for c in wfc_5min.columns:
        nan_idx = wfc_5min[c].isna()
        wfc_5min.loc[nan_idx,c] = 0
        wfc_5min[c] = butter_lowpass_filter(wfc_5min[c], lp_cutoff, fs, lp_order)
        wfc_5min.loc[nan_idx,c] = np.nan

    wfc_5min[wfc_5min < 0] = 0

    return wfc_5min


def get_scaled_filtered_noise(m, n=24 * 12):
    """ Create noise for filling high-frequency part of wind production spectrum
    n - number of samples wanted
    m - dict containing:
        spectrum - symmetric real spectrum used for scaling noise. noise intensity decreases (linearly?) with frequency
        std_real - steev of real part of white noise in frequency domain
        std_imag - stdev of imaginary part of noise in frequency domain
        fs - sampling frequency
        cutoff - cutoff frequency for lowpass filter
    """
    spectrum = m['spectrum']
    std_real = m['std_real']
    std_imag = m['std_imag']
    fs = m['fs']
    cutoff = m['cutoff']
    forder = m['forder']

    N = spectrum.__len__() # number of samples in spectrum

    # generate noise in time domain, scale to get correct stdev
    wn = np.random.normal(0, 1, size=n)
    wn_fft = fftshift(fft(wn))
    wn_fft = np.real(wn_fft) * std_real / np.std(np.real(wn_fft)) \
             + 1j * np.imag(wn_fft) * std_imag / np.std(np.imag(wn_fft))

    wn_freq = fftshift(fftfreq(n, d=1 / fs))
    spectrum_freq = fftshift(fftfreq(N,d=1/fs))

    # interpolate spectrum to new sample number
    wn_scale_factor = np.interp(wn_freq, spectrum_freq, spectrum) * np.power(n / N, 0.5)
    # scale wn spectrum and go back to time domain
    # Note: not sure why fft scales with sqrt of window length
    wn_scaled = ifft(ifftshift(wn_fft * wn_scale_factor))

    # check that noise is real
    max_imag = np.max(np.abs(np.imag(wn_scaled)))
    if max_imag > 1e-10:
        print(f'Scaled wn is non-real for n={n}, imaginary part: {max_imag:0.2e}')
    wn_scaled = np.real(wn_scaled)

    # filter noise, return high-frequency component
    wn_hp = wn_scaled - butter_lowpass_filter(wn_scaled, cutoff=cutoff, fs=fs, order=forder)

    return wn_hp

class WindModel:

    def __init__(self,name='basemodel',path='D:/Data/AEMO/Results',wpd_db='D:/Data/aemo_new.db',wfc_db='D:/Data/aemo_new.db',
                 default_model = 'wind_model_default'):

        self.name = name
        self.default_options()
        # setup databases
        self.wpd_db = Database(wpd_db)
        self.wfc_db = Database(wfc_db)

        self.path = Path(path) / f'wind_model_{name}'
        self.path.mkdir(exist_ok=True,parents=True)

        # always load values from the default model
        # if os.path.isfile(Path(path) / f'{default_model}.pkl'):
        #     with open(Path(path) / f'{default_model}.pkl','rb') as f:
        #         m = pickle.load(f)
        #     for p in m:
        #         self.__setattr__(p,m[p])


    def default_options(self):


        self.units = ['ARWF1', 'MACARTH1', 'BALDHWF1']
        self.startdate = '20190801'
        self.enddate = '20200731'
        self.lead_times = [32,16,1]
        self.eps_figures = True
        self.plot_titles = True

        # low-pass filter parameters
        self.lp_cutoff = 1
        self.lp_order = 6
        self.fs = 12

        # covariance parameters
        self.cov_scale=None
        self.cov_exclude_zero_days=True
        self.cov_nhours_exclude = 10
        self.cov_resolution = '30min' # 30min or 5min
        # parameters for lp-filtering wind forecast to use for mapping errors to uniform distribution
        self.cov_quant_cutoff = 0.1
        self.cov_quant_forder = 6
        self.cov_quant_filter = True
        self.cov_quant_interp = 'linear' # 'linear' or 'rect'
        # min and max values for uniform distribution when mapping uniform to multivariate normal
        self.cov_un_min = 0.0001
        self.cov_un_max = 0.9999


        # high-frequency noise parameters
        self.hf_ndays = 10
        self.hf_scale = 0.3
        self.hf_std_scale = 1
        self.hf_fit_cutoff = 1
        self.hf_linear_scale = True
        self.hf_add_noise_avg = False
        self.hf_nbins = 3
        self.hf_lead_time = 1 # lead time for forecast used to create regimes when fitting noise
        self.hf_binvar = 'fc' # fc/pd, to use forecast or production for creating regimes
        self.hf_noise_zeroprod = False
        self.hf_nzero_nonoise = 2 # for periods with more than this number of periods with
        # negative production after adding covariance error no noise is added
        # parameters for plot of noise-fit
        self.hf_plot_figures = True
        self.hf_plot_hours = 10
        self.hf_plot_offset = 0
        self.hf_plot_nscen = 3

        # start of operation day for aemo
        self.start_hour = '0430'

        # parameters for quantile regression
        self.quant_nbins = 3
        self.quant_set_bins = None # manually set limit of bins
        self.quant_spline_order = 3
        self.quant_qvals = [0.05, 0.2, 0.5, 0.8, 0.95]
        self.quant_plots = True
        self.quant_solver_output = False
        self.quant_cofit = True
        self.quant_resolution = '30min'
        self.quant_remove_outliers = True
        self.quant_outlier_tol = 0.02
        self.quant_remove_zero_periods = True
        self.quant_nzero_hours_remove = 10
        self.quant_labels = False
        self.quant_fit_boundary = True
        self.quant_q1_nbins = 2 # number of bins for 1th quantile (remaining bins are taken as feasible boundary)
        self.quant_q0_nbins = 2 # number of bins for 0th quantile
        self.quant_maxerr_fc1 = 0.5 # maximum value of the forecast error when forecast = 1, imposed as constraint 
                                    # when fitting quantiles, when fitting boundary quantiles
        

    def load_model(self):

        if os.path.exists(self.path / f'wind_model_{self.name}.pkl'):
            with open(self.path / f'wind_model_{self.name}.pkl', 'rb') as f:
                m = pickle.load(f)
            for p in m:
                self.__setattr__(p,m[p])

    def save_model(self):

        m = {}
        m_pars = ['units','startdate','enddate','capacity','cov','noise','cov_quant','cor_quant',
                  'lead_times','lp_cutoff','lp_order','fs','cov_scale','cov_exclude_zero_days','cov_resolution',
                  'cov_nhours_exclude',
                  'hf_ndays','hf_std_scale','hf_fit_cutoff','hf_linear_scale','hf_add_noise_avg','hf_nbins','hf_lead_time',
                  'hf_binvar',
                  'quant_fits','quant_coeff','quant_bins','quant_fit_boundary','quant_q1_nbins','quant_q0_nbins',
                  'quant_nbins','quant_spline_order','quant_qvals','quant_plots','quant_solver_output','quant_cofit']

        save_pars = [p for p in m_pars if hasattr(self,p)]
        for p in save_pars:
            m[p] = self.__getattribute__(p)
        with open(self.path / f'wind_model_{self.name}.pkl','wb') as f:
            pickle.dump(m,f)

    def load_data(self):

        wfc_end = (str_to_date(self.enddate) + datetime.timedelta(days=1)).strftime('%Y%m%d')
        self.wfc = self.wfc_db.select_forecast_data_full(startdate=self.startdate,
                                                         enddate=wfc_end,
                                                         categories=self.units,
                                                         lead_times=self.lead_times)


        wpd_start = f'{self.startdate}:0000'
        wpd_end = (str_to_date(f'{self.enddate}:00') + datetime.timedelta(days=2)).strftime('%Y%m%d:%H%M')
        self.wpd = self.wpd_db.select_data(starttime=wpd_start,
                                           endtime=wpd_end,
                                           table_type='dispatch',
                                           categories=self.units)

        # normalize data
        self.capacity = self.wpd.max()

        self.wpd = self.wpd / self.capacity
        for col in self.wfc.columns:
            self.wfc[col] = self.wfc[col] / self.capacity[col[0]]


        self.nDaysData = np.round((str_to_date(self.enddate)-str_to_date(self.startdate)).days) + 1

    # def get_capacity(self):
    #
    #     self.capacity = self.wpd_db.query_max_values(table_type='dispatch',categories=self.units)


    def filter_data(self):
        """ Low-pass filter 5 min production data """
        self.wpd_lp = pd.DataFrame(dtype=float, index=self.wpd.index, columns=self.wpd.columns)
        for col in self.wpd.columns:
            x = self.wpd[col].copy()
            # Replace nan-values with zero when filtering, then reinsert nan-values
            xnan = x.isna()
            nnan = xnan.sum()
            if nnan > 0:
                print(f'Warning: replacing {nnan} nan values with zero in wpd for filtering')
                x = x.fillna(0)
            self.wpd_lp[col] = butter_lowpass_filter(x, self.lp_cutoff, self.fs, self.lp_order)
            if nnan > 0:
                self.wpd_lp.loc[xnan,col] = np.nan

        """ LP-filter 30 min forecast data, for """

    def find_valid_data(self):
        #!! TODO: Create single function to determine which data range (i.e. which days) to include in different parts of the model
        #!! TODO: then replace various methods for exluding nan/zero data in fit_noise, fit_quantiles, and fit_covariance
        # for each day in specified period find:
        # start index of day in data (forecast and production)

        # also find list of days for which:
        # 1. forecast has nan value
        # 2. production has nan value
        # 3. forecast is zero fo more than x hours
        # 4. production is zero for more than x hours
        pass

    def fit_covariance(self):
        """ Fit covariance matrix, but in domain for transformed normal variable, obtained by doing

        err -[quantile fit]-> uniform -[inverse of cdf]-> standard normal distribution

        Also does fit for unmapped data
        """

        from scipy.stats import norm

        """ Fit covariance matrix using LP-filtered data """
        cov_resolution = self.cov_resolution

        start_hour = self.start_hour
        nT30 = 24 * 2  # Number of 30-min periods per day
        nT5 = 24 * 12  # number of 5-min periods per day


        if cov_resolution == '30min':
            nT = nT30
        else:
            nT = nT5

        wpd = self.wpd
        wpd_filt = self.wpd_lp
        units = self.units

        d_cov = {}
        d_cor = {}

        # %% choose which 24-hour periods to include for computing covariance
        t30 = self.wfc.index
        start_idx_30 = 0  # applies only to 30-min time values
        while t30[start_idx_30].strftime('%H%M') < start_hour:
            start_idx_30 += 1

        # compute number of complete days in data
        # note that wfc contains one extra day of data, needed to interpolate forecasts 5-min values in last hour
        nDaysData = np.floor_divide(t30.__len__() - start_idx_30, nT30) - 1
        end_idx_30 = start_idx_30 + nDaysData * nT30

        err_idx = pd.date_range(start=t30[start_idx_30], end=t30[end_idx_30], freq=cov_resolution)

        # %%
        for lead_time in self.lead_times:
            # get data with specific lead time
            wfc = self.wfc.loc[:, (slice(None), lead_time)]
            wfc.columns = wfc.columns.get_level_values(0)

            # replace nan values with zero, keep track of index
            wfc_nan = wfc.isna()
            wfc[wfc_nan] = 0
            nan_days = []
            for t in wfc_nan.index:
                if t.strftime('%H%M') == start_hour and wfc_nan.loc[t,:].sum() > 0:
                    # exclude this day
                    nan_days.append(t.strftime('%Y%m%d'))

            # % compute error
            if cov_resolution == '30min':
                # fit covariance with 30 min resolution
                pred_err = (wfc.loc[err_idx, :] - wpd_filt.loc[err_idx, :])

            else:
                # interpolate forecasts to 5-min resolution
                # Note: wfc will contain data from 4:30 on day 1 until 4:00 on day N+1
                wfc_5min = pd.DataFrame(dtype=float,
                                        index=err_idx,
                                        columns=wfc.columns)
                time_idxs_30min = pd.date_range(start=err_idx[0], end=err_idx[-1], freq='30min')
                if self.cov_quant_interp == 'linear':
                    # linear interpolation between 30-min values
                    wfc_5min.loc[time_idxs_30min, :] = wfc.loc[time_idxs_30min, :]
                    wfc_5min = wfc_5min.interpolate(method='linear')
                else:
                    # rectangular step profile, then low-pass filtered
                    # wfc_5min.loc[err_idx,self.units] = self.wfc_5min.loc[err_idx,(self.units,lead_time)]
                    wfc_5min.loc[err_idx,:] = make_5min_forecast(wfc.loc[time_idxs_30min,:],self.lp_cutoff,self.fs,self.lp_order)

                pred_err = (wfc_5min - wpd_filt.loc[wfc_5min.index, :])


            # lowpass filter forecast to use for transformation to uniform distribution
            wfc_lp = pd.DataFrame(index=err_idx, columns=self.units, dtype=float)
            # fc_lp_sel = pd.DataFrame(index=err_idx, columns=wfc.columns, dtype=float)
            for wf in self.units:
                if self.cov_quant_filter:
                    if cov_resolution == '30min':
                        wfc_lp[wf] = butter_lowpass_filter(wfc.loc[err_idx,wf],
                                                           cutoff=self.cov_quant_cutoff,
                                                           fs=2,
                                                           order=self.cov_quant_forder)
                    else:
                        wfc_lp[wf] = butter_lowpass_filter(wfc_5min[wf],
                                                           cutoff=self.cov_quant_cutoff,
                                                           fs=12,
                                                           order=self.cov_quant_forder)
                else:
                    if cov_resolution == '30min':
                        wfc_lp[wf] = wfc.loc[err_idx,wf]
                    else:
                        wfc_lp[wf] = wfc_5min[wf]
                # fc_lp_sel[wf] = fc_lp.loc[err_idx, wf]

            # f = plt.figure()
            # for i, wf in enumerate(self.units):
            #     ax = f.add_subplot(2, 2, i + 1)
            #     fc[wf].plot(label='data')
            #     fc_lp_sel[wf].plot(label='LP')
            #     plt.grid()
            #     plt.legend()

            # %%

            # map error to uniform distribution using fitted quantiles
            err_un = pd.DataFrame(dtype=float, index=pred_err.index, columns=pred_err.columns)
            # err_un_lp = pd.DataFrame(dtype=float, index=pred_err.index, columns=pred_err.columns)
            for wf in self.units:
                err_un.loc[:, wf] = cdf2uniform(fc=wfc_lp[wf], er=pred_err[wf], bins=self.quant_bins[lead_time][wf],
                                                fit=pd.DataFrame(self.quant_fits[lead_time][wf],
                                                                 columns=['q'] + self.quant_coeff))
                # err_un_lp.loc[:, wf] = cdf2uniform(fc=fc_lp_sel[wf], er=pred_err[wf], bins=self.quant_bins[lead_time][wf],
                #                                    fit=pd.DataFrame(self.quant_fits[lead_time][wf],
                #                                                     columns=['q'] + self.quant_coeff))

            # map uniform distribution to multivariate normal distribution

            err_un[err_un < self.cov_un_min] = self.cov_un_min
            err_un[err_un > self.cov_un_max] = self.cov_un_max
            # err_un_lp[err_un_lp < un_min] = un_min
            # err_un_lp[err_un_lp > un_max] = un_max

            # %%
            err_norm = pd.DataFrame(dtype=float, index=pred_err.index, columns=pred_err.columns)
            # err_norm_lp = pd.DataFrame(dtype=float, index=pred_err.index, columns=pred_err.columns)
            for wf in self.units:
                err_norm[wf] = norm.ppf(err_un[wf])
                # err_norm_lp[wf] = norm.ppf(err_un_lp[wf])

            # %%
            histbins = 20
            xvals = np.linspace(-6, 6, 100)
            yvals = norm.pdf(xvals)

            f = plt.gcf()
            f.clf()
            f.set_size_inches(18,8)
            for i, wf in enumerate(self.units):
                ax = f.add_subplot(2,3,i+1)
                plt.hist(err_norm[wf], bins=histbins, edgecolor='black', linewidth=1.2, density=True)
                # plt.density(err_norm[wf],bins=histbins,edgecolor='black',linewidth=1.2)
                plt.plot(xvals, yvals, lw=1.5)
                plt.title(f'{wf}')
            # plt.grid()
            #
            # f = plt.figure()
            for i, wf in enumerate(self.units):
                ax = f.add_subplot(2,3, i + 4)
                plt.hist(pred_err[wf] / pred_err[wf].std(), bins=histbins, edgecolor='black', linewidth=1.2,
                         density=True)
                plt.plot(xvals, yvals, lw=1.5)
                # plt.title(f'{wf}')
            if self.plot_titles:
                plt.suptitle('Error distribution')
            plt.savefig(self.path / f'error_distributions.png')
            # plt.grid()

            # %%
            # for each day, check if it should be used in cov calculation
            days_include = []
            for didx in range(nDaysData):
                time_idxs_day = err_idx[didx * nT:(didx + 1) * nT]
                include = True
                if err_un.loc[time_idxs_day, :].isna().sum().sum() > 0:
                    include = False  # exclude days with missing values
                if self.cov_exclude_zero_days:
                    for u in units:
                        # criteria for excluding day:
                        # more than 5 periods with exactly correct wfc
                        # if (pred_err.loc[time_idxs,u] < 1e-4).sum() > 5:
                        # more than 10 hours with zero production
                        if (wpd.loc[time_idxs_day, u] < 1e-2).sum() > nT / 24 * self.cov_nhours_exclude:
                            # if False:
                            include = False
                            break
                if time_idxs_day[0].strftime('%Y%m%d') in nan_days: # remove days with missing forecast data
                    include = False
                if include:
                    days_include.append(didx)

            nDays = days_include.__len__()

            nVars = nT * units.__len__()

            fit_data = pd.DataFrame(dtype=float, columns=range(nDays), index=range(nVars))
            fit_data2 = pd.DataFrame(dtype=float, columns=range(nDays), index=range(nVars)) # unprocessed data
            for idx, didx in enumerate(days_include):
                for pidx in range(units.__len__()):
                    fit_data.iloc[range(pidx * nT, (pidx + 1) * nT), idx] = np.array(
                        err_norm.iloc[didx * nT:(didx + 1) * nT, pidx]
                    )
                    fit_data2.iloc[range(pidx * nT, (pidx + 1) * nT), idx] = np.array(
                        pred_err.iloc[didx * nT:(didx + 1) * nT, pidx]
                    )

            cov = np.cov(fit_data)
            cov2 = np.cov(fit_data2)

            corr = np.corrcoef(fit_data)
            corr2 = np.corrcoef(fit_data2)

            # %% scale covariance matrix
            if self.cov_scale is not None:
                scale_up = self.cov_scale
                slope = (np.sqrt(1 + scale_up) - 1) / (nT - 1)
                intercept = (nT - np.sqrt(1 + scale_up)) / (nT - 1)

                cov_scaled = np.zeros(cov.shape)
                cov2_scaled = np.zeros(cov.shape)
                # scale matrix
                for i in range(nVars):
                    ti = np.remainder(i, nT) + 1
                    scale_i = slope * ti + intercept
                    for j in range(nVars):
                        tj = np.remainder(j, nT) + 1
                        scale_j = slope * tj + intercept
                        cov_scaled[i, j] = cov[i, j] * scale_i * scale_j
                        cov2_scaled[i, j] = cov2[i, j] * scale_i * scale_j

                cov = cov_scaled
                cov2 = cov2_scaled

            # %%
            fig = plt.gcf()
            fig.set_size_inches(14,6.5)
            fig.clf()

            # ax.cla()
            ax1 = fig.add_subplot(1,2,1)
            ax2 = fig.add_subplot(1,2,2)
            fig.subplots_adjust(top=0.9)

            im = ax2.imshow(corr)
            fig.colorbar(im,ax=ax2)
            if self.plot_titles:
                ax2.set_title('Mapped data',y=1)

            # divider = make_axes_locatable(ax)
            # cax = divider.append_axes("right", size="5%", pad=0.05)
            # plt.colorbar(im,cax=cax)


            im = ax1.imshow(corr2)
            fig.colorbar(im,ax=ax1)
            if self.plot_titles:
                ax1.set_title('Original data',y=1)

            # divider = make_axes_locatable(ax)
            # cax = divider.append_axes("right", size="5%", pad=0.05)
            # plt.colorbar(im,cax=cax)


            # fig.subplots_adjust(right=0.8)
            # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            # fig.colorbar(im, cax=cbar_ax)

            title_str = 'Error covariance: '
            for p in units:
                title_str += f'{p}, '
            # plt.suptitle(title_str)
            plt.savefig(Path(self.path) / f'cov_{self.name}_l{lead_time}.png')
            if self.eps_figures:
                plt.savefig(Path(self.path) / f'cov_{self.name}_l{lead_time}.eps')

            d_cov[f'l{lead_time}'] = cov
            d_cor[f'l{lead_time}'] = corr

        self.cov_quant = d_cov
        self.cor_quant = d_cor

    def fit_quantiles(self):
        """ Fit quantiles to error distribution, as function of forecasted wind power """

        #!! TODO: Implement quantile fit with Pyomo to allow other solvers than Gurobi

        import gurobipy as gp
        from gurobipy import GRB

        quant_fits = {}
        quant_bins = {}

        end_margin = 1e-3 # margin between quantile curves, if cofitting is used
        pdf_max = 5 # clip values higher than this in pdf, for good plot visualization

        if self.quant_set_bins is not None:
            # bins set manually, ensure nbins parameter is consistent
            self.quant_nbins = self.quant_set_bins.__len__() - 1

        nbins = self.quant_nbins
        spline_order = self.quant_spline_order
        quantiles = self.quant_qvals
        nquant = quantiles.__len__()
        if self.quant_fit_boundary:
            quantiles_full = [quantiles[i] for i in range(1,nquant-1)]
            quant0 = quantiles[0]
            quant1 = quantiles[-1]
        else:
            quantiles_full = self.quant_qvals
        nquant_full = quantiles_full.__len__()

        qreg_resolution = self.quant_resolution
        start_hour = self.start_hour
        nT30 = 24 * 2  # Number of 30-min periods per day

        t30 = self.wfc.index
        start_idx_30 = 0  # applies only to 30-min time values
        while t30[start_idx_30].strftime('%H%M') < start_hour:
            start_idx_30 += 1

        # compute number of complete days in data
        # note that wfc contains one extra day of data, needed to interpolate forecasts 5-min values in last hour
        nDaysData = np.floor_divide(t30.__len__() - start_idx_30, nT30) - 1
        end_idx_30 = start_idx_30 + nDaysData * nT30

        err_idx = pd.date_range(start=t30[start_idx_30], end=t30[end_idx_30], freq=qreg_resolution)
        # nobs = err_idx.__len__()

        spline_coeff = spline_order + 1

        coeff = [f'b{k + 1}{3 - i}' for k in range(nbins) for i in range(spline_coeff)]
        ncoeff = coeff.__len__()

        # parameters used when fitting extreme quantiles
        ncoeff_q0 = spline_coeff*self.quant_q0_nbins
        ncoeff_q1 = spline_coeff*self.quant_q1_nbins
        coeff0_fit = [coeff[i] for i in range(ncoeff_q0)]
        coeff1_fit = [coeff[i] for i in range(ncoeff-ncoeff_q1,ncoeff)]


        print('-- Fitting quantile curves -- ')
        for lead_time in self.lead_times:

            quant_fits[lead_time] = {}
            quant_bins[lead_time] = {}
            if qreg_resolution == '30min':
                # get data with specific lead time
                wfc = self.wfc.loc[:, (slice(None), lead_time)]
            else:
                wfc = self.wfc_5min.loc[:, (slice(None), lead_time)]
            wfc.columns = wfc.columns.get_level_values(0)

            for wf in self.units:
                print(f'L{lead_time} {wf}')
                # remove observations with nan forecast or production
                err_idx_nonan = [t for t in err_idx if wfc.loc[t,:].isna().sum() == 0 and self.wpd_lp.loc[t,:].isna().sum() == 0]

                # get data for this unit

                df = pd.DataFrame(dtype=float, index=err_idx_nonan, columns=['fc', 'er'])
                df.loc[:, 'fc'] = wfc.loc[err_idx_nonan, wf]
                df.loc[:, 'er'] = wfc.loc[err_idx_nonan, wf] - self.wpd_lp.loc[err_idx_nonan, wf]

                if self.quant_remove_zero_periods:
                    # remove periods with zero production for more than X hours
                    prod = self.wpd.loc[err_idx_nonan,wf]
                    sidx = 0 # start of current zero period
                    remove_flag = False
                    keep_vals = pd.Series(1,index=err_idx_nonan,dtype=bool)
                    for idx in range(prod.__len__()):

                        if not remove_flag and prod.iat[idx] <= 0: # start of new zero period
                            remove_flag = True
                            sidx = idx
                        elif remove_flag and (prod.iat[idx] > 0 or idx == prod.__len__() - 1): # end of zero period
                            remove_flag = False
                            plen = idx - sidx
                            # set values to 0 if length exceeds requirement
                            if plen >= self.quant_nzero_hours_remove*12:
                                # hf_dummy[(u,s)].iloc[idx:sidx] = False
                                keep_vals.loc[err_idx_nonan[sidx]:err_idx_nonan[idx-1]] = False
                    keep_idxs = [t for t in keep_vals.index if keep_vals.at[t]]
                    exclude_idxs = [t for t in keep_vals.index if not keep_vals.at[t]]
                    df_nhours_exclude = df.loc[exclude_idxs,:]
                    df = df.loc[keep_idxs,:]

                # days_include = []
                # for didx in range(nDaysData):
                #     time_idxs_day = err_idx[didx * nT:(didx + 1) * nT]
                #     include = True
                #     if err_un.loc[time_idxs_day, :].isna().sum().sum() > 0:
                #         include = False  # exclude days with missing values
                #     if self.cov_exclude_zero_days:
                #         for u in units:
                #             # criteria for excluding day:
                #             # more than 5 periods with exactly correct wfc
                #             # if (pred_err.loc[time_idxs,u] < 1e-4).sum() > 5:
                #             # more than 10 hours with zero production
                #             if (wpd.loc[time_idxs_day, u] < 1e-2).sum() > nT / 24 * self.cov_nhours_exclude:
                #                 # if False:
                #                 include = False
                #                 break
                #     if time_idxs_day[0].strftime('%Y%m%d') in nan_days: # remove days with missing forecast data
                #         include = False
                #     if include:
                #         days_include.append(didx)


                # remove outliers
                if self.quant_remove_outliers:
                    incl_idx = (df.fc - df.er) > self.quant_outlier_tol

                    excl_idx = (df.fc > 0.3) & (df.fc - df.er < self.quant_outlier_tol)
                    df_excl = df.loc[ excl_idx,:]
                    # remove those values where  fc - er < tol
                    df = df.loc[ excl_idx == False,:]
                nobs = df.__len__()

                if self.quant_set_bins is None:
                    # create bins by quantiles
                    # split forecast into bins for splines
                    wfc_dig, bins = pd.qcut(df.fc, nbins, retbins=True, labels=False)
                else:
                    # use manually set bins
                    bins = np.array(self.quant_set_bins)
                    wfc_dig = pd.cut(df.fc,self.quant_set_bins,labels=False).astype(int)


                obs_idx_q0 = [i for i in range(nobs) if wfc_dig.iat[i] < self.quant_q0_nbins]
                obs_idx_q1 = [i for i in range(nobs) if wfc_dig.iat[i] >= nbins - self.quant_q1_nbins]

                nobs_q0 = (wfc_dig < self.quant_q0_nbins).sum()
                # nobs_q0 = obs_q0_idx.__len__()
                nobs_q1 = (wfc_dig >= nbins - self.quant_q1_nbins).sum()
                # fit all quantiles curves simultaneously

                gm = gp.Model("quant_reg")
                if not self.quant_solver_output:
                    gm.setParam('OutputFlag', 0)


                beta = gm.addVars(range(nquant_full), range(ncoeff), name='beta', lb=-np.inf, ub=np.inf)
                rp = gm.addVars(range(nquant_full), range(nobs), name='rp')
                rn = gm.addVars(range(nquant_full), range(nobs), name='rn')

                if self.quant_fit_boundary:

                    # 0th quantile: fit bins starting from 0
                    beta0 = gm.addVars(range(ncoeff_q0),name='beta0',lb=-np.inf,ub=np.inf)
                    rp0 = gm.addVars(range(nobs_q0),name='rp0')
                    rn0 = gm.addVars(range(nobs_q0),name='rn0')
                    # 1th quantile: fit bins until nbins
                    beta1 = gm.addVars(range(ncoeff_q1),name='beta1',lb=-np.inf,ub=np.inf)
                    rp1 = gm.addVars(range(nobs_q1),name='rp1')
                    rn1 = gm.addVars(range(nobs_q1),name='rn1')

                gm.update()

                # %
                obj = gp.LinExpr()
                for i, q in enumerate(quantiles_full):
                    obj.add(q * rp.sum(i, '*') + (1 - q) * rn.sum(i, '*'))

                if self.quant_fit_boundary:
                    obj.add(quant0 * rp0.sum() + (1-quant0) * rn0.sum())
                    obj.add(quant1 * rp1.sum() + (1-quant1) * rn1.sum())

                gm.setObjective(obj)

                # %
                constr_dev = {}  # constraint to force rn,rp to take deviation between model and actual value
                for qidx, q in enumerate(quantiles_full):
                    # setup constraints
                    for i in range(nobs):
                        yi = df.er.iat[i]
                        x = np.array(
                            [df.fc.iat[i] ** (spline_order - ii) for ii in
                             range(spline_order + 1)])  # powers of x-vales
                        si = wfc_dig.iat[i]  # index of spline
                        idx = si * spline_coeff  # index of first spline coefficient
                        est = gp.quicksum(beta[qidx, idx + ii] * x[ii] for ii in range(spline_coeff))
                        constr_dev[qidx, i] = gm.addLConstr(lhs=yi - est,
                                                            sense=GRB.EQUAL,
                                                            rhs=rp[qidx, i] - rn[qidx, i],
                                                            name=f'cs[{qidx},{i}]')

                if self.quant_fit_boundary:
                    constr_dev_q0 = {}
                    for i in range(nobs_q0):
                        iobs = obs_idx_q0[i] # index of observation
                        yi = df.er.iat[iobs]
                        xi = df.fc.iat[iobs]
                        x = np.array(
                            [xi ** (spline_order - ii) for ii in
                             range(spline_order + 1)])  # [x^3, x^2, x, 1]
                        si = wfc_dig.iat[iobs] # index of spline
                        # q0: first x splines are fit
                        idx = si * spline_coeff
                        est = gp.quicksum(beta0[idx + ii] * x[ii] for ii in range(spline_coeff))
                        constr_dev_q0[i] = gm.addLConstr(lhs=yi - est,
                                                         sense=GRB.EQUAL,
                                                         rhs=rp0[i] - rn0[i],
                                                         name=f'cs_q0[{i}]')
                    constr_dev_q1 = {}
                    for i in range(nobs_q1):
                        iobs = obs_idx_q1[i] # index of observation
                        yi = df.er.iat[iobs]
                        xi = df.fc.iat[iobs]
                        x = np.array(
                            [xi ** (spline_order - ii) for ii in
                             range(spline_order + 1)])  # [x^3, x^2, x, 1]
                        # Note: For q1, first x splines are excluded
                        si = wfc_dig.iat[iobs] - (nbins - self.quant_q1_nbins) # index of spline
                        # q0: first x splines are fit
                        idx = si * spline_coeff
                        est = gp.quicksum(beta1[idx + ii] * x[ii] for ii in range(spline_coeff))
                        constr_dev_q0[i] = gm.addLConstr(lhs=yi - est,
                                                         sense=GRB.EQUAL,
                                                         rhs=rp1[i] - rn1[i],
                                                         name=f'cs_q1[{i}]')

                # add constraints for continuity
                constr_diff0 = {}
                constr_diff1 = {}
                constr_diff2 = {}
                for qidx, q in enumerate(quantiles_full):
                    for i in range(nbins - 1):
                        xi = bins[i + 1]  # point at which to evaluate derivatives
                        idx1 = i * spline_coeff
                        idx2 = (i + 1) * spline_coeff
                        constr_diff0[(qidx, i + 1)] = gm.addLConstr(
                            lhs=xi ** 3 * beta[qidx, idx1] + xi ** 2 * beta[qidx, idx1 + 1] + xi * beta[
                                qidx, idx1 + 2] + beta[qidx, idx1 + 3],
                            sense=GRB.EQUAL,
                            rhs=xi ** 3 * beta[qidx, idx2] + xi ** 2 * beta[qidx, idx2 + 1] + xi * beta[
                                qidx, idx2 + 2] + beta[qidx, idx2 + 3],
                            name=f'diff0[{qidx},{i + 1}]'
                        )
                        constr_diff1[(qidx, i + 1)] = gm.addLConstr(
                            lhs=3 * xi ** 2 * beta[qidx, idx1] + 2 * xi * beta[qidx, idx1 + 1] + beta[
                                qidx, idx1 + 2],
                            sense=GRB.EQUAL,
                            rhs=3 * xi ** 2 * beta[qidx, idx2] + 2 * xi * beta[qidx, idx2 + 1] + beta[
                                qidx, idx2 + 2],
                            name=f'diff1[{qidx},{i + 1}]')
                        constr_diff2[(qidx, i + 1)] = gm.addLConstr(
                            lhs=6 * xi * beta[qidx, idx1] + 2 * beta[qidx, idx1 + 1],
                            sense=GRB.EQUAL,
                            rhs=6 * xi * beta[qidx, idx2] + 2 * beta[qidx, idx2 + 1],
                            name=f'diff2[{qidx},{i + 1}]')

                if self.quant_fit_boundary:
                    # continuity constraints for q0
                    constr_diff0_q0 = {}
                    constr_diff1_q0 = {}
                    constr_diff2_q0 = {}
                    # q0: nbin_q0 - 1 continuity constraints between different splines
                    for i in range(max(0,self.quant_q0_nbins-1)):
                        xi = bins[i + 1]  # point at which to evaluate derivatives
                        idx1 = i * spline_coeff
                        idx2 = (i + 1) * spline_coeff
                        constr_diff0_q0[i + 1] = gm.addLConstr(
                            lhs=xi ** 3 * beta0[idx1] + xi ** 2 * beta0[idx1 + 1] + xi * beta0[
                                idx1 + 2] + beta0[idx1 + 3],
                            sense=GRB.EQUAL,
                            rhs=xi ** 3 * beta0[idx2] + xi ** 2 * beta0[idx2 + 1] + xi * beta0[
                                idx2 + 2] + beta0[idx2 + 3],
                            name=f'diff0_q0[{i + 1}]'
                        )
                        constr_diff1_q0[i + 1] = gm.addLConstr(
                            lhs=3 * xi ** 2 * beta0[idx1] + 2 * xi * beta0[idx1 + 1] + beta0[
                                idx1 + 2],
                            sense=GRB.EQUAL,
                            rhs=3 * xi ** 2 * beta0[idx2] + 2 * xi * beta0[idx2 + 1] + beta0[
                                idx2 + 2],
                            name=f'diff1_q0[{i + 1}]')
                        constr_diff2_q0[i + 1] = gm.addLConstr(
                            lhs=6 * xi * beta0[idx1] + 2 * beta0[idx1 + 1],
                            sense=GRB.EQUAL,
                            rhs=6 * xi * beta0[idx2] + 2 * beta0[idx2 + 1],
                            name=f'diff2_q0[{i + 1}]')
                    # continuity constraint for last spline (0th and 1st derivative)
                    xi = bins[self.quant_q0_nbins]
                    idx = spline_coeff*(self.quant_q0_nbins-1)
                    constr_q0_d0 = gm.addLConstr(
                        lhs=xi ** 3 * beta0[idx] + xi ** 2 * beta0[idx + 1] + xi * beta0[
                            idx + 2] + beta0[idx + 3],
                        sense=GRB.EQUAL,
                        rhs=xi-1,
                        name=f'q0_d0'
                    )
                    constr_q0_d1 = gm.addLConstr(
                        lhs=3 * xi ** 2 * beta0[idx] + 2 * xi * beta0[idx + 1] + beta0[
                            idx + 2],
                        sense=GRB.EQUAL,
                        rhs=1,
                        name=f'q0_d1'
                    )

                    # continuity constraints for q1
                    constr_diff0_q1 = {}
                    constr_diff1_q1 = {}
                    constr_diff2_q1 = {}
                    # q1: nbin_q1 - 1 continuity constraints between different splines
                    for i in range(max(0,self.quant_q1_nbins-1)):
                        xi = bins[i + 1 + nbins - self.quant_q1_nbins]  # point at which to evaluate derivatives
                        idx1 = i * spline_coeff
                        idx2 = (i + 1) * spline_coeff
                        constr_diff0_q1[i + 1] = gm.addLConstr(
                            lhs=xi ** 3 * beta1[idx1] + xi ** 2 * beta1[idx1 + 1] + xi * beta1[
                                idx1 + 2] + beta1[idx1 + 3],
                            sense=GRB.EQUAL,
                            rhs=xi ** 3 * beta1[idx2] + xi ** 2 * beta1[idx2 + 1] + xi * beta1[
                                idx2 + 2] + beta1[idx2 + 3],
                            name=f'diff0_q1[{i + 1}]'
                        )
                        constr_diff1_q1[i + 1] = gm.addLConstr(
                            lhs=3 * xi ** 2 * beta1[idx1] + 2 * xi * beta1[idx1 + 1] + beta1[
                                idx1 + 2],
                            sense=GRB.EQUAL,
                            rhs=3 * xi ** 2 * beta1[idx2] + 2 * xi * beta1[idx2 + 1] + beta1[
                                idx2 + 2],
                            name=f'diff1_q1[{i + 1}]')
                        constr_diff2_q1[i + 1] = gm.addLConstr(
                            lhs=6 * xi * beta1[idx1] + 2 * beta1[idx1 + 1],
                            sense=GRB.EQUAL,
                            rhs=6 * xi * beta1[idx2] + 2 * beta1[idx2 + 1],
                            name=f'diff2_q1[{i + 1}]')
                    # continuity constraint for first spline (0th and 1st derivative)
                    xi = bins[nbins - self.quant_q1_nbins]
                    idx = 0
                    constr_q1_d0 = gm.addLConstr(
                        lhs=xi ** 3 * beta1[idx] + xi ** 2 * beta1[idx + 1] + xi * beta1[
                            idx + 2] + beta1[idx + 3],
                        sense=GRB.EQUAL,
                        rhs=xi,
                        name=f'q1_d0'
                    )
                    constr_q1_d1 = gm.addLConstr(
                        lhs=3 * xi ** 2 * beta1[idx] + 2 * xi * beta1[idx + 1] + beta1[
                            idx + 2],
                        sense=GRB.EQUAL,
                        rhs=1,
                        name=f'q1_d1'
                    )

                constr_end_0 = {}  # make sure end points are consistent, i.e. that quantile curves are in increasing order at 0,1
                constr_end_1 = {}
                # note: assumes quantilesa are in increasing order
                for qidx in range(nquant_full - 1):
                    # at 0:
                    constr_end_0[qidx] = gm.addLConstr(
                        lhs=beta[qidx, 3] + end_margin,
                        sense=GRB.LESS_EQUAL,
                        rhs=beta[qidx + 1, 3],
                        name=f'constr_end_0[{qidx}]'
                    )
                    constr_end_1[qidx] = gm.addLConstr(
                        lhs=beta.sum(qidx,
                                     [spline_coeff * (nbins - 1) + i for i in range(spline_coeff)]) + end_margin,
                        sense=GRB.LESS_EQUAL,
                        rhs=beta.sum(qidx + 1, [spline_coeff * (nbins - 1) + i for i in range(spline_coeff)]),
                        name=f'constr_end_1[{qidx}]',
                    )
                if self.quant_fit_boundary:
                    constr_q1_fc0 = gm.addLConstr(
                        lhs=beta[nquant_full-1,3],
                        sense=GRB.LESS_EQUAL,
                        rhs=0,
                        name='constr_q1_fc0'
                    )
                    constr_q1_fc1 = gm.addLConstr(
                        lhs=beta.sum(nquant_full-1,[spline_coeff * (nbins - 1) + i for i in range(spline_coeff)]) + end_margin,
                        sense=GRB.LESS_EQUAL,
                        rhs=gp.quicksum(beta1[spline_coeff*(self.quant_q1_nbins-1)+i] for i in range(spline_coeff)),
                        name='constr_q1_fc1'
                    )
                    constr_q0_fc0 = gm.addLConstr(
                        lhs=beta0[3],
                        sense=GRB.LESS_EQUAL,
                        rhs=beta[0,3],
                        name='constr_q0_fc0'
                    )
                    constr_q0_fc1 = gm.addLConstr(
                        lhs=beta.sum(0, [spline_coeff * (nbins - 1) + i for i in range(spline_coeff)]),
                        sense=GRB.GREATER_EQUAL,
                        rhs=0,
                        name='constr_q0_fc1'
                    )

                    # limit value of last spline for q1 at forecast=1
                    constr_q1_fc1_max = gm.addLConstr(
                        lhs=gp.quicksum(beta1[spline_coeff*(self.quant_q1_nbins-1)+i] for i in range(spline_coeff)),
                        sense=GRB.LESS_EQUAL,
                        rhs=self.quant_maxerr_fc1,
                        name='constr_q1_fc1_max'
                    )



                gm.update()
                res = gm.optimize()
                if self.quant_fit_boundary:
                    pass
                    splines = []
                    # add q0:
                    ispline = [quant0]
                    # add fitted splines first
                    for i in range(ncoeff_q0):
                        ispline.append(beta0[i].X)
                    # add straight line for remaining bins: error = forecast - 1
                    for ib in range(nbins - self.quant_q0_nbins):
                        for i in range(spline_coeff):
                            if i > spline_coeff-2: # constant
                                ispline.append(-1)
                            elif i > spline_coeff-3: # linear term
                                ispline.append(1)
                            else:# other terms
                                ispline.append(0)
                    splines.append(ispline)
                    # add quantile curves with full splines
                    for iq,q in enumerate(quantiles_full):
                        splines.append([q] + [beta[iq, i].X for i in range(ncoeff)])
                    # splines = [[q] + [beta[iq, i].X for i in range(ncoeff)] for iq, q in enumerate(quantiles_full)]

                    # add 1th quantile curve
                    ispline = [quant1]
                    # add straight lines: error = forecast
                    for ib in range(nbins - self.quant_q1_nbins):
                        for i in range(spline_coeff):
                            if i > spline_coeff-2: # constant
                                ispline.append(0)
                            elif i > spline_coeff-3: # linear term
                                ispline.append(1)
                            else: # other terms
                                ispline.append(0)
                    # add fitted splines
                    for i in range(ncoeff_q1):
                        ispline.append(beta1[i].X)
                    splines.append(ispline)

                else:
                    splines = [[q] + [beta[iq, i].X for i in range(ncoeff)] for iq, q in enumerate(quantiles_full)]



                # save results
                quant_fits[lead_time][wf] = splines
                quant_bins[lead_time][wf] = bins

                # plot results
                if self.quant_plots:
                    sshape = 'o'
                    # %% plot
                    fits2 = pd.DataFrame(splines, columns=['q'] + coeff)

                    # % plot
                    x = np.linspace(0,1,101)

                    fig = plt.gcf()
                    fig.clf()
                    fig.set_size_inches(14,5)
                    # spec = fig.add_gridspec(ncols=3,nrows=1,width_ratios=[4,0.1,5])
                    spec = fig.add_gridspec(ncols=2,nrows=1,width_ratios=[4,5])


                    # ax = fig.add_subplot(1,2,1)
                    ax = fig.add_subplot(spec[0])
                    # fig, ax = plt.subplots(figsize=(8, 6))

                    for i in fits2.index:
                        y, dig = eval_spline2(x, bins, fits2.loc[i, :])
                        if self.quant_labels:
                            label = f'{fits2.at[i, "q"]:0.3f}'
                            color = f'C{i + 1}'
                            linewidth = 1
                        else:
                            if i == 0:
                                label = 'quantiles'
                            else:
                                label = '_nolegend_'
                            color = 'black'
                            linewidth = 0.8
                        ax.plot(x, y, linestyle='-', color=color, label=label, linewidth = linewidth)

                    # plot division between splines
                    y = [df.er.min(), df.er.max()]
                    for xval in bins[1:bins.__len__() - 1]:
                        ax.plot([xval, xval], y, linestyle='dashed', color='black')

                    if self.eps_figures:
                        alpha = 0.3
                    else:
                        alpha = 0.3
                    ax.scatter(df.fc, df.er, alpha=alpha, label='data',marker=sshape)
                    # plot removed data
                    if self.quant_remove_outliers:
                        ax.scatter(df_excl.fc,df_excl.er, alpha=alpha, color='red', label='outliers',marker=sshape)
                    if self.quant_remove_zero_periods:
                        ax.scatter(df_nhours_exclude.fc,df_nhours_exclude.er, alpha=alpha, color='red', label='_nolabel_',marker=sshape)

                    ax.legend()
                    ax.set_xlabel('Forecast')
                    ax.set_ylabel('Error')
                    ax.grid()
                    plt.ylim([-1,1])
                    plt.xlim([0,1])
                    plt.tight_layout()

                    # ax = fig.add_subplot(1,2,2)
                    ax = fig.add_subplot(spec[1])

                    pdf, quant = get_pdf_2d(bins, fits2, cut_extreme=True)

                    fits_bounds = fits2.iloc[[0,fits2.__len__()-1],:]
                    fits_bounds.index = range(fits_bounds.__len__())
                    xvals = np.linspace(0,1,500)
                    boundary,s = eval_splines_2d(x=xvals,bins=bins,fit=fits_bounds,cut_extreme=True)

                    pdf_plt = pdf.copy()
                    pdf_plt[pdf > pdf_max] = pdf_max
                    plt.imshow(pdf_plt, extent=[0, 1, -1, 1], aspect='auto', cmap='YlGnBu')
                    plt.colorbar()
                    plt.xlabel('Forecast')
                    plt.ylabel('Error')

                    ax.plot(xvals,boundary,color='black',linestyle='dashed')

                    plt.tight_layout()
                    if self.plot_titles:
                        plt.suptitle(f'{wf} lead time {lead_time}')
                    plt.savefig(self.path / f'quantiles_L{lead_time}_{wf}.png')
                    if self.eps_figures:
                        plt.savefig(self.path / f'quantiles_L{lead_time}_{wf}.eps')



                    #%% plot distribution of data

                    # %%
                    un = np.random.uniform(size=(1, df.__len__()))[0]

                    er_random = uniform2cdf(np.array(df.fc), un, bins=bins, fit=fits2,cut_extreme=True)
                    un_data = cdf2uniform(np.array(df.fc), np.array(df.er), bins=bins, fit=fits2)

                    # %% plot scatter data
                    # f = plt.figure()
                    f = plt.gcf()
                    f.clf()
                    gs = f.add_gridspec(2, 2)
                    f.set_size_inches(12, 10)
                    f.add_subplot(gs[0, 0])
                    plt.scatter(df.fc, df.er, alpha=alpha, marker=sshape)
                    plt.xlim([0, 1])
                    plt.ylim([-1, 1])
                    plt.grid()
                    plt.title('Data')
                    plt.xlabel('Forecast')
                    plt.ylabel('Error')

                    f.add_subplot(gs[0, 1])
                    plt.scatter(df.fc, er_random, alpha=alpha, marker=sshape)
                    plt.xlim([0, 1])
                    plt.ylim([-1, 1])
                    plt.grid()
                    plt.title('Random samples')
                    plt.xlabel('Forecast')
                    plt.ylabel('Error')

                    # % plot actual data transformed to uniform distribution
                    ax = f.add_subplot(gs[1, :])

                    ax.hist(un_data, bins=10, edgecolor='black', linewidth=1.2)
                    plt.title('Distribution of F(x) for data')
                    plt.suptitle(f'{wf} lead time {lead_time}')
                    plt.savefig(self.path / f'distribution_L{lead_time}_{wf}.png')

                    f = plt.gcf()
                    f.clf()
                    f.set_size_inches(5,3)
                    ax = f.add_subplot(1,1,1)
                    ax.hist(un_data, bins=20, edgecolor='black', linewidth=1, density=True)
                    if self.plot_titles:
                        plt.title('Distribution of F(x)')
                    # plt.suptitle(f'{wf} lead time {lead_time}')
                    plt.grid()
                    plt.tight_layout()
                    plt.savefig(self.path / f'distribution_bar_L{lead_time}_{wf}.png')
                    if self.eps_figures:
                        plt.savefig(self.path / f'distribution_bar_L{lead_time}_{wf}.eps')


        self.quant_fits = quant_fits
        self.quant_bins = quant_bins
        self.quant_coeff = coeff

    def make_5min_forecast(self):

        self.wfc_5min = make_5min_forecast(self.wfc,self.lp_cutoff,self.fs,self.lp_order)

    def fit_noise(self):
        path = self.path / 'noise_bins'
        path.mkdir(exist_ok=True)

        """ Fit high-frequency noise """
        fs = self.fs
        cutoff = self.lp_cutoff
        forder = self.lp_order
        wpd = self.wpd

        nbins = self.hf_nbins
        lead_time = self.hf_lead_time
        wfc = self.wfc.loc[:, (slice(None), lead_time)]
        wfc.columns = wfc.columns.get_level_values(0)

        nT = 24 * fs
        nT30 = 24 * 2

        # %%

        fc_avg = pd.DataFrame(index=[wfc.index[i * nT30] for i in range(self.nDaysData)], columns=self.units, dtype=float)
        pd_avg = pd.DataFrame(index=fc_avg.index, columns=self.units, dtype=float)
        exclude_days_nan = []
        for i in range(self.nDaysData):
            fc_avg.iloc[i, :] = wfc.iloc[range(i * nT30, (i + 1) * nT30), :].mean()
            timerange = pd.date_range(start=fc_avg.index[i],
                                      end=fc_avg.index[i].to_pydatetime() + datetime.timedelta(
                                          seconds=3600 * 24 - 5 * 60),
                                      freq='5min')
            pd_avg.iloc[i, :] = wpd.loc[timerange, :].mean()

            if wpd.loc[timerange,:].isna().sum().sum() > 0:
                exclude_days_nan.append(timerange[0])

        print(exclude_days_nan)
        model = {}
        # %%
        for unit in self.units:

            model[unit] = {}
            # divide days by average production

            # exclude days with constant (0) production
            exclude_days = [t for t in fc_avg.index if pd_avg.at[t, unit] == 0]
            include_days = [t for t in fc_avg.index if t not in exclude_days and t not in exclude_days_nan]
            fc_avg_unit = fc_avg.loc[include_days, unit]
            pd_avg_unit = pd_avg.loc[include_days, unit]

            if self.hf_binvar == 'fc':
                wind_avg_dig, bins = pd.qcut(fc_avg_unit, nbins, retbins=True, labels=False)
            else:
                wind_avg_dig, bins = pd.qcut(pd_avg_unit, nbins, retbins=True, labels=False)
            model[unit]['bins'] = bins

            for ibin in range(nbins):
                # ibin = 0

                ndays = (wind_avg_dig == ibin).sum()
                days = [i for i in wind_avg_dig.index if wind_avg_dig.at[i] == ibin]
                # % create time series

                pd_t = np.zeros(shape=(ndays, nT))
                pd_lp_t = np.zeros(shape=(ndays, nT))
                pd_f = np.zeros(shape=(ndays, nT), dtype=complex)
                pd_fs = np.zeros(shape=(ndays, nT), dtype=complex)
                pd_lp_fs = np.zeros(shape=(ndays, nT), dtype=complex)

                for i, t_day in enumerate(days):
                    pd_t[i] = np.array(wpd.loc[pd.date_range(start=t_day,
                                                             end=t_day.to_pydatetime() + datetime.timedelta(
                                                                 seconds=3600 * 24 - 5 * 60),
                                                             freq='5min'), unit])
                    pd_lp_t[i] = butter_lowpass_filter(pd_t[i], cutoff, fs, forder)
                    pd_f[i] = fft(pd_t[i])
                    pd_fs[i] = fftshift(pd_f[i])
                    pd_lp_fs[i] = fftshift(fft(pd_lp_t[i]))

                # %
                freq = fftshift(fftfreq(nT, 1 / fs))

                cutoff_idx = 0
                for i in range(nT):
                    if freq[i] <= self.hf_fit_cutoff:
                        cutoff_idx += 1
                nvals = nT - cutoff_idx  # number of values to keep from spectrum

                from scipy.optimize import curve_fit

                yvals = np.abs(np.reshape(pd_fs[:, cutoff_idx:], (nvals * ndays)))
                xvals = np.tile(freq[cutoff_idx:], ndays)

                ylog = np.log(yvals)
                xlog = np.log(xvals)

                # use polyfit for initial values, but slope is off so it can't be used for final fit
                fit = np.polyfit(xlog, ylog, deg=1)

                func = lambda x, p0, p1: x * p0 + p1
                res = curve_fit(func, xlog, ylog, p0=fit, method=None)
                # freq_linear_fit = res[0]
                # %%
                f = plt.gcf()
                f.set_size_inches(12,7)
                # f = plt.figure()
                plt.clf()
                # plt.rcParams['figure.constrained_layout.use'] = True
                gs = f.add_gridspec(2, 1)

                # original and filtered production, FFT
                ax1 = f.add_subplot(gs[0, 0])

                # f = plt.figure()
                # f.add_subplot(2, 1, 1)
                xvals_plot = np.tile(freq[nT // 2 + 1:], ndays)
                yvals_plot = np.abs(np.reshape(pd_fs[:, nT // 2 + 1:], ((nT // 2 - 1) * ndays)))
                if self.eps_figures:
                    alpha = 1
                else:
                    alpha = 0.5
                plt.plot(np.log(xvals_plot), np.log(yvals_plot), label='data', color='C0', alpha=alpha, marker='o',
                         linestyle='none')
                # plt.scatter(np.log(freq[nT // 2 + 1:]), np.log(np.abs(pd_fs[N // 2 + 1:])), label='data', color='C0')
                xplot = np.linspace(np.min(xlog), np.max(xlog), 100)
                plt.plot(xplot, np.polyval(fit, xplot), color='C1', marker=None, label='curve fit')
                # plt.plot(xplot, np.polyval(res[0], xplot), color='C2', marker=None, label='curve fit')
                plt.grid()
                plt.legend()
                plt.xlabel('Freq [log(1/Hour)]')
                plt.tight_layout()
                # %

                spec_scale = np.array([spectrum_func(f, fit) for f in freq])

                # plot spectrum (linear space)
                # f.add_subplot(2, 1, 2)
                ax2 = f.add_subplot(gs[1, 0])
                plt.semilogy(np.tile(freq, ndays), np.abs(np.reshape(pd_fs, newshape=(ndays * nT))),
                             color='C0', linestyle='none', marker='o', label='data', alpha=alpha)
                plt.semilogy(freq, spec_scale, color='C1', label='curve fit')
                plt.grid()
                plt.legend()
                plt.xlabel('Freq [1/Hour]')
                plt.tight_layout()
                plt.savefig(path / f'spectrum_scale_{unit}_b{ibin}.png')

                # %%
                pd_norm_fs = pd_fs / spec_scale

                # compute std of real and imaginary part for high frequencies
                y_f_real_hf = np.real(pd_norm_fs[:, cutoff_idx:])
                noise_std_real = np.std(y_f_real_hf) * self.hf_std_scale
                noise_avg_real = np.average(y_f_real_hf)

                y_f_imag_hf = np.imag(pd_norm_fs[:, cutoff_idx:])
                noise_std_imag = np.std(y_f_imag_hf) * self.hf_std_scale
                noise_avg_imag = np.average(y_f_imag_hf)

                # %% save model
                model[unit][ibin] = {
                    'spectrum': spec_scale,
                    # 'freq_linear_fit': freq_linear_fit,
                    'std_real': noise_std_real,
                    'std_imag': noise_std_imag,
                    'avg_real': noise_avg_real,
                    'avg_imag': noise_avg_imag,
                }

                # %
                # %%
                if self.hf_plot_figures:
                    # %%

                    plot_day_bin_idx = 0  # index of day to plot (among days in current bin)

                    start = days[plot_day_bin_idx]
                    end = start.to_pydatetime() + datetime.timedelta(seconds=3600 * 24 - 60 * 5)
                    timerange = pd.date_range(start=start, end=end, freq='5min')
                    nspectra = 4

                    # white noise signal
                    wn_t = np.random.normal(0, 1, size=nT)
                    wn_f = fft(wn_t)
                    wn_fs = fftshift(wn_f)
                    # scale white noise to give same std
                    if self.hf_add_noise_avg:
                        wn_std_fs = np.real(wn_fs) * noise_std_real / np.std(
                            np.real(wn_fs)) + noise_avg_real - np.average(np.real(wn_fs)) \
                                    + 1j * (np.imag(wn_fs) * noise_std_imag / np.std(
                            np.imag(wn_fs)) + noise_avg_imag - np.average(np.imag(wn_fs)))
                    else:
                        wn_std_fs = np.real(wn_fs) * noise_std_real / np.std(np.real(wn_fs)) \
                                    + 1j * np.imag(wn_fs) * noise_std_imag / np.std(np.imag(wn_fs))

                    wn_scale_fs = wn_std_fs * spec_scale

                    # %%
                    # create frequency domain version of filtered noise, to use in plot
                    wn_scale_t = ifft(ifftshift(wn_scale_fs))
                    imag_max = np.max(np.abs(np.imag(wn_scale_t)))
                    if imag_max > 1e-10:
                        print(f'Maximum imaginary part of wn_scaled: {imag_max:0.2e}')
                    wn_scale_t = np.real(wn_scale_t)

                    # low-pass filter noise and keep high-frequency part
                    wn_low_t = butter_lowpass_filter(wn_scale_t, cutoff=cutoff, fs=fs, order=forder)
                    wn_high_t = wn_scale_t - wn_low_t

                    # fft of final noise
                    wn_high_f = fftshift(fft(wn_high_t))

                    # % ############ Plot spectra and time series ####################

                    # f = plt.figure(constrained_layout=True)
                    f = plt.gcf()
                    f.clf()
                    # plt.rcParams['figure.constrained_layout.use'] = True
                    # f = plt.figure(constrained_layout=True)
                    f.set_size_inches(25 / cm_per_inch, 14 / cm_per_inch)
                    # f.clf()
                    # %%
                    gs = f.add_gridspec(2, 2)
                    freq_label = 'Freq (1/Hour)'

                    ## F1 - SPECTRA OF FEW DAYS

                    # original and filtered production, FFT
                    ax1 = f.add_subplot(gs[0, 0])

                    # pd_lp_f = fft(pd_lp_t)
                    pd_t_plot = np.array(wpd.loc[timerange, unit])
                    pd_t_lp_plot = butter_lowpass_filter(pd_t_plot, cutoff, fs, forder)

                    pd_fs_plot = fftshift(fft(pd_t_plot))
                    pd_fs_lp_plot = fftshift(fft(pd_t_lp_plot))

                    freq_plot = np.tile(freq[nT // 2 + 1:], ndays)
                    pd_f_plot = np.abs(np.reshape(pd_fs[:, nT // 2 + 1:], ((nT // 2 - 1) * ndays)))
                    pd_f_lp_plot = np.abs(np.reshape(pd_lp_fs[:, nT // 2 + 1:], ((nT // 2 - 1) * ndays)))

                    # plt.loglog(freq_plot, 2.0 / np.power(nT, 0.5) * pd_f_plot, label='Production 5-min',
                    #            color='black', alpha=0.4,marker='.',linestyle='none')
                    # plt.loglog(freq_plot, 2.0 / np.power(nT, 0.5) * pd_f_lp_plot, label='Production LP-filtered',
                    #            color='black', alpha=1, linewidth=0.8,marker='*',linestyle='none')

                    for i in range(nspectra):
                        if i == 0:
                            label1 = 'Production 5-min'
                            label2 = 'Production LP-filtered'
                        else:
                            # label1 = '_nolegend_'
                            label1 = ' '
                            label2 = '_nolegend_'
                        label1 = f'Day {i+1}'
                        ax1.loglog(freq[nT//2+1:], 2.0 / np.power(nT, 0.5) * np.abs(pd_fs[i,nT//2+1:]), label=label1,
                                   color=f'C{i}', alpha=1, marker=None, linestyle='solid')
                        # plt.loglog(freq[nT//2+1:], 2.0 / np.power(nT, 0.5) * np.abs(pd_lp_fs[i,nT//2+1:]), label=label2,
                        #            color='black', alpha=1, linewidth=0.8, marker='none', linestyle='solid')
                    #%%
                    # plt.loglog(freq[nT // 2 + 1:], 2.0 / np.power(nT, 0.5) * np.abs(pd_fs_plot[nT // 2 + 1:]),
                    #            label='Production 5-min',
                    #            color='black', alpha=0.4, marker='none', linestyle='solid')
                    # plt.loglog(freq[nT // 2 + 1:], 2.0 / np.power(nT, 0.5) * np.abs(pd_fs_lp_plot[nT // 2 + 1:]),
                    #            label='Production LP-filtered',
                    #            color='black', alpha=1, linewidth=0.8, marker='none', linestyle='solid')

                    ax1.legend()
                    ax1.grid()
                    # plt.xlim(xlims)
                    ax1.set_xlabel(freq_label)
                    if self.plot_titles:
                        ax1.set_title('5-min wind data FFT')
                    # plt.tight_layout()


                    ## F2: White noise, scaling factor, and final noise
                    ax2 = f.add_subplot(gs[0, 1], sharey = ax1)
                    # ax2 = f.add_subplot(gs[0, 1])
                    if self.eps_figures:
                        alpha = 1
                    else:
                        alpha = 0.8
                    plt.loglog(freq[nT // 2 + 1:], 2.0 / np.power(nT, 0.5) * np.abs(wn_std_fs[nT // 2 + 1:]),
                               label='White noise', color='black',
                               alpha=1,linewidth=0.8)
                    plt.loglog(freq[nT // 2 + 1:], 2.0 / np.power(nT, 0.5) * np.abs(wn_high_f[nT // 2 + 1:]),
                               label='Scaled and filtered noise',
                               color='C1', linewidth=1,
                               alpha=alpha)
                    plt.loglog(freq[nT // 2 + 1:], 2.0 / np.power(nT, 0.5) * spec_scale[nT // 2 + 1:],
                               label='Scale factor', color='C0',
                               linestyle='dashed', linewidth=2, alpha=alpha, marker=None)
                    plt.legend()
                    plt.grid()
                    # plt.xlim(xlims)
                    plt.xlabel(freq_label)
                    if self.plot_titles:
                        ax2.set_title('Noise FFT')
                    plt.tight_layout()

                    ## F3: time series: original production and samples
                    ax3 = f.add_subplot(gs[1, :])

                    if self.eps_figures:
                        alpha = 1
                    else:
                        alpha = 0.8

                    for i in range(self.hf_plot_nscen):
                        if i == 0:
                            label = 'scenarios'
                        else:
                            label = '_nolegend_'
                        noise_hp = self.create_noise(spectrum=spec_scale, std_real=noise_std_real, std_imag=noise_std_imag,
                                                     nsamp=nT, nscen=1)[0]

                        prod = self.wpd_lp[unit].loc[timerange] + noise_hp
                        prod[prod < 0] = 0
                        prod.plot(ax=ax3, label=label, color='C1', linewidth=0.9, alpha = alpha)

                    # plot original data
                    wpd[unit].loc[timerange].plot(ax=ax3, label='Production', linewidth=1,
                                                  color='black',alpha=alpha)
                    self.wpd_lp[unit].loc[timerange].plot(ax=ax3, label='LP filtered production',
                                                          color='C0', linestyle='dashed', linewidth=3.5,
                                                          alpha=alpha)

                    plt.legend()
                    plt.grid()
                    if self.plot_titles:
                        ax3.set_title('Wind power scenarios')

                    # f.suptitle(f'{unit}')
                    plt.tight_layout()
                    plt.savefig(path / f'hf_{unit}_bin{ibin}.png')
                    if self.eps_figures:
                        plt.savefig(path / f'hf_{unit}_bin{ibin}.eps')

            # %% plot spectra for all bins
            f = plt.gcf()
            f.clf()
            f.set_size_inches(6,5)
            # f = plt.figure()
            for i in range(nbins):
                plt.loglog(freq[nT // 2 + 1:], model[unit][i]['spectrum'][nT // 2 + 1:], label=f'bin{i}')
            plt.grid()
            plt.legend()
            plt.xlabel('Freq (1/hour)')
            plt.title(f'Noise scaling, {unit}')
            plt.tight_layout()
            plt.savefig(path / f'noise_spectra_{unit}')

        self.noise = model

    def create_noise(self,nsamp,nscen,spectrum,std_real,std_imag):

        # def get_scaled_filtered_noise(m, n=24 * 12):
        """ Create noise for filling high-frequency part of wind production spectrum
        n - number of samples wanted
        m - dict containing:
            spectrum - symmetric real spectrum used for scaling noise. noise intensity decreases (linearly?) with frequency
            std_real - stdev of real part of white noise in frequency domain
            std_imag - stdev of imaginary part of noise in frequency domain
            fs - sampling frequency
            cutoff - cutoff frequency for lowpass filter
        """
        # spectrum = m['spectrum']
        # std_real = m['std_real']
        # std_imag = m['std_imag']
        # fs = m['fs']
        # cutoff = m['cutoff']
        # forder = m['forder']

        fs = self.fs
        cutoff = self.lp_cutoff
        forder = self.lp_order

        N = spectrum.__len__()  # number of samples in spectrum

        # generate noise in time domain, scale to get correct stdev
        wn = np.random.normal(0, 1, size=(nscen,nsamp))
        wn_hp = np.zeros(wn.shape)
        for i in range(nscen):
            wn_fft = fftshift(fft(wn[i]))
            wn_fft = np.real(wn_fft) * std_real / np.std(np.real(wn_fft)) \
                     + 1j * np.imag(wn_fft) * std_imag / np.std(np.imag(wn_fft))

            wn_freq = fftshift(fftfreq(nsamp, d=1 / fs))
            spectrum_freq = fftshift(fftfreq(N, d=1 / fs))

            # interpolate spectrum to new sample number
            wn_scale_factor = np.interp(wn_freq, spectrum_freq, spectrum) * np.power(nsamp / N, 0.5)
            # scale wn spectrum and go back to time domain
            # Note: not sure why fft scales with sqrt of window length
            wn_scaled = ifft(ifftshift(wn_fft * wn_scale_factor))

            # check that noise is real
            max_imag = np.max(np.abs(np.imag(wn_scaled)))
            if max_imag > 1e-10:
                print(f'Scaled wn is non-real for nsamp={nsamp}, imaginary part: {max_imag:0.2e}')
            wn_scaled = np.real(wn_scaled)

            # filter noise, return high-frequency component
            wn_hp[i] = wn_scaled - butter_lowpass_filter(wn_scaled, cutoff=cutoff, fs=fs, order=forder)

        return wn_hp

    def generate_scenarios(self,nscen = 10,date = '20190801',seed=None,use_hf_model = True,scen_base='forecast',lead_time = 1,cut_extreme=True,qrange=(0.05,0.95),
                                    return_info=False):
        """ Generate scenarios from model
        Using quantile fit

        scen_base: The synthetic errors are added to the scenario base variable to create the scenarios
            forecast - use forecast as base for scenarios
            production - use production data as base for scenarios

        Return:
            scenarios [range(1,nscen+1)]
            data: [forecast,production]
        """
        info = {}
        starttime = f'{date}:{self.start_hour}'

        # %% get low-frequency scenarios
        cov = self.cor_quant[f'l{lead_time}']

        units = self.units
        if self.cov_resolution == '30min':
            nT = 24 * 2
            tstep = 6
        else:
            nT = 24 * 12
            tstep = 1

        # create time axis
        nVars = cov.shape[0]
        timerange = pd.date_range(start=str_to_date(starttime),
                                  end=str_to_date(starttime) + datetime.timedelta(hours=24),
                                  freq='5min')
        endtime = timerange[-1].strftime('%Y%m%d:%H%M')


        cols = pd.MultiIndex.from_product([self.units,['fc','pd','pd_lp','max','min']],names=['unit','type'])
        data = pd.DataFrame(index=timerange,columns=cols,dtype=float)

        wpd = self.wpd_db.select_data(table_type='dispatch', categories=units, starttime=starttime,
                                      endtime=endtime) / self.capacity
        wfc = self.wfc_db.select_forecast_data_full(startdate=date,
                                                    enddate=(str_to_date(date) + datetime.timedelta(hours=24)).strftime(
                                                        '%Y%m%d'),
                                                    lead_times=[lead_time],
                                                    categories=units)
        wfc.columns = wfc.columns.get_level_values(0)
        wfc = wfc.iloc[0:wfc.__len__() // 2 + 1, :] / self.capacity
        for wf in m.units:
            data.loc[:,(wf,'pd')] = wpd[wf]
            data.loc[:,(wf,'pd_lp')] = butter_lowpass_filter(wpd[wf],self.lp_cutoff,self.fs,self.lp_order)
            # data.loc[:,(wf,'fc')] = wfc[wf]
            # data.loc[:,(wf,'fc_lp')] = butter_lowpass_filter(wfc[wf],self.lp_cutoff,self.fs,self.lp_order)
        data.loc[:,(slice(None),'fc')] = np.array(make_5min_forecast(wfc, self.lp_cutoff, self.fs, self.lp_order))

        # Note: Should not cut off distribution when generating quantile curves, otherwise it will never be possible
        # to get the curves to go to min/max production
        for wf in m.units:
            fit = pd.DataFrame(m.quant_fits[lead_time][wf], columns=['q'] + m.quant_coeff)
            bins = m.quant_bins[lead_time][wf]
            data.loc[:, (wf, 'max')] = - uniform2cdf(fc=data.loc[:, (wf, 'fc')],
                                                     un=qrange[0] * np.ones(timerange.__len__()),
                                                     bins=bins, fit=fit, cut_extreme=False) + data.loc[:, (wf, 'fc')]
            data.loc[:, (wf, 'min')] = - uniform2cdf(fc=data.loc[:, (wf, 'fc')],
                                                     un=qrange[1] * np.ones(timerange.__len__()),
                                                     bins=bins, fit=fit, cut_extreme=True) + data.loc[:, (wf, 'fc')]

            data.loc[data[(wf, 'min')] < 0, (wf, 'min')] = 0
            data.loc[data[(wf, 'max')] > 1, (wf, 'max')] = 1
        # if self.scen_base == 'production':  # use actual production instead of forecast
        #     # get actual production
        #     wind_data = self.wpd_db.select_data(table_type='dispatch', categories=units, starttime=starttime,
        #                                      endtime=endtime) / self.capacity
        #
        #     # filter out high-frequency component from production data
        #     wind_data_filt = pd.DataFrame(dtype=float, index=wind_data.index, columns=wind_data.columns)
        #     if use_hf_model:
        #         for u in units:
        #             wind_data_filt[u] = butter_lowpass_filter(wind_data[u], self.lp_cutoff, self.fs, self.lp_order)
        # else:  # use forecast as base for constructing scenarios
        #     wfc = self.wfc_db.select_forecast_data_full(startdate=date,
        #                                                 enddate=(str_to_date(date)+datetime.timedelta(hours=24)).strftime('%Y%m%d'),
        #                                                 lead_times=[lead_time],
        #                                                 categories=self.units)
        #     wfc.columns = wfc.columns.get_level_values(0)
        #     wfc = wfc.iloc[0:wfc.__len__() // 2 + 1, :] / m.capacity
        #     wind_data = make_5min_forecast(wfc, self.lp_cutoff, self.fs, self.lp_order)
        #     wind_data_filt = wind_data

        # chose data for scenario 0, either filtered or unfiltered, production or forecast
        wpd_sc0 = pd.DataFrame(dtype=float, index=timerange, columns=self.units)
        for u in units:
            if scen_base == 'production':
                if use_hf_model:
                    wpd_sc0[u] = data.loc[:,(u,'pd_lp')]
                else:
                    wpd_sc0[u] = data.loc[:,(u,'pd')]
            else:
                wpd_sc0[u] = data.loc[:,(u,'fc')]

        multi_cols = pd.MultiIndex.from_product(
            [units, range(1,nscen + 1)], names=['unit', 'scenario']
        )
        rls = pd.DataFrame(index=timerange, columns=multi_cols, dtype=float)

        for u in units:
            # enter production as scenario 0
            # rls.loc[:, (u, 0)] = wind_data[u]
            # rls.loc[:, (u, 0)] = wpd_sc0[u]
            # enter enter initial value into all scenarios
            # rls.loc[timerange[0], (u, slice(None))] = rls.at[timerange[0], (u, 0)]
            rls.loc[timerange[0], (u, slice(None))] = wpd_sc0.at[timerange[0],u]

        # %% generate high-frequency component of scenarios
        if not seed is None:
            np.random.seed(seed=seed)

        hf_rls = pd.DataFrame(0.0, index=timerange, columns=multi_cols)
        if use_hf_model:
            info['hf_bin'] = {}
            for u in units:
                # check if noise has bin fitted using regimes:
                if 0 in self.noise[u]:
                    # use regimes
                    pass
                    bins = self.noise[u]['bins']
                    if self.hf_binvar == 'fc':
                        avg_prod = data.loc[:,(u,'fc')].mean()
                    else:
                        avg_prod = data.loc[:,(u,'pd')].mean()
                    ibin = 0
                    while ibin < self.hf_nbins - 1 and avg_prod > bins[ibin+1]:
                        ibin += 1
                    info['hf_bin'][u] = ibin
                    arr = self.create_noise(
                        nsamp=24 * 12,
                        nscen=nscen,
                        spectrum=self.noise[u][ibin]['spectrum'],
                        std_real=self.noise[u][ibin]['std_real'],
                        std_imag=self.noise[u][ibin]['std_imag'])
                    # print(f'Using hf bin {ibin} for {u}')

                else:
                    arr = self.create_noise(
                        nsamp=24 * 12,
                        nscen=nscen,
                        spectrum=self.noise[u]['spectrum'],
                        std_real=self.noise[u]['std_real'],
                        std_imag=self.noise[u]['std_imag'])
                for sidx in range(1, arr.__len__() + 1):
                    hf_rls.loc[timerange[1:], (u, sidx)] = arr[sidx - 1]

        # %% generate low-frequency component of scenarios
        rands = np.random.multivariate_normal(mean=np.zeros(nVars), cov=cov, size=nscen)

        # % transform to uniform distribution
        from scipy.stats import norm
        uni = norm.cdf(rands)

        # f = plt.figure()
        # plt.hist(uni[0],bins=10)
        # % transform to error

        err = np.zeros(shape=rands.shape, dtype=float)
        tidxs = [timerange[i] for i in range(tstep, timerange.__len__(), tstep)]

        for ii, wf in enumerate(units):
            bins = self.quant_bins[lead_time][wf]
            fit = pd.DataFrame(self.quant_fits[lead_time][wf], columns=['q'] + self.quant_coeff)
            # Note: Take actual production as forecast, assuming zero mean error
            fc = wpd_sc0.loc[tidxs, wf]
            for i in range(rands.shape[0]):
                err[i, ii * nT:(ii + 1) * nT] = uniform2cdf(fc=fc, un=uni[i, ii * nT:(ii + 1) * nT], bins=bins, fit=fit, cut_extreme=cut_extreme)

        # %% plot errors generated for all wf
        # f = plt.figure()
        # for ii,wf in enumerate(units):
        #
        #     ax = f.add_subplot(2,2,ii+1)
        #
        #     plt.plot(np.transpose(err[:,ii*nT:(ii+1)*nT]))
        #     plt.grid()
        #     plt.title(f'{wf}')

        # %% dataframe with erros, only reshaping structure of data
        # Note that when doing quantile fit error = forecast - production => prod = forecast - error,
        # hence add error with negative sign
        columns = pd.MultiIndex.from_product([units, range(1, nscen + 1)], names=['unit', 'scen'])
        errs = pd.DataFrame(dtype=float, index=timerange, columns=columns)
        if self.cov_resolution == '30min':  # then interpolate errors between 30-min intervals
            for sidx in range(nscen):
                for vidx, er in enumerate(err[sidx]):
                    tidx = int(np.remainder(vidx, nT))
                    u = units[np.floor_divide(vidx, nT)]
                    errs.at[timerange[tidx * 6 + 1], (u, sidx + 1)] = - er
        else:  # rands with 5-min resolution
            for sidx in range(nscen):
                for uidx, u in enumerate(units):
                    errs.loc[errs.index[1:], (u, sidx + 1)] = - err[sidx][uidx * (nVars // 3):(uidx + 1) * (nVars // 3)]

        if self.cov_resolution == '30min':
            errs_full = errs.interpolate(method='linear')
        else:
            errs_full = errs
        # %%

        # nzero_req = 4 # all periods with more than this number of zero values will not have high-frequency noise added
        scenarios_lp = pd.DataFrame(0.0,index=timerange,columns=errs_full.columns)
        hf_dummy = pd.DataFrame(1,index=timerange,columns=errs_full.columns,dtype=bool)
        for u in units:
            for s in range(1,nscen+1):
                scen = wpd_sc0.loc[:,u] + errs_full.loc[:,(u,s)]
                scenarios_lp.loc[:,(u,s)] = scen

                # check which periods should have noise added
                sidx = 0 # start of current zero period
                zero_flag = False
                for idx in range(scenarios_lp.__len__()):

                    if not zero_flag and scen.iat[idx] <= 0: # start of new zero period
                        zero_flag = True
                        sidx = idx
                    elif zero_flag and (scen.iat[idx] > 0 or idx == scen.__len__() - 1): # end of zero period
                        zero_flag = False
                        plen = idx - sidx
                        # set values to 0 if length exceedds requirement
                        if plen >= self.hf_nzero_nonoise:
                            # hf_dummy[(u,s)].iloc[idx:sidx] = False
                            hf_dummy.loc[timerange[sidx]:timerange[idx-1],(u,s)] = False

        # get production from errors
        for u in units:
            for i in errs_full.index[1:]:
                for s in range(1, nscen + 1):
                    if self.hf_noise_zeroprod or hf_dummy.at[i,(u,s)]: # add hf noise
                        realz = np.min(
                            [1, np.max([0, wpd_sc0.at[i, u] \
                                        + errs_full.at[i, (u, s)] + self.hf_scale * hf_rls.at[i, (u, s)]])])
                    else: # don't add hf noise
                        realz = np.min([1, np.max([0, wpd_sc0.at[i, u] + errs_full.at[i, (u, s)] ])])
                    rls.at[i, (u, s)] = realz

        if return_info:
            return rls,data,info
        else:
            return rls,data

    def plot_production(self,unit='MACARTH1',date='20190810'):

        # unit = 'MACARTH1'

        # starttime = '20190810:00'
        # endtime = '20190802:00'
        timerange = pd.date_range(start=str_to_date(date),
                                  end=str_to_date(date) + datetime.timedelta(seconds=24 * 3600 - 5 * 60),
                                  freq='5min')

        wpd = m.wpd.loc[timerange, unit] * m.capacity[unit]

        wpd_avg = pd.Series(0.0, index=wpd.index)

        for t in wpd_avg.index:
            wpd_avg.at[t] = wpd.loc[[i for i in wpd.index if i.hour == t.hour]].mean()

        # %%
        f, ax = plt.subplots()
        f.set_size_inches(6, 4)
        wpd.plot(ax=ax, label='5 min')
        if self.eps_figures:
            alpha = 1
        else:
            alpha = 0.8
        wpd_avg.plot(ax=ax, linestyle='dashed', linewidth=3, alpha=alpha, label='hourly')
        plt.grid()
        plt.ylabel('MW')
        plt.legend()
        plt.savefig(m.path / f'production_{unit}.png')
        plt.savefig(m.path / f'production_{unit}.eps')

    def validate(self,startdate='20190811',ndays=100,use_hf_model=True,tag='default',seed=None,print_diff=False):
        # validate_scenarios(startdate='20190811',fig_path=Path('D:Data/AEMO/Figures/'),ndays=100):
        print('VALIDATE SCENARIOS')
        # fig_path = Path('D:Data/AEMO/Figures/')
        # date = '20190811'
        date=startdate
        # ndays = 10
        nunits = self.units.__len__()
        N = 12*24


        unit = self.units[0]

        # initialize arrays for data storage
        d = {}
        for unit in self.units:
            d[unit] = {}
            d[unit]['pd'] = np.zeros(shape=(ndays,N))
            d[unit]['sc'] = np.zeros(shape=(ndays,N))
            d[unit]['pd_er'] = np.zeros(shape=(ndays,N))
            d[unit]['sc_er'] = np.zeros(shape=(ndays,N))
        # wpd = np.zeros(shape=(ndays,N))
        # wfc = np.zeros(shape=(ndays,N))
        exclude_days = []
        for i in range(ndays):
            if np.remainder(i,5) == 0:
                print(f'Scenario {i}')
            if i == 0: # only use seed for first scenario
                rls,data = self.generate_scenarios_quantfit(nscen=1,cut_extreme=True,date=date,use_hf_model=use_hf_model,seed=seed)
            else:
                rls,data = self.generate_scenarios_quantfit(nscen=1,cut_extreme=True,date=date,use_hf_model=use_hf_model)

            if data.isna().sum().sum() > 0:
                # missing data, probably missing forecast -> exclude this day from comparison
                exclude_days.append(i)
            else:
                for unit in self.units:
                    wpd = data[(unit,'pd')].iloc[1:]
                    wfc = data[(unit,'fc')].iloc[1:]
                    wsc = rls[(unit,1)].iloc[1:]

                    d[unit]['pd'][i] = wpd
                    d[unit]['sc'][i] = wsc
                    d[unit]['pd_er'][i] = wfc-wpd
                    d[unit]['sc_er'][i] = wfc-wsc

            date = (str_to_date(date) + datetime.timedelta(hours=24)).strftime('%Y%m%d')

        ndays_data = ndays - exclude_days.__len__()
        idays = [i for i in range(ndays) if i not in exclude_days]
        #%%
        ffts = {}
        # ffts_abs = {}
        for unit in self.units:
            # compute average fft
            fft_pd = np.zeros(N,dtype=np.complex)
            fft_sc = np.zeros(N,dtype=np.complex)
            fft_pd_abs = np.zeros(N,dtype=float)
            fft_sc_abs = np.zeros(N,dtype=float)
            # ffts = np.zeros(shape=(ndays,N))
            freq = fftfreq(N, 1 / 12)
            for i in range(ndays):
                fft_pd += fft(d[unit]['pd_er'][i])/ndays_data
                fft_sc += fft(d[unit]['sc_er'][i])/ndays_data
                fft_pd_abs += np.abs(fft(d[unit]['pd_er'][i]))/ndays_data
                fft_sc_abs += np.abs(fft(d[unit]['sc_er'][i]))/ndays_data

            ffts[f'{unit}_pd'] = fft_pd
            ffts[f'{unit}_sc'] = fft_sc
            ffts[f'{unit}_pd_abs'] = fft_pd_abs
            ffts[f'{unit}_sc_abs'] = fft_sc_abs
        #     ffts.loc[:,s] = fft(np.array(rls[(unit, s)].iloc[1:]))
        #

        #%% sort model and data errors
        sort = {}
        for unit in self.units:
            sort[f'{unit}_pd'] = np.sort(np.reshape(d[unit]['pd_er'][idays],(ndays_data*N)))
            sort[f'{unit}_sc'] = np.sort(np.reshape(d[unit]['sc_er'][idays],(ndays_data*N)))

        #%% plot spectrum
        f,axs = plt.subplots(2,3,sharey=False)
        f.set_size_inches(12.5,9)
        # f = plt.figure()
        for ax,unit in zip(axs[1],self.units):
            # ax = f.add_subplot(2,3,pidx+1,sharey)
            # ax.cla()
            diff = np.sum(2/np.sqrt(N)*np.abs(ffts[f'{unit}_pd_abs'][:N//2]-ffts[f'{unit}_sc_abs'][:N//2]))


            ax.loglog(freq[:N//2],2/np.sqrt(N)*np.abs(ffts[f'{unit}_pd_abs'][:N//2]),label='data',alpha=1,
                      color = 'C1',linewidth=1.5)
            ax.loglog(freq[:N//2],2/np.sqrt(N)*np.abs(ffts[f'{unit}_sc_abs'][:N//2]),label='scenarios',
                      linewidth=0.6,color='black')

            # plt.grid()
            # plt.legend()
            ax.set_xlabel('Frequency (1/Hour)')
            ax.grid()
            if print_diff:
                ax.legend(title=f'diff: {diff:0.3f}')
            else:
                ax.legend()
            # ax.set_title(unit)
            # plt.savefig(fig_path / f'spectrum_{unit}.png')
        #
        # ax = f.add_subplot(1, 2, 2)
        #
        # avg_spec = np.zeros(ndays, dtype=complex)
        # for s in range(1, nscen + 1):
        #     avg_spec += 1 / nscen * ffts.loc[:,s]

        #% plot sorted errors
        for ax,unit in zip(axs[0],self.units):
            # f,ax = plt.subplots()
            # ax.cla()
            ax.plot(sort[f'{unit}_pd'],sort[f'{unit}_sc'],linestyle='none',marker='x')
            ax.plot([-0.6,0.6],[-0.6,0.6],color='black',linewidth=0.8)
            ax.grid()
            ax.set_xlabel('True forecast error')
            ax.set_ylabel('Scenario forecast error')
            ax.set_title(unit)

        plt.tight_layout()
        plt.savefig(self.path / f'validation_{tag}.png')
        if self.eps_figures:
            plt.savefig(self.path / f'validation_{tag}.eps')


def plot_wind_scenarios(wind,tag='fit1',wcap=[1,1,1],wbus=[3,5,16]):
    # %% plot wind scenarios (total wind)

    fig_path = Path('C:/Users/elisn/Box Sync/Research/ramp capability uc/Figures')

    fig_path.mkdir(exist_ok=True, parents=True)

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex=True)
    f.set_size_inches(w=25 / cm_per_inch, h=20 / cm_per_inch)

    units = list(wind.keys())
    # wcap = data['wind_capacity']
    nT = wind[units[0]].__len__()
    sidxs = list(wind[units[0]].columns[1:])
    delta = nT / 24

    # %% sum total wind
    wind_tot = pd.DataFrame(0.0, index=range(1, nT + 1), columns=sidxs)
    for sidx in sidxs:
        for bus, name, cap in zip(wbus, units, wcap):
            wind_tot.loc[:, sidx] = wind_tot.loc[:, sidx] + np.array(wind[name][sidx]) * cap

    # %%
    x = np.arange(1, nT + 1) / delta
    y = np.array(wind_tot)

    ax1.plot(x, y, color='k')
    # wind_tot.plot(ax=ax1,color='k')
    ax1.grid()
    ax1.set_title('Total wind')
    ax1.legend([])
    ax1.set_ylabel('MWh')

    for ax, bus, name, cap in zip([ax2, ax3, ax4], wbus, units, wcap):
        y = cap * np.array(wind[name].iloc[:, 1:])
        ax.plot(x, y, color='k')

        ax.set_title(f'Bus {bus} - {name}')
        ax.legend([])
        ax.set_ylabel('MWh')
        ax.grid()

    plt.savefig(fig_path / f'wind_scenarios_{tag}.png')
    plt.savefig(fig_path / f'wind_scenarios_{tag}.eps')



# %%

def plot_scenarios(name='basemodel',nscen=20,date='20190707',plot_avg_spectrum = True):
    """ Generate scenarios, plot time series and ffts """

    m = WindModel(name=name)
    m.load_model()

    rls = m.generate_scenarios(nscen, date=date)

    fs = m.fs
    # units = list(set(rls.columns.get_level_values(0)))
    units = m.units
    scenarios = list(set(rls.columns.get_level_values(1)))

    # plot time series
    f = plt.figure()
    f.set_size_inches(20/cm_per_inch,15/cm_per_inch)

    for idx, unit in enumerate(units):
        plt.subplot(2, 2, idx + 1)

        ax = plt.gca()

        for sidx in range(1,nscen+1):
            if sidx == 1:
                label = 'scenarios'
            else:
                label= '_nolegend_'
            if nscen < 10:
                color = f'C{sidx}'
            else:
                color = 'black'
            rls.loc[:,(unit,sidx)].plot(ax=ax, color=color, label=label, linewidth=0.8)
        rls[unit].iloc[:, 0].plot(ax=ax, color='red', linewidth=1.2, linestyle='solid', alpha=1,label='data')
        # (wind_data[unit] / capacity[unit]).plot(ax=ax, color='black', linestyle='--')
        ax.set_title(f'{unit}')
        plt.legend()
        plt.grid()
    # plt.suptitle(f'{nscen} wind power scenarios generated from AEMO {date}')
    plt.tight_layout()
    plt.suptitle(f'{nscen} wind power scenarios generated from AEMO {date}')

    plt.savefig(m.path / 'scenarios_ts.png')


    # plot ffts
    f = plt.figure()
    f.subplots_adjust(top=0.8)
    N = rls.__len__() - 1
    ffts = pd.DataFrame(0.0,index=range(N),columns=rls.columns)
    for u in units:
        for s in scenarios:
            ffts.loc[:,(u,s)] = fft(rls[(u,s)].iloc[1:])

    freq = fftfreq(N,1/fs)
    for idx,u in enumerate(units):

        ax = f.add_subplot(2,2,idx+1)

        avg_spec = np.zeros(N,dtype=complex)
        for s in range(1,scenarios.__len__()):
            avg_spec += 1/nscen * ffts.loc[:,(u,s)]

        if plot_avg_spectrum:
            plt.loglog( freq[:N//2], 2/np.sqrt(N)*np.abs(ffts.loc[:N // 2 - 1, (u, 0)]), color='red')
            plt.loglog( freq[:N//2], 2/np.sqrt(N)*np.abs(avg_spec[:N//2]), color='black', linewidth=2 , alpha=1)
        else:
            plt.loglog( freq[:N//2], 2/np.sqrt(N)*np.abs(ffts.loc[:N//2-1,(u,s)]), color='black', linewidth=0.5  ,label='_nolegend_')
            plt.loglog( freq[:N//2], 2/np.sqrt(N)*np.abs(ffts.loc[:N//2-1,(u,0)]) ,color='red')

        plt.grid()
        ax.set_title(f'{u}')
    plt.suptitle('FFTs of wind scenarios')
    plt.savefig(m.path / 'scenarios_fft.png')


def plot_scenarios2(rls,data=None,fig_path=Path('D:/Data/AEMO/Figures'),plot_avg_spectrum = True,tag='default',units=None):
    """ Generate scenarios, plot time series and ffts """

    # m = WindModel(name=name)
    # m.load_model()
    #
    # rls = m.generate_scenarios(nscen, date=date)
    fs = 12
    # fs = m.fs
    if units is None:
        units = list(set(rls.columns.get_level_values(0)))
    # units = m.units
    scenarios = list(set(rls.columns.get_level_values(1)))
    nscen = scenarios.__len__()

    # plot time series
    f = plt.figure()
    f.set_size_inches(15 / cm_per_inch, 30 / cm_per_inch)

    for idx, unit in enumerate(units):
        # plt.subplot(2, 2, idx + 1)

        # ax = plt.gca()
        if idx > 0:
            ax = f.add_subplot(3,1,idx+1,sharex=ax)
        else:
            ax = f.add_subplot(3,1,idx+1)


        if data is not None:
            data[(unit,'fc')].plot(ax=ax, color='C0', linewidth=2.5, linestyle='dashed',alpha=1,label='forecast')
            data[(unit,'pd')].plot(ax=ax, color='black', linewidth=1, linestyle='solid', alpha=1, label='outcome')

        for sidx in range(1, nscen + 1):
            if sidx == 1:
                label = 'scenarios'
            else:
                label = '_nolegend_'
            if nscen < 10:
                color = f'C{sidx}'
            else:
                color = 'orange'
            rls.loc[:, (unit, sidx)].plot(ax=ax, color=color, label=label, linewidth=0.8,alpha=1)


        # (wind_data[unit] / capacity[unit]).plot(ax=ax, color='black', linestyle='--')
        ax.set_title(f'{unit}')
        plt.legend()
        plt.ylim([0,1])
        plt.grid()
    # plt.suptitle(f'{nscen} wind power scenarios generated from AEMO {date}')
    plt.tight_layout()
    # plt.suptitle(f'{nscen} wind power scenarios generated from AEMO')

    plt.savefig(fig_path / f'scenarios_ts_{tag}.png')

    #%%
    # plot ffts
    f = plt.figure()
    f.subplots_adjust(top=0.8)
    N = rls.__len__() - 1
    ffts = pd.DataFrame(0.0, index=range(N), columns=rls.columns)
    for u in units:
        for s in scenarios:
            ffts.loc[:, (u, s)] = fft(np.array(rls[(u, s)].iloc[1:]))

    freq = fftfreq(N, 1 / fs)
    for idx, u in enumerate(units):

        ax = f.add_subplot(2, 2, idx + 1)

        avg_spec = np.zeros(N, dtype=complex)
        for s in range(1, nscen+1):
            avg_spec += 1 / nscen * ffts.loc[:, (u, s)]

        if plot_avg_spectrum:
            if data is not None:
                pd_fft = fft(np.array(data[(u,'pd')].iloc[1:]))
                plt.loglog(freq[:N // 2], 2 / np.sqrt(N) * np.abs(pd_fft[:N // 2]), color='red')
            # plt.loglog(freq[:N // 2], 2 / np.sqrt(N) * np.abs(ffts.loc[:N // 2 - 1, (u, 0)]), color='red')
            plt.loglog(freq[:N // 2], 2 / np.sqrt(N) * np.abs(avg_spec[:N // 2]), color='black', linewidth=2,
                       alpha=1)
        else:
            for s in range(1, scenarios.__len__()):
                plt.loglog(freq[:N // 2], 2 / np.sqrt(N) * np.abs(ffts.loc[:N // 2 - 1, (u, s)]), color='black',
                           linewidth=0.5, label='_nolegend_')
            # plt.loglog(freq[:N // 2], 2 / np.sqrt(N) * np.abs(ffts.loc[:N // 2 - 1, (u, 0)]), color='red')
            if data is not None:
                pd_fft = fft(np.array(data[(u,'pd')].iloc[1:]))
                plt.loglog(freq[:N // 2], 2 / np.sqrt(N) * np.abs(pd_fft[:N // 2]), color='red')

        plt.grid()
        ax.set_title(f'{u}')
    plt.suptitle('FFTs of wind scenarios')
    plt.savefig(fig_path / f'scenarios_fft_{tag}.png')

def plot_scenarios_single_unit(rls,data=None,fig_path=Path('D:/Data/AEMO/Figures'),plot_avg_spectrum=True):

    # unit = 'ARWF1'
    # fig_path=Path('D:/Data/AEMO/Figures')
    # plot_avg_spectrum=True
    # tag='default'

    fs = 12
    # fs = m.fs
    scenarios = list(set(rls.columns.get_level_values(1)))
    units = list(set(rls.columns.get_level_values(0)))

    nscen = scenarios.__len__()

    # plot time series
    f = plt.figure()
    f.set_size_inches(18 / cm_per_inch, 8 / cm_per_inch)

    for unit in units:
        f.clf()
        ax = f.add_subplot(1, 2, 1)

        if data is not None:
            data[(unit, 'fc')].plot(ax=ax, color='C0', linewidth=2.5, linestyle='dashed', alpha=1, label='forecast')
            data[(unit, 'pd')].plot(ax=ax, color='black', linewidth=1.2, linestyle='solid', alpha=1, label='outcome')

        for sidx in range(1, nscen + 1):
            if sidx == 1:
                label = 'scenarios'
            else:
                label = '_nolegend_'
            if nscen < 10:
                color = f'C{sidx}'
            else:
                color = 'orange'
            # color = 'black'
            rls.loc[:, (unit, sidx)].plot(ax=ax, color=color, label=label, linewidth=0.8, alpha=0.8)


        # remove some tick labels
        xax = ax.get_xaxis()
        xlabs = xax.get_minorticklabels()
        xticks = xax.get_minor_ticks()
        for t in [xticks[i] for i in range(1, xticks.__len__(), 2)]:
            t.label.set_visible(False)
        xticks[-1].label.set_visible(False)

        # (wind_data[unit] / capacity[unit]).plot(ax=ax, color='black', linestyle='--')
        plt.legend()
        plt.ylim([0, 1])
        plt.grid()
        # plt.suptitle(f'{nscen} wind power scenarios generated from AEMO {date}')
        plt.tight_layout()
        # plt.suptitle(f'{nscen} wind power scenarios generated from AEMO')
        plt.title('Wind production')



        # % fft

        # f.subplots_adjust(top=0.8)
        N = rls.__len__() - 1
        ffts = pd.DataFrame(0.0, index=range(N), columns=scenarios)
        for s in scenarios:
            ffts.loc[:,s] = fft(np.array(rls[(unit, s)].iloc[1:]))

        freq = fftfreq(N, 1 / fs)

        ax = f.add_subplot(1, 2, 2)

        avg_spec = np.zeros(N, dtype=complex)
        for s in range(1, nscen + 1):
            avg_spec += 1 / nscen * ffts.loc[:,s]

        if plot_avg_spectrum:
            if data is not None:
                pd_fft = fft(np.array(data[(unit, 'pd')].iloc[1:]))
                plt.loglog(freq[:N // 2], 2 / np.sqrt(N) * np.abs(pd_fft[:N // 2]), color='black', linewidth = 1, label='outcome')
            # plt.loglog(freq[:N // 2], 2 / np.sqrt(N) * np.abs(ffts.loc[:N // 2 - 1, (u, 0)]), color='red')
            plt.loglog(freq[:N // 2], 2 / np.sqrt(N) * np.abs(avg_spec[:N // 2]), color='C1', linewidth=2,
                       alpha=0.7, label = 'scenario average')
        else:
            for s in range(1, scenarios.__len__()):
                plt.loglog(freq[:N // 2], 2 / np.sqrt(N) * np.abs(ffts.loc[:N // 2 - 1, (unit, s)]), color='black',
                           linewidth=0.5, label='_nolegend_')
            # plt.loglog(freq[:N // 2], 2 / np.sqrt(N) * np.abs(ffts.loc[:N // 2 - 1, (u, 0)]), color='red')
            if data is not None:
                pd_fft = fft(np.array(data[(unit, 'pd')].iloc[1:]))
                plt.loglog(freq[:N // 2], 2 / np.sqrt(N) * np.abs(pd_fft[:N // 2]), color='red')

        plt.grid()
        ax.set_title(f'Spectra')
        plt.legend()
        plt.tight_layout()
        # plt.suptitle('FFTs of wind scenarios')
        # plt.savefig(fig_path / f'scenarios_fft_{tag}.png')

        plt.savefig(fig_path / f'scenarios_{unit}.png')
        plt.savefig(fig_path / f'scenarios_{unit}.eps')


def plot_scenarios_minmax(rls, data, fig_path = Path('D:/Data/AEMO/Figures'),tag='default',titles=False,quantiles=(0.02,0.98)):

    # unit = 'ARWF1'
    # fig_path=Path('D:/Data/AEMO/Figures')
    # plot_avg_spectrum=True
    # tag='default'

    # fs = m.fs
    units = list(set(rls.columns.get_level_values(0)))
    scenarios = list(set(rls.columns.get_level_values(1)))
    nscen = scenarios.__len__()

    f = plt.figure()
    f.set_size_inches(6.5, 5)

    for i,wf in enumerate(units):
        f.clf()
        # ax = f.add_subplot(2,2,i+1)
        ax = f.add_subplot(1,1,1)

        for ii in range(1,nscen+1):
            if ii == 1:
                label = 'scenarios'
            else:
                label = '_nolegend_'
            rls.loc[:,(wf,ii)].plot(ax=ax,color='C1',linewidth=0.8,label=label, alpha = 0.6)
        data.loc[:,(wf,'fc')].plot(ax=ax,label='forecast',color='C0',linestyle='dashed',linewidth=3, alpha=0.9)
        data.loc[:,(wf,'pd')].plot(ax=ax,label='outcome',color='black', alpha=0.9)
        data.loc[:,(wf,'min')].plot(ax=ax,linestyle='dashed',color='black',label=f'{quantiles[0]:0.0f} %')
        data.loc[:,(wf,'max')].plot(ax=ax,linestyle='dashed',color='black',label=f'{quantiles[1]:0.0f} %')

        plt.grid()
        plt.legend()
        if titles:
            plt.title(f'{wf}')

        plt.savefig(fig_path / f'scenarios_minmax_{tag}_{wf}.png')
        plt.savefig(fig_path / f'scenarios_minmax_{tag}_{wf}.eps')

def fit_model():

    m = WindModel(name='paper')

    ## OPTIONS ##
    m.units = ['ARWF1','MACARTH1','BALDHWF1']
    m.startdate = '20190801'
    # m.startdate = '20200301'
    # m.enddate = '20200228'
    # m.enddate = '20200331'
    m.enddate = '20191030'
    m.eps_figures = True
    m.plot_titles = False
    m.cov_resolution = '5min'
    m.cov_quant_filter = True
    m.cov_quant_cutoff = 0.1
    m.cov_quant_interp = 'rect'
    m.hf_linear_scale = True
    m.hf_nbins = 3
    m.hf_lead_time = 1  # lead time for forecast used to create regimes when fitting noise
    m.hf_binvar = 'fc'  # fc/pd, to use forecast or production for creating regimes
    m.hf_plot_figures = True
    m.hf_scale = 0.3
    m.quant_resolution = '30min'
    m.quant_solver_output = True
    m.quant_plots = True
    m.quant_cofit = True
    m.quant_remove_outliers = True
    m.quant_remove_zero_periods = True
    m.quant_nzero_hours_remove = 10
    m.quant_qvals = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    m.quant_nbins = 3
    m.quant_set_bins = [-0.1,0.3,0.7,1.1]
    m.lead_times = [1]
    m.scen_base = 'forecast'

    ## FIT MODEL ##
    # load data and preprocessing
    m.load_data()
    m.filter_data()
    # fitting model
    m.fit_noise()
    m.fit_quantiles()
    m.fit_covariance()

    m.save_model() # save model objects as pickle file
    # Saved model can be loaded with: m.load_model()

    # validate model
    seed = 1
    m.validate(ndays=60,startdate='20190801',tag='v1',use_hf_model=True,seed=seed)

    # generate scenarios
    date = '20190811'
    nscen = 10
    rls,data = m.generate_scenarios(nscen=nscen,cut_extreme=True,date=date,qrange=(0.02,0.98),use_hf_model=True)
    plot_scenarios_minmax(rls,data,m.path,tag='default',quantiles=(2,98))

if __name__ == "__main__":

    pd.set_option('display.max_rows',20)
    pd.set_option('display.max_columns',None)

    #%%
    m = WindModel(name='v1')
    m.load_model()

    date = '20190811'
    nscen = 10
    rls,data = m.generate_scenarios(nscen=nscen,cut_extreme=True,date=date,qrange=(0.02,0.98),use_hf_model=True)
    plot_scenarios_minmax(rls,data,m.path,tag='default',quantiles=(2,98))


    #%%
    # m = WindModel(name='paper')
    # m.units = ['ARWF1','MACARTH1','BALDHWF1']
    # m.startdate = '20190801'
    # # m.startdate = '20200301'
    # m.enddate = '20200228'
    # # m.enddate = '20200331'
    # m.eps_figures = True
    # m.plot_titles = False
    # m.cov_resolution = '5min'
    # m.cov_quant_filter = True
    # m.cov_quant_cutoff = 0.1
    # m.cov_quant_interp = 'rect'
    # m.hf_linear_scale = True
    # m.hf_nbins = 3
    # m.hf_lead_time = 1  # lead time for forecast used to create regimes when fitting noise
    # m.hf_binvar = 'fc'  # fc/pd, to use forecast or production for creating regimes
    # m.hf_scale = 0.3
    # m.hf_plot_figures = True
    # m.quant_resolution = '30min'
    # m.quant_solver_output = True
    # m.quant_plots = True
    # m.quant_cofit = True
    # m.quant_remove_outliers = True
    # m.quant_remove_zero_periods = True
    # m.quant_nzero_hours_remove = 10
    # # m.quant_qvals = np.arange(0.05,1,0.1)
    # m.quant_qvals = np.array([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    # # m.quant_qvals = np.array([0.05, 0.2, 0.8, 0.95])
    # m.quant_nbins = 3
    # m.quant_set_bins = [-0.1,0.3,0.7,1.1]
    # # m.quant_set_bins = None
    # m.lead_times = [1]
    # m.scen_base = 'forecast'
    #
    # m.load_data()
    # m.filter_data()
    # # m.make_5min_forecast()
    # m.load_model()
    # # #
    # # m.fit_noise()
    # m.fit_quantiles()
    # # m.fit_covariance()
    # # #
    # # # m.hf_nzero_nonoise = 2
    # # # m.hf_scale = 0.5
    # # # m.validate(ndays=100,tag='hf50pc_nonoise2',use_hf_model=True,startdate='20190801')
    # # #
    # # # m.validate(ndays=100,tag='hf100pc',use_hf_model=True)
    # #
    # # # m.hf_scale = 0.5
    # # # m.validate(ndays=100,tag='hf50pc2',use_hf_model=True)
    # #
    # seed = 1
    # m.hf_scale = 1
    # m.hf_noise_zeroprod = True
    # m.validate(ndays=100,startdate='20200301',tag='v1',use_hf_model=True,seed=seed,print_diff=True)
    # #
    # m.hf_scale = 0.5
    # m.hf_noise_zeroprod = True
    # m.validate(ndays=100,startdate='20200301',tag='v1_50pc',use_hf_model=True,seed=seed)
    #
    # m.hf_scale = 0.3
    # m.hf_noise_zeroprod = False
    # m.validate(ndays=100,startdate='20200301',tag='v1_30pc',use_hf_model=True,seed=seed)
    #
    # m.hf_scale = 0
    # m.hf_noise_zeroprod = True
    # m.validate(ndays=100,startdate='20200301',tag='v1_0pc',use_hf_model=True,seed=seed)
    #
    #
    # # # # # # #
    # m.save_model()
    #
    # m = WindModel()
    # m.load_model()
    # m.startdate = '20190801'
    # m.enddate = '20190831'
    # m.load_model()
    #
    # plot_scenarios(name='noise_envelope',nscen=5,date='20190707')

    # m.fit_noise_bins()

    # m.scen_base = 'production'
    # rls1,data1 = m.generate_scenarios_quantfit()
    #
    #%%
    # date = '20190811'
    # nscen = 10
    # m.hf_scale = 0.3
    # m.hf_noise_zeroprod = False
    # rls,data = m.generate_scenarios_quantfit(nscen=nscen,cut_extreme=True,date=date,qrange=(0.02,0.98),use_hf_model=True)
    # plot_scenarios_minmax(rls,data,m.path,tag='default',quantiles=(2,98))

    #%% plot wind farms

    # wpd = m.wpd_db.select_data(starttime='20200301:00',endtime='20200331:00',table_type='dispatch',categories=m.units)



