#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced EEG feature extraction with preprocessing (forward-fill + ICA),
band-wise features (Delta/Theta/Alpha/Beta/Gamma), Hjorth, Shannon entropy,
and concentration heuristic (Beta/(Theta+Alpha)).

This module exposes generate_feature_vectors_from_samples(file_path, nsamples, period, state, remove_redundant, cols_to_ignore)
compatible with the original project API.
"""

import numpy as np
import scipy
import scipy.signal
from scipy.signal import butter, filtfilt
from scipy import stats
from sklearn.decomposition import FastICA, PCA
import warnings
from sklearn.exceptions import ConvergenceWarning


def matrix_from_csv_file(file_path):
    csv_data = np.genfromtxt(file_path, delimiter=',')
    full_matrix = csv_data[1:]
    return full_matrix


def get_time_slice(full_matrix, start=0., period=1.):
    rstart = full_matrix[0, 0] + start
    # find indices where timestamp <= target; handle empty result gracefully
    inds0 = np.where(full_matrix[:, 0] <= rstart)[0]
    if inds0.size == 0:
        raise IndexError('no start index found for requested time slice')
    index_0 = inds0.max()
    inds1 = np.where(full_matrix[:, 0] <= rstart + period)[0]
    if inds1.size == 0:
        raise IndexError('no end index found for requested time slice')
    index_1 = inds1.max()
    duration = full_matrix[index_1, 0] - full_matrix[index_0, 0]
    return full_matrix[index_0:index_1, :], duration


def feature_mean(matrix):
    ret = np.mean(matrix, axis=0).flatten()
    names = ['mean_' + str(i) for i in range(matrix.shape[1])]
    return ret, names


def feature_stddev(matrix):
    ret = np.std(matrix, axis=0, ddof=1).flatten()
    names = ['std_' + str(i) for i in range(matrix.shape[1])]
    return ret, names


def feature_moments(matrix):
    skw = stats.skew(matrix, axis=0, bias=False)
    krt = stats.kurtosis(matrix, axis=0, bias=False)
    ret = np.append(skw, krt)
    names = ['skew_' + str(i) for i in range(matrix.shape[1])]
    names.extend(['kurt_' + str(i) for i in range(matrix.shape[1])])
    return ret, names


def feature_min(matrix):
    ret = np.min(matrix, axis=0).flatten()
    names = ['min_' + str(i) for i in range(matrix.shape[1])]
    return ret, names


def feature_max(matrix):
    ret = np.max(matrix, axis=0).flatten()
    names = ['max_' + str(i) for i in range(matrix.shape[1])]
    return ret, names


def feature_covariance_matrix(matrix):
    covM = np.cov(matrix.T)
    indx = np.triu_indices(covM.shape[0])
    ret = covM[indx]
    names = []
    for i in np.arange(0, covM.shape[1]):
        for j in np.arange(i, covM.shape[1]):
            names.extend(['covM_' + str(i) + '_' + str(j)])
    return ret, names, covM


def feature_eigenvalues(covM):
    ret = np.linalg.eigvals(covM).flatten()
    # Try to convert to real when imaginary parts are numerically negligible
    try:
        ret = np.real_if_close(ret, tol=1000)
    except Exception:
        pass
    # If still complex, fall back to absolute(real) to get a stable real feature
    if np.iscomplexobj(ret):
        ret = np.real(np.abs(ret))
    names = ['eigenval_' + str(i) for i in range(covM.shape[0])]
    return np.asarray(ret, dtype=float), names


def feature_logcov(covM):
    # Regularize covariance to improve numerical stability before logm
    try:
        n = covM.shape[0]
        trace = np.trace(covM)
        # scale eps relative to trace to adapt to magnitude of matrix
        eps = 1e-6 * (trace / float(n) if trace != 0 else 1e-6)
        covM_reg = covM + eps * np.eye(n)
    except Exception:
        covM_reg = covM
    # logm can sometimes produce small imaginary components or numerical warnings;
    # call it inside a warnings context and coerce to a stable real matrix below.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        log_cov = scipy.linalg.logm(covM_reg)
    # try to coerce matrix log to real when imaginary parts are negligible
    try:
        log_cov_real = np.real_if_close(log_cov, tol=1000)
    except Exception:
        log_cov_real = log_cov
    # if still complex, take the real part (regularization should have reduced complex components)
    if np.iscomplexobj(log_cov_real):
        log_cov_used = np.real(log_cov_real)
    else:
        log_cov_used = log_cov_real
    indx = np.triu_indices(log_cov_used.shape[0])
    ret = log_cov_used[indx]
    names = []
    for i in np.arange(0, log_cov.shape[1]):
        for j in np.arange(i, log_cov.shape[1]):
            names.extend(['logcovM_' + str(i) + '_' + str(j)])
    return np.asarray(ret, dtype=float), names, np.asarray(log_cov_used, dtype=float)


class HeadsetError(Exception):
    """Custom exception for headset connection issues"""
    pass

def feature_fft(matrix, period=1., mains_f=50., filter_mains=True, filter_DC=True,
                normalise_signals=True, ntop=10, get_power_spectrum=True):
    N = matrix.shape[0]
    T = period / N
    if normalise_signals:
        data_range = np.max(matrix) - np.min(matrix)
        if data_range == 0 or not np.isfinite(data_range):
            raise HeadsetError("Device may be incorrectly worn or not worn. Check electrode connections.")
            # Return a matrix of NaNs to indicate invalid data
            return np.full((N//2, matrix.shape[1]), np.nan), np.linspace(0.0, 1.0 / (2.0 * T), N//2), None
        matrix = -1 + 2 * (matrix - np.min(matrix)) / data_range
    fft_values = np.abs(scipy.fft.fft(matrix, axis=0))[0:N // 2] * 2 / N
    freqs = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
    if filter_DC:
        fft_values = fft_values[1:]
        freqs = freqs[1:]
    if filter_mains:
        indx = np.where(np.abs(freqs - mains_f) <= 1)
        fft_values = np.delete(fft_values, indx, axis=0)
        freqs = np.delete(freqs, indx)
    indx = np.argsort(fft_values, axis=0)[::-1]
    indx = indx[:ntop]
    ret = freqs[indx].flatten(order='F')
    names = []
    for i in np.arange(fft_values.shape[1]):
        names.extend(['topFreq_' + str(j) + "_" + str(i) for j in np.arange(1, 11)])
    if (get_power_spectrum):
        ret = np.hstack([ret, fft_values.flatten(order='F')])
        for i in np.arange(fft_values.shape[1]):
            names.extend(['freq_' + "{:03d}".format(int(j)) + "_" + str(i) for j in 10 * np.round(freqs, 1)])
    return ret, names


# --- New helpers

def hjorth_parameters(x):
    x = np.asarray(x)
    var_x = np.var(x)
    diff2 = np.diff(x)
    var_diff = np.var(diff2)
    activity = var_x
    mobility = np.sqrt(var_diff / var_x) if var_x > 0 else 0.0
    diff3 = np.diff(diff2)
    var_diff2 = np.var(diff3)
    complexity = np.sqrt(var_diff2 / var_diff) if var_diff > 0 else 0.0
    return activity, mobility, complexity


def shannon_entropy(x, bins=64):
    h, _ = np.histogram(x, bins=bins, density=True)
    h = h[h > 0]
    p = h / np.sum(h)
    return -np.sum(p * np.log2(p))


def rms(x):
    x = np.asarray(x)
    return np.sqrt(np.mean(x ** 2))


def bandpass_filter(data, low, high, fs, order=4):
    nyq = 0.5 * fs
    lowcut = low / nyq
    highcut = high / nyq
    if lowcut <= 0:
        b, a = butter(order, highcut, btype='low')
    else:
        b, a = butter(order, [lowcut, highcut], btype='band')
    return filtfilt(b, a, data, axis=0)


def forward_fill_timeseries(matrix):
    m = matrix.copy()
    sig = m[:, 1:]
    sig[~np.isfinite(sig)] = np.nan
    for col in range(sig.shape[1]):
        coldata = sig[:, col]
        valid = np.where(np.isfinite(coldata))[0]
        if valid.size == 0:
            sig[:, col] = 0.0
            continue
        for i in range(1, len(coldata)):
            if not np.isfinite(coldata[i]):
                coldata[i] = coldata[i - 1]
        if not np.isfinite(coldata[0]):
            first = valid[0]
            coldata[:first] = coldata[first]
        sig[:, col] = coldata
    m[:, 1:] = sig
    return m


def apply_ica_and_remove_artifacts(sig_matrix, n_components=None, kurtosis_thresh=5.0):
    X = sig_matrix.copy()
    n_comp = n_components if n_components is not None else min(X.shape[1], X.shape[0])
    try:
        # Tune FastICA parameters for better convergence
        # increase max_iter to reduce ConvergenceWarning on difficult signals
        ica = FastICA(n_components=n_comp, random_state=0, max_iter=2000, tol=1e-4)
        # suppress FastICA ConvergenceWarning locally (we fallback to PCA if ICA fails)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=ConvergenceWarning)
            S = ica.fit_transform(X)
        A = ica.mixing_
        k = stats.kurtosis(S, axis=0, bias=False)
        remove_idx = np.where(np.abs(k) > kurtosis_thresh)[0]
        if remove_idx.size > 0:
            S[:, remove_idx] = 0.0
        X_clean = np.dot(S, A.T)
        return X_clean
    except Exception:
        # Fall back to PCA-based denoising: zero the first principal component (often large artifacts)
        try:
            pca_n = min(n_comp, X.shape[1])
            pca = PCA(n_components=pca_n, random_state=0)
            S_p = pca.fit_transform(X)
            # Zero the first principal component to remove dominant artifact energy
            if S_p.shape[1] > 0:
                S_p[:, 0] = 0.0
            X_pca = pca.inverse_transform(S_p)
            return X_pca
        except Exception:
            return sig_matrix


def calc_feature_vector(matrix, state):
    # matrix: nsamples x nchannels
    h1, h2 = np.split(matrix, [int(matrix.shape[0] / 2)])
    q1, q2, q3, q4 = np.split(matrix, [int(0.25 * matrix.shape[0]), int(0.50 * matrix.shape[0]), int(0.75 * matrix.shape[0])])
    var_names = []
    x, v = feature_mean(matrix)
    var_names += v
    var_values = x
    x, v = feature_stddev(matrix)
    var_names += v
    var_values = np.hstack([var_values, x])
    x, v = feature_moments(matrix)
    var_names += v
    var_values = np.hstack([var_values, x])
    x, v = feature_max(matrix)
    var_names += v
    var_values = np.hstack([var_values, x])
    x, v = feature_min(matrix)
    var_names += v
    var_values = np.hstack([var_values, x])
    x, v, covM = feature_covariance_matrix(matrix)
    var_names += v
    var_values = np.hstack([var_values, x])
    x, v = feature_eigenvalues(covM)
    var_names += v
    var_values = np.hstack([var_values, x])
    x, v, log_cov = feature_logcov(covM)
    var_names += v
    var_values = np.hstack([var_values, x])
    x, v = feature_fft(matrix)
    var_names += v
    var_values = np.hstack([var_values, x])

    # band-wise features
    fs = int(round(matrix.shape[0] / 1.0)) if matrix.shape[0] > 0 else 1
    bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30),
        'Gamma': (30, 45)
    }
    band_means = {}
    for bname, (low, high) in bands.items():
        try:
            bp = bandpass_filter(matrix, low, high, fs)
        except Exception:
            bp = np.zeros_like(matrix)
        powers = np.mean(bp ** 2, axis=0)
        band_means[bname] = np.mean(powers)
        p_mean = np.mean(powers)
        p_median = np.median(powers)
        p_std = np.std(powers, ddof=1) if powers.size > 1 else 0.0
        p_skew = stats.skew(powers) if powers.size > 0 else 0.0
        p_kurt = stats.kurtosis(powers) if powers.size > 0 else 0.0
        p_rms = rms(powers)
        concat = bp.flatten()
        h_activity, h_mobility, h_complexity = hjorth_parameters(concat)
        ent = shannon_entropy(concat)
        vals = [p_mean, p_median, p_std, p_skew, p_kurt, p_rms,
                h_activity, h_mobility, h_complexity, ent]
        var_values = np.hstack([var_values, np.array(vals)])
        var_names += [f"{bname}_mean_pow", f"{bname}_median_pow", f"{bname}_std_pow",
                      f"{bname}_skew_pow", f"{bname}_kurt_pow", f"{bname}_rms_pow",
                      f"{bname}_hjorth_activity", f"{bname}_hjorth_mobility",
                      f"{bname}_hjorth_complexity", f"{bname}_entropy"]

    beta = band_means.get('Beta', 1e-9)
    theta = band_means.get('Theta', 1e-9)
    alpha = band_means.get('Alpha', 1e-9)
    conc = beta / (theta + alpha + 1e-12)
    var_values = np.hstack([var_values, np.array([conc])])
    var_names += ['concentration_beta_div_theta_plus_alpha']

    if state is not None:
        var_values = np.hstack([var_values, np.array([state])])
        var_names += ['Label']

    # Ensure returned feature vector is purely real floats (drop negligible imaginary parts)
    try:
        var_values = np.real_if_close(var_values, tol=1000)
    except Exception:
        pass
    if np.iscomplexobj(var_values):
        # fallback: take absolute of complex entries and cast to real
        var_values = np.real(np.abs(var_values))
    var_values = np.asarray(var_values, dtype=float)

    return var_values, var_names


def generate_feature_vectors_from_samples(file_path, nsamples, period,
                                          state=None,
                                          remove_redundant=True,
                                          cols_to_ignore=None):
    matrix = matrix_from_csv_file(file_path)
    t = 0.
    previous_vector = None
    ret = None
    headers = None
    while True:
        try:
            s, dur = get_time_slice(matrix, start=t, period=period)
            if cols_to_ignore is not None:
                s = np.delete(s, cols_to_ignore, axis=1)
        except IndexError:
            break
        if len(s) == 0:
            break
        if dur < 0.9 * period:
            break

        s_ff = forward_fill_timeseries(s)
        sig = s_ff[:, 1:]
        sig_clean = apply_ica_and_remove_artifacts(sig)
        try:
            ry, rx = scipy.signal.resample(sig_clean, num=nsamples, t=s_ff[:, 0], axis=0)
        except Exception:
            try:
                ry, rx = scipy.signal.resample(s[:, 1:], num=nsamples, t=s[:, 0], axis=0)
            except Exception:
                ry = np.zeros((nsamples, s.shape[1] - 1))

        t += 0.5 * period

        r, headers = calc_feature_vector(ry, state)

        if previous_vector is not None:
            feature_vector = np.hstack([previous_vector, r])
            if ret is None:
                ret = feature_vector
            else:
                ret = np.vstack([ret, feature_vector])

        previous_vector = r
        if state is not None:
            previous_vector = previous_vector[:-1]

    if headers is None:
        headers = []

    feat_names = ["lag1_" + s for s in headers[:-1]] + headers

    if remove_redundant and ret is not None:
        to_rm = ["lag1_mean_q3_", "lag1_mean_q4_", "lag1_mean_d_q3q4_",
                 "lag1_max_q3_", "lag1_max_q4_", "lag1_max_d_q3q4_",
                 "lag1_min_q3_", "lag1_min_q4_", "lag1_min_d_q3q4_"]
        for i in range(len(to_rm)):
            for j in range(ry.shape[1]):
                rm_str = to_rm[i] + str(j)
                if rm_str in feat_names:
                    idx = feat_names.index(rm_str)
                    feat_names.pop(idx)
                    ret = np.delete(ret, idx, axis=1)

    return ret, feat_names


def generate_feature_vectors_from_matrix(full_matrix: np.ndarray, nsamples: int, period: float,
                                         state=None, remove_redundant=True, cols_to_ignore=None):
    """Same behavior as generate_feature_vectors_from_samples but accepts the
    numeric matrix already loaded (including headerless rows like matrix_from_csv_file returns).
    This allows in-memory feature extraction without writing CSV files.
    """
    matrix = full_matrix
    t = 0.
    previous_vector = None
    ret = None
    headers = None
    while True:
        try:
            s, dur = get_time_slice(matrix, start=t, period=period)
            if cols_to_ignore is not None:
                s = np.delete(s, cols_to_ignore, axis=1)
        except IndexError:
            break
        if len(s) == 0:
            break
        if dur < 0.9 * period:
            break

        s_ff = forward_fill_timeseries(s)
        sig = s_ff[:, 1:]
        sig_clean = apply_ica_and_remove_artifacts(sig)
        try:
            ry, rx = scipy.signal.resample(sig_clean, num=nsamples, t=s_ff[:, 0], axis=0)
        except Exception:
            try:
                ry, rx = scipy.signal.resample(s[:, 1:], num=nsamples, t=s[:, 0], axis=0)
            except Exception:
                ry = np.zeros((nsamples, s.shape[1] - 1))

        t += 0.5 * period

        r, headers = calc_feature_vector(ry, state)

        if previous_vector is not None:
            feature_vector = np.hstack([previous_vector, r])
            if ret is None:
                ret = feature_vector
            else:
                ret = np.vstack([ret, feature_vector])

        previous_vector = r
        if state is not None:
            previous_vector = previous_vector[:-1]

    if headers is None:
        headers = []

    feat_names = ["lag1_" + s for s in headers[:-1]] + headers

    if remove_redundant and ret is not None:
        to_rm = ["lag1_mean_q3_", "lag1_mean_q4_", "lag1_mean_d_q3q4_",
                 "lag1_max_q3_", "lag1_max_q4_", "lag1_max_d_q3q4_",
                 "lag1_min_q3_", "lag1_min_q4_", "lag1_min_d_q3q4_"]
        for i in range(len(to_rm)):
            for j in range(ry.shape[1]):
                rm_str = to_rm[i] + str(j)
                if rm_str in feat_names:
                    idx = feat_names.index(rm_str)
                    feat_names.pop(idx)
                    ret = np.delete(ret, idx, axis=1)

    return ret, feat_names
