# __author__ = 'jihyunp'

import os
import re
import numpy as np
from scipy.misc import factorial

def split_data(data, split):
    """

    Parameters
    ----------
    data : np.array
    split : tuple

    Returns
    -------

    """
    N = data.shape[0]
    if sum(split) != N:
        const = int(np.ceil(N / float(sum(split))))
    else:
        const = 1
    starts = np.cumsum(np.r_[0, split[:-1]] * const)
    ends = np.cumsum(split) * const
    if ends[-1] > N:
        ends[-1] = N
    splits = [data[s:e] for s, e in zip(starts, ends)]
    return splits


def check_path_and_make_dirs(path):
    """
    Check if the directories in the path exists and if not, makedirs
    Parameters
    ----------
    path : str

    Returns
    -------

    """
    dirname = os.path.dirname(path)
    if dirname != "":
        if not os.path.isdir(dirname):
            os.makedirs(dirname)


def loglik_bernoulli(p, y):
    """
    Parameters
    ----------
    p : np.ndarray
    y : np.ndarray

    Returns
    -------
    float
    """
    p += 0.000001
    p[p >= 1] = 0.999999
    ll_arr = y*np.log(p) + (1-y)*np.log(1-p)
    return np.sum(ll_arr)


def loglik_poisson(lam, k):
    """

    Parameters
    ----------
    lam : np.ndarray
        lambda (rates)
    k : np.ndarray
        number of events

    Returns
    -------
    float
    """
    lam += 0.000001
    lam[lam >= 1] = 0.999999
    ll_arr = k * np.log(lam) + (-lam) - np.log(factorial(k))
    return np.sum(ll_arr)


def expit(a):
    return 1/(1+np.exp(-a))


def logit(a):
    a += 0.000001
    a[a>=1] = 0.999999
    return np.log(a/(1-a))


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    # return [ atoi(c) for c in re.split('(\d+)', text) ]
    return atoi(''.join(re.findall(r"[0-9]+", text)))


def custom_sort(x, y):
    """

    Parameters
    ----------
    x : str
    y : str

    Returns
    -------

    """
    if (type(x) == str) and (type(y) == str):
        if (y == 'pre') or (y == 'post'):
            return -1
        elif x.endswith('pre') and y.endswith('post'):
            return -1
        else:
            return 0
    else:
        return 0

def plus_minus_sort(x, y):
    """

    Parameters
    ----------
    x : str
    y : str

    Returns
    -------

    """
    if (type(x) == str) and (type(y) == str):
        if cmp(x[0], y[0]) == 0:
            if x.endswith('+') or y.endswith('-'):
                return -1
            elif x.endswith('-') or y.endswith('+'):
                return +1
            else:
                return cmp(x, y)
        else:
            return cmp(x, y)
    else:
        return cmp(x, y)


def set_plot_frame(fig, xlabel=None, ylabel=None, text_font=None, axis_font=None):
    """
    Setting a large plot outside small subplots.

    Parameters
    ----------
    fig
    xlabel
    ylabel
    text_font
    axis_font

    Returns
    -------

    """
    ax = fig.add_subplot(111)
    [ax.spines[x].set_color('none') for x in ax.spines.keys()] # color='none' is identical to color='w'
    ax.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
    ax.set_xlabel(xlabel, labelpad=10, **axis_font)
    ax.set_ylabel(ylabel, labelpad=20, **axis_font)
    return fig
