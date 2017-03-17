# __author__ = 'jihyunp'

from datetime import datetime
import numpy as np
import csv
import os

import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError
from glm_gd import GLM, GLMResult
from utils import check_path_and_make_dirs, expit, logit, \
    loglik_poisson, loglik_bernoulli


class StudentChangePoint():

    def __init__(self, data_fname = './data.csv', result_dir='./result', padding=10, binary=False):
        """

        Parameters
        ----------
        data_fname : str
            File path to the data, where the data is a matrix of numbers in .csv form.
            The matrix should have the size (n_students x n_days)
        result_dir : str
            File path to the results (output)
        padding : int
            Number of days that you want to skip at both ends (start and end)
            when searching the changepoint tau_i's
        binary : bool
            True if you want to use the Bernoulli model using binarized data
        """

        self.padding = padding
        self.binary = binary
        self.result_dir = result_dir
        check_path_and_make_dirs(self.result_dir)

        # load data
        self.rawdata = None
        self.n_students, self.n_days = 0, 0
        self.idxs = None
        self.glm_mat = None
        self.load_data(data_fname, binary)
        if np.max(self.rawdata) == 1:
            self.binary = True
            # You're forced to use Bernoulli model when there are only 0 or 1
        self.glm_mat = self.get_matrix_for_glm_mixed()

        # Things that are used in changedetection
        # These will have values after running function 'naive_changepoint_detection()'
        self.detected_cp_arr = None  # detected changepoint locations
        self.mcp_max_ll_mat, self.m0_ll_mat = None, None  # LogLik values for model w/ cp and model w/o cp
        self.mcp_min_bic_mat, self.m0_bic_mat = None, None  # BIC values for model w/ cp and model w/o cp
        self.alpha_i_mat = None  # alpha_i's. There are three columns.
        self.better_w_cp_sidxs, self.better_wo_cp_idxs = None, None # student indices w/ detected change, w/o detected change


    def load_data(self, data_fname, binary=False):
        print '\nLoading the data from '+ data_fname
        reader = csv.reader(open(data_fname, 'r'), delimiter=',')
        data = []
        for line in reader:
            data.append(map(float, line))
        data = np.array(data)
        self.rawdata = data
        self.n_students = data.shape[0]
        self.n_days = data.shape[1]
        self.idxs = range(self.n_students)
        if binary:
            self.rawdata[np.where(data)] = 1.0


    def get_matrix_for_glm_mixed(self):
        """
        Convert the raw (N x T) matrix into (NT x K) matrix
        so that we can use this as an input to glm.

        Returns
        -------
        Returns two numpy arrays
          1: endogenous response variable. The dependent variable. length NT
          2: X matrix : (NT x 3) size array
             , where T is the number of observations (days), N is number of groups (students).
             First column  : student index (i)
             Second column  : number of clicks per day (X_it)
             Third column : mu_t (population mean rate)
        """
        # This will give you the same result as we actually estimate the intercept
        #  (in fact it might be better since we can add a small number for padding)
        mat = self.rawdata
        population_rate = mat.sum(axis=0) / float(self.n_students)
        population_rate += 0.000001

        if self.binary:
            # Log odds (alpha_t)
            mu_t = np.log(population_rate / (1 - population_rate))
        else:
            mu_t = np.log(population_rate)

        X_it= mat.flatten('C')
        id_days_X = np.vstack((np.repeat(self.idxs, self.n_days), X_it)).T
        mu_t_NT = np.tile(mu_t, (1, self.n_students)).T

        mat_for_glm= np.concatenate((id_days_X, mu_t_NT), axis=1).tolist()
        return np.array(mat_for_glm)


    def get_one_student_data(self, glm_mat, sidx):
        if glm_mat.shape[0] == (self.n_days * self.n_students):
            if (0 <= sidx < self.n_students):
                sidx_st = self.n_days * sidx
                sidx_end = self.n_days * (sidx + 1)
                result = glm_mat[sidx_st:sidx_end, :]
                return result


    def run_glm(self, glm_data, binary):
        if binary:
            model = GLM(glm_data[:, 1], np.ones((glm_data.shape[0], 1)), offset=glm_data[:, 2],
                         family='Bernoulli_logit')
            res = model.fit()
        else:
            if np.sum(glm_data[:,1]) == 0:
                ll = loglik_poisson(np.zeros(glm_data.shape[0]), np.zeros(glm_data.shape[0]))
                ai = [-27.0]  # Just fix the intercept to -27 (small enough to give estimates close to 0)
                res = GLMResult(llf=ll, fittedvals=None, params=ai)
            else:
                # Use the package. GD version doesn't seem to be working well.
                model = sm.GLM(glm_data[:, 1], np.ones((glm_data.shape[0], 1)), offset=glm_data[:, 2],
                                family=sm.families.Poisson(link=sm.genmod.families.links.log))
                res = model.fit()
        return res


    def naive_changepoint_detection(self, plot=True, debug=False):
        """

        Parameters
        ----------
        plot  : bool
            Plot individual student's result while running it.
            The plots will be generated in 'result_dir/student_plots'
        debug : bool
            Print student index while running

        """
        st_time = datetime.now()

        if self.binary:
            mod = 'Bernoulli'
        else:
            mod = 'Poisson'
        print("Running "+ mod + " changepoint detection.")

        days_limit = self.n_days
        N_obs = self.n_days

        cp_to_try = range(self.padding, (days_limit - self.padding))
        segments_bic_mat = np.zeros((self.n_students,len(cp_to_try)))
        segments_ll_mat = np.zeros((self.n_students,len(cp_to_try)))
        segments_min_bic_mat = np.zeros(self.n_students)
        segments_min_bic_mat_idx = np.zeros(self.n_students)
        segments_max_ll_mat = np.zeros(self.n_students)
        no_segment_bic_mat = np.zeros(self.n_students)
        no_segment_ll_mat = np.zeros(self.n_students)
        detected_cp_arr = np.zeros(self.n_students)
        alpha_i_mat = np.zeros((self.n_students, 3))  # each row: segment1 | segment2

        for sidx in range(self.n_students):
            if debug:
                print(sidx)
            data_full = self.get_one_student_data(self.glm_mat, sidx)
            data = data_full[:days_limit, :]

            Ctotal_arr = []
            ll_arr = []
            glmres_arr1 = []
            glmres_arr2 = []

            for cp in cp_to_try:
                # First part
                res1 = self.run_glm(data[:cp, :], self.binary)
                # Second part
                res2 = self.run_glm(data[cp:, :], self.binary)

                k = 3  # 2 additional params introduced by adding a changepoint
                Ctot = -2*(res1.llf + res2.llf) + k * np.log(N_obs)
                Ctotal_arr.append(Ctot)
                ll_arr.append(res1.llf + res2.llf)
                glmres_arr1.append(res1)
                glmres_arr2.append(res2)

            segments_bic_mat[sidx,:] = Ctotal_arr
            segments_ll_mat[sidx,:] = ll_arr
            minbic = np.sort(Ctotal_arr)[ 0]
            min_idx = np.argsort(Ctotal_arr)[0]

            segments_min_bic_mat[sidx] = minbic
            segments_min_bic_mat_idx[sidx] = min_idx
            segments_max_ll_mat[sidx] = ll_arr[min_idx]

            detected_cp = min_idx + self.padding
            detected_cp_arr[sidx] = detected_cp

            res0 = self.run_glm(data, self.binary)
            no_segment_bic_mat[sidx] = -2 * res0.llf + 1 * np.log(N_obs)
            no_segment_ll_mat[sidx] = res0.llf

            # alpha_i's (estimated intercept coefficients for both models)
            glm0_ai = res0.params[0]
            glm1_ai= glmres_arr1[min_idx].params[0]
            glm2_ai= glmres_arr2[min_idx].params[0]
            alpha_i_mat[sidx, :] = [glm1_ai, glm2_ai, glm0_ai]

            if plot:
                if np.sum(data[:, 1]) == 0:
                    continue
                folder_path = os.path.join(self.result_dir, 'student_plots')
                check_path_and_make_dirs(folder_path + '/')
                filename = os.path.join(folder_path, "indiv_plot_student_"+ str(sidx) + ".pdf")
                self.plot_indiv_student_activity(sidx, filename, detected_cp, [glm1_ai, glm2_ai, glm0_ai],
                                                 segments_min_bic_mat[sidx], no_segment_bic_mat[sidx],
                                                 xlim=None, ylim=None, ylim2=None,
                                                 legend_loc='upper left', legend_loc2=None,
                                                 subtract_mean=False)
        self.detected_cp_arr = detected_cp_arr
        self.mcp_max_ll_mat = segments_max_ll_mat
        self.mcp_min_bic_mat = segments_min_bic_mat
        self.m0_bic_mat = no_segment_bic_mat
        self.m0_ll_mat = no_segment_ll_mat
        self.alpha_i_mat = alpha_i_mat
        self.better_w_cp_sidxs = np.where(self.m0_bic_mat > self.mcp_min_bic_mat)[0]  # better with segments
        self.better_wo_cp_idxs = np.where(self.m0_bic_mat <= self.mcp_min_bic_mat)[0]
        print('  ---> took ' + str((datetime.now() - st_time).seconds) + ' seconds.')


    def plot_indiv_student_activity(self, sidx, filename, detected_cp_loc, alpha_i,
                                    bic_with_cp, bic_wo_cp,
                                    xlim=None, ylim=None, ylim2=None,
                                    legend_loc='upper left', legend_loc2=None, subtract_mean=False):
        """
        Plotting individual student's activity. Used inside of the function 'naive_changepoint_detection()'
        """
        X_NT = self.rawdata
        X1t = X_NT[sidx, :]

        alpha_i1, alpha_i2, alpha_i0 = alpha_i
        avg_clicks = np.sum(X_NT, axis=0) / float(X_NT.shape[0])
        if self.binary:
            mu_t_hat = logit(avg_clicks + 0.0000001)
        else:
            mu_t_hat = np.log(avg_clicks + 0.0000001)

        ai1_arr = np.ones(detected_cp_loc) * (alpha_i1 + mu_t_hat[:detected_cp_loc])
        ai2_arr = np.ones(self.n_days - detected_cp_loc) * (alpha_i2 + mu_t_hat[detected_cp_loc:])
        ai0_arr = alpha_i0 + mu_t_hat
        logodds_wo_cp = ai0_arr
        logodds_w_cp = np.concatenate((ai1_arr, ai2_arr))

        fig, axs = plt.subplots(2, 1, figsize=(9, 8), gridspec_kw={'height_ratios': [2, 1]})
        ax_lam = axs[0]
        ax_lam.plot(logodds_wo_cp, color='gray', linewidth=3, alpha=0.6, label="M1, BIC=" + str(round(bic_wo_cp, 2)))
        ax_lam.plot(logodds_w_cp, color='#0C4EC9', linewidth=3, label="M2, BIC=" + str(round(bic_with_cp, 2)))
        ax_lam.axvline(x=detected_cp_loc, ymin=0, ymax=1, color='red', linewidth=3, alpha=0.8)
        ax_lam.grid(alpha=0.3)
        ax_lam.legend(loc=legend_loc, fontsize=14)
        if ylim is not None:
            ax_lam.set_ylim(ylim[0], ylim[1])
        if xlim is None:
            ax_lam.set_xlim(-1, self.n_days)
        else:
            ax_lam.set_xlim(xlim[0], xlim[1])
        ax_lam.set_ylabel(r"$\hat{\mu}_t + \hat{\alpha_i}$", fontsize=18)
        ax_lam.tick_params(labelsize=12)

        ax_raw = axs[1]
        if self.binary:
            X1t_ones = np.where(X1t == 1)[0]
            for xx in X1t_ones[:-1]:
                ax_raw.axvline(x=xx, ymin=0, ymax=1, color="black", linewidth=3, alpha=0.7)
            ax_raw.axvline(x=X1t_ones[-1], ymin=0, ymax=1, color="black", linewidth=3, alpha=0.7)
            ax_raw.set_yticks([])
        else:
            if subtract_mean:
                X_diff = X1t - avg_clicks
                ax_raw.bar(range(len(X_diff)), X_diff, width=0.6, color="black", alpha=0.7,
                           edgecolor='white', align='center')
                ax_raw.set_ylabel("$x_{it} - \hat{\lambda}_t$", fontsize=19)
            else:
                ax_raw.bar(range(len(X1t)), X1t, width=0.6, color="black", alpha=0.7,
                           edgecolor='white', align='center')
                ax_raw.set_ylabel(r"$x_{it}$", fontsize=19)
            ax_raw.grid(alpha=0.3)
            if ylim2 is not None:
                ax_raw.set_ylim(ylim2[0], ylim2[1])

        ax_raw.axvline(x=detected_cp_loc - 0.5, ymin=0, ymax=1, color='red', linewidth=3, alpha=0.8,
                       label="DETECTED CP")

        if legend_loc2 is None:
            legend_loc2 = legend_loc
        ax_raw.legend(loc=legend_loc2, fontsize=14)

        if xlim is None:
            ax_raw.set_xlim(-1, self.n_days)
        else:
            ax_raw.set_xlim(xlim[0], xlim[1])
        ax_raw.tick_params(labelsize=12)

        ax_raw.set_xlabel('DAYS', fontsize=15)
        plt.savefig(filename)
        plt.close()


if __name__ == "__main__":

    # Bernoulli model
    cp_bin = StudentChangePoint(data_fname='./test_data.csv', binary=True, result_dir='result_bin')
    cp_bin.naive_changepoint_detection(plot=True, debug=False) # Disable 'plot' (plot=False) for faster run.

    # Poisson model
    cp_cnts = StudentChangePoint(data_fname='./test_data.csv', binary=False, result_dir='result_cnts')
    cp_cnts.naive_changepoint_detection(plot=True, debug=False) # Disable 'plot' (plot=False) for faster run.
