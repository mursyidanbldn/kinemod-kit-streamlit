# KINEMOD-KIT: Integrated Kinetic Modeling and Sensitivity Analysis of a Multi-Stage POME Treatment System
# Created and Designed by: Rizky Mursyidan Baldan (2025)
# Objective: This script provides the backend logic for the KINEMOD-KIT Streamlit application.

# ==============================================================================
# 1. Import Libraries
# ==============================================================================
import json
import os
import warnings

import numpy as np
import pandas as pd
from SALib.analyze import morris as morris_analyzer
from SALib.analyze import sobol
from SALib.sample import morris as morris_sampler
from SALib.sample import saltelli
from scipy import stats
from scipy.optimize import curve_fit, differential_evolution, fsolve
from sklearn.metrics import mean_squared_error, r2_score

# --- Global Settings ---
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings(
    'ignore', '.*The iteration is not making good progress.*')
pd.options.display.float_format = '{:.2f}'.format

# ==============================================================================
# 2. Objective Functions for RBC Optimization
# ==============================================================================


def _calculate_penalized_error(y_true, predictions, days):
    valid_mask = ~np.isnan(predictions)
    if not np.any(valid_mask):
        return 1e7
    y_true_valid, predictions_valid, days_valid = y_true[
        valid_mask], predictions[valid_mask], days[valid_mask]
    rmse = np.sqrt(mean_squared_error(y_true_valid, predictions_valid))
    solver_penalty = (len(predictions) - np.sum(valid_mask)) * rmse * 10
    mean_penalty = 0.5 * \
        ((np.mean(y_true_valid) - np.mean(predictions_valid))**2)
    relative_error = np.abs(
        y_true_valid - predictions_valid) / (y_true_valid + 1e-9)
    outlier_penalty = max(0, (np.sum(relative_error > 0.35) - 4)) * 10000
    floor_mask = (y_true_valid > 500) & (predictions_valid < 100)
    floor_penalty = np.sum(
        (y_true_valid[floor_mask] - predictions_valid[floor_mask]) * 50)
    lower_bound, upper_bound = 1500, 4000
    window_mask = (days_valid >= 23) & (days_valid <= 30)
    lower_violations = lower_bound - \
        predictions_valid[(window_mask) & (predictions_valid < lower_bound)]
    upper_violations = predictions_valid[(window_mask) & (
        predictions_valid > upper_bound)] - upper_bound
    time_window_penalty = (np.sum(lower_violations) +
                           np.sum(upper_violations)) * 100
    return rmse + solver_penalty + mean_penalty + outlier_penalty + floor_penalty + time_window_penalty


def _rbc_objective_function_orig(kin_params_array, rbc_model_instance, param_keys, data):
    kin_params = dict(zip(param_keys, kin_params_array))
    predictions = data.apply(lambda row: rbc_model_instance.predict({
        'So': row['COD_Filt_Eff'], 'HRT_days': row['HRT_hours'] / 24.0, 'Xa': row['Xa'], 'Xs': row['Xs']
    }, kin_params), axis=1)
    return _calculate_penalized_error(data['COD_Final'].values, predictions.values, data['Day'].values)


def _rbc_objective_function_ph(kin_params_array, rbc_model_instance, param_keys, data):
    kin_params = dict(zip(param_keys, kin_params_array))
    predictions = data.apply(lambda row: rbc_model_instance.predict({
        'So': row['COD_Filt_Eff'], 'HRT_days': row['HRT_hours'] / 24.0, 'Xa': row['Xa'], 'Xs': row['Xs'], 'pH': row['pH_Eff_RBC']
    }, kin_params), axis=1)
    return _calculate_penalized_error(data['COD_Final'].values, predictions.values, data['Day'].values)

# ==============================================================================
# 3. Model Classes for Each Treatment Stage
# ==============================================================================


class UASBModel:
    def predict(self, x_data, params):
        U_max, K_B, K_I = params['U_max'], params['K_B'], params['K_I']
        OLR, VFA_ALK_Ratio = x_data
        epsilon = 1e-9
        inhibition = 1 / (1 + (VFA_ALK_Ratio / (K_I + epsilon)))
        base_rate = (U_max * OLR) / (K_B + OLR + epsilon)
        return base_rate * inhibition

    def solve_parameters(self, df_data, bounds):
        def fit_model(x_data, U_max, K_B, K_I):
            return self.predict(x_data, {'U_max': U_max, 'K_B': K_B, 'K_I': K_I})
        x_data = (df_data['OLR_UASB'], df_data['VFA_ALK_Ratio'])
        y_data = df_data['SRR_UASB']
        try:
            popt, _ = curve_fit(fit_model, x_data, y_data, p0=[np.mean(b) for b in bounds],
                                bounds=([b[0] for b in bounds], [b[1] for b in bounds]), method='trf', max_nfev=8000)
            preds = fit_model(x_data, *popt)
            r2, rmse = r2_score(y_data, preds), np.sqrt(
                mean_squared_error(y_data, preds))
            return {'U_max': popt[0], 'K_B': popt[1], 'K_I': popt[2]}, True, f"R²={r2:.2f}, RMSE={rmse:.2f}"
        except (RuntimeError, ValueError):
            return {}, False, "Fit Error"


class FilterModel:
    def predict(self, x_data, params):
        R_cod_tss, k_ads = params['R_cod_tss'], params['k_ads']
        TSS_in, TSS_out, S_in = x_data
        pCOD_in = TSS_in * R_cod_tss
        sCOD_in = np.maximum(0, S_in - pCOD_in)
        pCOD_removed_filt = (TSS_in - TSS_out) * R_cod_tss
        sCOD_removed_ads = k_ads * sCOD_in
        return pCOD_removed_filt + sCOD_removed_ads

    def solve_parameters(self, df_data, cod_uasb_pred, bounds):
        def fit_model(x_data, R_cod_tss, k_ads):
            return self.predict(x_data, {'R_cod_tss': R_cod_tss, 'k_ads': k_ads})
        x_data = (df_data['TSS_UASB_Eff'],
                  df_data['TSS_Filt_Eff'], cod_uasb_pred)
        y_data = cod_uasb_pred - df_data['COD_Filt_Eff']
        try:
            popt, _ = curve_fit(fit_model, x_data, y_data, p0=[np.mean(b) for b in bounds],
                                bounds=([b[0] for b in bounds], [b[1] for b in bounds]), method='trf', max_nfev=5000)
            preds = fit_model(x_data, *popt)
            r2, rmse = r2_score(y_data, preds), np.sqrt(
                mean_squared_error(y_data, preds))
            return {'R_cod_tss': popt[0], 'k_ads': popt[1]}, True, f"R²={r2:.2f}, RMSE={rmse:.2f}"
        except (RuntimeError, ValueError):
            return {}, False, "Fit Error"


class RBCModel:
    def __init__(self, V, Ab): self.V, self.Ab, self.epsilon = V, Ab, 1e-9

    def _model_equation_for_root(self, Se, op_params, kin_params):
        if Se <= 0:
            return 1e6
        So, Xa, Xs, theta_i = op_params['So'], op_params['Xa'], op_params['Xs'], op_params['HRT_days']
        mu_a = ((kin_params['umxa'] * Se) / (kin_params['Ku'] + Se + self.epsilon)) * (
            (0.5 * kin_params['Ko'] + kin_params['O']) / (kin_params['Ko'] + kin_params['O'] + self.epsilon))
        mu_s = (kin_params['umxs'] * Se) / \
            (kin_params['Ks'] + Se + self.epsilon)
        return ((So - Se) / (theta_i + self.epsilon)) - ((self.Ab * mu_a * Xa) / (kin_params['Ya'] * self.V + self.epsilon)) - ((mu_s * Xs) / (kin_params['Ys'] + self.epsilon))

    def predict(self, op_params, kin_params):
        for guess in [op_params['So'] * 0.5, op_params['So'] * 0.1, 1.0]:
            try:
                sol, _, ier, _ = fsolve(self._model_equation_for_root, x0=guess, args=(
                    op_params, kin_params), xtol=1e-6, full_output=True)
                if ier == 1 and sol[0] > 1.0:
                    return sol[0]
            except Exception:
                continue
        return np.nan

    def solve_parameters(self, df_data, bounds):
        param_keys = ['Ko', 'Ks', 'Ku', 'O', 'Ya', 'Ys', 'umxa', 'umxs']
        try:
            result = differential_evolution(_rbc_objective_function_orig, bounds=bounds, args=(
                self, param_keys, df_data), strategy='best1bin', maxiter=250, popsize=25, tol=0.01, mutation=(0.5, 1), recombination=0.7, disp=False, workers=-1)
            kin_params = dict(zip(param_keys, result.x))
            preds = df_data.apply(lambda row: self.predict(
                {'So': row['COD_Filt_Eff'], 'HRT_days': row['HRT_hours'] / 24.0, 'Xa': row['Xa'], 'Xs': row['Xs']}, kin_params), axis=1)
            valid_mask = ~preds.isna()
            r2, rmse = r2_score(df_data['COD_Final'][valid_mask], preds[valid_mask]), np.sqrt(
                mean_squared_error(df_data['COD_Final'][valid_mask], preds[valid_mask]))
            return kin_params, True, f"R²={r2:.2f}, RMSE={rmse:.2f}"
        except (RuntimeError, ValueError):
            return {}, False, "Fit Error"


class RBCModelWithpH(RBCModel):
    def _calculate_ph_inhibition(self, pH, kin_params):
        pH_min, pH_max = kin_params['pH_min'], kin_params['pH_max']
        if pH_min >= pH_max:
            return 0
        numerator = 1 + 2 * (10**(0.5 * (pH_min - pH_max)))
        denominator = 1 + 10**(pH - pH_max) + 10**(pH_min - pH)
        return numerator / denominator if denominator > self.epsilon else 0

    def _model_equation_for_root(self, Se, op_params, kin_params):
        if Se <= 0:
            return 1e6
        So, Xa, Xs, theta_i, pH = op_params['So'], op_params[
            'Xa'], op_params['Xs'], op_params['HRT_days'], op_params['pH']
        tau_pH = self._calculate_ph_inhibition(pH, kin_params)
        mu_a = tau_pH * ((kin_params['umxa'] * Se) / (kin_params['Ku'] + Se + self.epsilon)) * (
            (0.5 * kin_params['Ko'] + kin_params['O']) / (kin_params['Ko'] + kin_params['O'] + self.epsilon))
        mu_s = tau_pH * (kin_params['umxs'] * Se) / \
            (kin_params['Ks'] + Se + self.epsilon)
        return ((So - Se) / (theta_i + self.epsilon)) - ((self.Ab * mu_a * Xa) / (kin_params['Ya'] * self.V + self.epsilon)) - ((mu_s * Xs) / (kin_params['Ys'] + self.epsilon))

    def solve_parameters(self, df_data, bounds):
        param_keys = ['Ko', 'Ks', 'Ku', 'O', 'Ya',
                      'Ys', 'umxa', 'umxs', 'pH_min', 'pH_max']
        try:
            result = differential_evolution(_rbc_objective_function_ph, bounds=bounds, args=(
                self, param_keys, df_data), strategy='best1bin', maxiter=250, popsize=25, tol=0.01, mutation=(0.5, 1), recombination=0.7, disp=False, workers=-1)
            kin_params = dict(zip(param_keys, result.x))
            preds = df_data.apply(lambda row: self.predict(
                {'So': row['COD_Filt_Eff'], 'HRT_days': row['HRT_hours'] / 24.0, 'Xa': row['Xa'], 'Xs': row['Xs'], 'pH': row['pH_Eff_RBC']}, kin_params), axis=1)
            valid_mask = ~preds.isna()
            r2, rmse = r2_score(df_data['COD_Final'][valid_mask], preds[valid_mask]), np.sqrt(
                mean_squared_error(df_data['COD_Final'][valid_mask], preds[valid_mask]))
            return kin_params, True, f"R²={r2:.2f}, RMSE={rmse:.2f}"
        except (RuntimeError, ValueError):
            return {}, False, "Fit Error"

# ==============================================================================
# 4. Main Integrated Reactor Class
# ==============================================================================


class IntegratedReactor:
    """Orchestrates the entire modeling and analysis workflow for the Streamlit app."""

    def __init__(self, data_file, reactor_constants):
        self.data_file = data_file
        self.df = None
        self.steady_df = None
        self.params_uasb, self.params_filter, self.params_rbc_orig, self.params_rbc_ph = {}, {}, {}, {}
        self.uasb_model = UASBModel()
        self.filter_model = FilterModel()
        self.rbc_model_orig = RBCModel(
            V=reactor_constants['V_RBC'], Ab=reactor_constants['A_RBC'])
        self.rbc_model_ph = RBCModelWithpH(
            V=reactor_constants['V_RBC'], Ab=reactor_constants['A_RBC'])
        self.data_loaded, self.params_estimated, self.predictions_run = False, False, False

    def load_and_prepare_data(self):
        try:
            # The data_file can now be a file path or an in-memory buffer
            self.df = pd.read_csv(self.data_file, sep=';')
        except FileNotFoundError:
            return False

        epsilon = 1e-9
        self.df['OLR_UASB'] = self.df['COD_in'] / \
            (self.df['HRT_UASB'] + epsilon)
        self.df['VFA_ALK_Ratio'] = self.df['VFA_Eff'] / \
            (self.df['ALK_Eff'] + epsilon)
        self.df['SRR_UASB'] = (
            self.df['COD_in'] - self.df['COD_UASB_Eff']) / (self.df['HRT_UASB'] + epsilon)
        self.steady_df = self.df[self.df['Steady_State'] == 1].copy().replace(
            [np.inf, -np.inf], np.nan).dropna()
        self.data_loaded = True
        return True

    def run_parameter_estimation(self, optimization_bounds):
        # Stage 1: UASB
        p_uasb, s_uasb, _ = self.uasb_model.solve_parameters(
            self.steady_df, optimization_bounds['UASB'])
        self.params_uasb = p_uasb if s_uasb else {}

        # Stage 2: Filter
        if s_uasb:
            cod_uasb_pred = self.steady_df['COD_in'] - self.uasb_model.predict(
                (self.steady_df['OLR_UASB'], self.steady_df['VFA_ALK_Ratio']), self.params_uasb) * self.steady_df['HRT_UASB']
            p_filt, s_filt, _ = self.filter_model.solve_parameters(
                self.steady_df, cod_uasb_pred, optimization_bounds['Filter'])
            self.params_filter = p_filt if s_filt else {}

        # Stage 3 & 4: RBC
        rbc_train_df = self.steady_df[~self.steady_df['Day'].isin(
            [4, 5, 10, 22, 24])].copy()
        p_rbc_o, s_rbc_o, _ = self.rbc_model_orig.solve_parameters(
            rbc_train_df, optimization_bounds['RBC'])
        self.params_rbc_orig = p_rbc_o if s_rbc_o else {}

        p_rbc_ph, s_rbc_ph, _ = self.rbc_model_ph.solve_parameters(
            rbc_train_df, optimization_bounds['RBC_pH'])
        self.params_rbc_ph = p_rbc_ph if s_rbc_ph else {}

        self.params_estimated = True
        self.predictions_run = False
        return True

    def save_parameters(self, filename="calibrated_params.json"):
        if not self.params_estimated:
            return
        params_to_save = {
            "uasb": self.params_uasb, "filter": self.params_filter,
            "rbc_orig": self.params_rbc_orig, "rbc_ph": self.params_rbc_ph
        }
        with open(filename, 'w') as f:
            json.dump(params_to_save, f, indent=4)

    def load_parameters(self, filename="calibrated_params.json"):
        if not os.path.exists(filename):
            return False
        with open(filename, 'r') as f:
            params_data = json.load(f)
        self.params_uasb = params_data.get("uasb", {})
        self.params_filter = params_data.get("filter", {})
        self.params_rbc_orig = params_data.get("rbc_orig", {})
        self.params_rbc_ph = params_data.get("rbc_ph", {})
        self.params_estimated = True
        self.predictions_run = False
        return True

    def run_full_predictions(self):
        if not self.params_estimated:
            return False
        srr_pred = self.uasb_model.predict(
            (self.df['OLR_UASB'], self.df['VFA_ALK_Ratio']), self.params_uasb)
        self.df['COD_UASB_Pred'] = self.df['COD_in'] - \
            srr_pred * self.df['HRT_UASB']
        cod_removed_filt = self.filter_model.predict(
            (self.df['TSS_UASB_Eff'], self.df['TSS_Filt_Eff'], self.df['COD_UASB_Pred']), self.params_filter)
        self.df['COD_Filt_Pred'] = self.df['COD_UASB_Pred'] - \
            cod_removed_filt
        if self.params_rbc_orig:
            self.df['COD_Final_Pred_Orig'] = self.df.apply(lambda row: self.rbc_model_orig.predict(
                {'So': row['COD_Filt_Pred'], 'HRT_days': row['HRT_hours'] / 24.0, 'Xa': row['Xa'], 'Xs': row['Xs']}, self.params_rbc_orig), axis=1)
        if self.params_rbc_ph:
            self.df['COD_Final_Pred_pH'] = self.df.apply(lambda row: self.rbc_model_ph.predict(
                {'So': row['COD_Filt_Pred'], 'HRT_days': row['HRT_hours'] / 24.0, 'Xa': row['Xa'], 'Xs': row['Xs'], 'pH': row['pH_Eff_RBC']}, self.params_rbc_ph), axis=1)
        self.predictions_run = True
        return True

    def _gsa_model_wrapper(self, param_vector, param_keys, df_subset, rbc_model='orig'):
        temp_params = dict(zip(param_keys, param_vector))
        df = df_subset.copy()
        try:
            if any(k in self.params_uasb for k in temp_params):
                srr_pred = self.uasb_model.predict(
                    (df['OLR_UASB'], df['VFA_ALK_Ratio']), {k: temp_params.get(k, self.params_uasb[k]) for k in self.params_uasb})
                df['COD_UASB_Pred'] = df['COD_in'] - srr_pred * df['HRT_UASB']
            else:
                df['COD_UASB_Pred'] = df['COD_UASB_Eff']

            if any(k in self.params_filter for k in temp_params):
                cod_rem_filt = self.filter_model.predict(
                    (df['TSS_UASB_Eff'], df['TSS_Filt_Eff'], df['COD_UASB_Pred']), {k: temp_params.get(k, self.params_filter[k]) for k in self.params_filter})
                df['COD_Filt_Pred'] = df['COD_UASB_Pred'] - cod_rem_filt
            else:
                df['COD_Filt_Pred'] = df['COD_Filt_Eff']

            model = self.rbc_model_ph if rbc_model == 'ph' else self.rbc_model_orig
            rbc_base_params = self.params_rbc_ph if rbc_model == 'ph' else self.params_rbc_orig
            rbc_params_for_run = {k: temp_params.get(
                k, rbc_base_params[k]) for k in rbc_base_params}

            def predict_row(row):
                op_params = {
                    'So': row['COD_Filt_Pred'], 'HRT_days': row['HRT_hours']/24.0, 'Xa': row['Xa'], 'Xs': row['Xs']}
                if rbc_model == 'ph':
                    op_params['pH'] = row['pH_Eff_RBC']
                if any(pd.isna(v) for v in op_params.values()):
                    return np.nan
                return model.predict(op_params, rbc_params_for_run)
            preds = df.apply(predict_row, axis=1)
            return np.nanmean(preds) if not preds.isnull().all() else np.nan
        except Exception:
            return np.nan

    def _execute_gsa(self, params, param_keys, rbc_model_key, N_morris=100, N_sobol=512):
        gsa_bounds = [[params[k] * 0.75, params[k] * 1.25] if params[k]
                      > 0 else [params[k]-0.1, params[k]+0.1] for k in param_keys]
        problem = {'num_vars': len(param_keys),
                   'names': param_keys, 'bounds': gsa_bounds}
        param_values_morris = morris_sampler.sample(
            problem, N=N_morris, num_levels=4)
        param_values_sobol = saltelli.sample(
            problem, N=N_sobol, calc_second_order=True)

        Y_morris = np.array([self._gsa_model_wrapper(
            p, param_keys, self.steady_df, rbc_model_key) for p in param_values_morris])
        Y_sobol = np.array([self._gsa_model_wrapper(
            p, param_keys, self.steady_df, rbc_model_key) for p in param_values_sobol])

        if np.isnan(Y_morris).sum() > len(Y_morris) * 0.9 or np.isnan(Y_sobol).sum() > len(Y_sobol) * 0.9:
            return None, None, None

        Y_morris, Y_sobol = np.nan_to_num(Y_morris, nan=np.nanmean(
            Y_morris)), np.nan_to_num(Y_sobol, nan=np.nanmean(Y_sobol))

        Mi = morris_analyzer.analyze(
            problem, param_values_morris, Y_morris, conf_level=0.95)
        Si = sobol.analyze(problem, Y_sobol, calc_second_order=True)

        return Mi, Si, problem

    def _execute_mc(self, params, param_keys, rbc_model_key, N=1000):
        samples = {key: np.random.uniform(
            params[key]*0.75, params[key]*1.25, N) for key in param_keys}
        samples_df = pd.DataFrame(samples)

        outputs = [self._gsa_model_wrapper(
            samples_df.iloc[i].values, param_keys, self.steady_df, rbc_model_key) for i in range(N)]

        samples_df['Output'] = outputs

        if samples_df['Output'].notna().sum() < N * 0.1:
            return None, None

        samples_df.dropna(inplace=True)
        correlations = {key: stats.spearmanr(samples_df[key], samples_df['Output'])[
            0] for key in param_keys}
        results_df = pd.DataFrame(list(correlations.items()), columns=[
                                  'Parameter', 'Spearman_Correlation'])

        return results_df, samples_df[param_keys]

    def run_predictions_with_new_params(self, df_original, params_uasb, params_filter, params_rbc_orig, params_rbc_ph):
        """Re-runs only the prediction steps with a new set of parameters."""
        df = df_original.copy()

        # Recalculate predictions based on the new parameters
        srr_pred = self.uasb_model.predict(
            (df['OLR_UASB'], df['VFA_ALK_Ratio']), params_uasb)
        df['COD_UASB_Pred'] = df['COD_in'] - srr_pred * df['HRT_UASB']

        cod_removed_filt = self.filter_model.predict(
            (df['TSS_UASB_Eff'], df['TSS_Filt_Eff'], df['COD_UASB_Pred']), params_filter)
        df['COD_Filt_Pred'] = df['COD_UASB_Pred'] - cod_removed_filt

        if params_rbc_orig:
            df['COD_Final_Pred_Orig'] = df.apply(lambda row: self.rbc_model_orig.predict(
                {'So': row['COD_Filt_Pred'], 'HRT_days': row['HRT_hours'] / 24.0, 'Xa': row['Xa'], 'Xs': row['Xs']}, params_rbc_orig), axis=1)
        if params_rbc_ph:
            df['COD_Final_Pred_pH'] = df.apply(lambda row: self.rbc_model_ph.predict(
                {'So': row['COD_Filt_Pred'], 'HRT_days': row['HRT_hours'] / 24.0, 'Xa': row['Xa'], 'Xs': row['Xs'], 'pH': row['pH_Eff_RBC']}, params_rbc_ph), axis=1)

        return df
