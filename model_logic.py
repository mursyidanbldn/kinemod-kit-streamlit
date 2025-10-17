# KINEMOD-KIT: Integrated Kinetic Modeling and Sensitivity Analysis of a Multi-Stage POME Treatment System
# Created and Designed by: Rizky Mursyidan Baldan (2025)
# Objective: This script presents a complete, object-oriented workflow for analyzing the performance
# of a three-stage POME treatment reactor.
#
# Main Features:
# 1. Object-Oriented Design: Each treatment unit (UASB, Filter, RBC) is a separate class.
# 2. Robust Parameter Estimation: Uses 'curve_fit' for non-linear least squares optimization.
# 3. Comprehensive Validation: Evaluates each stage using R², RMSE, and statistical ANOVA tests.
# 4. Dual Sensitivity Analysis: Employs both GSA (Morris/Sobol) and Monte Carlo methods.
# 5. Interactive UI: A full command-line interface for a dynamic user experience.
# 6. Beautiful Visualizations: All plots are now rendered using Seaborn for a clean and professional look.

# ==============================================================================
# 1. Import Libraries
# ==============================================================================
from rich.align import Align
from rich.text import Text
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.table import Table
from rich.console import Console
import json
import time
import warnings
import os
import glob
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit, fsolve, differential_evolution
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error
from SALib.sample import morris as morris_sampler, saltelli
from SALib.analyze import morris as morris_analyzer, sobol
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
# Use TkAgg backend for better compatibility in various environments (especially for interactive plots)
matplotlib.use('Agg')


# --- Initialize Rich Console ---
console = Console()

# ==============================================================================
# 2. Futuristic Terminal Output Class
# ==============================================================================


class TermFX:

    @staticmethod
    def print_header(text):
        console.print(Panel(Align.center(f"[bold white on magenta] {text.upper()} [/]"),
                            border_style="magenta", expand=False))

    @staticmethod
    def print_subheader(text):
        console.print(f"\n[bold cyan]» {text}[/bold cyan]")
        console.print(f"[dark_cyan]{'─' * (len(text) + 2)}[/]")

    @staticmethod
    def print_success(text):
        console.print(f"[bold green]✓[/bold green] {text}")

    @staticmethod
    def print_warning(text):
        console.print(f"[bold yellow]![/bold yellow] {text}")

    @staticmethod
    def print_error(text):
        console.print(f"[bold red]✗[/bold red] {text}")

    @staticmethod
    def animated_startup_banner():
        """
        Displays a futuristic startup banner with a multi-stage animation:
        1. A "scanner" sweeps across to reveal the text.
        2. A subtitle is typed out.
        3. The border pulses to signal completion.
        """

        banner_lines = [
            r"██╗  ██╗██╗███╗   ██╗███████╗███╗   ███╗ ██████╗ ██████╗       ██╗  ██╗██╗████████╗ ",
            r"██║ ██╔╝██║████╗  ██║██╔════╝████╗ ████║██╔═══██╗██╔══██╗      ██║ ██╔╝██║╚══██╔══╝ ",
            r"█████╔╝ ██║██╔██╗ ██║█████╗  ██╔████╔██║██║   ██║██║  ██║█████╗█████╔╝ ██║   ██║    ",
            r"██╔═██╗ ██║██║╚██╗██║██╔══╝  ██║╚██╔╝██║██║   ██║██║  ██║╚════╝██╔═██╗ ██║   ██║    ",
            r"██║  ██╗██║██║ ╚████║███████╗██║ ╚═╝ ██║╚██████╔╝██████╔╝      ██║  ██╗██║   ██║    ",
            r"╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚══════╝╚═╝     ╚═╝ ╚═════╝ ╚═════╝       ╚═╝  ╚═╝╚═╝   ╚═╝    "
        ]
        subtitle = Text("UASB-FILTRATION-RBC REACTOR KINETIC MODELING KIT BY RIZKY MURSYIDAN BALDAN",
                        style="bold magenta", justify="left", no_wrap=True, overflow="ellipsis", end="\n")

        # --- Animation Settings ---
        off_style = "grey23"
        on_style = "bold cyan"
        scan_style = "bold black on bright_cyan"
        scanner_width = 10
        width = len(banner_lines[0])

        # Create the panel and text objects that will be updated
        combined_text = Text(justify="center")
        panel = Panel(Align.center(combined_text),
                      border_style="bold purple", padding=(1, 2))

        with Live(panel, console=console, screen=False, auto_refresh=False, vertical_overflow="visible") as live:
            # 1. Scanner Animation
            for i in range(width + scanner_width + 1):
                new_text = Text(justify="center")
                for line in banner_lines:
                    styled_line = Text()
                    for j, char in enumerate(line):
                        if i > j and (i - scanner_width) < j:
                            # Character is inside the scanner
                            style = scan_style
                        elif (i - scanner_width) >= j:
                            # Character is behind the scanner (powered on)
                            style = on_style
                        else:
                            # Character is ahead of the scanner (powered off)
                            style = off_style
                        styled_line.append(char, style=style)
                    new_text.append(styled_line)
                    new_text.append("\n")

                panel.renderable = Align.center(new_text)
                live.update(panel, refresh=True)
                time.sleep(0.005)

            # 2. Animate the subtitle with a typing effect
            banner_renderable = Text.from_markup(
                "\n".join(f"[{on_style}]{line}[/]" for line in banner_lines),
                justify="left"
            )

            # Start with empty subtitle
            subtitle_renderable = Text("", justify="left")

            # Typing animation for subtitle
            for char in subtitle:
                subtitle_renderable.append(char)
                combined_renderable = Text("\n\n").join([
                    banner_renderable,
                    Text.from_markup(
                        f"[bold magenta]{subtitle_renderable}[/]", justify="left")
                ])
                panel.renderable = Align.center(combined_renderable)
                live.update(panel, refresh=True)
                time.sleep(0.005)

            # Final pulse effect (flash border color)
            final_combined = Text("\n\n").join([
                banner_renderable,
                Text.from_markup(
                    f"[bold magenta]{subtitle}[/]", justify="right")
            ])
            for _ in range(2):
                panel = Panel(Align.center(final_combined),
                              border_style="bold bright_magenta", padding=(1, 2))
                live.update(panel, refresh=True)
                time.sleep(0.2)

                panel = Panel(Align.center(final_combined),
                              border_style="bold purple", padding=(1, 2))
                live.update(panel, refresh=True)
                time.sleep(0.2)

            # 3. Final "pulse" effect
            for _ in range(2):
                panel.border_style = "bold bright_magenta"
                live.update(panel, refresh=True)
                time.sleep(0.2)
                panel.border_style = "bold purple"
                live.update(panel, refresh=True)
                time.sleep(0.2)


# --- REFACTOR: Set standard Seaborn theme for all plots ---
sns.set_theme(style="whitegrid")
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings(
    'ignore', '.*The iteration is not making good progress.*')
pd.options.display.float_format = '{:.2f}'.format

# ==============================================================================
# 3. Objective Functions for RBC Optimization (No Changes)
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
# 4. Model Classes for Each Treatment Stage (No Changes)
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
# 5. Main Integrated Reactor Class
# ==============================================================================


class IntegratedReactor:
    """Orchestrates the entire modeling and analysis workflow."""

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
        TermFX.print_header("Data Ingestion & Pre-processing")
        if not os.path.exists(self.data_file):
            TermFX.print_error(f"Data file '{self.data_file}' not found.")
            return False
        with console.status(f"[bold cyan]Loading data stream from '{self.data_file}'...[/]"):
            time.sleep(1)
            self.df = pd.read_csv(self.data_file, sep=';')
        TermFX.print_success(
            f"Data stream loaded: {self.df.shape[0]} records, {self.df.shape[1]} columns.")

        epsilon = 1e-9
        self.df['OLR_UASB'] = self.df['COD_in'] / \
            (self.df['HRT_UASB'] + epsilon)
        self.df['VFA_ALK_Ratio'] = self.df['VFA_Eff'] / \
            (self.df['ALK_Eff'] + epsilon)
        self.df['SRR_UASB'] = (
            self.df['COD_in'] - self.df['COD_UASB_Eff']) / (self.df['HRT_UASB'] + epsilon)
        self.steady_df = self.df[self.df['Steady_State'] == 1].copy().replace(
            [np.inf, -np.inf], np.nan).dropna()
        TermFX.print_success(
            f"Pre-processing complete. Isolated {len(self.steady_df)} steady-state vectors.")
        self.data_loaded = True
        return True

    def run_parameter_estimation(self, optimization_bounds):
        TermFX.print_header("Parameter Estimation Protocol")
        statuses = {"UASB": {"status": "⏳ Pending...", "metrics": ""}, "Filter": {"status": "⏳ Pending...", "metrics": ""}, "RBC v1.0": {
            "status": "⏳ Pending...", "metrics": ""}, "RBC v1.1 (pH)": {"status": "⏳ Pending...", "metrics": ""}}

        def generate_status_table(s):
            table = Table(title="Live Calibration Status",
                          style="purple", title_style="bold magenta", box=None)
            table.add_column("Stage", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Metrics", style="yellow")
            for stage, info in s.items():
                table.add_row(stage, info["status"], info["metrics"])
            return table

        with Live(generate_status_table(statuses), console=console, screen=False, refresh_per_second=10) as live:
            # Stage 1: UASB
            statuses["UASB"]["status"] = "[yellow]Calibrating...[/]"
            live.update(generate_status_table(statuses))
            p_uasb, s_uasb, m_uasb = self.uasb_model.solve_parameters(
                self.steady_df, optimization_bounds['UASB'])
            self.params_uasb = p_uasb if s_uasb else {}
            statuses["UASB"] = {
                "status": "[bold green]✓ Done[/]" if s_uasb else "[bold red]✗ Failed[/]", "metrics": m_uasb}
            live.update(generate_status_table(statuses))

            # Stage 2: Filter
            if s_uasb:
                statuses["Filter"]["status"] = "[yellow]Calibrating...[/]"
                live.update(generate_status_table(statuses))
                cod_uasb_pred = self.steady_df['COD_in'] - self.uasb_model.predict(
                    (self.steady_df['OLR_UASB'], self.steady_df['VFA_ALK_Ratio']), self.params_uasb) * self.steady_df['HRT_UASB']
                p_filt, s_filt, m_filt = self.filter_model.solve_parameters(
                    self.steady_df, cod_uasb_pred, optimization_bounds['Filter'])
                self.params_filter = p_filt if s_filt else {}
                statuses["Filter"] = {
                    "status": "[bold green]✓ Done[/]" if s_filt else "[bold red]✗ Failed[/]", "metrics": m_filt}
                live.update(generate_status_table(statuses))
            else:
                statuses["Filter"]["status"] = "[grey70]Skipped[/]"
                live.update(generate_status_table(statuses))

            # Stage 3 & 4: RBC
            rbc_train_df = self.steady_df[~self.steady_df['Day'].isin(
                [4, 5, 10, 22, 24])].copy()
            statuses["RBC v1.0"]["status"] = "[yellow]Calibrating...[/]"
            live.update(generate_status_table(statuses))
            p_rbc_o, s_rbc_o, m_rbc_o = self.rbc_model_orig.solve_parameters(
                rbc_train_df, optimization_bounds['RBC'])
            self.params_rbc_orig = p_rbc_o if s_rbc_o else {}
            statuses["RBC v1.0"] = {
                "status": "[bold green]✓ Done[/]" if s_rbc_o else "[bold red]✗ Failed[/]", "metrics": m_rbc_o}
            live.update(generate_status_table(statuses))

            statuses["RBC v1.1 (pH)"]["status"] = "[yellow]Calibrating...[/]"
            live.update(generate_status_table(statuses))
            p_rbc_ph, s_rbc_ph, m_rbc_ph = self.rbc_model_ph.solve_parameters(
                rbc_train_df, optimization_bounds['RBC_pH'])
            self.params_rbc_ph = p_rbc_ph if s_rbc_ph else {}
            statuses["RBC v1.1 (pH)"] = {
                "status": "[bold green]✓ Done[/]" if s_rbc_ph else "[bold red]✗ Failed[/]", "metrics": m_rbc_ph}
            live.update(generate_status_table(statuses))

        self.params_estimated = True
        self.predictions_run = False
        TermFX.print_success("Parameter estimation protocol complete.")
        return True

    def save_parameters(self, filename="calibrated_params.json"):
        if not self.params_estimated:
            TermFX.print_error("No parameters have been estimated to save.")
            return
        params_to_save = {
            "uasb": self.params_uasb, "filter": self.params_filter,
            "rbc_orig": self.params_rbc_orig, "rbc_ph": self.params_rbc_ph
        }
        with open(filename, 'w') as f:
            json.dump(params_to_save, f, indent=4)
        TermFX.print_success(f"Parameters successfully saved to '{filename}'.")

    def load_parameters(self, filename="calibrated_params.json"):
        if not os.path.exists(filename):
            TermFX.print_error(f"Parameter file '{filename}' not found.")
            return False
        with open(filename, 'r') as f:
            params_data = json.load(f)
        self.params_uasb = params_data.get("uasb", {})
        self.params_filter = params_data.get("filter", {})
        self.params_rbc_orig = params_data.get("rbc_orig", {})
        self.params_rbc_ph = params_data.get("rbc_ph", {})
        self.params_estimated = True
        self.predictions_run = False
        TermFX.print_success(
            f"Parameters successfully loaded from '{filename}'.")
        self.display_estimated_parameters()
        return True

    def run_full_predictions(self):
        if not self.params_estimated:
            TermFX.print_error(
                "Cannot run predictions without estimated parameters.")
            return False
        with console.status("[bold cyan]Running full system simulation...[/]"):
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
            time.sleep(1)
        self.predictions_run = True
        TermFX.print_success("Full system simulation is complete.")
        return True

    def display_performance_and_equations(self):
        if not self.params_estimated:
            TermFX.print_warning("Parameters have not been estimated yet.")
            return
        if not self.predictions_run:
            self.run_full_predictions()
        self.evaluate_performance_as_table()
        self.display_final_equations_as_table()

    def display_estimated_parameters(self):
        if not self.params_estimated:
            TermFX.print_warning(
                "No parameters available. Please run or load an estimation first.")
            return
        TermFX.print_header("Calibrated Parameter Matrix")
        table = Table(title="[bold]Calibrated Kinetic Parameters[/bold]", style="purple",
                      title_style="bold magenta", show_header=True, header_style="bold cyan")
        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Parameter", style="green")
        table.add_column("Calibrated Value", justify="right", style="yellow")
        for k, v in self.params_uasb.items():
            table.add_row("UASB", k, f"{v:.4f}")
        for k, v in self.params_filter.items():
            table.add_row("Filter", k, f"{v:.4f}")
        if (self.params_uasb or self.params_filter) and (self.params_rbc_orig or self.params_rbc_ph):
            table.add_section()
        for k, v in self.params_rbc_orig.items():
            table.add_row("RBC Original", k, f"{v:.4f}")
        if self.params_rbc_orig and self.params_rbc_ph:
            table.add_section()
        for k, v in sorted(self.params_rbc_ph.items()):
            table.add_row("RBC pH-Modified", k, f"{v:.4f}")
        console.print(table)

    def evaluate_performance_as_table(self):
        TermFX.print_header("Model Goodness-of-Fit Evaluation")
        table = Table(title="[bold]Performance Metrics[/bold]",
                      style="purple", title_style="bold magenta")
        table.add_column("Model Stage", style="cyan")
        table.add_column("R-squared", style="yellow")
        table.add_column("RMSE (mg/L)", style="yellow")
        table.add_column("ANOVA p-value", style="yellow")
        table.add_column("Significance", style="green")
        stages = {"UASB": ("COD_UASB_Eff", "COD_UASB_Pred"), "Filter": ("COD_Filt_Eff", "COD_Filt_Pred"), "RBC (Original)": (
            "COD_Final", "COD_Final_Pred_Orig"), "RBC (pH Mod)": ("COD_Final", "COD_Final_Pred_pH")}
        for name, (actual_col, pred_col) in stages.items():
            if pred_col not in self.df.columns or self.df[pred_col].isnull().all():
                continue
            eval_df = self.df[[actual_col, pred_col]].copy().dropna()
            if len(eval_df) < 2:
                continue
            y_true, y_pred = eval_df[actual_col], eval_df[pred_col]
            r2, rmse = r2_score(y_true, y_pred), np.sqrt(
                mean_squared_error(y_true, y_pred))
            _, p_value = stats.f_oneway(y_true, y_pred)
            sig_text = "[red]Different (p<0.05)[/red]" if p_value < 0.05 else "[green]Not Different[/green]"
            table.add_row(name, f"{r2:.3f}", f"{rmse:.2f}",
                          f"{p_value:.3f}", sig_text)
        console.print(table)

    def display_final_equations_as_table(self):
        TermFX.print_header("Calibrated Kinetic Equations")
        table = Table(title="[bold]Final Model Equations[/bold]",
                      style="purple", title_style="bold magenta", show_lines=True)
        table.add_column("Stage", style="cyan", justify="center")
        table.add_column("Component", style="green")
        table.add_column("Calibrated Equation", style="white")
        if self.params_uasb:
            p = self.params_uasb
            table.add_row("UASB", "Substrate Removal Rate (SRR)",
                          f"SRR = (([yellow]{p['U_max']:.3f}[/yellow] * OLR) / ([yellow]{p['K_B']:.3f}[/yellow] + OLR)) * (1 / (1 + (VFA_ALK / [yellow]{p['K_I']:.3f}[/yellow])))")
        if self.params_filter:
            p = self.params_filter
            table.add_row("Filter", "COD Removal",
                          f"COD_Removed = (TSS_in - TSS_out) * [yellow]{p['R_cod_tss']:.3f}[/yellow] + [yellow]{p['k_ads']:.3f}[/yellow] * sCOD_in")
        if self.params_rbc_ph:
            p = self.params_rbc_ph
            table.add_row("RBC (pH)", "pH Inhibition (τ_pH)",
                          f"τ_pH = (1 + 2*10^([yellow]{0.5*(p['pH_min'] - p['pH_max']):.2f}[/yellow])) / (1 + 10^(pH-[yellow]{p['pH_max']:.2f}[/yellow]) + 10^([yellow]{p['pH_min']:.2f}[/yellow]-pH))")
            table.add_row("RBC (pH)", "Growth Rate (μ_a)",
                          f"μ_a = τ_pH * (([yellow]{p['umxa']:.3f}[/yellow]*Se)/([yellow]{p['Ku']:.3f}[/yellow]+Se)) * (([yellow]{0.5*p['Ko']:.3f}[/yellow]+[yellow]{p['O']:.3f}[/yellow])/([yellow]{p['Ko']:.3f}[/yellow]+[yellow]{p['O']:.3f}[/yellow]))")
            table.add_row("RBC (pH)", "Growth Rate (μ_s)",
                          f"μ_s = τ_pH * ([yellow]{p['umxs']:.3f}[/yellow] * Se) / ([yellow]{p['Ks']:.3f}[/yellow] + Se)")
            table.add_row("RBC (pH)", "Overall Equation",
                          f"(So - Se)/HRT = (μ_a * C1) + (μ_s * C2)")
        console.print(table)

    def visualize_time_series(self):
        TermFX.print_subheader("Generating Time-Series Plot...")
        fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
        fig.suptitle('Model Prediction vs. Measured Data')

        # UASB
        sns.lineplot(data=self.df, x='Day', y='COD_UASB_Pred',
                     ax=axes[0], label='Model Prediction')
        sns.scatterplot(data=self.df, x='Day', y='COD_UASB_Eff',
                        ax=axes[0], label='Measured Data')
        axes[0].set_title('UASB Effluent')
        axes[0].set_ylabel('Effluent COD (mg/L)')

        # Filter
        sns.lineplot(data=self.df, x='Day', y='COD_Filt_Pred',
                     ax=axes[1], label='Model Prediction')
        sns.scatterplot(data=self.df, x='Day', y='COD_Filt_Eff',
                        ax=axes[1], label='Measured Data')
        axes[1].set_title('EFB Filter Effluent')
        axes[1].set_ylabel('Effluent COD (mg/L)')

        # RBC
        sns.lineplot(data=self.df, x='Day', y='COD_Final_Pred_Orig',
                     ax=axes[2], label='Original Model', linestyle='--')
        sns.lineplot(data=self.df, x='Day', y='COD_Final_Pred_pH',
                     ax=axes[2], label='pH Model')
        sns.scatterplot(data=self.df, x='Day', y='COD_Final',
                        ax=axes[2], label='Measured Data')
        axes[2].set_title('Final (RBC) Effluent')
        axes[2].set_ylabel('Effluent COD (mg/L)')
        axes[2].set_xlabel('Time (Days)')

        for ax in axes:
            ax.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        output_filename = "timeseries_plot.png"
        plt.savefig(output_filename)
        TermFX.print_success(f"Time-series plot saved to '{output_filename}'.")
        plt.show()

    def visualize_parity_plots(self):
        TermFX.print_subheader("Generating Parity Plots...")
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Parity Plots for Each Treatment Stage')

        stages = {
            'UASB': ('COD_UASB_Eff', 'COD_UASB_Pred'),
            'Filter': ('COD_Filt_Eff', 'COD_Filt_Pred')
        }

        for i, (title, (m_col, p_col)) in enumerate(stages.items()):
            ax = axes[i]
            data = self.df.dropna(subset=[m_col, p_col])
            sns.regplot(data=data, x=m_col, y=p_col, ax=ax)

            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),
                np.max([ax.get_xlim(), ax.get_ylim()]),
            ]
            ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0, label='1:1 Line')
            ax.set_aspect('equal')
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_title(title)
            ax.set_xlabel(f'Measured {title} COD (mg/L)')
            ax.set_ylabel(f'Predicted {title} COD (mg/L)')
            ax.legend()

        # RBC Plot
        ax_rbc = axes[2]
        rbc_data = self.df.dropna(
            subset=['COD_Final', 'COD_Final_Pred_Orig', 'COD_Final_Pred_pH'])
        sns.regplot(data=rbc_data, x='COD_Final', y='COD_Final_Pred_Orig', ax=ax_rbc,
                    scatter=False, line_kws={'linestyle': '--'}, label='Original Model')
        sns.regplot(data=rbc_data, x='COD_Final', y='COD_Final_Pred_pH',
                    ax=ax_rbc, scatter=False, label='pH Model')
        sns.scatterplot(data=rbc_data, x='COD_Final', y='COD_Final_Pred_Orig',
                        ax=ax_rbc, marker='s', label='Original Points')
        sns.scatterplot(data=rbc_data, x='COD_Final', y='COD_Final_Pred_pH',
                        ax=ax_rbc, marker='^', label='pH Points')

        lims = [
            np.min([ax_rbc.get_xlim(), ax_rbc.get_ylim()]),
            np.max([ax_rbc.get_xlim(), ax_rbc.get_ylim()]),
        ]
        ax_rbc.plot(lims, lims, 'k--', alpha=0.75, zorder=0, label='1:1 Line')
        ax_rbc.set_aspect('equal')
        ax_rbc.set_xlim(lims)
        ax_rbc.set_ylim(lims)
        ax_rbc.set_title('Final (RBC) Effluent')
        ax_rbc.set_xlabel('Measured Final COD (mg/L)')
        ax_rbc.set_ylabel('Predicted Final COD (mg/L)')
        ax_rbc.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        output_filename = "parity_plots.png"
        plt.savefig(output_filename)
        TermFX.print_success(f"Parity plots saved to '{output_filename}'.")
        plt.show()

    def _gsa_model_wrapper(self, param_vector, param_keys, df_subset, rbc_model='orig'):
        temp_params = dict(zip(param_keys, param_vector))
        df = df_subset.copy()
        try:
            # This is a simplified prediction chain for speed; it may not reflect all interdependencies
            # accurately if some parameters appear in multiple models (which they don't currently).
            if any(k in self.params_uasb for k in temp_params):
                srr_pred = self.uasb_model.predict(
                    (df['OLR_UASB'], df['VFA_ALK_Ratio']), {k: temp_params.get(k, self.params_uasb[k]) for k in self.params_uasb})
                df['COD_UASB_Pred'] = df['COD_in'] - srr_pred * df['HRT_UASB']
            else:  # Fallback if UASB params not in current SA set
                df['COD_UASB_Pred'] = df['COD_UASB_Eff']

            if any(k in self.params_filter for k in temp_params):
                cod_rem_filt = self.filter_model.predict(
                    (df['TSS_UASB_Eff'], df['TSS_Filt_Eff'], df['COD_UASB_Pred']), {k: temp_params.get(k, self.params_filter[k]) for k in self.params_filter})
                df['COD_Filt_Pred'] = df['COD_UASB_Pred'] - cod_rem_filt
            else:  # Fallback
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

    def run_sensitivity_analysis(self, method):
        if not self.params_estimated:
            TermFX.print_error(
                "Cannot run analysis without estimated parameters.")
            return

        header = "Global Sensitivity Analysis (GSA)" if method == 'gsa' else "Monte Carlo Sensitivity Protocol"
        TermFX.print_header(header)

        console.print(
            f"[bold cyan]Select RBC model for {method.upper()} analysis:[/]\n  [1] RBC v1.0 (Original)\n  [2] RBC v1.1 (pH-Modified)")
        choice = console.input(
            "[bold green]>[/bold green] [cyan]Enter choice: [/cyan]")

        if choice not in ['1', '2']:
            TermFX.print_error("Invalid selection.")
            return

        rbc_model_key = 'ph' if choice == '2' else 'orig'
        params = {**self.params_uasb, **self.params_filter, **
                  (self.params_rbc_ph if choice == '2' else self.params_rbc_orig)}

        if not params:
            TermFX.print_error("Selected model has no parameters. Aborting.")
            return

        param_keys = sorted(params.keys())

        if method == 'gsa':
            self._execute_gsa(params, param_keys, rbc_model_key)
        elif method == 'mc':
            self._execute_mc(params, param_keys, rbc_model_key)

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

        # NEW: Check for widespread solver failure
        if np.isnan(Y_morris).sum() > len(Y_morris) * 0.9 or np.isnan(Y_sobol).sum() > len(Y_sobol) * 0.9:
            return None, None, None  # Signal failure

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

        # NEW: Check for widespread solver failure
        # If less than 10% of runs succeeded
        if samples_df['Output'].notna().sum() < N * 0.1:
            return None  # Signal failure

        samples_df.dropna(inplace=True)
        correlations = {key: stats.spearmanr(samples_df[key], samples_df['Output'])[
            0] for key in param_keys}
        results_df = pd.DataFrame(list(correlations.items()), columns=[
                                  'Parameter', 'Spearman_Correlation'])

        return results_df, samples_df[param_keys]

    def load_and_display_gsa_results(self, model_key, output_type='table'):
        files = [f"gsa_morris_results_{model_key}.csv",
                 f"gsa_sobol_total_results_{model_key}.csv", f"gsa_sobol_first_results_{model_key}.csv"]
        if not all(os.path.exists(f) for f in files):
            TermFX.print_error(
                f"Saved GSA result files for model '{model_key}' not found. Please run the analysis first.")
            return
        Mi_df = pd.read_csv(files[0]).rename(
            columns={'Unnamed: 0': 'names'}).set_index('names')
        Si_total_df = pd.read_csv(files[1]).rename(
            columns={'Unnamed: 0': 'names'}).set_index('names')
        Si_first_df = pd.read_csv(files[2]).rename(
            columns={'Unnamed: 0': 'names'}).set_index('names')

        Si = {
            'ST': Si_total_df['ST'].values, 'S1': Si_first_df['S1'].values,
            'ST_conf': Si_total_df['ST_conf'].values, 'S1_conf': Si_first_df['S1_conf'].values,
        }
        problem = {'names': Si_total_df.index.tolist()}
        TermFX.print_success(f"Loaded GSA results for model '{model_key}'.")

        if output_type == 'table':
            self.display_gsa_results_table(Mi_df, Si, problem)
        elif output_type == 'chart':
            self.visualize_gsa_results(Mi_df, Si, problem)

    def load_and_display_mc_results(self, model_key, output_type='table'):
        filename = f"mc_sensitivity_results_{model_key}.csv"
        if not os.path.exists(filename):
            TermFX.print_error(
                f"Saved Monte Carlo result file for model '{model_key}' not found. Please run the analysis first.")
            return
        results_df = pd.read_csv(filename)
        TermFX.print_success(
            f"Loaded Monte Carlo results for model '{model_key}'.")

        if output_type == 'chart':
            self.display_mc_results_plot(results_df)
        else:  # Default to table
            console.print(results_df)

    def display_gsa_results_table(self, Mi, Si, problem):
        TermFX.print_subheader("Morris Analysis Results")
        morris_table = Table(
            title="Morris Sensitivity Indices", style="purple")
        morris_table.add_column("Parameter", style="cyan")
        morris_table.add_column("mu_star", justify="right", style="yellow")
        morris_table.add_column("sigma", justify="right", style="green")
        for index, row in Mi.sort_values('mu_star', ascending=False).iterrows():
            morris_table.add_row(
                index, f"{row['mu_star']:.4f}", f"{row['sigma']:.4f}")
        console.print(morris_table)

        TermFX.print_subheader("Sobol Analysis Results")

        sobol_df = pd.DataFrame({
            'S1': Si['S1'], 'ST': Si['ST']
        }, index=problem['names']).sort_values('ST', ascending=False)

        sobol_table = Table(title="Sobol Sensitivity Indices", style="purple")
        sobol_table.add_column("Parameter", style="cyan")
        sobol_table.add_column(
            "S1 (First-order)", justify="right", style="yellow")
        sobol_table.add_column(
            "ST (Total-order)", justify="right", style="green")

        for param, row in sobol_df.iterrows():
            sobol_table.add_row(param, f"{row['S1']:.4f}", f"{row['ST']:.4f}")
        console.print(sobol_table)

    def visualize_gsa_results(self, Mi, Si, problem):
        TermFX.print_subheader("Visualizing GSA Results")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Global Sensitivity Analysis (GSA) Results')

        # Morris Plot
        sns.scatterplot(data=Mi, x='mu_star', y='sigma', s=80, ax=ax1)
        for i in range(len(Mi)):
            ax1.text(Mi['mu_star'][i] + 0.01 * Mi['mu_star'].max(),
                     Mi['sigma'][i], Mi.index[i], fontsize=9)
        ax1.set_title('Morris Elementary Effects')
        ax1.set_xlabel('μ* (Overall Influence)')
        ax1.set_ylabel('σ (Interaction/Non-linear Effects)')

        # Sobol Plot
        sobol_df = pd.DataFrame({
            'S1': Si['S1'], 'ST': Si['ST']
        }, index=problem['names'])
        sobol_df_melted = sobol_df.reset_index().melt(
            id_vars='index', var_name='Index', value_name='Value')

        sns.barplot(data=sobol_df_melted, y='index',
                    x='Value', hue='Index', ax=ax2)
        ax2.set_title('Sobol Sensitivity Indices')
        ax2.set_xlabel('Sobol Index Value')
        ax2.set_ylabel('Parameter')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        output_filename = "gsa_results_plot.png"
        plt.savefig(output_filename)
        TermFX.print_success(f"GSA results plot saved to '{output_filename}'.")
        plt.show()

    def display_mc_results_plot(self, results_df):
        TermFX.print_subheader("Visualizing Monte Carlo Results")
        fig, ax = plt.subplots(figsize=(8, 6))

        df_sorted = results_df.sort_values(
            by="Spearman_Correlation", key=abs, ascending=True)

        df_sorted['Sign'] = np.where(
            df_sorted['Spearman_Correlation'] > 0, 'Positive', 'Negative')

        sns.barplot(
            data=df_sorted,
            y='Parameter',
            x='Spearman_Correlation',
            hue='Sign',
            dodge=False,
            ax=ax
        )

        ax.set_title('Monte Carlo Sensitivity Analysis')
        ax.set_xlabel('Spearman Rank Correlation')
        ax.set_ylabel('Parameter')
        ax.axvline(0, color='k', linestyle='--', alpha=0.7)
        ax.legend(title='Effect Direction')

        plt.tight_layout()
        output_filename = "mc_results_plot.png"
        plt.savefig(output_filename)
        TermFX.print_success(f"Monte Carlo plot saved to '{output_filename}'.")
        plt.show()

    # --- NEW METHODS FOR DATA MANAGEMENT AND EXPLORATION ---
    def delete_files(self, pattern, file_description):
        TermFX.print_subheader(f"Deleting {file_description}")
        files_to_delete = glob.glob(pattern)
        if not files_to_delete:
            TermFX.print_warning(
                f"No {file_description} files found to delete.")
            return
        for f in files_to_delete:
            try:
                os.remove(f)
                TermFX.print_success(f"File '{f}' successfully deleted.")
            except OSError as e:
                TermFX.print_error(f"Failed to delete file '{f}': {e}")

    def view_configuration(self, config):
        TermFX.print_header("Current Configuration")
        table = Table(title="[bold]Configuration Parameters[/bold]",
                      style="purple", title_style="bold magenta")
        table.add_column("Category", style="cyan")
        table.add_column("Parameter", style="green")
        table.add_column("Value", style="yellow")

        table.add_row("Data", "data_file", config["data_file"])
        for key, val in config["reactor_constants"].items():
            table.add_row("Reactor Constants", key, str(val))

        for stage, bounds in config["optimization_bounds"].items():
            table.add_section()
            for i, bound in enumerate(bounds):
                table.add_row(
                    f"Optimization Bounds ({stage})", f"Param {i+1}", f"({bound[0]:.2f}, {bound[1]:.2f})")
        console.print(table)

    def show_data_summary(self):
        TermFX.print_header("Data Summary Statistics")
        if self.df is None:
            TermFX.print_error("Data has not been loaded.")
            return
        console.print(self.df.describe().transpose())

    def show_correlation_heatmap(self):
        TermFX.print_subheader("Generating Correlation Heatmap...")
        if self.df is None:
            TermFX.print_error("Data has not been loaded.")
            return

        fig, ax = plt.subplots(figsize=(16, 12))
        sns.heatmap(self.df.corr(numeric_only=True), annot=True, fmt=".2f", cmap='viridis',
                    annot_kws={"size": 9}, ax=ax)
        fig.suptitle('Correlation Heatmap of Input Variables')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        fig.tight_layout(rect=[0.05, 0.05, 1, 0.96])

        output_filename = "correlation_heatmap.png"
        plt.savefig(output_filename)
        TermFX.print_success(
            f"Correlation heatmap saved to '{output_filename}'.")
        plt.show()

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
