"""This file provide tools to adjust parameter and record running history in Alternative Direction Multiplier Method (ADMM)
"""

import sys
import time

import numpy as np
import logging
from typing import List, Any, Union
from matplotlib import pyplot as plt
from contextlib import contextmanager
from dot_surface_socp.config import LOG_LEVELS

from numpy import ndarray
from tqdm import tqdm  # show progress of accuracy
from math import log10


class AdjustAdmmParam:
    """A class to help adjust parameters in ADMM
    """

    def __init__(self):
        self.last_it = -1
        self._sigma_upper_bound = 10 ** 3
        self._sigma_lower_bound = 10 ** (-3)
        self._scale_times_matrix = 0

    # ==== Adjust penalty parameter
    def is_to_adjust(self, current_it):
        """Determine whether to adjust the penalty parameter based on current iteration and
        iterations passed since last adjustment.

        Parameters:
            current_it: Current iteration number

        Returns:
            bool: True if penalty parameter should be adjusted, False otherwise
        """
        passed_it = current_it - self.last_it

        # Adjustment frequency increases with iteration count
        if (current_it < 20 and passed_it >= 3) or \
                (current_it < 50 and passed_it >= 7) or \
                (current_it < 100 and passed_it >= 11) or \
                (current_it < 200 and passed_it >= 17) or \
                (current_it < 500 and passed_it >= 31) or \
                (passed_it >= 43):
            self.last_it = current_it
            return True

        return False

    def get_updated_value(self, sigma, prim_dual_gap):
        """Adjust ADMM penalty parameter according to primal-dual gap
        """

        # Update with safeguard
        adjust_factor = self.__get_adjust_factor(prim_dual_gap)
        sigma = max(min(sigma * adjust_factor, self._sigma_upper_bound), self._sigma_lower_bound)

        return sigma

    @staticmethod
    def __get_adjust_factor(prim_dual_gap):
        """Get adjusting factor of ADMM penalty parameter
        """

        # Only need to code for the case of "primal-dual gap >= 1"
        # The case of "primal-dual gap <= 1" will get the inverse factor of the case of "primal-dual gap >= 1"
        if prim_dual_gap < 1.:
            prim_dual_gap = 1. / prim_dual_gap
            prim_win = True
        else:
            prim_win = False

        # Get adjusting factor according to primal-dual gap
        match prim_dual_gap:
            case xi if xi > 50:     factor = 2.00
            case xi if xi > 35:     factor = 1.75
            case xi if xi > 20:     factor = 1.60
            case xi if xi > 10:     factor = 1.40
            case xi if xi > 5:      factor = 1.35
            case xi if xi > 3:      factor = 1.32
            case xi if xi > 2.5:    factor = 1.28
            case xi if xi > 2:      factor = 1.26
            case xi if xi > 1.5:    factor = 1.20
            case xi if xi > 1.2:    factor = 1.10
            case _:                 factor = 1.00

        # Recover factor
        if prim_win:
            return 1. / factor
        else:
            return factor
    
    # ==== Scaling
    @staticmethod
    def is_to_scale(current_it):
        if current_it == 10 or \
                current_it == 50 or \
                current_it % 100 == 50:
            return True
        else:
            return False
    
    def is_to_scale_matrix(self, current_it, current_kkt: list[float] = [0], min_it: int = 100, max_scale_times: int = 1, tol: float = 5e-3):
        if current_it >= min_it and \
                self._scale_times_matrix < max_scale_times and \
                max(current_kkt) < tol:
            self._scale_times_matrix += 1
            return True
        else:
            return False

    @staticmethod
    def compute_scale_factor(
        prim_norm: Union[list[float], ndarray, float],
        dual_norm: Union[list[float], ndarray, float],
        prim_norm_target: Union[list[float], ndarray, float] = None,
        dual_norm_target: Union[list[float], ndarray, float] = None,
        msg: str = "Norm of prim and dual"
    ):
        """Compute scaling factors for primal and dual variables to reach target norms
        
        This function computes scaling factors to adjust primal and dual variables to reach 
        desired target norms. If target norms are not provided, variables will be scaled to 1.0.
        The function also logs the current and target norms for debugging purposes.

        Parameters:
            prim_norm: Current norm values of primal variables, can be scalar or array-like
            dual_norm: Current norm values of dual variables, can be scalar or array-like  
            prim_norm_target: Target norm value for primal variables, defaults to 1.0 if None
            dual_norm_target: Target norm value for dual variables, defaults to 1.0 if None
            msg: Message prefix for debug logging, defaults to "Info of scaling"

        Returns:
            tuple: (prim_rescale_factor, dual_rescale_factor) - Scaling factors for primal and dual variables
        """
        
        # Format all norm values into a debug message
        debug_msg = [msg]
        
        def __format_norm(name: str, value):
            if isinstance(value, (float, int)):
                debug_msg.append(f"{name}: {value:.2e}")
            else:
                debug_msg.append(f"{name}: [{', '.join([f'{x:.2e}' for x in value])}]")
        
        __format_norm("Prim Norm", prim_norm)
        __format_norm("Dual Norm", dual_norm)
        
        if prim_norm_target is not None:
            __format_norm("Standard Level of Prim Norm", prim_norm_target)
        
        if dual_norm_target is not None:
            __format_norm("Standard Level of Dual Norm", dual_norm_target)
            
        logging.log(LOG_LEVELS["scaling"], "\n".join(debug_msg))

        # Compute scaling factor
        if prim_norm_target is None:
            prim_norm_target = 1.0

        if dual_norm_target is None:
            dual_norm_target = 1.0

        prim_rescale_factor = np.max(prim_norm) / prim_norm_target
        dual_rescale_factor = np.max(dual_norm) / dual_norm_target

        return prim_rescale_factor, dual_rescale_factor


class RunningHistory:
    """A class to help record Optimization Algorithm running history and to help visualize them
    """

    def __init__(self, max_record_numbers: int, kkt_labels: List[str], name: str,
                       kkt_short_labels: List[str] = None, use_linear_progress: bool = False):
        assert kkt_short_labels is None or len(kkt_short_labels) == len(kkt_labels), \
            "kkt_short_labels must be None or have the same length as kkt_labels."

        # Private value
        self._kkt_num = 0
        self._max_num = max_record_numbers
        self._unknown_value = np.inf
        self._start_time_global = self._unknown_value # global clock to time the algorithm and the kkt errors
        self._tol_progress = None # show progress of solution accuracy
        self._use_linear_progress = use_linear_progress # Whether to use linear progress bar
        self._separate_symbol = lambda s: f"{'----'} {s} ".ljust(42, '-')
        self._sub_separate_symbol = '-' * 42

        # Public value
        self.kkt_entry_num      = len(kkt_labels) # The number of kkt constraint
        self.kkt_errors         = np.ones((max_record_numbers, self.kkt_entry_num)) * self._unknown_value
        self.kkt_iteration      = np.ones(max_record_numbers) * self._unknown_value # The number of iteration which producing corresponding kkt errors
        self.kkt_time           = np.ones(max_record_numbers) * self._unknown_value # The time producing corresponding kkt errors
        self.kkt_labels         = kkt_labels
        self.kkt_short_labels   = kkt_short_labels if kkt_short_labels is not None else kkt_labels
        self.name               = name
        self.running_time       = self._unknown_value
        self.last_record_it     = -1 # The last iteration number that has been recorded

        self.steps_time: dict   = {} # Total running time of each step
        self.history: dict      = {} # Other running history besides kkt errors, which will be filled in self.record()

        logging.basicConfig(level=LOG_LEVELS["info"], format='%(message)s')

    # ================
    # Time for the whole algorithm
    def start(self):
        """Start running algorithm
        """

        self._start_time_global = time.perf_counter()

    def end(self):
        """Trim running history: truncate the redundant pre-allocated data
        """

        self.running_time = time.perf_counter() - self._start_time_global

        # Truncation
        self.kkt_errors = self.kkt_errors[:self._kkt_num, :]
        self.kkt_iteration = self.kkt_iteration[:self._kkt_num]
        self.kkt_time = self.kkt_time[:self._kkt_num]

        for key in self.history.keys():
            self.history[key] = self.history[key][:self._kkt_num]

        # Close
        if self._tol_progress is not None:
            self._tol_progress.close()
            print(self._separate_symbol("Finish performing"))
            sys.stdout.flush()
            time.sleep(0.1)
    
    def get_running_time(self):
        """Gets the time that the algorithm has been run
        """
        return time.perf_counter() - self._start_time_global

    # ================
    @contextmanager
    def timer(self, tag):
        start_time = time.perf_counter()
        yield
        if tag in self.steps_time:
            self.steps_time[tag] += time.perf_counter() - start_time
        else:
            self.steps_time[tag] = time.perf_counter() - start_time

    def __transfer_tol_to_progress(self, tol):
        """Transfer tolerance (0 < tolerance < 1) to the integer type progress
        """
        if hasattr(self, '_use_linear_progress') and self._use_linear_progress:
            return self.__transfer_tol_to_progress_linear(tol)
        return self.__transfer_tol_to_progress_sublinear(tol)

    @staticmethod
    def __transfer_tol_to_progress_linear(tol):
        """Transfer tolerance (0 < tolerance < 1) to the integer type progress (linear rate)
        """
        return round(1000.0 * log10(1.0 / tol))
    
    @staticmethod
    def __transfer_tol_to_progress_sublinear(tol):
        """Transfer tolerance (0 < tolerance < 1) to the integer type progress (sublinear rate)
        """
        return round(1000.0 * (1.0 / tol) ** 0.5)

    def __format_condition_names(self, conditions):
        """Format condition indices/names for display
        
        Args:
            conditions: List of condition indices (int) or names (str)
            
        Returns:
            str: Formatted string of condition names
        """
        if not conditions:
            return "None"
        
        if isinstance(conditions[0], int):
            # Convert indices to short labels
            names = [self.kkt_short_labels[i] if i < len(self.kkt_short_labels) else f"KKT-{i}" 
                    for i in conditions]
        else:
            # Already names
            names = conditions
        
        # Limit display length
        if len(names) <= 2:
            return ", ".join(names)
        elif len(names) <= 4:
            return ", ".join(names[:2]) + f" + {len(names)-2} more"
        else:
            return f"{names[0]}, {names[1]} + {len(names)-2} others"

    def __get_condition_index(self, condition_name):
        """Get condition index from name"""
        try:
            return self.kkt_short_labels.index(condition_name)
        except ValueError:
            try:
                return self.kkt_labels.index(condition_name)
            except ValueError:
                return -1
            
    def __create_progress_bar(self):
        return tqdm(
            total=self.__transfer_tol_to_progress(self._target_tol),
            ncols=150,
            desc=f'Tol={self._target_tol:.2e}',
            bar_format='[{desc}{postfix}]|{bar}|{percentage:4.1f}%',
            leave=True
        )

    def create_tol_progress(self, target_tol):
        """Create the progress object to show progress during iteration"""
        print(self._separate_symbol("Starting to perform ..."))
        
        # Store target tolerance for later use
        self._target_tol = target_tol
        self._tol_progress = self.__create_progress_bar()
        logging.log(LOG_LEVELS["kkt"], "---- Iteration Start ".ljust(42, '-'))

    def show_tol_progress(self, current_it, current_err, active_idx=None, converged_idx=None):
        """Show the progress of solution accuracy w.r.t. target tolerance
        
        Args:
            current_it: Current iteration number
            current_err: Current maximum error value
            active_idx: List of indices or names of currently active KKT conditions
            converged_idx: List of indices or names of recently converged conditions
        """
        
        # Handle converged conditions - close current progress bar and start new one
        if converged_idx and len(converged_idx) > 0:
            if self._tol_progress is not None:
                # Show convergence message
                converged_names = self.__format_condition_names(converged_idx)
                self._tol_progress.set_postfix_str(f"Converged: {converged_names}")
                self._tol_progress.close()
                
                print(f"Conditions converged at iteration {current_it}: {converged_names}\n")
            
            # Start new progress bar for remaining conditions
            converged_indices = converged_idx if (converged_idx and isinstance(converged_idx[0], int)) else \
                               [self.__get_condition_index(name) for name in converged_idx] if converged_idx else []
            
            # Get previously converged conditions (stored in instance)
            if not hasattr(self, '_converged_conditions'):
                self._converged_conditions = set()
            self._converged_conditions.update(converged_indices)
            
            remaining_conditions = [i for i in range(self.kkt_entry_num) 
                                  if i not in self._converged_conditions]
            
            if len(remaining_conditions) > 0:
                self._tol_progress = self.__create_progress_bar()
            else:
                print("All KKT conditions have converged!")
                return
        
        # Update progress bar
        if self._tol_progress is not None:
            self._tol_progress.n = self.__transfer_tol_to_progress(current_err)

            # Calculate elapsed time since algorithm start
            elapsed_time = time.perf_counter() - self._start_time_global
            
            # Build postfix string manually to control order
            is_over_one_hour = (elapsed_time // 3600 >= 1)
            time_str = time.strftime('%M:%S', time.gmtime(elapsed_time)) if not is_over_one_hour else \
                time.strftime('%H:%M:%S', time.gmtime(elapsed_time))
            postfix_parts = [
                f"Acc: {current_err:.2e}",
                f"Time: {time_str}",
                f"Iter: {current_it} ({elapsed_time / (current_it + 1):.4f} sec/it)",
            ]

            if active_idx is not None:
                postfix_parts.append(f"Checking: {self.__format_condition_names(active_idx)}")

            self._tol_progress.set_postfix_str(", ".join(postfix_parts))
            self.__verbose_logging()

    def __verbose_logging(self):
        """Log the detailed information during iteration
        """
        idx = self._kkt_num - 1
        msg_it = f"{self.kkt_iteration[idx]:4.0f}"
        msg_kkt = " ".join([f"{error:6.2e}" for error in self.kkt_errors[idx, :]])
        logging.log(LOG_LEVELS["kkt"], f"Iteration: {msg_it} - KKT: {msg_kkt}")

    # ================
    # Record running history
    def record(self, current_it: int = None,
                     kkt_errors: Union[List, ndarray] = None,
                     history: dict[str, Any] = None):
        """Record kkt errors and their corresponding iteration number and time
        """

        if kkt_errors is None or current_it is None:
            raise ValueError("Argument `kkt_errors` or `current_it` must be provided.")
        
        if current_it < self.last_record_it:
            raise ValueError(f"Current iteration {current_it} is smaller than last recorded iteration {self.last_record_it}. "
                             f"Please check the input arguments.")
        elif current_it == self.last_record_it:
            # If the current iteration is the same as the last recorded iteration,
            # we will overwrite the last record
            self._kkt_num -= 1
             

        if self._kkt_num < self._max_num:
            self.last_record_it = current_it

            self.kkt_errors[self._kkt_num, :] = kkt_errors
            self.kkt_iteration[self._kkt_num] = current_it
            self.kkt_time[self._kkt_num] = time.perf_counter() - self._start_time_global
        else:
            raise ValueError(f"Current recorded items number {self._kkt_num} is bigger than max valid number {self._max_num}."
                             f"There is no redundant space to store the running history.")

        if history is not None:
            for key, val in history.items():
                if key not in self.history:
                    self.history[key] = np.ones_like(self.kkt_iteration) * self._unknown_value
                self.history[key][self._kkt_num] = val

        self._kkt_num += 1
    
    def get_current_kkt_errors(self):
        """Get the current kkt errors
        """
        if self._kkt_num == 0:
            # raise ValueError("No kkt errors recorded yet.")
            return np.ones(self.kkt_errors.shape[1]) * self._unknown_value
        else:
            return self.kkt_errors[self._kkt_num - 1, :]

    # ================
    # Show/Print running history
    def show_kkt_errors(self, filename: str = None, is_show_when_save: bool = False, x_axis: str = 'iteration',
                        title: str = None, x_label: str = None, y_label: str = None):
        """Show kkt error curves

        Usage
            1. show_kkt_errors() to *show* kkt error curves
            2. show_kkt_errors(file_name="name.ext") to *save* figure as a picture file (*not show* the figure)
            3. show_kkt_errors(file_name="name.ext", is_show_when_save=True) to *show* and *save* the figure of kkt error curves
            4. show_kkt_errors(x_axis=...)
                x_axis="iteration"(default), to set the x-axis label as iteration numbers
                x_axis="time", to set the x-axis label as iteration time
            5. fig = show_kkt_errors(...) to get the figure handle
        """

        # Choose type of x-axis
        if x_axis == 'iteration':
            x_data = self.kkt_iteration
            x_default_label = "Iteration numbers"
        elif x_axis == 'time':
            x_data = self.kkt_time
            x_default_label = "Iteration time [seconds]"
        else:
            raise ValueError(f"x_axis {x_axis} is not supported.")

        # Plot
        fig = plt.figure()

        for n in range(self.kkt_entry_num):
            kkt_errors = self.kkt_errors[:, n]
            kkt_errors[kkt_errors < 10**(-10)] = 0.0 # If error < 1e-10, we consider it has been exactly solved
            plt.semilogy(x_data, kkt_errors, label=self.kkt_short_labels[n])

        plt.title(title if isinstance(title, str) else self.name)
        plt.xlabel(x_label if isinstance(x_label, str) else x_default_label)
        plt.ylabel(y_label if isinstance(y_label, str) else "Karush–Kuhn–Tucker errors")
        plt.legend()

        # Show / Save
        if isinstance(filename, str):
            if is_show_when_save:
                fig.show()

            format_fig = self._get_saved_fig_opts_matplotlib()
            fig.savefig(filename, **format_fig)
        else:
            fig.show()

        # Close
        plt.close(fig)

    @staticmethod
    def _get_saved_fig_opts_matplotlib():
        return {
            # "format": "pdf",
            "bbox_inches": "tight",
            # "transparent": True,
            # "dpi": 600,
        }

    def print_steps_time(
            self,
            tag_tips: str = "Time of each step",
            tag_step_time: str = "Time of steps",
            tag_total_time: str = "Total Time",
            tag_total_iteration: str = "Total Iteration"
        ):
        """Show total time-consuming of each step
        """

        total_time = self.running_time
        total_iteration = self.kkt_iteration[-1]
        step_labels, step_time = list(self.steps_time.keys()), list(self.steps_time.values())
        sum_step_time = sum(step_time)
        sum_step_time_100_iteration = 100.0 * sum_step_time / total_iteration

        # Get the max length of step labels
        max_len = max([len(label) for label in step_labels + [tag_step_time, tag_total_time, tag_total_iteration]])

        # Cat the message
        msg_step_time = "\n".join([
            f"{step_label:<{max_len}}: {step_time:>7.2f} sec ({100.0 * step_time / total_time:5.2f}%) "
            f"({100.0 * step_time / total_iteration:<5.2f} sec/100-iterations)"
            for step_label, step_time in zip(step_labels, step_time)
        ])
        msg_total_time = \
            f"{tag_step_time.ljust(max_len)}: {sum_step_time:>7.2f} sec ({100.0 * sum_step_time / total_time:5.2f}%) ({sum_step_time_100_iteration:<5.2f} sec/100-iterations)\n"\
            f"{tag_total_time.ljust(max_len)}: {total_time:>7.2f} sec ({100.0:5.2f}%)\n"\
            f"{tag_total_iteration.ljust(max_len)}: {total_iteration:>7.0f} iterations"

        logging.log(LOG_LEVELS["info"],
            f"{self._separate_symbol(tag_tips)}\n"
            f"{msg_step_time}\n"
            f"{self._sub_separate_symbol}\n"
            f"{msg_total_time}"
        )

    def print_end_history(self):
        """Print the running history at the end
        """

        # Relative kkt errors
        kkt_errors = self.kkt_errors[-1, :]
        max_len = max([len(label) for label in self.kkt_labels])
        msg_kkt = "\n".join([f"{label:<{max_len}}: {error:>6.2e}" for error, label in zip(kkt_errors, self.kkt_labels)])

        logging.log(LOG_LEVELS["info"],
            f"{self._separate_symbol("The kkt errors at end")}\n"
            f"{msg_kkt}"
        )

        # Other running history
        if self.history:
            msg_history = "\n".join([f"{key}: {history_item[-1]:.6e}" for key, history_item in self.history.items()])
            logging.log(LOG_LEVELS["info"],
                f"{self._separate_symbol("Other history at end")}\n"
                f"{msg_history}"
            )


if __name__ == '__main__':
    """A simple demo to use RunningHistory class
    """

    max_iteration = 1000
    iterations = 500

    # Initialization an instance for RunningHistory
    run_history = RunningHistory(
        max_record_numbers=max_iteration,
        kkt_labels=["Primal Feasibility", "Dual Feasibility", "Complementary"],
        name="Alternative Direction Multiplier Method",
    )

    # Synthetic kkt errors
    rng = np.random.default_rng(42)

    noise = rng.normal(loc=0.0, scale=0.5, size=iterations)
    prim_errors = [0.4 / (n + 1) ** 2 * (1.0 + noise[n] / (n+1) ** 0.5) for n in range(iterations)]

    noise = rng.normal(loc=0.0, scale=0.5, size=iterations)
    dual_errors = [0.5 / (n + 1) ** 2 * (1.0 + noise[n] / (n+1) ** 0.5) for n in range(iterations)]

    noise = rng.normal(loc=0.0, scale=0.5, size=iterations)
    comp_errors = [0.6 / (n + 1) ** 2 * (1.0 + noise[n] / (n+1) ** 0.5) for n in range(iterations)]

    # Fake computing (to simulate ADMM)
    step1 = lambda n: time.sleep(0.001)
    step2 = lambda n: time.sleep(0.002)
    step3 = lambda n: time.sleep(0.003)

    # Collect running history
    run_history.start()
    run_history.create_tol_progress(target_tol=1e-6)
    for n in range(iterations):
        with run_history.timer(tag="Step1"):
            step1(n)

        with run_history.timer(tag="Step2"):
            step2(n)

        with run_history.timer(tag="Step3"):
            step3(n)

        kkt_errors = [prim_errors[n], dual_errors[n], comp_errors[n]]
        run_history.record(kkt_errors=kkt_errors, current_it=n)
        run_history.show_tol_progress(n, max(kkt_errors))

    # Trim running history
    run_history.end()

    # Show running history
    run_history.show_kkt_errors(x_axis='iteration')
    run_history.show_kkt_errors(x_axis='time')
    run_history.print_end_history()
    run_history.print_steps_time()
