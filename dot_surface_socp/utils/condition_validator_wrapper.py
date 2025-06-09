"""This module provides tools for adaptive strategies for condition validation.
"""

from typing import Dict, Any, Union
import numpy as np
from dot_surface_socp.utils.condition_validator import ConditionValidator


class AdaptiveValidatorWrapper:
    """Adaptive validator wrapper.
    
    This class is designed to wrap the ConditionValidator class
    in order to provide adaptive strategies for condition validation.
    It adjusts validation intervals based on how close the error is to the tolerance.

    Usage
    -----
    >>> validator = ConditionValidator(conditions)
    >>> adaptive_validator = AdaptiveValidatorWrapper(validator)
    >>> 
    >>> while not done:
    >>>     is_passed, info_dict = adaptive_validator.validate()
    >>>     ... get current error and tolerance
    >>>     adaptive_validator.set_error_and_tolerance(error, tolerance)
    """
    def __init__(self, validator: ConditionValidator, default_interval: int = 1, min_interval: int = 1, max_interval: int = 37):
        """Initialize the adaptive validator wrapper.
        
        Args:
            validator: The ConditionValidator instance to wrap
            default_interval: Default interval between validations (default: 1)
            min_interval: Minimum interval between validations (default: 1)
            max_interval: Maximum interval between validations (default: 37)
        """
        self.validator = validator
        self.default_interval = default_interval
        self.current_interval = default_interval
        self.iteration_counter = 0
        self.last_error = None
        self.target_tolerance = None
        self.min_interval = min_interval
        self.max_interval = max_interval
    
    def set_error_and_tolerance(self, error: Union[float, np.ndarray], tolerance: Union[float, np.ndarray]) -> None:
        """Set the current error and target tolerance for adaptive validation.
        
        Args:
            error: Current iteration error value(s)
            tolerance: Target tolerance value(s)
        """
        self.last_error = error
        self.target_tolerance = tolerance
        self._update_interval()
    
    def _update_interval(self) -> None:
        """Update the validation interval based on error and tolerance.
        
        The closer the error is to the tolerance, the smaller the interval will be.
        As error approaches tolerance from above, we check more frequently.
        """
        if self.last_error is None or self.target_tolerance is None:
            self.current_interval = self.default_interval
            return
        
        if not isinstance(self.last_error, np.ndarray):
            error = np.array([self.last_error])
        else:
            error = self.last_error
            
        if not isinstance(self.target_tolerance, np.ndarray):
            tolerance = np.array([self.target_tolerance])
        else:
            tolerance = self.target_tolerance
        
        # Calculate ratio of error to tolerance (clip to avoid division by zero)
        ratio = np.max(error / np.maximum(tolerance, 1e-10))
        
        # When error is below tolerance, we've converged - check at minimum interval
        if ratio <= 1.0:
            self.current_interval = self.min_interval
            return
            
        # For error above tolerance, we use a logarithmic scale to determine interval
        # log_ratio ranges from 0 (when ratio=1) to log(max_ratio)
        log_ratio = np.log10(ratio)
        
        if log_ratio > 1.0:  # ratio > 10
            # Far from convergence: use maximum interval
            self.current_interval = self.max_interval
        else: # 1 < ratio <= 10
            # As error approaches tolerance, decrease interval linearly
            # When ratio=1 (log_ratio=0), interval=min_interval
            # When ratio=10 (log_ratio=1), interval=max_interval
            self.current_interval = max(
                self.min_interval,
                int(self.min_interval + log_ratio * (self.max_interval - self.min_interval))
            )
    
    def should_validate(self) -> bool:
        """Determine if validation should be performed in the current iteration.
        
        Returns:
            bool: True if validation should be performed, False otherwise
        """
        should_check = (self.iteration_counter % self.current_interval) == 0
        self.iteration_counter += 1
        return should_check
    
    def validate(self, *args, **kwargs) -> tuple[bool, Dict[int, Any]]:
        """Perform validation if the current iteration requires it.
        
        This method wraps the validator's validate method and only calls it
        when the adaptive strategy determines it's necessary.
        
        Args:
            *args: Arguments to pass to the validator's validate method
            **kwargs: Keyword arguments to pass to the validator's validate method
            
        Returns:
            tuple: (is_test_all_passed, detailed_info) from the validator,
                  or (False, {}) if validation was skipped
        """
        if self.should_validate() or kwargs.get("required_conditions", []):
            return self.validator.validate(*args, **kwargs)
        return False, {}

    def get_num_conditions(self) -> int:
        """Get the number of conditions being validated."""
        return self.validator.get_num_conditions()
    
    def reset_counter(self) -> None:
        """Reset the counter of the adaptive validator, which can ensure the next validation been performed.
        """
        self.iteration_counter = 0
    
    def reset(self) -> None:
        """Reset the adaptive validator to its initial state."""
        self.current_interval = self.default_interval
        self.iteration_counter = 0
        self.last_error = None
        self.target_tolerance = None
    
    def set_interval_bounds(self, min_interval: int = 1, max_interval: int = 10) -> None:
        """Set the minimum and maximum validation intervals.
        
        Args:
            min_interval: Minimum interval between validations
            max_interval: Maximum interval between validations
        """
        self.min_interval = max(1, min_interval)
        self.max_interval = max(self.min_interval, max_interval)
