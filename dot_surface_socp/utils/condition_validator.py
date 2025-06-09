"""Condition Validation Speedup Tool

This module implements the circular queue optimization for condition validation.
The approach leverages the asymmetry between AND logic (all conditions must pass)
and OR logic (any condition can fail) to significantly accelerate convergence checking.

This tool is general-purpose and can be used for any set of conditions 
that need to be validated together.
"""

from typing import Union, List, Callable, Optional, Dict, Any
import numpy as np
import logging
from collections import OrderedDict

# Set up logger for this module
logger = logging.getLogger(__name__)

class BasicConditionWrapper:
    """Wrapper class to standardize basic condition functions for use with ConditionValidator.
    
    The condition is considered passed if the function returns True.
    """
    
    def __init__(self, condition_function: Callable, name: str = ""):
        """
        Initialize wrapper for a basic condition.
        
        Args:
            condition_function: Function that returns a boolean indicating condition satisfaction
            name: Descriptive name for the condition
        """
        self.condition_function = condition_function
        self.name = name
    
    def __call__(self) -> bool:
        """Evaluate the basic condition.
            
        Returns:
            bool: True if condition passed, False otherwise
        """
        try:
            return self.condition_function()
        except Exception as e:
            # Handle any computation errors gracefully
            logger.debug(f"Error in condition '{self.name}': {e}")
            return False


class ErrorConditionWrapper(BasicConditionWrapper):
    """Wrapper class to standardize error-based condition functions for use with ConditionValidator.
    
    This class helps convert existing error computation functions to the required format: tolerance -> bool.
    The condition is considered passed if the computed error is below the tolerance threshold.
    After each call, one can retrieve the last computed error value using `get_last_error()`.
    """
    
    def __init__(self, error_function: Callable, tolerance: float, name: str = ""):
        """Initialize wrapper for an error-based condition with a given tolerance.
        
        Args:
            error_function: Function that computes the error value (should return a float)
            tolerance: Tolerance threshold for condition satisfaction
            name: Descriptive name for the condition
        """
        self.error_function = error_function
        self.name = name
        self.tolerance = tolerance  # Tolerance threshold for condition satisfaction
        self.last_error_value = None  # Store the last computed error value
    
    def __call__(self) -> bool:
        """Evaluate the error-based condition.
        
        Args:
            tolerance: Tolerance threshold
            error_output: Optional list to store the computed error value [error_value]
            
        Returns:
            bool: True if condition passed (error_value < tolerance), False otherwise
        """
        try:
            error_value = self.error_function()
            self.last_error_value = error_value
            return (error_value < self.tolerance)
        except Exception as e:
            # Handle any computation errors gracefully
            logger.debug(f"Error in condition '{self.name}': {e}")
            self.last_error_value = float('inf')
            return False
    
    def get_last_error(self) -> Optional[float]:
        """Get the last computed error value."""
        return self.last_error_value
    
    def reset_last_error(self, value: Optional[float] = None):
        """Reset the last error value to a specific value (default to None)."""
        self.last_error_value = value
    
    def get_and_reset_last_error(self, value: Optional[float] = None) -> Optional[float]:
        """Get the last error value and reset it to a specific value (default to None)."""
        last_error, self.last_error_value = self.last_error_value, value
        return last_error
    

class ErrorConditionWrapperEx(ErrorConditionWrapper):
    """Extended wrapper class to standardize error-based condition functions for use with ConditionValidator.
    
    This class extends ErrorConditionWrapper to support multiple values of error for each condition.
    The user will specify the first value as the primary error, which is used to determine if the condition is satisfied.
    The next values is used to store the last computed error value for other purposes.
    """
    
    def __init__(self, error_function: Callable, tolerance: float, num_values: int, name: str = ""):
        """Initialize wrapper for an error-based condition with a given tolerance.
        
        Args:
            error_function: Function that computes the error value (should return a float)
            tolerance: Tolerance threshold for condition satisfaction
            num_values: Number of error values to store (typically 2 for primary and secondary errors)
            name: Descriptive name for the condition
        """
        # assert num_values >= 1, "num_values must be at least 1"
        # assert isinstance(num_values, int), "num_values must be an integer"

        self.error_function = error_function
        self.name = name
        self.tolerance = tolerance  # Tolerance threshold for condition satisfaction
        self.num_values = num_values
        self.last_error_value = [None,] * num_values  # Store the last computed multiple values of error
    
    def __call__(self) -> bool:
        """Evaluate the error-based condition.
        
        Args:
            tolerance: Tolerance threshold
            error_output: Optional list to store the computed error value [error_value]
            
        Returns:
            bool: True if condition passed (error_value < tolerance), False otherwise
        """
        try:
            self.last_error_value = self.error_function()
            error_value = self.last_error_value[0] # Use the first value as the primary error
            return (error_value < self.tolerance)
        except Exception as e:
            # Handle any computation errors gracefully
            logger.error(f"Error in condition '{self.name}': {e}")
            self.last_error_value = [float('inf'), float('inf')]
            return False
    
    def get_last_error(self) -> Optional[List[float]]:
        """Get the last computed error value."""
        return self.last_error_value
    
    def reset_last_error(self, value: Optional[List[float]] = None):
        """Reset the last error value of list to a specific value of list (default to [None,] * self.num_values)."""
        self.last_error_value = [None,] * self.num_values if value is None else value
    
    def get_and_reset_last_error(self, value: Optional[List[float]] = None) -> Optional[float]:
        """Get the last error value of list and reset it to a specific value of list (default to [None,] * self.num_values)."""
        last_errors, self.last_error_value = self.last_error_value, [None,] * self.num_values if value is None else value
        return last_errors


def max_of_list_with_none(values: List[Optional[float]]) -> Optional[float]:
    """Get the maximum value from a list, treating None as negative infinity."""
    filtered_values = [v for v in values if v is not None]
    return max(filtered_values) if filtered_values else None


class ErrorConditionCollector:
    """Utility class to collect error values from multiple ErrorConditionWrapper instances.
    
    This class allows you to aggregate error values from multiple conditions into a single list.
    """
    
    def __init__(self, conditions: List[ErrorConditionWrapper | ErrorConditionWrapperEx]):
        """Initialize the collector with a list of ErrorConditionWrapper instances.
        
        Args:
            conditions: List of ErrorConditionWrapper instances
        """
        self.conditions = conditions
    
    def get_errors(self) -> List[float] | List[List[float]]:
        """Collect error values from all wrapped conditions.
        
        Returns:
            List of last computed error values from each condition
        """
        return np.array([condition.get_and_reset_last_error() for condition in self.conditions])


class ConditionValidator:
    """Circular queue-based condition validator.
    
    This class implements a speedup methodology to validate any set of conditions during algorithm execution.
    - Uses circular queue to adaptively order condition checking
    - Early termination when any condition fails
    - Prioritizes conditions most likely to fail based on iteration history
    
    This validator is general-purpose and can be used for error conditions, convergence criteria,
    constraint satisfaction, or any other set of boolean conditions.
    
    Attributes:
        condition_functions: List of callable condition functions, each
        queue_front: Current front element index in circular queue
        queue_size: Number of conditions
    """
    
    def __init__(self, condition_functions: List[BasicConditionWrapper], optimize_queue_period: int = None):
        """Initialize the validator with condition functions.
        
        Args:
            condition_functions: List of functions that return (condition_passed: bool, error_value: float)
            optimize_queue_period: period for optimizing the queue order based on failure statistics (optional, defaults to None, meaning no periodic optimization)
        """
        self.condition_functions = condition_functions
        self.queue_size = len(condition_functions)
        self.queue_front = 0
        self.optimize_queue_period = optimize_queue_period
        
        # Index mapping for queue reordering
        self.original_to_current_index = list(range(self.queue_size))  # [0,1,2,3,...]
        self.current_to_original_index = list(range(self.queue_size))  # [0,1,2,3,...]
        
        # Statistics for adaptive optimization
        self.failure_counts = np.zeros(self.queue_size)
        self.total_checks = 0
        self.validation_times = 0
    
    def get_num_conditions(self) -> int:
        """Get the number of conditions being validated."""
        return self.queue_size
    
    def validate(self, max_checks: Optional[int] = None, required_conditions: Optional[List[int]] = None) -> tuple[bool, Dict[int, Any]]:
        """
        Perform condition validation using circular queue optimization.
        
        Args:
            max_checks: Maximum number of conditions to check (defaults to queue_size)
            required_conditions: List of condition indices that must be validated in this round (defaults to None)
            
        Returns 
        tuple: (is_test_all_passed, detailed_info)
            is_test_all_passed: Boolean indicating if all conditions passed
            detailed_info, a dictionary containing:
                - 'all_passed': Boolean indicating if all conditions passed
                - 'num_conditions_passed': Number of conditions actually checked
                - 'failing_condition': Index of first failing condition (if any)
                - 'early_termination': Boolean indicating if validation terminated early (i.e., not all conditions passed)
        """
        if max_checks is None:
            max_checks = self.queue_size
        
        if required_conditions is None:
            required_conditions = []
        
        # Convert original indices to current indices for required_conditions
        if required_conditions:
            required_conditions = [self.original_to_current_index[idx] for idx in required_conditions if idx < self.queue_size]
            
        is_test_all_passed = False
        num_passed = 0
        failing_conditions = []
        self.validation_times += 1
        checked_conditions = set()  # Track which conditions have been checked
        
        # Helper function to check a condition
        def check_condition(idx):
            nonlocal num_passed
            if idx in checked_conditions:
                return True 
            
            num_passed += 1
            self.total_checks += 1
            checked_conditions.add(idx)
            
            is_passed = self.condition_functions[idx]()
            if not is_passed:
                self.failure_counts[idx] += 1
                failing_conditions.append(idx)
            return is_passed

        # First check all required conditions
        is_required_passed = [check_condition(req_idx) for req_idx in required_conditions if req_idx < self.queue_size]

        # Continue with circular queue
        if all(is_required_passed) and num_passed < max_checks:
            start_front = self.queue_front
            while num_passed < max_checks:
                condition_index = self.queue_front % self.queue_size
                if condition_index not in checked_conditions:
                    if not check_condition(condition_index):
                        break  # Failed, exit early
                
                # Move to next condition
                self.queue_front = (self.queue_front + 1) % self.queue_size
                
                # Check if completed full cycle
                if self.queue_front == start_front:
                    is_test_all_passed = True
                    break
        elif all(is_required_passed) and num_passed >= max_checks:
            is_test_all_passed = True
        
        # Periodically optimize queue order based on failure statistics
        if self.optimize_queue_period and self.validation_times >= self.optimize_queue_period:
            self.optimize_queue_order()
            self.validation_times = 0
        
        # Logging
        if num_passed > 0:
            if is_test_all_passed:
                logger.debug(f"Validation: PASSED (checked {num_passed}/{self.queue_size} conditions)")
            else:
                # Convert current indices back to original indices for logging
                original_failing = [self.current_to_original_index[idx] for idx in failing_conditions]
                logger.debug(f"Validation: FAILED at condition {original_failing} (checked {num_passed}/{self.queue_size})")
        
        # Convert current indices back to original indices in the result
        original_failing_conditions = [self.current_to_original_index[idx] for idx in failing_conditions]
        
        detailed_info = {
            'all_passed': is_test_all_passed,
            'num_conditions_passed': num_passed,
            'failing_conditions': original_failing_conditions,
            'early_termination': not is_test_all_passed and num_passed < self.queue_size
        }
                
        return is_test_all_passed, detailed_info
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get validation statistics for analysis and optimization.
        
        Returns:
            Dictionary with failure rates, efficiency metrics, etc.
        """
        if self.total_checks == 0:
            return {'failure_rates': np.zeros(self.queue_size), 'total_checks': 0}
            
        failure_rates = self.failure_counts / max(self.total_checks, 1)
        
        return {
            'failure_rates': failure_rates,
            'failure_counts': self.failure_counts.copy(),
            'total_checks': self.total_checks,
            'most_failing_condition': np.argmax(failure_rates) if np.sum(failure_rates) > 0 else None,
            'average_checks_per_validation': self.total_checks / max(1, sum(self.failure_counts))
        }
    
    @staticmethod
    def _update_index_mapping(validator, sorted_indices):
        """Abstract function to update index mapping after reordering.
        
        Args:
            validator: ConditionValidator instance
            sorted_indices: List of indices in the new order
        """
        # Update index mapping
        new_original_to_current = [0] * validator.queue_size
        new_current_to_original = [0] * validator.queue_size
        
        for new_idx, old_idx in enumerate(sorted_indices):
            original_idx = validator.current_to_original_index[old_idx]
            new_original_to_current[original_idx] = new_idx
            new_current_to_original[new_idx] = original_idx
        
        validator.original_to_current_index = new_original_to_current
        validator.current_to_original_index = new_current_to_original
    
    def optimize_queue_order(self, queue_order: Optional[List[int]] = None):
        """Reorder the queue to prioritize conditions most likely to fail.

        Args:
            queue_order: Optional list of indices to reorder the queue.
            If None, uses current failure statistics.

        This implements the adaptive ordering benefit described in the methodology.
        """
        if queue_order is None:
            # Sort condition indices by failure rate (descending)
            failure_rates = self.failure_counts / max(self.total_checks, 1)
            sorted_indices = np.argsort(failure_rates)[::-1]

            logger.debug(f"Queue reordered based on failure rates: {failure_rates[sorted_indices]}")
        else:
            # Validate input queue_order
            if len(queue_order) != self.queue_size or set(queue_order) != set(range(self.queue_size)):
                raise ValueError("queue_order must contain all indices from 0 to queue_size-1 exactly once")
            
            sorted_indices = [self.original_to_current_index[i] for i in queue_order]

            logger.debug(f"Queue reordered based on provided order: {queue_order}")
        
        self.condition_functions = [self.condition_functions[i] for i in sorted_indices]
        self._update_index_mapping(self, sorted_indices)

        # Reset queue front and update failure counts accordingly
        self.queue_front = 0
        self.failure_counts = self.failure_counts[sorted_indices]
    
    def reset_statistics(self):
        """Reset failure statistics for fresh analysis."""
        self.failure_counts = np.zeros(self.queue_size)
        self.total_checks = 0
        self.validation_times = 0


def create_convergence_condition_validator(
    convergence_funcs: Union[OrderedDict[str, Callable], List[tuple[str, Callable]]],
    tolerance: Union[float, List[float]] = 1e-6,
    num_values: int = 1,
) -> tuple[ConditionValidator, ErrorConditionCollector]:
    """Factory function to create a ConditionValidator and ErrorConditionCollector for general convergence criteria.
    
    Args:
        convergence_funcs: List of tuples (name, callable function) or an OrderedDict mapping names to functions.
        tolerance: Tolerance threshold for convergence conditions (can be a single value or a list matching the number of conditions)
        
    Returns:
        ConditionValidator instance for convergence conditions
        ErrorConditionCollector instance to collect error values
    """

    if isinstance(tolerance, float):
        # If a single tolerance value is provided, convert to list
        tolerance = [tolerance] * len(convergence_funcs)
    elif isinstance(tolerance, list) and len(tolerance) != len(convergence_funcs):
        raise ValueError("If tolerance is a list, it must match the number of convergence functions.")
    
    if isinstance(convergence_funcs, Dict):
        Warning("Using OrderedDict for convergence_funcs is recommended for consistent ordering.\n"
                "Or one can use a list of tuples (name, function), which will be automatically converted to an OrderedDict.")
        convergence_funcs = OrderedDict(convergence_funcs)
    elif isinstance(convergence_funcs, list) and isinstance(convergence_funcs[0], tuple):
        # Convert list of tuples to OrderedDict
        convergence_funcs = OrderedDict(convergence_funcs)
    
    if num_values > 1:
        conditions = [
            ErrorConditionWrapperEx(func, tol, num_values=num_values, name=name)
            for (name, func), tol in zip(convergence_funcs.items(), tolerance)
        ]
    elif num_values == 1:
        conditions = [
            ErrorConditionWrapper(func, tol, name=name)
            for (name, func), tol in zip(convergence_funcs.items(), tolerance)
        ]
    else:
        raise ValueError("num_values must be at least 1 for ErrorConditionWrapper or ErrorConditionWrapperEx.")

    validator = ConditionValidator(conditions)
    collector = ErrorConditionCollector(conditions)
    return validator, collector


class ConvergenceConditionManager:
    """Simple manager for tracking convergence conditions and computing active/converged indices."""
    
    def __init__(self, validator: ConditionValidator, num_conditions: int, tolerance: float):
        self.validator = validator
        self.num_conditions = num_conditions
        self.tolerance = tolerance
        self.converged_mask = [False] * num_conditions
    
    def compute_conditions(self, detailed_info, kkt_errors, required_conditions=None):
        """Compute active and converged conditions based on current state.
        
        Returns:
            tuple: (active_conditions, converged_conditions) as lists of indices
        """
        active_conditions = []
        converged_conditions = []

        # Get base validator (Remove the possible decorators)
        validator = self.validator
        while True:
            if not hasattr(validator, 'validator'):
                break
            validator = validator.validator
        
        queue_size = validator.get_num_conditions()
        
        # Determine active conditions
        if required_conditions:
            active_conditions = required_conditions.copy()
        else:
            num_checked = detailed_info.get('num_conditions_passed', 0)

            if num_checked > 0:
                current_quene_front = getattr(validator, 'queue_front', 0)
                current_queue = getattr(validator, 'current_to_original_index', list(range(queue_size)))
                queue_front = current_queue[current_quene_front]

                for i in range(min(num_checked, queue_size)):
                    condition_idx = (queue_front + i) % queue_size
                    active_conditions.append(condition_idx)
        
        # Find newly converged conditions
        for i, kkt_error in enumerate(kkt_errors):
            if (kkt_error is not None and 
                    kkt_error <= self.tolerance and 
                    not self.converged_mask[i]):
                converged_conditions.append(i)
                self.converged_mask[i] = True
        
        return active_conditions, converged_conditions
    
    def reset(self):
        """Reset convergence tracking."""
        self.converged_mask = [False] * self.num_conditions
