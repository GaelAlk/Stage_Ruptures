from typing import List, Tuple

import numpy as np
import ruptures as rpt
from scipy.optimize import minimize


class Alpin:
    def __init__(self, estimator: rpt.base.BaseEstimator, feature_func=None):

        # check input arguments
        if feature_func is not None and not callable(feature_func):
            raise TypeError("The feature function must a callable or None.")

        self.feature_func = feature_func
        self.estimator = estimator

    def get_signal_features(self, signal: np.ndarray) -> np.ndarray:
        if self.feature_func is not None:
            return self.feature_func(signal)
        # default features: (signal length, constant)
        n_samples = signal.shape[0]
        return np.array([np.log(n_samples), 2.0])

    def target_func_single_signal(
        self, signal: np.ndarray, bkps: List[int], penalty_weights: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """Compute loss and gradient for a single signal and associated labelled
        breakpoints with the current penalty weights.

        Args:
            signal (np.ndarray): signal
            bkps (List[int]): associated breakpoints (label)
            penalty_weights (np.ndarray): current penalty weights

        Returns:
            Tuple[float, np.ndarray]: (loss value, gradient value)
        """
        features = self.get_signal_features(signal)
        pen = np.power(features, penalty_weights).prod()
        predicted_bkps = self.estimator.fit(signal=signal).predict(pen=pen)

        loss_val = (
            self.estimator.cost.sum_of_costs(bkps)
            - self.estimator.cost.sum_of_costs(predicted_bkps)
            + pen * (len(bkps) - len(predicted_bkps))
        )

        return loss_val, (len(bkps) - len(predicted_bkps)) * np.log(features)

    def target_func(self, penalty_weights) -> Tuple[float, np.ndarray]:
        """Return the loss and gradient for the given penalty weights.

        Args:
            penalty_weights (np.ndarray): array containing the current penalty weights

        Returns:
            Tuple[float, np.ndarray]: (loss value, gradient value)
        """
        global_loss_val = 0
        global_grad_val = np.zeros(self.n_features)
        for (signal, bkps) in zip(self.signal_list, self.bkps_list):
            loss_val, grad_val = self.target_func_single_signal(
                signal=signal, bkps=bkps, penalty_weights=penalty_weights
            )
            global_loss_val += loss_val
            global_grad_val += grad_val

        global_loss_val /= len(self.signal_list)
        global_grad_val /= len(self.signal_list)

        return global_loss_val, global_grad_val

    def fit(self, signal_list, bkps_list):
        """Compute the best penalty for the given list of signals and labels.

        Args:
            signal_list (list): list of signals.
            bkps_list (list): list of breakpoint lists.

        Returns:
            self: returns an instance of self.
        """
        # quick sanity check
        err_msg = f"Different number of signals ({len(signal_list)}) and labels ({len(bkps_list)})."
        assert len(signal_list) == len(bkps_list), err_msg

        # set attributes
        self.signal_list = signal_list
        self.bkps_list = bkps_list
        n_features = self.get_signal_features(self.signal_list[0]).shape
        self.n_features = n_features

        # minimize the loss
        solution = minimize(
            fun=self.target_func,
            x0=np.ones(self.n_features),
            method="L-BFGS-B",
            jac=True,
            tol=1e-7,
        )

        # record solution
        self.best_penalty_weights = solution.x
        return self

    def get_best_penalty(self, signal: np.ndarray) -> float:
        """Return the best penalty using the best found penalty weights.

        Args:
            signal (np.ndarray): signal of shape (n_samples, n_dims) or (n_samples,)

        Returns:
            float: penalty value
        """
        features = self.get_signal_features(signal)
        pen = np.power(features, self.best_penalty_weights).prod()
        return pen

    def predict(self, signal: np.ndarray):
        """Call predict on the estimator with the best found penalty weights.

        Args:
            signal (np.ndarray): signal to segment

        Returns:
            bkps (list): list of breakpoint indexes
        """
        bkps = self.estimator.fit(signal=signal).predict(
            pen=self.get_best_penalty(signal=signal)
        )
        return bkps
