import numpy as np
from scipy.optimize import minimize, Bounds

class Optimizer:
    """
    Parameters
    ----------
        assets: array-like
        two-dimensional (t x n) array of returns of n variables over t periods
        
    Returns
    -------
        array of fitted portfolio weights
    """
    def __init__(
        self,
        assets,
    ):
        self.assets = np.array(assets)

    def naive(
        self,
        *args,
        **kwargs
    ):
        """
        Implements naive diversification and returns equal weights among all assets
        """
        return np.ones(self.assets.shape[1]) / self.assets.shape[1]

    def rp(
        self,
        bounds = None,
        *args,
        **kwargs
    ):
        """
        Implements the risk-parity approach and returns weights proportional to the inverse volatility of each asset.
        If the bounds parameter is specified, the weights are adjusted to the lower (upper) bound if the unconstrained
        weight would lie below (above) the bound. Following that, all other weights that lie inside the bounds are scaled down 
        or up proportionally to ensure that the weights sum up to 1
        """
        volatilities = np.std(self.assets, axis=0)
        weights = (1/volatilities) / sum(1/volatilities)
        
        if bounds is None:
            return weights
        else:
            censored = []
            weights = list(weights)
            for index, ((lower, upper), weight) in enumerate(zip(bounds, weights)):
                if weight < lower:
                    weights[index] = lower
                    censored.append(True)
                elif weight > upper:
                    weights[index] = upper
                    censored.append(True)
                else:
                    censored.append(False)
            excess_weights = sum(weights) - 1
            uncensored_weights = sum([weight for (weight, boolean) in zip(weights, censored) if boolean is False])
            if uncensored_weights == 0 and excess_weights != 0:
                raise ValueError("""Bound specification leads to weights that do not sum up to 1. 
                Respecify or loosen the bounds in order to achieve correct results."""
                )
            scale_factor = 1 - excess_weights / uncensored_weights
            weights = [weight * scale_factor if boolean is False else weight for (weight, boolean) in zip(weights, censored)]
            bound_check = any([weight > upper or weight < lower for weight, (lower, upper) in zip(weights, bounds)])
            if bound_check:
                raise ValueError("""No possible solution without violating any bound specification. 
                Respecify or loosen the bounds in order to achieve correct results."""
                )
            
            return weights

    def minvar(
        self,
        constrained = True,
        bounds = None,
        *args,
        **kwargs
    ):
        """
        Returns weights of the minimum-variance portfolio in the mean-variance space
        
        Parameters
        ----------
        constrained : bool
            If False, short-sales are allowed and negative allocations may be returned
            If True, short-sales are not allowed and the bounds parameter is set to an array
            consisting of (0, 1) tuples for each asset
            default : False
        bounds : Array-like object containing tuples of minimum-maximum allocation boundary pairs 
                 for each asset; overwrites the possible bounds values set by constrained
            Example: [(0.3, 1), (0, 1), (0.2, 0.3)]
            default : None
        
        """
        if constrained is False and bounds is None:
            nominator = np.linalg.inv(np.cov(self.assets, rowvar=False)) @ np.ones(np.shape(self.assets)[1])
            denominator = np.ones(np.shape(self.assets)[1]).T @ np.linalg.inv(np.cov(self.assets, rowvar=False)) @ np.ones(np.shape(self.assets)[1])
            return nominator / denominator
        else:
            return _OptimizationWrapper(
                assets = self.assets,
                constrained = constrained,
                bounds = bounds,
                rule = "minvar"
            ).optimize()

    def rrt(
        self,
        *args,
        **kwargs
    ):
        """
        Implements Reward-to-Risk timing and returns weights proportional to the Sharpe Ratio of each asset
        """
        means = np.mean(self.assets, axis=0)
        volatilities = np.std(self.assets, axis=0)
        sr = means / volatilities
        return sr / sum(sr)

    def mean_variance(
        self,
        constrained = True,
        bounds = None,
        maximum_volatility = None,
        *args,
        **kwargs
    ):
        """
        Implements a mean-variance approach with possible short-sale constraints and minimum/maximum
        allocation boundaries for each asset and returns optimal weights that maximize the Sharpe Ratio,
        given the constraints. If no constraints are specified, the weights represent the ex-post tangency portfolio.
        
        Parameters
        ----------
        constrained : bool
            If False, short-sales are allowed and negative allocations may be returned
            If True, short-sales are not allowed and the bounds parameter is set to an array
            consisting of (0, 1) tuples for each asset
            default : False
            
        bounds : Array-like object containing tuples of minimum-maximum allocation boundary pairs 
                 for each asset; overwrites the possible bounds values set by constrained
            Example: [(0.3, 1), (0, 1), (0.2, 0.3)]
            default : None
        
        maximum_volatility : int or float
            Specifies whether the portfolio should have a maximum level of volatility
            default : None
        """
        if constrained is False and bounds is None and maximum_volatility is None:
            nominator = np.linalg.inv(np.cov(self.assets, rowvar=False)) @ np.mean(self.assets, axis=0)
            denominator = np.ones(np.shape(self.assets)[1]).T @ np.linalg.inv(np.cov(self.assets, rowvar=False)) @ np.mean(self.assets, axis=0)
            return nominator / denominator
        else:
            return _OptimizationWrapper(
                assets = self.assets,
                constrained = constrained,
                bounds = bounds,
                maximum_volatility = maximum_volatility,
                rule = "mv"
            ).optimize()


    def hrp(
        self,
        *args,
        **kwargs
    ):
        """
        Implements the Hierarchical Risk-Parity approach of De Prado (2016)
        which is a mixture of risk-parity and tangency weights built upon a clustered and
        hierarchical covariance matrix and has shown to outperform traditional rules out-of-sample
        """
        raise NotImplementedError


class _OptimizationWrapper:
    
    def __init__(
        self,
        assets,
        constrained = False,
        bounds = None,
        maximum_volatility = None,
        rule = None
    ):
        self.assets = assets
        self.constrained = constrained
        self.bounds = bounds
        if self.constrained is True and self.bounds is None:
            self.bounds = [(0, 1) for _ in range(np.shape(self.assets)[1])]
        self.maximum_volatility = maximum_volatility
        if rule not in ("minvar", "mv"):
            raise ValueError('Optimization rule has to be either "minvar" or "mv"')
        self.rule = rule

    def optimize(self):
        init_guess = Optimizer(self.assets).naive()
        kwargs = {}
        unit = {'type': 'eq', 'fun': self._unit_constraint}
        constraints = []
        constraints.append(unit)      
        if self.maximum_volatility is not None:
            vola_bound = {'type': 'ineq', 'fun': self._volatility_bound}
            constraints.append(vola_bound)
        if self.bounds is not None:
            kwargs["bounds"] = self.bounds
        
        results = minimize(fun = self._target_function,
                           x0 = init_guess,
                           method = "SLSQP",
                           constraints = constraints,
                           **kwargs
        )
        
        if results.success:
            return results.x
        else:
            return results
    
    def _target_function(self, weights):
        portfolio = self.assets @ weights
        if self.rule == "minvar":
            return np.std(portfolio)
        elif self.rule == "mv":
            return -np.mean(portfolio) / np.std(portfolio)

    def _unit_constraint(self, weights):
        return sum(weights) - 1

    def _volatility_bound(self, weights):
        cov_matrix = np.cov(self.assets, rowvar=False)
        variance = weights @ cov_matrix @ weights.T
        volatility = np.sqrt(variance)
        return -(variance - self.maximum_volatility)

