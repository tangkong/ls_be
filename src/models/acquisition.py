""" 
A collection of acquisition functions in the gpytorch interface.  
Analytic acquisition functions provide an acquisition landscape through the 
.forward() method, from which a maximizer is chosen.  

access to the posterior is possible via self._get_posterior()
access to training points is possible via X???
"""

import torch
from botorch.models.model import Model
from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.acquisition.objective import ScalarizedObjective # ???
from botorch.utils.transforms import t_batch_mode_transform
from typing import Optional

class MaxVariance(AnalyticAcquisitionFunction):
    def __init__(self, 
                model: Model,
                objective: Optional[ScalarizedObjective]=None,
                maximize: bool = True):
        super().__init__(model=model, objective=objective)
        self.maximize = maximize

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: torch.Tensor) -> torch.Tensor: 
        posterior = self._get_posterior(X=X) 
        mean = posterior.mean
        view_shape = mean.shape[:-2] if mean.dim() >= X.dim() else X.shape[:-2]
        variance = posterior.variance.clamp_min(1e-9).view(view_shape)

        return variance

class Random(AnalyticAcquisitionFunction):
    def __init__(self, 
                model: Model,
                objective: Optional[ScalarizedObjective]=None,
                maximize: bool = True):
        super().__init__(model=model, objective=objective)
        self.maximize = maximize

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: torch.Tensor) -> torch.Tensor: 
        posterior = self._get_posterior(X=X) 
        mean = posterior.mean
        view_shape = mean.shape[:-2] if mean.dim() >= X.dim() else X.shape[:-2]
        rands = torch.mul( mean - mean + 1, torch.rand_like(mean))
        rands = rands.clamp_min(1e-9).view(view_shape)

        return rands