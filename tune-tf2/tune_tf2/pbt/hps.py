import numpy as np

class HyperParam:
    """  """
    def __init__(self,
                 min_samp,
                 max_samp,
                 init=None,
                 sample_fn='loguniform',
                 explore_wt=0.2,
                 enforce_limits=False):
        """Represents constraints on hyperparameter 
        values that will be used during PBT.
        
        Parameters
        ----------
        min_samp : float
            The minimum allowed sample
        max_samp : float
            The maximum allowed sample
        init : float
            The initial value to use for PBT, by default 
            None initializes with a sample from the distribution.
        sample_fn : {'loguniform', 'uniform', 'randint'} or callable, optional
            The distribution from which to sample, by default 
            'loguniform'
        explore_wt : float, optional
            The maximum percentage increase or decrease for 
            a perturbation, by default 0.2
        enforce_limits : bool, optional
            Whether to limit exploration to within the sample_range, 
            by default False
        
        Raises
        ------
        ValueError
            When an invalid sample_fn is provided.
        """
        
        # check the sampling range
        assert min_samp < max_samp, \
            "`min_samp` must be smaller than `max_samp`."
        # set up the sampling function
        if callable(sample_fn):
            self.sample = sample_fn
        elif sample_fn in ['uniform', 'loguniform', 'randint']:
            if sample_fn == 'loguniform':
                base = 10
                logmin = np.log(min_samp) / np.log(base)
                logmax = np.log(max_samp) / np.log(base)
                self.sample = lambda _: base**(np.random.uniform(logmin, logmax))
            else:
                self.sample = lambda _: getattr(
                    np.random, sample_fn
                )(low=min_samp, high=max_samp)
        else:
            raise ValueError("Invalid `sample_fn` was specified.")
        # use the initial value if specified, otherwise use sampling function
        self.init = (lambda _: init) if init != None else self.sample
        # save other attributes
        self.min_bound = min_samp if enforce_limits else 0
        self.max_bound = max_samp if enforce_limits else np.inf
        self.explore_wt = explore_wt
