"""
Toy physics model and fitter implementation
"""

import os, sys, datetime

# from utils.plotting.standard_modules import *

# from utils.plotting.animation import AnimatedFigure
# from utils.maths.distribution import Distribution
# from utils.maths.fitting import fit, Scaling
from scipy import optimize
from scipy.stats import poisson, norm
import scipy.stats as stats
# from utils.maths.stats import get_chi2_critical_values_for_sigma
# from utils.cache_tools import Cachable
# from utils.filesys_tools import get_file_stem

import math
import numpy as np
import copy

import collections

#
# Added stuff we need
#


def get_bins(min_val,max_val,dtype=float,num=None,width=None) :
    '''
    Get binning between the specified min/max values.
    Must specify either `num` bins, or bin `width` to define the bins themselves.
    Can also specify `dtype`, where `int` is a special case
    '''

    #
    # Check inputs
    #
    #Check user provided num bins or bin width (but not both)
    if (num is not None) and (width is not None) : 
        raise Exception("get_bins : Must specify \"num\" of \"width\" kwarg, not both")
    if (num is None) and (width is None) : 
        raise Exception("get_bins : Must specify either \"num\" or \"width\" kwarg")

    #Check if distribution to be binned is an integer
    integer_binning = dtype == int

    #If int binning, check range values specified are ints
    # if integer_binning :
    #     if type(min_val) != int and not min_val.is_integer() : 
    #         raise Exception( "get_bins : Minimum must be an int when creating int bins" )
    #     if type(min_val) != int and not max_val.is_integer() : 
    #         raise Exception( "get_bins : Maximum must be an int when creating int bins" )
    #     if width is not None : 
    #         if type(width) != int and not width.is_integer() : raise Exception( "get_bins : Step size must be an int when creating int bins" )

    #If int binning, displace bin min/max by -/+0.5 such that bin centers are ints
    #TODO This doesn't work, fix it
    if integer_binning :
        min_val -= 0.5
        max_val += 0.5


    #
    # Case 1: User defined num bins
    #

    if num is not None :

        #Check n is an int
        if not isinstance(num,int): 
            raise Exception( "get_bins : Num bins \"n\" is not an int" )

        #If int binning, adjust max value to ensure width is an int
        if integer_binning :
            width = math.ceil( ( float(max_val) - float(min_val) ) / float(num) )
            max_val = min_val + ( width * num )


    #
    # Case 2: User defined bin width
    #

    #If num bins specified, get step size from range and n
    elif width is not None :

        #Get the corresponding n (may not be an int at this stage, handle later)
        num = math.ceil( ( float(max_val) - float(min_val) ) / float(width) )
        max_val = min_val + ( num * width )


    #return np.arange(min_val,max_val+step,step)
    return np.linspace(min_val,max_val,num+1)


class Scaling(object) :
    '''
    Define a scaling to map a data to array to the range [0,1].
    This is useful for minimizers, machine learning, etc.
    Note that an array extending beyond this range can be scaled using this class.
    '''

    def __init__(self,min_val,max_val) :
        self.min_val = min_val
        self.max_val = max_val
        assert self.min_val < self.max_val
        self.gradient = self.max_val - self.min_val
        self.intercept = self.min_val

    def scale(self,var_array) :
        # assert np.nanmin(var_array) >= self.min_val, "%0.3g >= %0.3g" % (np.min(var_array),self.min_val)
        # assert np.nanmax(var_array) <= self.max_val, "%0.3g <= %0.3g" % (np.max(var_array),self.max_val)
        return ( var_array - self.intercept ) / self.gradient 

    def unscale(self,var_array) :
        # assert np.nanmin(var_array) >= 0.
        # assert np.nanmax(var_array) <= 1.
        return ( var_array * self.gradient ) + self.intercept


class Histogram(object) :
    '''
    N-dimensional histogram class based on np.histogramdd, but adding:
       1) Ability to fill in steps (like TH::Fill in ROOT)
       2) Uncertainty/error handling
       3) Under/OverFlow (UOF) handling
       4) Histogram arithmetic (multiplication, additon, etc), including error propagation
       5) Useful functions for e.g. noemalising, integrating, projecting, etc (including error propagation)
    Tools for plotting these histograms (based on matplotlib) can be found in fridge/utils/plotting/hist.py
    References :
        [1] https://ddavis.fyi/blog/2018-02-08-numpy-histograms/
    Parameters :
      ndim : int
        Number of dimensions
      uof : bool
        Flag indicating if histogram should include Under/OverFlow bins
      use_buffer : bool
        Flag indicating that data should be buffered rather than directly filled to the histogram
        The user can flush manually using the `flush_buffer` method, or alternatively many of the member functions will flush automatically before being run.
    '''

    '''
    TODO list:
    - Add option for passing an "expression" to the histogram so it knows how to fill itself (like in Alex's production histos)
    - Dedicated plotinfo class, supporting e.g. axis projection (can support more than just histograms)
    - 3D and higher dimensions
    - Resurrect option to request boolean binning
    - More tests
    - Support for datetime
    - Possibly could be merged into dashi
    '''

    def __init__(self, ndim, bins, uof=True, x=None, y=None, weights=None, plotinfo=None, use_buffer=False) : #TODO Store plotinfo as kwargs instead? Can then use them if desired later...
        '''
        Must specify dimensionality and binning at construction.
        Optionally can also provide the data (and weights) a construction to fill the histpgram instantly.
        '''

        # Track whether hist has been filled at least once
        self._filled = False

        # Create the histogram, including initialising the binning
        self._init_hist(ndim, bins, uof)

        # Handle buffering
        self.use_buffer = use_buffer
        if self.use_buffer :
            self._init_buffer()

        # Store the plot info (which defines plotting properties)
        self.plotinfo = {} if plotinfo is None else plotinfo 

        # Fill the histogram with the values provided, if any are
        # Note that this handles 1D and 2D, and wiht or without weights, because the 'fill' function knows what to do here
        if x is not None : 
            self.fill(x=x, y=y, weights=weights)
            self.flush_buffer()
        else :
            assert y is None, "Cannot specify `y` unless have also specified `x`"
            assert weights is None, "Cannot specify `weights` unless have also specified `x`"


    @classmethod
    def populate(cls, ndim, bins, uof, hist, sigma2, plotinfo=None, use_buffer=False) :
        '''
        Alternative constructor to create this Histogram object using data from an existing histogram
        This existing histogram doesn't neccessarily need to be this type
        '''

        h = cls(ndim=ndim, bins=bins, uof=uof, use_buffer=use_buffer)

        assert h._hist.shape == hist.shape, "Hist shape mismatch : %s != %s" % (h._hist.shape, hist.shape)
        if sigma2 is not None :
            assert hist.shape == sigma2.shape
            assert h._sigma2.shape == sigma2.shape, "sigma2 shape mismatch : %s != %s" % (h._sigma2.shape, sigma2.shape)

        h._hist += hist
        if sigma2 is not None :
            h._sigma2 += sigma2

        h.plotinfo = plotinfo

        return h


    @classmethod
    def from_pisa(cls, pisa_map) :
        '''
        Alternative constructor, using a PISA Map as an input
        '''

        from pisa.core.map import Map, MapSet

        # Handle single-entry MapSet case
        if isinstance(pisa_map, MapSet) :
            assert len(pisa_map)
            pisa_map = pisa_map[0]

        # Check inputs
        assert isinstance(pisa_map, Map)

        # Get binning
        binning = pisa_map.binning
        ndim = binning.num_dims
        assert ndim < 3, "Only 1D and 2D histograms supported currently"
        bin_eges = [ edges.m for edges in binning.bin_edges ]

        h = cls(ndim=ndim, bins=bin_eges, uof=False)

        h._hist += pisa_map.nominal_values
        h._sigma2 += np.square(pisa_map.std_devs)

        return h


    def copy(self) :
        '''
        Return a copy of this histogram
        '''

        self.flush_buffer()

        return Histogram.populate(
            ndim=self.ndim,
            bins=self._bins_arg,
            uof=self.uof,
            hist=self._hist,
            sigma2=self._sigma2,
            plotinfo=self.plotinfo,
            use_buffer=self.use_buffer,
        )

    def __copy__(self) :
        return self.copy()

    def __deepcopy__(self,memo) :
        return self.copy()


    def _init_hist(self, ndim, bins, uof) :
        '''
        Setup the binning (including under/overflow) and create the histogram
        Get the main binning by performing a dummy call to the numoy histogram 
        function and letting it figure it out. The numpy binning stuff isn't in
        a standlone function so can't call just that bit.
        Add the under and overflow bins afterwards.
        Note that not supporting giving just a number of bins, as want histogram 
        to be valid even if never filled. So must provide bin edges.
        '''

        # Get dimensions
        self._ndim = ndim
        assert self._ndim in [1,2], "Cannot create histogram : Only 1D and 2D supported at this time"

        # If user specified special "bool" flag for the binning, convert to dedicated boolean binning
        if (bins is bool) or ( isinstance(bins, str) and (bins == "bool") ) :
            assert self.ndim == 1 #TODO Handle higher dimensional cases
            bins = [-0.5, 0.5, 1.5]
            uof = False

        # Store bins arg before we start messing with it, need it later 
        self._bins_arg = bins

        # Store flag indicating whether Under/OverFlow bins are being used
        self._uof = uof

        # Convert arg to array
        bins = np.asarray(bins)

        # Check no NaNs in bins
        #TODO Make work for 2D hists
        # assert not np.any(np.isnan(bins)), "Found NaN values in binning : %s" % bins

        # Handle nuance of 1D histograms with np.histogramdd
        # Want to be able to provide a 1D arrays for bins, not a single element 
        # list of arrays (support either case)
        if self.ndim == 1 and bins.ndim == 1 :
            bins = np.array([bins,])

        # Check the bins
        assert bins.shape[0] == self.ndim, "Bad binning provided, shape must be [%i,num bins], found %s" % (self.ndim,bins.shape)

        # Make a dummy call to the hist fiunction, using some dummy data of the correct shape
        tmp_sample = np.squeeze(np.zeros((self.ndim,2)))
        _, tmp_bin_edges = np.histogramdd( sample=tmp_sample, bins=bins )

        # Add under/overflow bins if required
        if self.uof :
            tmp_bin_edges = np.asarray([ np.concatenate([[-np.inf],e,[np.inf]]) for e in tmp_bin_edges ])

        # Store the bins as a member
        self._bin_edges = tmp_bin_edges

        # Create empty histogram and errors
        hist_shape = np.array([ e.size-1 for e in self._bin_edges ]) # 1 less bin that bin edge
        self._hist = np.zeros(hist_shape)
        self._sigma2 = np.zeros(hist_shape)
        self._sigma_upper = None
        self._sigma_lower = None


    def _init_buffer(self) :
        '''
        Initialize buffering. This allows histograms fill values to be buffered and the actually filling to be
        done less frequently, which can improve speed significantly (at the cost of needing more memory for the
        buffer itself).
        '''

        # Create buffers
        # Note that these may get set to None during filling depending on what type of data the user provides
        self._buffer_x = []
        self._buffer_y = [] if self.ndim > 1 else None
        self._buffer_weights = []


    def flush_buffer(self) :
        '''
        Flush the buffering (to fill the hist)
        '''

        if self.use_buffer :

            # No checks performed here, already done when buffers filled...

            # Only fill hist if there is any data
            if len(self._buffer_x) > 0 :
                self._fill(x=self._buffer_x, y=self._buffer_y, weights=self._buffer_weights)

            # Clear buffers
            self._buffer_x = []
            if self._buffer_y is not None :
                self._buffer_y = []
            if self._buffer_weights is not None :
                self._buffer_weights = []


    def __str__(self) :
        return str(self.hist)


    @property
    def filled(self) :
        return self._filled


    @property
    def ndim(self) :
        '''
        Return number of dimensions
        '''
        return self._ndim


    @property
    def uof(self) :
        '''
        Return flag indicating whether using Under/OverFlow bins or not
        '''
        return self._uof


    def _axis_index(self,axis) :
        '''
        Return the index for the specified axis
        User can specify e.g. 0,1,2, or "x","y","z", but the function always returns a number
        '''

        # If axis is None, eitehr compalin or default to only axis for 1D
        if axis is None :
            if self.ndim == 1 :
                return 0
            else :
                raise Exception("No axis specified for a multi-dimensional array")

        return_axis = None

        # Convert string axis label to an index
        if isinstance(axis, string_types) :
            if axis.lower() in ["x","i","u"] :
                return_axis = 0  
            elif axis.lower() in ["y","j","v"] :
                return_axis = 1  
            elif axis.lower() in ["z","k","w"] :
                return_axis = 2  
            else :
                raise Exception("Unrecognised axis '%s'" % axis)
        else :
            return_axis = axis

        # Check axis range
        assert return_axis < self.ndim, "Axis %i out of range fo %iD histogram" % (return_axis, self.ndim)

        return return_axis


    def bin_edges(self,axis=None,squeeze=True, grid=False) :
        '''
        Return the bin edges, omitting the over/underflow bins
        Shape is [num dimension,num bins+1]
        Can optionally specify a particular axis
        '''

        # Check args
        assert not ( (axis is not None) and grid ), "Cannot specify both `axis` and `grid` args"

        # Get the bin edges for each dimension and form an array
        # Remove the UOF bins if present
        if self.uof :
            edges = np.asarray([ e[1:-1] for e in self._bin_edges ])
        else :
            edges = self._bin_edges

        # Only return a single axis if requested
        if axis is not None :
            edges = [edges[self._axis_index(axis)]]

        # Squeeze out single element dimensions
        if squeeze :
            edges = np.squeeze(edges)

        # Combine into a grid
        if grid and (self.ndim > 1) :
            edges = np.meshgrid(*edges, indexing="ij")

        return edges


    def bin_centers(self, axis=None, squeeze=True, grid=False) :
        '''
        Return the bin centers, omitting the over/underflow bins
        Shape is [num dimension,num bins]
        Can optionally specify a particular axis
        '''

        #TODO log10 bin option

        # Check args
        assert not ( (axis is not None) and grid ), "Cannot specify both `axis` and `grid` args"

        # Get the bin centers for each dimension and form an array
        # Not including UOF bins here ("center" makes no sense)
        centers = np.asarray([ get_bin_centers(e) for e in self.bin_edges(axis=axis,squeeze=False) ])

        # Squeeze out single element dimensions
        if squeeze :
            centers = np.squeeze(centers)

        # Combine into a grid
        if grid and (self.ndim > 1) :
            centers = np.meshgrid(*centers, indexing="ij")

        return centers


    def bin_widths(self,axis=None,squeeze=True, grid=False) :
        '''
        Return the bin widths, omitting the over/underflow bins
        Shape is [num dimension,num bins]
        '''

        # Check args
        assert not ( (axis is not None) and grid ), "Cannot specify both `axis` and `grid` args"

        # Get the bin widths for each dimension and form an array
        # Not including UOF bins here ("width" makes no sense)
        widths = np.asarray([ np.abs(e[1:]-e[:-1]) for e in self.bin_edges(axis=axis,squeeze=False) ])

        # Squeeze out single element dimensions
        if squeeze :
            widths = np.squeeze(widths)

        # Combine into a grid
        if grid and (self.ndim > 1) :
            widths = np.meshgrid(*widths, indexing="ij")

        return widths



    @property
    def shape(self) :
        return self.hist.shape

    @property
    def num_bins(self) : 
        return self.hist.size

    @property
    def _hist_uarray(self) :
        '''
        Return underlying histogram + errors as uarray, including Under/OverFlow (UOF) bins
        '''
        assert UNCERTAINTIES_AVAIL, "This functionality requires the `uncertainties` module"

        self.flush_buffer()

        return uarray(self._hist, self._sigma)


    @property
    def hist(self) :
        '''
        Return the histogram, excluding Under/OverFlow (UOF) bins
        Most of the time this is the default behaviour required by a user
        '''

        self.flush_buffer()

        if self.uof :
            return self._hist[ tuple([slice(1,-1)]*self.ndim) ]
        else :
            return self._hist


    @property
    def hist_uarray(self) :
        '''
        Return underlying histogram + errors as uarray, excluding Under/OverFlow (UOF) bins
        '''
        assert UNCERTAINTIES_AVAIL, "This functionality requires the `uncertainties` module"
        return uarray(self.hist,self.sigma)


    @property
    def sigma2(self) :
        '''
        Return the sigma2 values, excluding Under/OverFlow (UOF) bins
        Most of the time this is the default behaviour required by a user
        '''

        self.flush_buffer()

        if self.uof :
            return self._sigma2[ tuple([slice(1,-1)]*self.ndim) ]
        else :
            return self._sigma2


    @property
    def sigma_upper(self) :
        '''        
        Return the upper bound values, excluding Under/OverFlow (UOF) bins
        Most of the time this is the default behaviour required by a user 
        '''

        self.flush_buffer()

        if self._sigma_upper is None:
            return None
        elif self.uof :
            return self._sigma_upper[ tuple([slice(1,-1)]*self.ndim) ]
        else :
            return self._sigma_upper

    @property
    def sigma_lower(self) :
        '''
        Return the lower bound values, excluding Under/OverFlow (UOF) bins
        Most of the time this is the default behaviour required by a user
        '''

        self.flush_buffer()

        if self._sigma_lower is None:
            return None
        elif self.uof :
            return self._sigma_lower[ tuple([slice(1,-1)]*self.ndim) ]
        else :
            return self._sigma_lower

    @property
    def _sigma(self) :
        '''
        Return the sigma, including Under/OverFlow (UOF) bins
        '''

        self.flush_buffer()

        return np.sqrt(self._sigma2)


    @property
    def sigma(self) :
        '''
        Return the sigma, excluding Under/OverFlow (UOF) bins
        '''
        return np.sqrt(self.sigma2)


    @property
    def variance(self) :
        '''
        Alias
        '''
        return self.sigma2


    def fill(self, x, y=None, weights=None) :
        '''
        Fill the histogram with the data (and optionally weights) provided
        Can be called multiple times
        '''

        self._filled = True


        #
        # Check inputs
        #

        # Check input arguments match the dimensions
        if self.ndim == 1 and y is not None :
            raise Exception("Cannot fill histogram : x and y values provided for a 1D histogram")

        # Handle pint quantities
        if hasattr(x,"magnitude") : 
            x = x.magnitude
        if hasattr(y,"magnitude") : 
            y = y.magnitude
        if hasattr(weights,"magnitude") : 
            weights = weights.magnitude

        # Handle boolean values
        #TODO doesn't work, fix
        # if np.dtype(np.asarray(x)) is bool :
        #     x = x.astype(int)
        # if y is not None :
        #     if np.dtype(np.asarray(y)) is bool :
        #         y = y.astype(int)


        # Handle datetimes
        #TODO Doesn't work, fix this
        '''
        def convertDatetimesToNumbers(vals) :
            if vals is not None :
            if len(vals) > 0 :
                if isinstance(vals[0],datetime.datetime) :
                to_timestamp = np.vectorize(lambda x: (x - datetime.datetime(1970, 1, 1)).total_seconds())
                return to_timestamp(vals)
            return vals
        x = convertDatetimesToNumbers(x)
        y = convertDatetimesToNumbers(y)
        '''


        #
        # Buffer
        #

        # Check if using buffer
        if self.use_buffer :

            # For optional args (y, weights), they must either all be None or never be None
            if y is None :
                self._buffer_y = None
            else :
                assert self._buffer_y is not None, "`y` must always be None or never be None when buffering (cannot combine)"

            if weights is None :
                self._buffer_weights = None
            else :
                assert self._buffer_weights is not None, "`weights` must always be None or never be None when buffering (cannot combine)"

            # Need type to be a list here #TODO is this inefficient?
            if isinstance(x, np.ndarray) :
                x = x.tolist()
            assert isinstance(x, list)

            if y is not None :
                if isinstance(y, np.ndarray) :
                    y = y.tolist()
                assert isinstance(y, list)

            if weights is not None :
                if isinstance(weights, np.ndarray) :
                    weights = weights.tolist()
                assert isinstance(weights, list)

            # Fill the buffers
            self._buffer_x.extend(x)
            if y is not None :
                self._buffer_y.extend(y)
            if weights is not None :
                self._buffer_weights.extend(weights)

        else :

            # Just fill
            self._fill(x=x, y=y, weights=weights)



    def _fill(self, x, y=None, weights=None) :
        '''
        Called by `fill`, user should not fill directly
        '''

        #
        # Create a histogram
        #

        # Populate a local histogram for these input values and add them to the overall values for this histogram
        data = np.asarray(x)
        if y is not None :
            if hasattr(np,"stack") :
                data = np.stack([data,y])
            else :
                data = _numpy_stack([data,y]) #Horror hack for old numpy version
            data = data.T

        # Define default weights if none given (required for `binned_statistic_dd` call below)
        if weights is None : 
            weights = np.ones_like(x)
        weights = np.asarray(weights)

        # Fill a histogram from the data passed to this function
        new_hist,_ = np.histogramdd( sample=data, bins=self._bin_edges, weights=weights )
 
        # Get sum of weights squared, for calculating bin count errors
        new_sigma2, _, _ = stats.binned_statistic_dd( sample=data, values=weights, bins=self._bin_edges, statistic=lambda w : np.sum(np.square(w)) )


        #
        # Add to overall histogram
        #

        self._hist = self._hist + new_hist
        self._sigma2 = self._sigma2 + new_sigma2


    def compatible(self,other_hist) :
        '''
        Check if another hist is compatible with this one e.g.
        same number of dimensions and same binning.
        This can be used to test if can do airthmetic between 
        the hists amongst other things.
        '''

        compatible = True

        # Check the two hists have the same dimension
        if self.ndim != other_hist.ndim :
            compatible = False

        # Check both hists have the same binning values
        # Do this for each dimension individually (makes it easier to handle numpy shape stuff in 1D vs ND)
        for this_e,other_e in zip(self._bin_edges, other_hist._bin_edges) :
            if not np.array_equal(this_e,other_e) :
                compatible = False
                break

        return compatible


    def _update_uarray(self,hist_uarray) :
        '''
        Update the underlying histogram data structures from a uarray
        Used by various arihtmetic functions, users should not be using this directly
        '''
        assert UNCERTAINTIES_AVAIL, "This functionality requires the `uncertainties` module"
        self._hist = nominal_values(hist_uarray)
        self._sigma2 = np.square(std_devs(hist_uarray))


    def __add__(self,other) : 
        '''
        Add something to this histogram; another hist, an array, a scalar
        Using the `uncertainties` module for error propagation
        '''
        assert UNCERTAINTIES_AVAIL, "This functionality requires the `uncertainties` module"

        self.flush_buffer()
        if isinstance(other,Histogram) :
            other.flush_buffer()

        output_hist = self.copy()

        A = output_hist._hist_uarray
        B = other._hist_uarray if isinstance(other,Histogram) else other
        result = A + B
        output_hist._update_uarray(result)

        return output_hist

    def __radd__(self,other):
        return self.__add__(other)


    def __sub__(self,other) : 
        '''
        S8ubtract something from this histogram; another hist, an array, a scalar
        Using the `uncertainties` module for error propagation
        '''

        assert UNCERTAINTIES_AVAIL, "This functionality requires the `uncertainties` module"

        self.flush_buffer()
        if isinstance(other,Histogram) :
            other.flush_buffer()

        output_hist = self.copy()

        A = output_hist._hist_uarray
        B = other._hist_uarray if isinstance(other,Histogram) else other
        result = A - B
        output_hist._update_uarray(result)

        return output_hist

    def __rsub__(self,other):
        return self.__sub__(other)


    def __mul__(self,other) :
        '''
        Multiply something by this histogram; another hist, an array, a scalar
        Using the `uncertainties` module for error propagation
        '''

        assert UNCERTAINTIES_AVAIL, "This functionality requires the `uncertainties` module"

        self.flush_buffer()
        if isinstance(other,Histogram) :
            other.flush_buffer()

        output_hist = self.copy()

        A = output_hist._hist_uarray
        B = other._hist_uarray if isinstance(other,Histogram) else other
        result = A * B
        output_hist._update_uarray(result)

        return output_hist

    def __rmul__(self,other) :
        return self.__mul__(other)


    def __truediv__(self,other) : 
        '''
        Divide something by this histogram; another hist, an array, a scalar
        Using the `uncertainties` module for error propagation
        Handling divide by zero
        '''
        assert UNCERTAINTIES_AVAIL, "This functionality requires the `uncertainties` module"

        self.flush_buffer()
        if isinstance(other,Histogram) :
            other.flush_buffer()

        # Copy self to get the hist to return
        output_hist = self.copy()
        A = output_hist._hist_uarray

        # Get the denominator
        B = other._hist_uarray if isinstance(other,Histogram) else other

        # Do the division, handling divide by 0 errors
        result = unc_div(A,B)

        # Write back to the hist instance
        output_hist._hist = nominal_values(result)
        output_hist._sigma2 = np.square(std_devs(result))

        return output_hist


    def __floordiv__(self,other) : 
        '''
        Also specify floor'd version of division (py3 requirement)
        Remove uncertainty in this case as is no longer valid after this operation
        '''
        new_hist = self.__truediv__(other) # py2 compatibility
        new_hist._hist = np.floor(new_hist)
        new_hist._sigma2 = np.full_like( new_hist._hist, np.NaN )
        return new_hist


    def __div__(self,other) : 
        return self.__truediv__(other) # py2 compatibility

    # def __rdiv__(self,other) : 
    #     return self.__div__(other)


    def transpose(self) :
        '''
        Return a histogram that is a transpose of this one
        '''
        assert self.ndim > 1, "`transpose` not supported for 1D histograms"

        self.flush_buffer()

        new_hist = self.copy()
        new_hist._hist = self._hist.T
        new_hist._sigma2 = self._sigma2.T
        new_hist._bin_edges = self._bin_edges.T
        #TODO plotinfo
        return new_hist

    @property
    def T(self) :
        '''
        Alias, a la numpy
        '''
        return self.transpose()


    def norm(self,axis=None) :
        '''
        Return a normalised histogram
        Optionally can normalise in a single axis only
        '''

        assert UNCERTAINTIES_AVAIL, "This functionality requires the `uncertainties` module"

        self.flush_buffer()

        # Normalise
        # Use nominal values only, e.g. do not increase uncertainty
        # If user only wants to nomalise along a particular axis, project first
        if axis is None :
            return self / self.sum().nominal_value
        else :
            return self / self.proj(axis=axis).hist # This is the nominal value only


    def norm_to_peak(self) :
        '''
        Return a histogram normalised w.r.t. the peak (e.g. the max value)
        '''

        assert UNCERTAINTIES_AVAIL, "This functionality requires the `uncertainties` module"

        self.flush_buffer()

        # Normalise w.r.t the maximum bin count (e.g. this bin with now = 1)
        # Use nominal values only, e.g. do not increase uncertainty
        return self / nominal_values( np.nanmax(self.hist) )


    def cum(self,axis=None,cdf=False) :
        '''
        Return a cumulative version of this hist (in the axis specified)
        Optionally show as Cumulative Distribution Function (CDF)
        '''

        assert UNCERTAINTIES_AVAIL, "This functionality requires the `uncertainties` module"

        self.flush_buffer()

        output_hist = self.copy()

        # Get the index for this axis
        axis = self._axis_index(axis)

        # Get cumulative histogram
        cumsum_uarray = np.cumsum(output_hist._hist_uarray,axis=axis)

        # Convert to CDF
        # Do this by dividing by the summed bin count across the other dimensions for each bin in the specified axis
        # Only use the nomial values of the denominator, don't double count errors #TODO Is this correct (also for `norm` function)?
        if cdf :

            # If CDF limiting value is 0., set CDF=0 rather than inf
            div0_result = 0.

            # Get the sum for each bin in this axis
            bin_sums = np.sum(output_hist._hist,axis=axis)

            # Now divide through
            #TODO Make the calulation below more general (general problem of how to divide a numpy array along an arb. axis)
            if self.ndim == 1 :
                cumsum_uarray = unc_div( cumsum_uarray, bin_sums, div0_result=div0_result )
            else :
                for i in range(0,len(bin_sums)) :
                    if axis == 0 :
                        cumsum_uarray[:,i] = unc_div( cumsum_uarray[:,i], bin_sums[i], div0_result=div0_result )
                    elif axis == 1 :
                        cumsum_uarray[i,:] = unc_div( cumsum_uarray[i,:], bin_sums[i], div0_result=div0_result )
                    else :
                        raise Exception("CDF not yet impelemnted for histograms with more than 2 dimensions")

        output_hist._update_uarray(cumsum_uarray)

        return output_hist


    def cdf(self,axis=None) :
        '''
        Return a cumulative Distrubtion Function in the specified axis
        Alias to cum(cdf=True)
        '''
        return self.cum(axis=axis,cdf=True)

    '''
    #TODO Maybe plotinfo should be a class? Can then use same class for HistogramSet
    @staticmethod
    def _xproj_plotinfo(plotinfo) :
        new_plotinfo = copy.deepcopy(plotinfo)
        if "ylabel" in new_plotinfo :            
            new_plotinfo.pop("ylabel")
        if "ylog" in new_plotinfo :          
            new_plotinfo.pop("ylog")
        if "ycut" in new_plotinfo :          
            new_plotinfo.pop("ycut")
        if "zlabel" in new_plotinfo : 
            new_plotinfo["ylabel"] = new_plotinfo.pop("zlabel")
        if "zlog" in new_plotinfo : 
            new_plotinfo["ylog"] = new_plotinfo.pop("zlog")
        if "zcut" in new_plotinfo : 
            new_plotinfo["ycut"] = new_plotinfo.pop("zcut")
        return  new_plotinfo
    @staticmethod
    def _yproj_plotinfo(plotinfo) :
        new_plotinfo = copy.deepcopy(plotinfo)
        if "xlabel" in new_plotinfo : 
            new_plotinfo.pop("xlabel")
        if "xlog" in new_plotinfo : 
            new_plotinfo.pop("xlog")
        if "xcut" in new_plotinfo : 
            new_plotinfo.pop("xcut")
        if "ylabel" in new_plotinfo :            
            new_plotinfo["xlabel"] = new_plotinfo.pop("ylabel")
        if "ylog" in new_plotinfo :          
            new_plotinfo["xlog"] = new_plotinfo.pop("ylog")
        if "ycut" in new_plotinfo :          
            new_plotinfo["xcut"] = new_plotinfo.pop("ycut")
        if "zlabel" in new_plotinfo : 
            new_plotinfo["ylabel"] = new_plotinfo.pop("zlabel")
        if "zlog" in new_plotinfo : 
            new_plotinfo["ylog"] = new_plotinfo.pop("zlog")
        if "zcut" in new_plotinfo : 
            new_plotinfo["ycut"] = new_plotinfo.pop("zcut")
        return new_plotinfo
    '''

    def proj(self,axis) :
        '''
        Return a histogram that is a 1D projection onto the specified axis
        '''

        assert UNCERTAINTIES_AVAIL, "This functionality requires the `uncertainties` module"

        self.flush_buffer()

        assert axis is not None, "Must choose an axis for projection"
        
        axis = self._axis_index(axis)

        # Nothing to do for a 1D histogram
        if self.ndim == 1 : 
            return self 

        new_hist_uarray = sum([ np.sum(self._hist_uarray,axis=a) for a in range(self.ndim) if a != axis ])

        new_bin_edges = self.bin_edges(axis=axis)

        new_plotinfo = self.plotinfo #self._xproj_plotinfo(self.plotinfo) #TODO

        return Histogram.populate(1,bins=new_bin_edges,uof=self.uof,hist=nominal_values(new_hist_uarray),sigma2=np.squuare(std_devs(new_hist_uarray)),plotinfo=new_plotinfo)


    def sum(self) :
        '''
        Return the sum of the histogram
        '''

        self.flush_buffer()

        assert UNCERTAINTIES_AVAIL, "This functionality requires the `uncertainties` module"
        return np.sum(self._hist_uarray)


    def uof_sum(self) :
        '''
        Return the sum of Under/OverFlow (UOF) bins
        ''' 
        assert UNCERTAINTIES_AVAIL, "This functionality requires the `uncertainties` module"
        assert self.uof, "Not using Under/OverFlow bins"

        self.flush_buffer()

        return np.sum(self._hist_uarray) - np.sum(self.hist_uarray)


    def uof_per_axis(self) :
        '''
        Return under and overflow value for each axis
        NOT broken up into all bins, just total under and total overflow per axis
        '''

        assert UNCERTAINTIES_AVAIL, "This functionality requires the `uncertainties` module"
        assert self.uof, "Not using Under/OverFlow bins"

        self.flush_buffer()

        uof_indices = [0,-1]
        if self.ndim == 1 :
            return self._hist_uarray[uof_indices]
        elif self.ndim >= 2 :
            return np.array([ self.proj(axis)._hist_uarray[uof_indices] for axis in range(self.ndim) ])


    def clear_uof(self) :
        '''
        Return a copy of the histogram where the Under/OverFlow (UOF) bins have been cleared (e.g. set to zero)
        '''

        assert self.uof, "Not using Under/OverFlow bins"

        self.flush_buffer()

        output_hist = self.copy()

        uof_indices = [0,-1]

        #TODO Make more general...
        if self.ndim == 1 :
            output_hist._hist[uof_indices] = 0.
            output_hist._sigma2[uof_indices] = 0.
        elif self.ndim == 2 :
            output_hist._hist[uof_indices,:] = ufloat(0.,0.)
            output_hist._sigma2[uof_indices,:] = ufloat(0.,0.)
            output_hist._hist[:,uof_indices] = ufloat(0.,0.)
            output_hist._sigma2[:,uof_indices] = ufloat(0.,0.)

        return output_hist


    def integral(self) :
        '''
        Get the histogram integral (e.g. the summed volume of each each bin, i.e. the area under the curve)
        This does NOT include under/overflow bins (don't know the bin widths, e.g. integrating with binned interval)
        '''
        assert UNCERTAINTIES_AVAIL, "This functionality requires the `uncertainties` module"

        self.flush_buffer()

        integral = self.hist_uarray * self.bin_widths()
        return np.sum(integral)


    def mean(self,axis=None) :
        '''
        Return the histogram mean
        Optionally can provide an axis if only want mean in that axis
        Ignore the Under/OverFlow (UOF) bins, as do not know the bin center value so cannot compute mean
        '''

        assert UNCERTAINTIES_AVAIL, "This functionality requires the `uncertainties` module"

        self.flush_buffer()

        means = []            
        for axis in list(range(self.ndim)) if axis is None else ([self._axis_index(axis)]) :
            h = self.proj(axis=axis)
            means.append( unc_div( np.dot(h.hist_uarray,h.bin_centers()), h.sum() ) )
        return means[0] if self.ndim == 1 else np.squeeze(means)

    '''
    #TODO Think about these, need to be clear how these are not the bin content uncertainties, but are instead the spread of the hist
    @property
    def xstddev(self) :
        # Using definition from ROOT TH1F (https://root.cern.ch/doc/master/classTH1.html#aad1141a0a166972e1c032c07f46a21ee)
        hist = self.hist if self.ndim == 1 else self.xproj.hist
        mean = self.xmean
        centers = self.xcenters
        variance = ( (hist * centers**2 ).sum() / hist.sum() ) - (mean**2)
        return np.sqrt(variance)
    @property
    def ystddev(self) :
        if self.ndim == 1 : raise Exception("Cannot get y standard deviation for 1D histogram")
        hist = self.hist if self.ndim == 1 else self.yproj.hist
        mean = self.ymean
        centers = self.ycenters
        variance = ( (hist * centers**2 ).sum() / hist.sum() ) - (mean**2)
        return np.sqrt(variance)
    '''



    def cut(self, bin_edge_index=None, bin_edge=None) :
        '''
        Cut/split hist at some bin edge
        Returns the two sides of the split (less than, greater than)
        Can either provide the bin edge index or the numerical value (not both) 
        '''

        self.flush_buffer()

        # Check inputs
        assert not ( (bin_edge_index is None) and (bin_edge is None) ), "Must specify either `bin_edge` or `bin_edge_index`"
        assert not ( (bin_edge_index is not None) and (bin_edge is not None) ), "Must specify either `bin_edge` or `bin_edge_index` (not both)"

        #TODO Add support for 2D
        assert self.ndim == 1, "Does not yet support 2D or higher"

        # If user provide a bin edge value, get the corresponding index
        bin_edges = self.bin_edges(squeeze=False)[0] # This does NOT include UOF bins
        if bin_edge is not None :
            bin_edge_index = (np.abs(bin_edges - bin_edge)).argmin() # Find index of closest element
            assert np.isclose(bin_edges[bin_edge_index], bin_edge), "`bin_edge` value (%0.3g) provided does not match any bin edge in the histogram : %s" % (bin_edge,bin_edges) # Check it is sufficiently close #TODO arg to steer tolerance

        # Check index
        num_bin_edges = len(bin_edges)
        assert bin_edge_index < num_bin_edges, "`bin_edge_index` out-of-range"

        # Get the slice offset (depends on whether there are UOF bins)
        offset = 1 if self.uof else 0

        # Make the slices
        slice_lt = slice( None, bin_edge_index+offset )
        slice_gt = slice( bin_edge_index+offset, None)

        # Cut the bin edges (include any UOF bins)
        # Need to add new trailing edge for "<" hist
        bin_edges_lt = self._bin_edges[0][slice_lt]
        bin_edges_gt = self._bin_edges[0][slice_gt]
        bin_edges_lt = np.append(bin_edges_lt,[bin_edges_gt[0]])

        # Cut the histogram itself (+ errors)
        hist_lt = self._hist[slice_lt]
        sigma2_lt = self._sigma2[slice_lt]
        hist_gt = self._hist[slice_gt]
        sigma2_gt = self._sigma2[slice_gt]

        # Handle UOF
        if self.uof :
            # Need to add the leading/trailer UOF bin back in (with 0 entries because they were cut)
            bin_edges_lt = np.append( bin_edges_lt, [np.inf] )
            hist_lt = np.append( hist_lt, [0.] )
            sigma2_lt = np.append( sigma2_lt, [0.] )
            bin_edges_gt = np.append( [-np.inf], bin_edges_gt )
            hist_gt = np.append( [0.], hist_gt )
            sigma2_gt = np.append( [0.], sigma2_gt )
            # Also keep versions with no UOF bins for passing to the `_bins_arg` variable
            bin_edges_lt_no_uof = bin_edges_lt[1:-1]
            bin_edges_gt_no_uof = bin_edges_gt[1:-1]

        # Create the new hists
        new_hist_lt = self.copy()
        new_hist_lt._bin_edges = [bin_edges_lt]
        new_hist_lt._bins_arg = bin_edges_lt_no_uof #TODO wrong if UOF?
        new_hist_lt._hist = hist_lt
        new_hist_lt._sigma2 = sigma2_lt

        new_hist_gt = self.copy()
        new_hist_gt._bin_edges = [bin_edges_gt]
        new_hist_gt._bins_arg = bin_edges_gt_no_uof  #TODO wrong if UOF?
        new_hist_gt._hist = hist_gt
        new_hist_gt._sigma2 = sigma2_gt

        return new_hist_lt, new_hist_gt


    def clip(self,a_min,a_max) :
        '''
        Return a new hist with clipped values
        '''

        #TODO errors

        self.flush_buffer()

        new_hist = self.copy()
        np.clip( new_hist._hist, a_min=a_min, a_max=a_max, out=new_hist._hist )
        return new_hist


    def split(self, axis) :
        '''
        Split a 2D histogram into a series of 1D histograms, along the specified axis
        '''

        #TODO Doesn't currently support overflow/underflow bins

        assert self.ndim == 2, "Can only split 2D histograms"

        self.flush_buffer()

        output_hists = collections.OrderedDict()

        split_axis = self._axis_index(axis)
        keep_axis = 1 if split_axis == 0 else 0

        # Get the split bin edges
        split_bin_edges = self.bin_edges(axis=split_axis)
        if self.uof :
            split_bin_edges = np.concatenate([ [-np.inf], split_bin_edges, [np.inf] ])

        # Grab the binning for the split hist
        assert len(self._bins_arg) == self.ndim
        split_hist_bins_arg = self._bins_arg[keep_axis]

        # New dimensions
        split_hist_ndim = self.ndim - 1

        # Loop over bins in the split axis
        for bin_index in range(len(split_bin_edges)-1) :

            # Grab the hist and errors for this bin
            if split_axis == 1 :
                split_hist_hist = self._hist[:,bin_index]
                split_hist_sigma2 = self._sigma2[:,bin_index]
            else :
                split_hist_hist = self._hist[bin_index,:]
                split_hist_sigma2 = self._sigma2[bin_index,:]

            # Create a new hist with the split info
            hist = Histogram.populate(
                ndim=split_hist_ndim,
                bins=split_hist_bins_arg,
                uof=self.uof,
                hist=split_hist_hist,
                sigma2=split_hist_sigma2,
            )

            # Create a key from the edges
            if split_bin_edges[bin_index] == -np.inf :
                edges_key = "Underflow"
            elif split_bin_edges[bin_index+1] == np.inf :
                edges_key = "Overflow"
            else :
                edges_key = (split_bin_edges[bin_index], split_bin_edges[bin_index+1])

            # Add the new hist and the corresponding bin edges to the outputs
            output_hists[edges_key] = hist

        # Convert to a histogram set
        output_hists = HistogramSet.populate(hists_dict=output_hists)

        return output_hists


    def poisson_fluctuate(self, random_state=None) :
        '''
        Return a new histogram where he in counts have been flucutated according to a Poisson distribution
        Copied from PISA Map.fluctuate (https://github.com/IceCubeOpenSource/pisa/blob/master/pisa/core/map.py)
        '''

        from scipy.stats import poisson

        self.flush_buffer()

        new_hist = self.copy()

        # Update counts
        new_hist._hist = poisson.rvs( self._hist, random_state=random_state )

        # Update erorrs
        #TODO what to do here?

        return new_hist


    @property
    def xlim(self) :
        '''
        For 1D histograms, return the binning limit
        Useful for formattig plots
        '''
        #TODO also 2D
        assert self.ndim == 1
        edges = self.bin_edges(squeeze=True)
        return (edges[0], edges[-1])


def test_Histogram() :

    # A helper function
    def assert_hist(h,expected_vals,expected_sigma,msg) :
        if False :
            print("-------------")
            print((h.hist))
            print(expected_vals)
            print((h.sigma,h.sigma2))
            print(expected_sigma)
            print("-------------")
        assert np.allclose(h.hist, expected_vals), msg+" (hist values)"
        assert np.allclose(h.sigma, expected_sigma), msg+" (hist errors)"

    assert UNCERTAINTIES_AVAIL, "Histogram tests require `uncertainties` moodule"

    #
    # General functions (1D)
    #

    # Create test hist
    bins = get_bins(-0.5,2.5,num=3)
    h1 = Histogram(1,bins=bins)

    # Test empty before adding anything
    assert np.all(h1.hist==0.), "Histogram not empty" 
    assert np.all(h1.sigma==0.), "Histogram not empty" 
    
    # Add some data
    h1.fill([1.,2.])
    h1.fill([1.,1.])
    h1.fill([1.])
    assert_hist(h1,[0.,4.,1.],[0.,2.,1.],"Filling hist failed")

    # Add some weighted data
    h1.fill([0.],weights=[2.])
    assert_hist(h1,[2.,4.,1.],[2.,2.,1.],"Filling hist with weights failed")

    # Normalise
    assert_hist(h1.norm(),[2./7.,4./7.,1./7.],[2./7.,2./7.,1./7.],"Normalising hist failed")

    # Sum
    h_sum = h1.sum()
    assert h_sum.nominal_value == 7, "Hist sum failed (hist values)"
    assert h_sum.std_dev == 3, "Hist sum failed (hist errors)"

    # Integral
    h_int = h1.integral()
    assert h_int.nominal_value == 7., "Hist integral failed (hist values)"
    #TODO Test error

    # Mean
    h_mean = h1.mean()
    assert h_mean.nominal_value == 6./7., "Hist mean failed (hist values)"
    #TODO Test error

    # Cumulative hist
    #assert_hist(h1.cum(),[2.,6.,7.],[0.,0.,0.],"Cumulative hist failed")
    #TODO Test error

    # CDF
    #assert_hist(h1.cum(cdf=True),[2./7.,6./7.,7./7.],[0.,0.,0.],"CDF failed")
    #TODO Test error


    #
    # General functions (2D)
    #

    # Create a 1D and 2D hist for testing
    bins = get_bins(-0.5,2.5,num=3)
    h1 = Histogram(1,bins=bins)
    h2 = Histogram(2,bins=(bins,bins))

    #
    # 2D
    #

    h1 = Histogram(2,bins=(bins,bins),x=[1.,1.],y=[1.,1.])
    h2 = Histogram(2,bins=(bins,bins),x=[1.,2.],y=[1.,0.])

    h_sum = h1 + h2
    assert np.all( h_sum.hist == np.array([[0.,0.,0.],[0.,3.,0.],[1.,0.,0.]]) )


    #
    # Histogram arithmetic
    #

    # Compare to result from the `uncertainties` module to test
    try :
        from uncertainties import ufloat
        from uncertainties.unumpy import uarray, nominal_values, std_devs
    except :
        raise Exception("Cannot test `hist.py` : `uncertainties` module not available")

    bins = get_bins(0.,10.,num=5)
    hist1 = Histogram(1,bins=bins,x=np.random.uniform(bins[0],bins[-1],size=100))
    hist2 = Histogram(1,bins=bins,x=np.random.uniform(bins[0],bins[-1],size=100))

    u1 = uarray(hist1.hist,hist1.sigma)
    u2 = uarray(hist2.hist,hist2.sigma)

    h_add = hist1 + hist2
    u_add = u1 + u2
    assert np.array_equal(h_add.hist,nominal_values(u_add)), "Histogram addition values error" 
    assert np.array_equal(h_add.sigma,std_devs(u_add)), "Histogram addition sigma error" 

    h_subtract = hist1 - hist2
    u_subtract = u1 - u2
    assert np.array_equal(h_subtract.hist,nominal_values(u_subtract)), "Histogram subtraction values error" 
    assert np.array_equal(h_subtract.sigma,std_devs(u_subtract)), "Histogram subtraction sigma error" 

    h_multiply = hist1 * hist2
    u_multiply = u1 * u2
    assert np.array_equal(h_multiply.hist,nominal_values(u_multiply)), "Histogram multiplication values error" 
    assert np.array_equal(h_multiply.sigma,std_devs(u_multiply)), "Histogram multiplication sigma error" 

    h_divide = hist1 / hist2
    u_divide = u1 / u2
    assert np.array_equal(h_divide.hist,nominal_values(u_divide)), "Histogram division values error" 
    assert np.array_equal(h_divide.sigma,std_devs(u_divide)), "Histogram division sigma error" 

    #TODO test Historgam-array arithmetic (take care with UoF bins)


    #
    # Histogram sigma
    #

    bins = [0.,1.,2.]
    hist = Histogram(1,bins=bins,x=[0.5,0.5,1.])
    assert np.array_equal(hist.hist,[2.,1.]), "Histogram weights values error" 
    assert np.array_equal(hist.sigma,[np.sqrt(2.),1.]), "Histogram weights sigma error"

    bins = [0.,1.,2.]
    hist = Histogram(1,bins=bins,x=[0.5,1.],weights=[2.,1.])
    assert np.array_equal(hist.hist,[2.,1.]), "Histogram weights values error" 
    assert np.array_equal(hist.sigma,[2.,1.]), "Histogram weights sigma error"


    #
    # Histogram functions
    #

    # bins = [0.,1.,2.]
    # hist = Histogram(1,bins=bins,x=[0.5,0.5,1.])
    # hist = hist.norm()
    # assert np.array_equal(hist.hist,[2/3.,1/3.]), "Histogram norm error" 


#
# Test statistics
#


def get_poission_negative_log_likelihood(observed_hist, expected_hist):

    O = observed_hist.hist
    E = expected_hist.hist

    # PISA version
    neg_llh = np.sum(E - (O * np.log(E)))
    neg_llh -= np.sum(O - (O * np.log(O)))

    return neg_llh


#
# Main analysis class
#


class AnalysisBase():
    """
    Base class for a physics model, where the model has free parameters that can be fit to (pseudo)data

    This is basically a super simple version of PISA
    """

    def __init__(self, cache_dir=None, physics_params=None):

        # Caching
        if cache_dir is None:
            cache_dir = os.path.join(
                os.path.dirname(__file__), ".cache", self.__class__.__name__
            )
        # Cachable.__init__(self, cache_dir=cache_dir)

        # Store args
        self.physics_params = physics_params

        # Core state
        self.params = collections.OrderedDict()
        self.reset()

        # Init some defaults
        self.set_metric("poisson_llh", get_poission_negative_log_likelihood)

    def set_metric(self, name, func):
        self.metric_name = name
        self._metric_func = func

    def generate_mc_events(self, *args, **kwargs):

        # TODO For some reason the hash of the realoded events object is not the same as the original. Need to investigate why, but for the time being have disabled events caching...
        # Load cached events if available
        # events, func_call_hash = self.load_cached_results("generate_mc_events", locals())

        # # Run function if no cached results available
        # if events is None :
        if True:

            # Generate evets
            events = self._generate_mc_events(*args, **kwargs)

            # Check them
            assert isinstance(
                events, collections.abc.Mapping
            ), "`_generate_mc_events` must return a dict"
            assert len(events) > 0, "`_generate_mc_events` has returned no data"
            num_events = None
            for k, v in events.items():
                assert isinstance(
                    k, str
                ), "`_generate_mc_events` events dict key must be strings"
                assert isinstance(
                    v, np.ndarray
                ), "`_generate_mc_events` events dict svalues must be numpy arrays"
                if num_events is None:
                    num_events = v.size
                else:
                    assert (
                        v.size == num_events
                    ), "All arryas must have same number of events"

            # Save to cache
            # self.save_results_to_cache("generate_mc_events", func_call_hash, events)

        # Store as member
        self.events = events

        return events

    def _generate_mc_events(self):
        raise Exception(
            "Derived class must overload the `_generate_mc_events` function"
        )

    def pipeline(self, *args, **kwargs):
        hist = self._pipeline(events=self.events, *args, **kwargs)
        assert isinstance(hist, Histogram), "`pipeline` must return a `Histogram`"
        return hist

    def _pipeline(self, events):
        raise Exception("Derived class must overload the `_pipeline` function")

    def get_template(self):
        return self.pipeline()

    def get_asimov_data(self):
        return self.pipeline()

    def get_trial_data(self, trial_index):
        """
        Get trial data, e.g. with statistical fluctuations
        """

        # TODO option to regenerate MC (with the target num events) instead of rvs
        # TODO trial index

        # First get Asimov hist
        hist = self.pipeline()

        # Fluctuate bin counts
        random_state = np.random.RandomState(trial_index + 1)
        hist._hist = poisson.rvs(hist._hist, random_state=random_state)

        return hist

    def reset(self):

        # Reset minimization state variables
        self._num_iterations = 0
        self._animate = False

        # Reset params
        for p in self.free_params.values():
            p.reset()
            p.value = p.nominal_value

    def _minimizer_callback(self, x, data):
        """
        x = params
        args = other useful stuff
        """

        # Set the param values
        free_params = list(self.free_params.values())
        assert x.size == len(free_params)
        for i in range(x.size):
            free_params[i].scaled_value = x[i]

        # Get the template
        template = self.get_template()

        # Clip
        template = template.clip(a_min=1.0e-5, a_max=np.inf)
        data = data.clip(a_min=1.0e-5, a_max=np.inf)

        # Metric
        metric = self._metric_func(expected_hist=template, observed_hist=data)

        # Take any priors into account
        metric_penalty = 0.0
        for p in self.params.values():
            metric_penalty += p.prior_penalty()
        metric += metric_penalty

        # Animation
        if self._animate:
            self._animated_fig.get_ax().clear()
            self.plot_data_template_comparison(
                ax=self._animated_fig.get_ax(), data=data, template=template
            )
            self._animated_fig.get_ax().set_title("-LLH=%0.3g" % metric)
            self._animated_fig.snapshot()

        # Counter
        self._num_iterations += 1

        return metric

    def plot_data_template_comparison(self, ax, data, template):
        plot_hist(ax=ax, hist=template, color="red", errors="band", label="Template")
        plot_hist(
            ax=ax,
            hist=data,
            color="black",
            errors="band",
            linestyle="None",
            label="Data",
        )  # TODO errorbar

    def _fit(self, data, options=None, animate=False, minimizer_algorithm="SLSQP"):
        """
        Fitting function
        """

        # TODO take a copy of the params and pass to pipeline? rather than resetting

        start_time = datetime.datetime.now()

        # Get some default options (need to be well tuned to the specififc problem normally though
        # or else get things like 'Inequality constraints incompatible' )
        # if options is None :
        #     options = { "ftol":1.e-9, "eps":1.e-9 }

        # Reset params
        self.reset()

        # Handle animation
        self._animate = animate  # Needs to be a member to pass to the callback
        if self._animate:
            self._animated_fig = AnimatedFigure(nx=1, ny=1)
            self._animated_fig.start_capture(outfile="animation.mp4")

        # Get the (scaled) param bounds and initial values
        param_initial_guesses = [
            p.scaled_nominal_value for p in list(self.free_params.values())
        ]
        param_bounds = [p.scaled_bounds for p in list(self.free_params.values())]

        # Perform the minimization
        minimizer_results = optimize.minimize(
            self._minimizer_callback,
            x0=param_initial_guesses,
            bounds=param_bounds,
            args=data,
            method=minimizer_algorithm,
            options=options,
        )

        # Finish animation
        if self._animate:
            self._animated_fig.stop_capture()
            self._animated_fig = None

        # Extract results
        success = minimizer_results["success"]
        if not success:
            print("WARNING : Fit failed with error '%s'" % minimizer_results["message"])
        metric = minimizer_results["fun"]  # if success else np.NaN

        # Get best fit template and param values
        best_fit_template = self.get_template()
        best_fit_params = copy.deepcopy(self.params)

        # Time logging
        time_taken = datetime.datetime.now() - start_time

        # Collect results
        results = {
            "success": success,
            "metric": metric,
            "best_fit_template": best_fit_template,
            "best_fit_params": best_fit_params,
            "time_taken": time_taken,
            "num_iterations": self._num_iterations,
        }

        # Reset
        self.reset()

        return results

    def fit(self, *args, fit_null=False, **kwargs):

        # Load cached results if available
        results, func_call_hash = self.load_cached_results("fit", locals())

        # Run function if no cached results available
        if results is None:

            #
            # Perform fit
            #

            fit_results = self._fit(*args, **kwargs)

            results = collections.OrderedDict()
            results["fit"] = fit_results

            #
            # Perform null fit
            #

            if fit_null:

                assert self.physics_params is not None

                # Fix physics params
                for param_name in self.physics_params:
                    assert param_name in self.params
                    self.params[param_name].fixed = True

                # Fit
                null_fit_results = self._fit(*args, **kwargs)
                results["null_fit"] = null_fit_results

                # Unfix physics params
                for param_name in self.physics_params:
                    assert param_name in self.params
                    self.params[param_name].fixed = False

                # Compute mismodelling
                assert self.metric_name == "poisson_llh"
                ndof = len(self.physics_params)
                mismod_test_stat = 2.0 * (
                    null_fit_results["metric"] - fit_results["metric"]
                )  # Wilks theorem

                results["mismodeling"] = {"ndof": ndof, "test_stat": mismod_test_stat}

            # Save to cache
            self.save_results_to_cache("fit", func_call_hash, results)

        # Done
        return results

    def profile(self, data, scan, **fit_kw):
        """
        Run a profile metric scan (such as a profile likelihood)
        """

        # Load cached results if available
        results, func_call_hash = self.load_cached_results("profile", locals())

        # Run function if no cached results available
        if results is None:

            # Check scan
            self._check_scan(scan)
            assert len(scan) == 1, ">1 scan params not yet supported"

            # First do a fit with all params free
            free_fit_results = self._fit(data=data, **fit_kw)
            assert free_fit_results["success"]

            # Fix the scan params
            for param_name in scan.keys():
                assert not self.params[param_name].fixed
                self.params[param_name].fixed = True

            # Scan
            scan_results = []
            for param_name, param_values in scan.items():
                for param_val in param_values:

                    print("Profile scan point : %s" % param_val)

                    # Set param
                    self.params[param_name].value = param_val

                    # Fit
                    scan_point_results = self._fit(data=data, **fit_kw)
                    scan_results.append(scan_point_results)

            # Unfix the scan params and reset original vals
            for param_name in scan.keys():
                self.params[param_name].fixed = False
            self.reset()

            # Store the results
            results = {
                "free_fit_results": free_fit_results,
                "scan_points": scan,
                "scan_results": scan_results,
            }

            # Save to cache
            self.save_results_to_cache("profile", func_call_hash, results)

        # Done
        return results

    def _check_scan(self, scan):
        """
        Check the user-defined scan points
        """
        assert isinstance(scan, collections.abc.Mapping)
        for n, v in scan.items():
            assert n in self.free_params
            assert isinstance(v, np.ndarray)

    @property
    def free_params(self):
        return collections.OrderedDict(
            [(n, p) for n, p in self.params.items() if not p.fixed]
        )

    @property
    def fixed_params(self):
        return collections.OrderedDict(
            [(n, p) for n, p in self.params.items() if p.fixed]
        )

    def plot_fit(self):
        """
        Plot fit results
        """

        raise Exception("Needs updating")

        add_heading_page("Fit")

        all_figs = []

        #
        # Plot data vs template (single trial only)
        #

        trial_index = 0
        trial = self.trial_data[trial_index]

        fig = Figure(title="Trial %i" % trial_index)
        all_figs.append(fig)

        plot_hist(
            ax=fig.get_ax(),
            hist=trial["fit"]["best_fit_template"],
            color="red",
            errors="band",
            label="Fitted template",
        )
        plot_hist(
            ax=fig.get_ax(),
            hist=trial["data"],
            color="black",
            errors="bar",
            linestyle="None",
            label="Data",
        )

        fig.quick_format()

        #
        # Plot fit values for selected params
        #

        nx, ny = get_grid_dims(n=len(self.free_params))
        fig = Figure(
            nx=nx, ny=ny, title="Fitted values (%i trials)" % len(self.trial_data)
        )
        all_figs.append(fig)

        for i, param_name in enumerate(self.free_params.keys()):

            ax = fig.get_ax(i=i)

            param_fit_vals = [
                t["fit"]["best_fit_params"][param_name].value for t in self.trial_data
            ]

            hist = Histogram(
                ndim=1, bins=generate_bins(param_fit_vals, num=20), x=param_fit_vals
            )
            plot_hist(ax=ax, hist=hist, errors="band", label="Trial fits")

            ax.axvline(
                x=self.trial_data[0]["truth_values"][param_name],
                color="purple",
                label="Truth",
            )

            percentiles = np.percentile(
                param_fit_vals, [50.0 - (68.0 / 2.0), 50.0, 50.0 + (68.0 / 2.0)]
            )
            ax.axvline(x=percentiles[1], color="grey", linestyle="--", label=r"Median")
            ax.axvline(
                x=percentiles[0], color="grey", linestyle=":", label=r"$1 \sigma$"
            )
            ax.axvline(x=percentiles[2], color="grey", linestyle=":")

            ax.set_xlabel(param_name)

        fig.hide_unused_axes()
        fig.quick_format(ylabel="Num trials", ylim=(0.0, None))

        # Done
        return all_figs

    def plot_profile(self):
        """
        Plot profile results
        """

        raise Exception("Needs updating")

        add_heading_page("Profile")

        all_figs = []

        #
        # Plot profile
        #

        fig = Figure(nx=2, title="Profile scan (%i trials)" % len(self.trial_data))
        all_figs.append(fig)

        if self.num_trials > 0:

            #
            # Plot prifle for trials
            #

            # Get data
            ndim = len(self.trial_data[0]["profile"]["scan_points"])
            assert ndim == 1, "Only 1D plotting supported currently"
            scan_param_name = list(self.trial_data[0]["profile"]["scan_points"].keys())[
                0
            ]
            scan_points = list(self.trial_data[0]["profile"]["scan_points"].values())[0]
            scan_param_truth = self.trial_data[0]["truth_values"][scan_param_name]

            # Loop over trials
            trial_scan_metrics, trial_scan_test_stats = [], []
            for i_trial, trial in enumerate(self.trial_data):

                # Plot the metric
                scan_metrics = np.array(
                    [r["metric"] for r in trial["profile"]["scan_results"]]
                )
                fig.get_ax(x=0).plot(
                    scan_points,
                    scan_metrics,
                    color="orange",
                    alpha=0.1,
                    label=("Trials" if i_trial == 0 else None),
                )

                # Plot the test stat
                scan_test_stats = 2.0 * (
                    scan_metrics - trial["profile"]["free_fit_results"]["metric"]
                )
                fig.get_ax(x=1).plot(
                    scan_points,
                    scan_test_stats,
                    color="orange",
                    alpha=0.1,
                    label=("Trials" if i_trial == 0 else None),
                )

                # Also store for median calculation later
                trial_scan_metrics.append(scan_metrics)
                trial_scan_test_stats.append(scan_test_stats)

            # Median
            scan_metrics_median = np.median(trial_scan_metrics, axis=0)
            fig.get_ax(x=0).plot(
                scan_points, scan_metrics_median, color="red", label="Median"
            )
            scan_test_stats_median = np.median(trial_scan_test_stats, axis=0)
            fig.get_ax(x=1).plot(
                scan_points, scan_test_stats_median, color="red", label="Median"
            )

        if self.asimov_data is not None:

            #
            # Plot profle for trials
            #

            ndim = len(self.asimov_data["profile"]["scan_points"])
            assert ndim == 1, "Only 1D plotting supported currently"

            scan_param_name = list(self.asimov_data["profile"]["scan_points"].keys())[0]
            scan_points = list(self.asimov_data["profile"]["scan_points"].values())[0]

            scan_param_truth = self.asimov_data["truth_values"][
                scan_param_name
            ]  # TODO check matches trials

            scan_metrics = np.array(
                [r["metric"] for r in self.asimov_data["profile"]["scan_results"]]
            )
            fig.get_ax(x=0).plot(
                scan_points, scan_metrics, color="black", linestyle="--", label="Asimov"
            )

            scan_test_stats = 2.0 * (
                scan_metrics - self.asimov_data["profile"]["free_fit_results"]["metric"]
            )
            fig.get_ax(x=1).plot(
                scan_points,
                scan_test_stats,
                color="black",
                linestyle="--",
                label="Asimov",
            )

        # Overlay sigma lines
        critical_vals = get_chi2_critical_values_for_sigma(ndim, [1, 2, 3])
        color_scale = ColorScale("Greys_r", len(critical_vals) + 1)
        for k, v in critical_vals.items():
            fig.get_ax(x=1).axhline(
                v, color=color_scale.get_next(), label=r"$%i \sigma$" % k
            )

        # Formatting
        for ax in fig.get_all_ax():
            ax.axvline(scan_param_truth, color="purple", label="Truth")

        fig.get_ax(x=0).set_ylabel(r"$\rm{LLH}$")
        fig.get_ax(x=1).set_ylabel(r"$-2 \Delta \rm{LLH}$")

        fig.hide_unused_axes()
        fig.quick_format(xlabel=scan_param_name, ylim=(0.0, None))

        #
        # Plot free fits
        #

        fig = Figure(title="Free fits")
        all_figs.append(fig)

        scan_param_free_fit_results = [
            t["profile"]["free_fit_results"]["best_fit_params"][scan_param_name].value
            for t in self.trial_data
        ]

        percentiles = np.percentile(
            scan_param_free_fit_results,
            [50.0 - (68.0 / 2.0), 50.0, 50.0 + (68.0 / 2.0)],
        )

        hist = Histogram(
            ndim=1,
            bins=generate_bins(scan_param_free_fit_results, num=20),
            x=scan_param_free_fit_results,
        )
        plot_hist(ax=fig.get_ax(), hist=hist, errors="band", label="Trial fits")

        fig.get_ax().axvline(
            x=self.trial_data[0]["truth_values"][scan_param_name],
            color="purple",
            label="Truth",
        )

        fig.get_ax().axvline(
            x=percentiles[1], color="grey", linestyle="--", label=r"Median"
        )
        fig.get_ax().axvline(
            x=percentiles[0], color="grey", linestyle=":", label=r"$1 \sigma$"
        )
        fig.get_ax().axvline(x=percentiles[2], color="grey", linestyle=":")

        fig.quick_format(xlabel=scan_param_name)

        # Done
        return all_figs


class AnalysisParam(object):
    """
    Class representating a parameter in the model
    """

    # TODO Replace with utils.maths.fitting.Param?
    # TODO add a prior

    def __init__(self, value, bounds=None, fixed=False, prior_sigma=None):
        self.nominal_value = copy.deepcopy(value)
        self.value = value
        self.bounds = bounds
        self.fixed = fixed
        self.prior_sigma = prior_sigma  # This specifies a Gaussian prior

        if self.bounds is None:
            assert self.fixed  # TODO enforce as part of a setter
        else:
            assert len(self.bounds) == 2
            assert np.all(np.isfinite(self.bounds))
            self.scaling = Scaling(min_val=self.bounds[0], max_val=self.bounds[1])

    def reset(self):
        self.value = self.nominal_value

    @property
    def scaled_value(self):
        assert not self.fixed
        return self.scaling.scale(self.value)

    @scaled_value.setter
    def scaled_value(self, sv):
        assert not self.fixed
        self.value = self.scaling.unscale(sv)

    @property
    def scaled_nominal_value(self):
        assert not self.fixed
        return self.scaling.scale(self.nominal_value)

    @property
    def scaled_bounds(self):
        assert not self.fixed
        return tuple(
            [self.scaling.scale(self.bounds[0]), self.scaling.scale(self.bounds[1])]
        )

    def prior_penalty(self):
        """
        Penalty term from a Gaussian prior
        This assumes the metric is LLH
        """
        if self.prior_sigma is not None:
            x = self.value
            m = self.nominal_value
            s = self.prior_sigma
            return (x - m) ** 2 / (
                2 * s ** 2
            )  # Removed - sign, sign using -llh as metric
        else:
            return 0.0


class Hypersurface():
    """
    A simple hypersurface implementation

    Actually is only a line right now...
    """

    def __init__(self, bins, gradient_bounds, intercept_bounds):

        self.bins = bins

        assert self.bins.ndim == 1
        shape = self.bins.size - 1

        self.gradient = np.full(shape, np.NaN)
        self.intercept = np.full(shape, np.NaN)

        self.gradient_bounds = gradient_bounds
        self.intercept_bounds = intercept_bounds

    @property
    def shape(self):
        return self.gradient.shape

    def _func(self, value, gradient, intercept):
        return (gradient * value) + intercept

    def fit(self, nominal_mc, sys_mc):
        """
        Each MC set should be defined as:
            {
                "hist" : the histogram resulting from the MC
                "value" : the parameter value
            }
        """

        # TODO caching

        # Norm to the nominal
        self.hists = [nominal_mc["hist"]] + [s["hist"] for s in sys_mc]
        self.hists = [h / nominal_mc["hist"] for h in self.hists]

        # Get the param values
        x = [nominal_mc["value"]] + [s["value"] for s in sys_mc]

        # Steer fit
        p0 = [0.0, 0.0]  # TODO steerable
        bounds = [
            (self.gradient_bounds[0], self.intercept_bounds[0]),
            (self.gradient_bounds[1], self.intercept_bounds[1]),
        ]

        # Loop over bins
        for bin_idx in np.ndindex(self.shape):

            # Skip any with no data in the nominal hist
            if nominal_mc["hist"].hist[bin_idx] == 0.0:
                continue

            # Get the normalised bin counts
            y = [h.hist[bin_idx] for h in self.hists]
            y_sigma = [h.sigma[bin_idx] for h in self.hists]

            # Define callback
            def callback(x, *p):
                return self._func(value=x, gradient=p[0], intercept=p[1])

            # Fit
            popt, pcov = optimize.curve_fit(
                callback,
                x,
                y,
                sigma=y_sigma,
                p0=p0,
                bounds=bounds,
                maxfev=1000000,
                # method="dogbox", # lm, trf, dogbox
            )

            # Write fit results to the member variables
            self.gradient[bin_idx] = popt[0]
            self.intercept[bin_idx] = popt[1]

    def evaluate(self, value):
        return self._func(value=value, gradient=self.gradient, intercept=self.intercept)

    def __call__(self, value):
        return self.evaluate(value=value)

    def plot(self, ax, x, bin_idx, **kw):
        y = [self.evaluate(xx)[bin_idx] for xx in x]
        ax.plot(x, y, **kw)


#
# Simple example
#


class ExampleAnalysis(AnalysisBase):
    def __init__(self):

        super(ExampleAnalysis, self).__init__()

        # Define the free params of a normal distribution
        self.params["norm"] = AnalysisParam(value=1.0, bounds=(0.0, 10.0), fixed=False)
        self.params["mean"] = AnalysisParam(
            value=100.0, bounds=(80.0, 120.0), fixed=False
        )
        self.params["sigma"] = AnalysisParam(value=5.0, bounds=(0.1, 10.0), fixed=False)

    def _generate_mc_events(self, random_state=None):

        events = collections.OrderedDict()

        if random_state is None:
            random_state = np.random.RandomState()

        num_events = 1000000
        num_events_weighted = 100000

        events["true_x"] = random_state.uniform(
            self.params["mean"].bounds[0], self.params["mean"].bounds[1], num_events
        )
        # events["reco_x"] = andom_state.normal(events["true_x"], 5.)
        events["reco_x"] = events["true_x"]

        events["weights"] = np.full_like(
            events["true_x"], float(num_events_weighted) / float(num_events)
        )

        return events

    def _pipeline(self, events):

        reco_x = events["reco_x"]
        weights = events["weights"]

        # Re-weight to desired Gaussian
        weight_mod = np.exp(
            -0.5
            * np.square(
                (reco_x - self.params["mean"].value) / self.params["sigma"].value
            )
        )
        weight_mod *= self.params["norm"].value / np.nanmax(weight_mod)
        new_weights = weight_mod * weights

        # Make hist
        hist = Histogram(
            ndim=1,
            uof=False,
            bins=get_bins(
                self.params["mean"].bounds[0], self.params["mean"].bounds[1], num=20
            ),
            x=reco_x,
            weights=new_weights,
        )

        return hist


if __name__ == "__main__":

    from utils.script_tools import ScriptWrapper
    from utils.filesys_tools import replace_file_ext

    with ScriptWrapper(replace_file_ext(__file__, ".log")) as script:

        # Create model
        model = ExampleAnalysis()
        model.generate_mc_events()

        # Define a profile scan

        # Get some Asimov data
        data = model.get_asimov_data()  # TODO store true params

        # Fit the data
        results = model.fit(data)

        # Profile the data
        scan = {"sigma": np.linspace(3.0, 7.0, num=5)}
        model.profile(data=data, scan=scan)

        # Done
        print("")
        dump_figures_to_pdf(replace_file_ext(__file__, ".pdf"))
