import numpy as np
from math import isclose
from scipy import interpolate

class ReadTheDocumentation(UserWarning):
    pass

# ========== Chris' Probability distribution class =====================
class ProbabilityDistribution:
    """ Chris' Probability distribution class 
    
    This class stores, maintains and provides functions for the manipulation of data
    for a probabiltiy distribution set up within STEEL. For a specific breakdown of how
    this class works, please see it's member functions, but in general:

    1) To be useful, this class must be first initialized with the desired number of 
       'components', which represent separate parts of the probability distribution on
       the same domain, with the intention of allowing them to be disconnected. An example
       might be separate components for early and late type galaxies. If this is doesn't make
       much sense to you, you probably just need one component.

    2) The Probability distribution class is initalized with 3 integer values - these will
       generally correspond to redshift bins, mass bins and finally the desired resolution
       of the probability. This should be calibrated using 'addScale' which will add a scale
       to these values.
    
    3) Operations can then be performed on the distribution (as described below).
    
    """
    def __init__(self, zsteps, msteps, resolution):
        """Initialization of the probability distribution class.

        Note that the initialization of this object does not acutally add components, it just
        sets the preliminary values. The next step will almost always be adding the actual 
        components using the addDistributionComponents() function.

        Arguments:
        ---------
            zsteps (int) : Integer value representing the number of redshift steps in STEEL
            msteps (int) : Integer value representing the number of mass bin steps in STEEL
            resolution (int) : Integer value represeting the number of bins in the probability
                               distribution that will used. A reasonably large number is
                               approprate here.
        """
        self.zsteps = zsteps
        self.msteps = msteps
        self.resolution = resolution

        # Empty lists to contain (pointers to) components
        self.distributionComponents = []
        self.distributionComponentFractions = []

        # Empty lists for Overflows. Overflows essentially act like separate componenets that
        # store values that go out of range of the main probabilty array (to maintain tests). 
        self.distributionOverflows = []
        self.distributionOverflowFractions = []

        self.remainder_store = []

        self.testAccuracy = 10**-8

    def addDistributionComponents(self, n = 1):
        """ Function to add n probability distribution components to the object

        Arguments:
        ---------
            n (int) : Integer value representing the desired number of distribution components
                      to be added.
        """
        # Input validation
        assert n >= 1, "The number of distribution components must be one or more"
        if n > 10:
            warnings.warn("You (probably) don't want to be adding this many components", ReadTheDocumentation)
        n = int(n) # Cast to integer to catch strange use cases.

        self.distributionComponentsCount = n

        for i in range(n):
            # Vector per distribution
            self.distributionComponents.append(np.zeros((self.zsteps, self.msteps, self.resolution)))
            # Single Element per distribution
            self.distributionComponentFractions.append(np.zeros((self.zsteps, self.msteps)))
            self.remainder_store.append(np.zeros((self.zsteps, self.msteps)))
            self.distributionOverflows.append(np.zeros((self.zsteps, self.msteps)))
            self.distributionOverflowFractions.append(np.zeros((self.zsteps, self.msteps)))

    def addScale(self, start, stop):
        """ Function to initialize probabilty distribution scale. This function effectively calls
        np.linspace, where arguments start and stop are the limits of the array and the number of
        elements is the resolution specified when the object was created. For example, if I was
        setting up a sersic index probaility distribution, I might set start = 0 and stop = 10.
        If I had set the resolution to be 500 when creating the object, this would set the scale
        to be 500 linearly spaced points between 0 and 10.

        Arguments:
        ---------
            start (float/double) : Value representing the starting value of the probability scale
            stop (float/double) : Value representing the finishing value of the probabilty scale
        """
        self.Scale, self.ScaleStep = np.linspace(start, stop, self.resolution + 1, retstep = True)

    def inputValidate(self, j, k):
        """ Internal Function for validating calls to j and k pairs.

        Arguments:
        ---------
            j (int) : intex representing redshift
            k (int) : index representing mass bin
        """
        assert j >= 0 and j < self.zsteps, "function called with invalid j (redshift) index. Value was %i, range is 0-%i"\
                % (j, self.zsteps)
        assert k >= 0 and k < self.msteps, "function called with invalid k (mass bin) index. Value was %i, range is 0-%i"\
                % (k, self.msteps)

    def inputValidateDistro(self, i):
        """ Internal Function for validating calls to distribution components.

        Arguments:
        ---------
            i (int) : index representing distribution components
        """
        assert i >= 0 and i < self.distributionComponentsCount,\
                "function called with invalid i (component) index. Value is %i, range is 0-%i" % (i, self.distributionComponentsCount)

    def syncFractions(self, j, k):
        """ Function that recalculates the distribution fractions at j, k,

        This function should be called with care - it normally is only called internally. Note that this function
        acts across all components.

        Arguments:
        ---------
            j (int) : intex representing redshift
            k (int) : index representing mass bin
        """
        self.inputValidate(j, k)
        # Processing
        fracTotal = 0
        for i in range(self.distributionComponentsCount):
            self.distributionComponentFractions[i][j, k] = np.sum(self.distributionComponents[i][j, k, :]) * self.ScaleStep
            self.distributionOverflowFractions[i][j, k] = self.distributionOverflows[i][j, k] * self.ScaleStep
            fracTotal += self.distributionComponentFractions[i][j, k] + self.distributionOverflowFractions[i][j, k]
        # Check Fractions are still consistent
        assert isclose(fracTotal, 1, abs_tol = self.testAccuracy),\
                "fraction sync failed, total is %f != 1, z iteration %i of %i"\
                % (fracTotal, self.zsteps-1-i, self.zsteps-1)

    def fullCheck(self, j, k):
        """ Function that does a full check of the distribution function

        All distributions and overflows will be integrated over, and the result will
        be checked to be equal to one.
       
        Arguments:
        ---------
            j (int) : intex representing redshift
            k (int) : index representing mass bin
        """
        self.inputValidate(j, k)
        total = 0
        for i in range(self.distributionComponentsCount):
            total += np.sum(self.distributionComponents[i][j, k, :])
            total += self.distributionOverflows[i][j, k]
        total *= self.ScaleStep
        assert isclose(total, 1, abs_tol = self.testAccuracy),\
                "fullCheck failed, integral is %f != 1. z iteration %i of %i"\
                % (total, self.zsteps-j, self.zsteps)

    def initializeGaussian(self, i, j, k, location, scale, magnitude):
        """ Function to overlay a gaussian distribution atop one distribution component.

        It is up to you to ensure that this maintains consistency.
        
        Arguments:
        ---------
            i (int) : index representing distribution component
            j (int) : intex representing redshift
            k (int) : index representing mass bin
            location (float/dobule) : the central location of the gaussian, in units of self.Scale
            scale (float/double) : the scale (spread) of the gaussian, in units of self.Scale
            magnitude (float/double) : the normaliztion (total integral) of the gaussian.
        """
        self.inputValidate(j, k)
        self.inputValidateDistro(i)

        # Stomp those floating point errors
        assert magnitude <= 1 + self.testAccuracy, "Initializing a gaussian with magnitude > 1 will necessarily violate probabilities summing to one "
        if magnitude > 1:
            magnitude = 1
        assert magnitude >= 0 - self.testAccuracy, "Iniitalization cannot be made to less than zero"
        if magnitude < 0:
            magnitude = 0
        
        if magnitude != 0: # Pointless, so let's not bother with the computations
            gaussian = np.random.normal(loc = location, scale = scale, size = 10000)
            hist = np.histogram(gaussian, self.Scale)[0]
            hist_norm = magnitude * hist/(np.sum(hist) * self.ScaleStep) # this is normalized by bin width too - not sure about this
            self.distributionComponents[i][j, k, :] += hist_norm
            self.syncFractions(j, k)

    def copyFromPreviousStep(self, j, k, increment = 1):
        """ Function to update distribution to be equal to previous redshift step.
        
        Do not use this where it would confict with boundary conditions 

        Arguments:
        ---------
            j (int) : index representing redshift
            k (int) : index representing mass bin
            increment (int) : adjustment representing the definition of the 'previous' redshit  bin.
        """
        self.inputValidate(j, k)
        self.inputValidate(j + increment, k)
        self.fullCheck(j + increment, k)

        for i in range(self.distributionComponentsCount):
            self.distributionComponents[i][j, k] = self.distributionComponents[i][j + increment, k, :]
            self.distributionComponentFractions[i][j, k] = self.distributionComponentFractions[i][j + increment, k]
            self.remainder_store[i][j, k] = self.remainder_store[i][j + increment, k]
            self.distributionOverflows[i][j, k] = self.distributionOverflows[i][j + increment, k]
            self.distributionOverflowFractions[i][j, k] = self.distributionOverflowFractions[i][j + increment, k]

    def scaleDistributionByFraction(self, i, j, k, Fraction):
        """ Function to reduce one distribution component by a defined fraction.

        It is up to you to maintain consistency here.

        Arguments:
        ---------
            i (int) : index representing distribution component
            j (int) : index representing redshift
            k (int) : index representing mass bin

        """
        self.inputValidateDistro(i)
        self.inputValidate(j, k)

        assert Fraction >= 0 - self.testAccuracy, "Fraction cannot be negative, Value %f" % (Fraction)
        if Fraction < 0:
            Fraction = 0
        assert Fraction <= 1 + self.testAccuracy, "Fraction must be less than 1. Value: %f"% (Fraction)
        if Fraction > 1:
            Fraction = 1

        if Fraction != 1: # Pointless, so let's not bother with the computations
            self.distributionComponents[i][j, k, :] *= Fraction
            self.distributionComponentFractions[i][j, k] *= Fraction
            total = np.sum(self.distributionComponents[i][j, k]) * self.ScaleStep
            assert isclose(total, self.distributionComponentFractions[i][j, k], abs_tol = self.testAccuracy),\
                    "Distibution scaling failed, integral %f != %f. z iteration %i of %i"\
                    % (total, self.distributionComponentFractions[i][j, k], self.zsteps-j, self.zsteps)

    def moveDistribution(self, i, j, k, delta):
        """ Function to move a distribution horizontally.

        Note that movement in the negative direction is not currently supported, and will fail (for now).

        Arguments:
        ---------
            i (int) : index representing distribution component
            j (int) : index representing redshift
            k (int) : index representing mass bin
            delta (float/double) : value representing the shift (in units of self.Scale).
        """
        self.inputValidateDistro(i)
        self.inputValidate(j, k)
        self.fullCheck(j, k)
        assert delta >= 0, "delta cannot yet be negative, as a precaution until underflow buffers are implemented"

        # Remainder Management
        assert self.remainder_store[i][j, k] < self.ScaleStep, "Remainder store is larger than step"
        delta += self.remainder_store[i][j, k]

        # Bin shifting
        self.shift_n_bins = int(np.floor(delta/self.ScaleStep)) # Number of bins to move up the nearest integer
        self.remainder_store[i][j, k] = delta - self.shift_n_bins * self.ScaleStep # Store any remainder

        # Overflow
        index_shift = self.resolution - self.shift_n_bins
        self.distributionOverflows[i][j, k] += np.sum(self.distributionComponents[i][j, k, index_shift:])
        self.distributionComponentFractions[i][j, k] -=  np.sum(self.distributionComponents[i][j, k, index_shift:]) * self.ScaleStep
        self.distributionOverflowFractions[i][j, k]  = self.distributionOverflows[i][j, k] * self.ScaleStep
        self.distributionComponents[i][j, k, index_shift:] = 0

        self.fullCheck(j, k)
        self.distributionComponents[i][j, k, :] = np.roll(self.distributionComponents[i][j, k, :], self.shift_n_bins)
        self.fullCheck(j, k)


    def extractCombinedDistribution(self):
        """ Function that aggregates each component and produces a single 3D array.

        Returns:
        -------
            Master (np.array[:, :, :]) : Array representing the entire probability distribution
        """
        master = np.zeros_like(self.distributionComponents[0][:, :, :])
        for k in range(self.distributionComponentsCount):
            master = np.add(master, self.distributionComponents[k][:, :, :])
        return master

    def extractElementDistribution(self, i, j):
        """ Function that aggregates each component but returns only distribution i,j.

        Arguments:
        ---------
            i (int) : index representing redshift
            j (int) : index representing the mass bin

        Returns:
        -------
            master (np.array([:]) : Array corresponding to associated probability distribution
        """
        self.inputValidate(i, j)
        master = np.zeros_like(self.distributionComponents[0][i, j, :])
        for k in range(self.distributionComponentsCount):
            master = np.add(master, self.distributionComponents[k][i, j, :])
        return master
    
    def Catalogue(self, i, j, number):
        """TODO: Docstring
        """
        self.inputValidate(i, j)
        prob = self.extractElementDistribution(i, j)
        
        sample = np.random.choice(self.Scale[0:-1], size = number, p = prob/np.sum(prob)) 
        
        '''
        prob_Cum = np.cumsum(prob)
        prob_Cum -= np.amin(prob_Cum)
        prob_Cum /= np.amax(prob_Cum)

        GetValue = interpolate.interp1d(prob_Cum, self.Scale[0:-1])
    
        sample = np.random.rand(number)
        sample = GetValue(sample)
        '''
        return sample

