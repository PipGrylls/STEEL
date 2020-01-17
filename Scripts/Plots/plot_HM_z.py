import inspect
import textwrap

import numpy as np
from numpy import ma

from matplotlib import cbook, docstring, rcParams
from matplotlib.ticker import (
    NullFormatter, ScalarFormatter, LogFormatterSciNotation, LogitFormatter,
    NullLocator, LogLocator, AutoLocator, AutoMinorLocator,
    SymmetricalLogLocator, LogitLocator)
from matplotlib.transforms import Transform, IdentityTransform
import matplotlib.pyplot as plt
import matplotlib.scale as mpls


# Function x**(1/2)
def forward(x):
    #return x**(1/2)
    return x**(1/2.1)

def inverse(x):
    #return x**2
    return x**2.1

z = np.loadtxt('Redshift_array.txt')
HM = np.loadtxt('HM_vdB.dat')

print(z[0])
print(z[z.size-1])

#plt.ion()
plt.figure()

#plt.plot(z_, np.repeat(1,z_.size),'.')

#'''
plt.plot(1+z,HM[:,1],color='lime')
plt.plot(1+z,HM[:,11],color='lime')
plt.plot(1+z,HM[:,21],color='lime')
plt.plot(1+z,HM[:,31],color='lime')
plt.plot(1+z,HM[:,41],color='lime')
#plt.xlim(0,10)
#plt.xscale('function', functions=(forward, inverse))
plt.xscale('log')
#'''

#ax.yaxis.set_major_locator(FixedLocator(np.arange(0, 1, 0.2)**2))
#ax.yaxis.set_major_locator(FixedLocator(np.arange(0, 1, 0.2)))

'''
x=np.arange(0,11,1)
y=np.repeat(1,x.size)
a=np.zeros(x.size)
a[0]=x[0]
bin=x[1]-x[0]
for i in range(1,x.size):
    a[i]=a[i-1]+bin/i
plt.plot(a,y,'o')
'''

plt.show()
