potsynth
========

**potsynth** evaluates spherical harmonic models of internal potential fields at large
numbers of scattered points. It supports geomagnetic models (IGRF, CHAOS, lithospheric
models) and gravity field models (EGM2008, EIGEN, XGM and others in the ICGEM format).

After a one-off set-up, the cost of each evaluation point does not depend on the
maximum degree :math:`L` of the model. With :math:`10^6` points and :math:`L = 185` it
is 15--50 times faster than an optimized direct sum. Its accuracy can be set anywhere
from about :math:`10^{-4}` down to the rounding error of double precision.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   intro.rst
   install.rst
   algorithm.rst
   c_api.rst
   python_api.rst
   examples.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
