********
Examples
********

Both examples build a synthetic lithospheric-type model: a dipole-dominated core field
plus a flat lithospheric spectrum up to degree 185, similar in size to CHAOS. They then
evaluate it at :math:`10^6` random points in the 250--800 km shell, and compare the
first 1000 points with the direct sum. To use a real model, replace the coefficients.

C
=

``examples/example_mag.c``. It is built with the library (``POTSYNTH_EXAMPLES``), or
separately:

.. code-block:: sh

   cc -O2 example_mag.c -lpotsynth -lm -o example_mag

.. literalinclude:: ../examples/example_mag.c
   :language: c
   :lines: 16-

Output on 16 threads:

.. code-block:: text

   set-up:      0.066 s
   method:      fast, window w = 11, radial rank = 9, grid 63.7 MB
   evaluation:  1000000 points in 0.077 s
   fast vs direct (1000 points): max difference 8.28e-07 nT (1.1e-11 relative)

         r [km]   colat [deg]  lon [deg]      B_N [nT]      B_E [nT]      B_C [nT]
       6647.140      93.1622   259.2005    23636.6470     9668.6087    10337.5273
       6621.698      22.1884   126.0955     8296.0260     9356.0822    45368.1189
       6835.938      62.8764   271.8356    11629.1472     1789.3621    29620.5851
       6636.556     119.9082   259.3058    26377.9781    14396.5709   -10635.1553

Python
======

``examples/example_mag.py``:

.. code-block:: sh

   POTSYNTH_LIBRARY=/path/to/libpotsynth.so PYTHONPATH=/path/to/potsynth/python \
       python example_mag.py

.. literalinclude:: ../examples/example_mag.py
   :language: python
   :lines: 11-

Output:

.. code-block:: text

   set-up:      0.067 s
   method:      fast, window w = 11, radial rank = 9, grid 63.7 MB
   evaluation:  1000000 points in 0.083 s
   fast vs direct (1000 points): max difference 9.50e-07 nT (1.4e-11 relative)

         r [km]   colat [deg]  lon [deg]      B_N [nT]      B_E [nT]      B_C [nT]
       7017.328      20.7620    80.6543    12491.1267    11416.9289    47991.4254
       6747.817     133.6640   273.3265    16105.7094    -2699.0329   -19713.7437
       6927.504      96.1811   223.7415    25283.8992      321.9703    -3081.1124
       6883.566     140.5038   204.5044    13025.9682     -156.4956   -27410.0374

   map of B_r at 450 km: shape (181, 360), range [-63271.3, 41021.3] nT

The last lines show broadcasting. ``model(a + 450.0, colat[:, None], lon[None, :])``
evaluates a 1-degree global map in one call.
