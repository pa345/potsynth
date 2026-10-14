********
Examples
********

Both examples evaluate a lithospheric magnetic field model at :math:`10^6` random points
in the 250--800 km shell, and compare the first 1000 points with the direct sum.

- **Model source:** with a file argument, the model is read from a static Swarm SHC
  file, for example LCS-1 (Olsen et al., 2017, degrees 1--185). Without an argument, a
  synthetic model of similar size is used.
- **Timings:** the outputs below are for LCS-1 on 16 threads of an Intel Core Ultra 7
  265H.

C
=

``examples/example_mag.c``. It is built with the library (``POTSYNTH_EXAMPLES``), or
separately:

.. code-block:: sh

   cc -O2 example_mag.c -lpotsynth -lm -o example_mag
   ./example_mag LCS-1.shc

.. literalinclude:: ../examples/example_mag.c
   :language: c
   :lines: 19-

Output:

.. code-block:: text

   model:       LCS-1.shc, degrees 1-185, epoch 2015
   set-up:      0.131 s
   method:      fast, window w = 11, radial rank = 12, grid 81.8 MB
   evaluation:  1000000 points in 0.153 s
   fast vs direct (1000 points): max difference 4.65e-10 nT (2.5e-11 relative)
         r [km]   colat [deg]  lon [deg]      B_N [nT]      B_E [nT]      B_C [nT]
       7083.303     102.1947   281.9157       -0.7021       -0.0320       -0.4513
       7060.342      34.5840    71.1185       -1.6776        0.4057        0.8405
       6805.573      57.5571    99.9989       -2.5544        1.4879       -1.9949
       6925.883      92.5910   226.3935       -1.2686       -0.7389       -1.8480

Python
======

``examples/example_mag.py``:

.. code-block:: sh

   POTSYNTH_LIBRARY=/path/to/libpotsynth.so PYTHONPATH=/path/to/potsynth/python \
       python example_mag.py LCS-1.shc

.. literalinclude:: ../examples/example_mag.py
   :language: python
   :lines: 12-

Output:

.. code-block:: text

   model:       LCS-1.shc, degrees 1-185, epoch 2015.0007
   set-up:      0.128 s
   method:      fast, window w = 11, radial rank = 12, grid 81.8 MB
   evaluation:  1000000 points in 0.165 s
   fast vs direct (1000 points): max difference 7.03e-10 nT (3.4e-11 relative)

         r [km]   colat [deg]  lon [deg]      B_N [nT]      B_E [nT]      B_C [nT]
       6902.702      84.5171   351.4295       -1.3117       -0.8520        2.3537
       7143.955      58.8638    24.6780       -0.5459        0.1744       -1.3910
       6700.488     146.0127     4.8576       -2.7507       -1.5084        2.2850
       7142.957     114.0623    60.5696        0.7894       -0.0423        0.4690

   map of B_r at 450 km: shape (181, 360), range [-17.5, 15.8] nT

The last lines show broadcasting. ``model(a + 450.0, colat[:, None], lon[None, :])``
evaluates a 1-degree global map in one call.

LCS-1 needs 12 radial functions at :math:`10^{-10}`, against about 9 for a
dipole-dominated model. A lithospheric model has no large low-degree terms, so its
field is dominated by high degrees, whose radial factors :math:`(a/r)^{l+2}` change
more across the shell.
