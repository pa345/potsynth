********
Examples
********

There are two pairs of examples, in C and Python: a magnetic field model and a gravity
field model.

Magnetic field
==============

Both magnetic examples evaluate a lithospheric magnetic field model at :math:`10^6` random points
in the 250--800 km shell, and compare the first 1000 points with the direct sum.

- **Model source:** with a file argument, the model is read from a static Swarm SHC
  file, for example LCS-1 (Olsen et al., 2017, degrees 1--185). Without an argument, a
  synthetic model of similar size is used.
- **Timings:** the outputs below are for LCS-1 on 16 threads of an Intel Core Ultra 7
  265H.

C
-

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
------

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

Gravity field
=============

Both gravity examples evaluate the potential :math:`V` and the acceleration
:math:`g_r, g_\theta, g_\phi` at :math:`10^6` random points in the 200--500 km shell, and
compare the first 1000 points with the direct sum, quantity by quantity.

- **Model source:** with a file argument, the model is read from an ICGEM ``.gfc`` file,
  for example EGM2008 (Pavlis et al., 2012), optionally truncated at a given degree.
  Without an argument, a synthetic Earth-like model up to degree 360 is used.
- **Outputs:** the outputs below are for EGM2008 truncated at degree 720, on 16 threads
  of an Intel Core Ultra 7 265H. ``npts_hint`` lets ``METHOD_AUTO`` choose the faster
  method for :math:`10^6` points (see :ref:`choosing-method`).

C
-

``examples/example_grav.c``:

.. code-block:: sh

   cc -O2 example_grav.c -lpotsynth -lm -o example_grav
   ./example_grav EGM2008.gfc 720

.. literalinclude:: ../examples/example_grav.c
   :language: c
   :lines: 19-

Output:

.. code-block:: text

   model:       EGM2008, degrees 0-720, GM = 3.9860044150e+14, a = 6378136.3000, tide system tide_free
   set-up:      1.889 s
   method:      fast, window w = 11, radial rank = 11, grid 1448.8 MB
   evaluation:  1000000 points in 0.269 s
   fast vs direct (1000 points), V      : max difference 1.01e-03 (1.7e-11 relative)
   fast vs direct (1000 points), g_r    : max difference 1.64e-10 (1.8e-11 relative)
   fast vs direct (1000 points), g_theta: max difference 1.96e-13 (1.4e-11 relative)
   fast vs direct (1000 points), g_phi  : max difference 7.87e-15 (2.3e-11 relative)
        r [km]  colat [deg]  lon [deg]      V [m^2/s^2]      g_r [mGal]   g_theta [mGal]     g_phi [mGal]
      6830.193     102.1947   281.9157    58382566.7555    -855479.6377        -501.1826          15.7969
      6817.669      34.5840    71.1185    58436954.9878    -856289.8625        1127.6632         -11.6479
      6678.704      57.5571    99.9989    59685986.0068    -893788.3076        1205.7504           6.0446
      6744.328      92.5910   226.3935    59129989.2152    -877583.7961        -112.0541          -7.5410

Each quantity is accurate to :math:`\varepsilon` relative to its own size, although
:math:`V` and :math:`g_\phi` differ by more than 11 orders of magnitude.

Python
------

``examples/example_grav.py``:

.. code-block:: sh

   POTSYNTH_LIBRARY=/path/to/libpotsynth.so PYTHONPATH=/path/to/potsynth/python \
       python example_grav.py EGM2008.gfc 720

.. literalinclude:: ../examples/example_grav.py
   :language: python
   :lines: 12-

Output:

.. code-block:: text

   model:       EGM2008, degrees 0-720, tide system tide_free
   set-up:      2.320 s
   method:      fast, window w = 11, radial rank = 11, grid 1448.8 MB
   evaluation:  1000000 points in 0.256 s
   fast vs direct (1000 points), V      : max difference 1.06e-03 (1.7e-11 relative)
   fast vs direct (1000 points), g_r    : max difference 1.63e-10 (1.8e-11 relative)
   fast vs direct (1000 points), g_theta: max difference 2.21e-13 (1.6e-11 relative)
   fast vs direct (1000 points), g_phi  : max difference 7.28e-15 (2.3e-11 relative)

   map of g_r at 250 km: shape (181, 360), range [-908722.6, -904571.6] mGal
