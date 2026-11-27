
============================
Compiling code ahead of time
============================

.. _pycc:

While Numba's main use case is :term:`Just-in-Time compilation`, it also
provides a facility for :term:`Ahead-of-Time compilation` (AOT).

.. note:: To use this feature the ``setuptools`` package is required at
          compilation time, but it is not a runtime dependency of the
          extension module produced.

.. note:: This module is pending deprecation. Please see
          :ref:`deprecation-numba-pycc` for more information.


Overview
========

Benefits
--------

#. AOT compilation produces a compiled extension module which does not depend
   on Numba: you can distribute the module on machines which do not have
   Numba installed (but NumPy is required).

#. There is no compilation overhead at runtime (but see the
   ``@jit`` :ref:`cache <jit-cache>` option), nor any overhead of importing
   Numba.

.. seealso::
   Compiled extension modules are discussed in the
   `Python packaging user guide <https://packaging.python.org/en/latest/guides/packaging-binary-extensions/>`_.


Limitations
-----------

#. AOT compilation only allows for regular functions, not :term:`ufuncs <ufunc>`.

#. You have to specify function signatures explicitly.

#. Each exported function can have only one signature (but you can export
   several different signatures under different names).

#. Exported functions do not check the types of the arguments that are passed
   to them; the caller is expected to provide arguments of the correct type.

#. AOT compilation produces generic code for your CPU's architectural family
   (for example "x86-64"), while JIT compilation produces code optimized
   for your particular CPU model.


Usage
=====

Standalone example
------------------

::

   from numba.pycc import CC

   cc = CC('my_module')
   # Uncomment the following line to print out the compilation steps
   #cc.verbose = True

   @cc.export('multf', 'f8(f8, f8)')
   @cc.export('multi', 'i4(i4, i4)')
   def mult(a, b):
       return a * b

   @cc.export('square', 'f8(f8)')
   def square(a):
       return a ** 2

   if __name__ == "__main__":
       cc.compile()


If you run this Python script, it will generate an extension module named
``my_module``.  Depending on your platform, the actual filename may be
``my_module.so``, ``my_module.pyd``, ``my_module.cpython-34m.so``, etc.

The generated module has three functions: ``multf``, ``multi`` and ``square``.
``multi`` operates on 32-bit integers (``i4``), while ``multf`` and ``square``
operate on double-precision floats (``f8``)::

   >>> import my_module
   >>> my_module.multi(3, 4)
   12
   >>> my_module.square(1.414)
   1.9993959999999997


Parallel compilation with oneTBB
--------------------------------

Pass ``parallel=True`` to ``cc.export()`` to enable the parallel compiler
passes for that export. For example::

   from numba import prange
   from numba.pycc import CC

   cc = CC('parallel_example')

   @cc.export('sum_array', 'f8(f8[::1])', parallel=True)
   def sum_array(a):
       total = 0.0
       for i in prange(a.size):
           total += a[i]
       return total

   if __name__ == '__main__':
       cc.compile()

Building requires the oneTBB headers and link library from oneTBB 2021.6 or
later. Set ``TBBROOT`` to the installation prefix if the compiler cannot find
them. The generated extension also needs the oneTBB shared library on the
runtime loader's search path. These requirements apply to both
``cc.compile()`` and ``cc.distutils_extension()``.

Set ``NUMBA_NUM_THREADS`` before importing the compiled extension to limit the
size of its oneTBB task arena. The default comes from oneTBB. Exports and
native helpers can use Numba's thread and chunk-size APIs. To change those
settings from Python without importing Numba at runtime, export a function
that calls the setters. Thread masks and chunk sizes are local to the calling
thread and extension. A positive chunk size lets oneTBB distribute smaller
chunks for uneven workloads.

The extension's runtime also handles calls to parallel helpers compiled by
Numba during the build. Mark at least one export with ``parallel=True`` when
the extension calls those helpers or uses the threading APIs.

The oneTBB JIT backend and oneTBB extensions share the fork lifecycle within
a Python interpreter. Forking from the thread that initialized oneTBB works
when its worker threads can be stopped before the fork. Use ``spawn`` for
portable multiprocessing and for programs that use other oneTBB clients.


Distutils integration
---------------------

You can also integrate the compilation step for your extension modules
in your ``setup.py`` script, using distutils or setuptools::

   from distutils.core import setup

   from source_module import cc

   setup(...,
         ext_modules=[cc.distutils_extension()])


The ``source_module`` above is the module defining the ``cc`` object.
Extensions compiled like this will be automatically included in the
build files for your Python project, so you can distribute them inside
binary packages such as wheels or Conda packages. Note that in the case of
using conda, the compilers used for AOT need to be those that are available
in the Anaconda distribution.


Signature syntax
----------------

The syntax for exported signatures is the same as in the ``@jit``
decorator.  You can read more about it in the :ref:`types <numba-types>`
reference.

Here is an example of exporting an implementation of the second-order
centered difference on a 1d array::

   @cc.export('centdiff_1d', 'f8[:](f8[:], f8)')
   def centdiff_1d(u, dx):
       D = np.empty_like(u)
       D[0] = 0
       D[-1] = 0
       for i in range(1, len(D) - 1):
           D[i] = (u[i+1] - 2 * u[i] + u[i-1]) / dx**2
       return D

.. (example from http://nbviewer.ipython.org/gist/ketch/ae87a94f4ef0793d5d52)

You can also omit the return type, which will then be inferred by Numba::

   @cc.export('centdiff_1d', '(f8[:], f8)')
   def centdiff_1d(u, dx):
       # Same code as above
       ...

