API reference
=============

Top-level package
-----------------

.. automodule:: flatwalk
   :no-members:

The names below are re-exported from :mod:`flatwalk` for convenience;
their full documentation lives with the source modules listed in this
page. For worked usage see the :doc:`examples <auto_examples/index>` and
:doc:`tutorials <auto_tutorials/index>`; the snippets below link the ones
that exercise each object.

.. minigallery:: flatwalk.WLDriver flatwalk.RewlDriver flatwalk.Bin1D
   :add-heading: Examples and tutorials using the core API


Binning
-------

.. automodule:: flatwalk.binning
   :members:
   :show-inheritance:


Driver and configuration
------------------------

.. automodule:: flatwalk.core
   :members:
   :show-inheritance:


Walker
------

.. automodule:: flatwalk.walker
   :members:
   :show-inheritance:


Diagnostics
-----------

.. automodule:: flatwalk.diagnostics
   :members:
   :show-inheritance:


Checkpoint I/O
--------------

.. automodule:: flatwalk.io
   :members:
   :show-inheritance:


Replica-exchange Wang-Landau
----------------------------

.. automodule:: flatwalk.rewl
   :members:
   :show-inheritance:


Exchange handler interface (extension hook)
-------------------------------------------

.. automodule:: flatwalk.exchange
   :members:
   :show-inheritance:

.. minigallery:: flatwalk.RewlDriver flatwalk.make_windows flatwalk.join_g
   :add-heading: Examples and tutorials using replica exchange
