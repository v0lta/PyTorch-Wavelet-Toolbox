.. _ref-modes:

.. currentmodule:: ptwt.constants

Boundary handling modes
=======================

As is typical the algorithms in this toolbox are designed to be applied
to signal tensors of finite size.
This requires some handling of the signal boundaries to apply the
wavelet transform convolutions.

This toolbox implements two different approaches to boundary handling:

* signal extension via padding
* using boundary filters for coeffients on the signal boundary

Signal extension via padding
----------------------------

.. _`modes.padding`:

Signal extensions by padding are applied using :func:`torch.nn.functional.pad`.
The following modes of padding are supported:

.. autoclass:: BoundaryMode


Boundary wavelets
-----------------

Boundary filters are another way to deal with signals of finite length.
The :ref:`getting started section <sec-boundary-wavelets>`
of the docs provide and introduction to the main concepts.


.. _`modes.boundary wavelets`:

.. autodata:: ExtendedBoundaryMode

.. autodata:: PaddingMode

.. autoclass:: OrthogonalizeMethod
