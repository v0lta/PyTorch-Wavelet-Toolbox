.. PyTorch-Wavelet-Toolbox documentation master file, created by
   sphinx-quickstart on Thu Oct 14 15:19:29 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _PyTorch install instructions: https://pytorch.org/get-started/locally/

PyTorch Wavelet Toolbox (``ptwt``)
==================================

``ptwt`` brings wavelet transforms to PyTorch. The code is open-source, follow the link above
to go to the source. This package is listed in the Python Package Index (PyPI). Its best installed via pip.
GPU support depends on PyTorch. To take advantage of GPU-processing follow the `PyTorch install instructions`_.
Install the version that best suits your systems hardware setup. Once PyTorch ist set up, type the following
to get started:

.. code-block:: sh

    pip install ptwt

This documentation aims to explain the foundations of wavelet theory, introduce the ``ptwt`` package by example, and
deliver a complete documentation of all functions. Readers who are already familiar with the theory should directly
jump to the examples or the API-documentation using the navigation on the left.

``ptwt`` is built to be `PyWavelets <https://pywavelets.readthedocs.io/en/latest/>`_ compatible.
It should be possible to switch back and forth with relative ease.

If you use this work in a scientific context, please cite the following thesis:

.. code-block::

   @phdthesis{handle:20.500.11811/9245,
   urn: https://nbn-resolving.org/urn:nbn:de:hbz:5-63361,
   author = {{Moritz Wolter}},
   title = {Frequency Domain Methods in Recurrent Neural Networks for Sequential Data Processing},
   school = {Rheinische Friedrich-Wilhelms-Universität Bonn},
   year = 2021,
   month = jul,
   url = {https://hdl.handle.net/20.500.11811/9245}
   }

   @thesis{Blanke2021,
   author        = {Felix Blanke},
   title         = {{Randbehandlung bei Wavelets für Faltungsnetzwerke}},
   type          = {Bachelor's Thesis},
   annote        = {Gbachelor},
   year          = {2021},
   school        = {Institut f\"ur Numerische Simulation, Universit\"at Bonn}
   }

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Getting Started

   intro
   intro_cwt
   examples


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Public API

   ptwt

