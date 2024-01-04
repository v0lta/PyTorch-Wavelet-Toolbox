.. PyTorch-Wavelet-Toolbox documentation master file, created by
   sphinx-quickstart on Thu Oct 14 15:19:29 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _PyTorch install instructions: https://pytorch.org/get-started/locally/

Pytorch Wavelet Toolbox (``ptwt``) 
==================================

``ptwt`` brings wavelet transforms to Pytorch. The code is open-source, follow the link above
to go to the source. This package is listed in the Python Package Index. Its best intalled via pip.
GPU support depends on PyTorch. To take advantage of GPU-processing follow the `PyTorch install instructions`_.
Install the version, that best suits your systems hardware setup. Once PyTorch ist set up type

.. code-block:: sh

    pip install ptwt

to get started.


This documentation aims to explain the foundations of wavelet theory, introduce the ``ptwt`` package by example and
deliver a complete documentation of all functions. Readers who are already familiar with the theory should directly
jump to the examples or the API-documentation using the navigation on the left.


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Getting Started

   intro
   examples


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Public API

   ptwt

