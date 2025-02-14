.. PyTorch-Wavelet-Toolbox documentation master file, created by
   sphinx-quickstart on Thu Oct 14 15:19:29 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. _PyTorch install instructions: https://pytorch.org/get-started/locally/

PyTorch Wavelet Toolbox (``ptwt``)
==================================

``ptwt`` brings wavelet transforms to PyTorch. The code is open-source. Follow the GitHub link above
to go to the source. This package is listed in the Python Package Index (PyPI). It's best installed via pip.
GPU support depends on PyTorch. To take advantage of GPU-processing, follow the `PyTorch install instructions`_.
Install the version that best suits your system's hardware setup. Once PyTorch is set up, type the following
to get started:

.. code-block:: sh

    pip install ptwt

This documentation aims to explain the foundations of wavelet theory, introduce the ``ptwt`` package by example, and
deliver a complete documentation of all functions. Readers who are already familiar with the theory should directly
jump to the :ref:`examples <sec-examples>` or the :ref:`API documentation <ref-index>` using the navigation on the left.

``ptwt`` is built to be `PyWavelets <https://pywavelets.readthedocs.io/en/latest/>`_ compatible.
It should be possible to switch back and forth with relative ease.

If you use this work in a scientific context, please cite us!
We have a  BibTeX entry on the :ref:`citation`-page.


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Getting Started

   intro
   intro_boundary_wavelets
   common_wavelets
   intro_cwt
   literature
   examples


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Public API

   ref/index


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Community

   citation
   contributing
   release_notes
