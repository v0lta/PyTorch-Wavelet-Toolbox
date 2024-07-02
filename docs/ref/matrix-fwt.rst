.. _ref-matrix-fwt:

.. currentmodule:: ptwt

Sparse-matrix based Fast Wavelet Transform (FWT)
================================================

1d decomposition using ``MatrixWavedec``
----------------------------------------

.. autoclass:: MatrixWavedec
   :members:
   :special-members: __call__
   :undoc-members:
   :show-inheritance:

2d decomposition using ``MatrixWavedec2``
-----------------------------------------

.. autoclass:: MatrixWavedec2
   :members:
   :special-members: __call__
   :undoc-members:
   :show-inheritance:

3d decomposition using ``MatrixWavedec3``
-----------------------------------------

.. autoclass:: MatrixWavedec3
   :members:
   :special-members: __call__
   :undoc-members:
   :show-inheritance:


Sparse-matrix FWT base class
----------------------------
All sparse-matrix decomposition classes extend :class:`ptwt.matmul_transform.BaseMatrixWaveDec`.

.. autoclass:: ptwt.matmul_transform.BaseMatrixWaveDec
   :members:
   :undoc-members:
