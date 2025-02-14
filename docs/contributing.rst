.. _contributing:

Contributing to the Toolbox
============================

Contributions to the PyTorch-Wavelet-Toolbox are always welcome!

Development workflow:
---------------------
We use nox to run our unit tests. Before creating your pull request follow these three steps.

1. Make sure all unit tests are passing.
    Run:
    .. code-block:: bash

        nox -s test

    to check.

2. Help yourself by running,
    .. code-block:: bash

        nox -s format

    to take care of linting issues, with an automatic fix.

3. Afterward, run
    .. code-block:: bash

        nox -s lint

    to learn where manual fixes are required for style compatibility.


Once everything checks out, you can create a pull request. We will review it as soon as possible.
