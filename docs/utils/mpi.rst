=========
MPI Tools
=========

.. contents:: Table of Contents

Core MPI Utilities
==================

.. automodule:: spinup.utils.mpi_tools
    :members:


MPI + PyTorch Utilities
=======================

``spinup.utils.mpi_pytorch`` contains tools for data-parallel PyTorch optimization across MPI processes. The two main ingredients are syncing parameters and averaging gradients before they are used by the adaptive optimizer.

The pattern for using these tools:

1) At the beginning of the training script, call ``setup_pytorch_for_mpi()``.

2) After constructing a PyTorch module, call ``sync_params(module)``.

3) During gradient descent, call ``mpi_avg_grads`` after the backward pass:

.. code-block:: python

    optimizer.zero_grad()
    loss = compute_loss(module)
    loss.backward()
    mpi_avg_grads(module)   # averages gradient buffers across MPI processes!
    optimizer.step()


.. automodule:: spinup.utils.mpi_pytorch
    :members:
