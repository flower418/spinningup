==========================================
Benchmarks for Spinning Up Implementations
==========================================

.. contents:: Table of Contents

We benchmarked the Spinning Up algorithm implementations in five environments from the Gymnasium MuJoCo task suite: HalfCheetah, Hopper, Walker2d, Swimmer, and Ant.

Performance in Each Environment
===============================

HalfCheetah
-----------

.. figure:: ../images/plots/pyt/pytorch_halfcheetah_performance.svg
    :align: center

    3M timestep benchmark for HalfCheetah-v3.

Hopper
------

.. figure:: ../images/plots/pyt/pytorch_hopper_performance.svg
    :align: center

    3M timestep benchmark for Hopper-v3.

Walker2d
--------

.. figure:: ../images/plots/pyt/pytorch_walker2d_performance.svg
    :align: center

    3M timestep benchmark for Walker2d-v3.

Swimmer
-------

.. figure:: ../images/plots/pyt/pytorch_swimmer_performance.svg
    :align: center

    3M timestep benchmark for Swimmer-v3.

Ant
---

.. figure:: ../images/plots/pyt/pytorch_ant_performance.svg
    :align: center

    3M timestep benchmark for Ant-v3.


Experiment Details
==================

The experiments were benchmarked with Spinning Up's PyTorch implementations.

.. note::
    These benchmarks are from the original Spinning Up release. New benchmarks with updated environments (v5) and the PyTorch-only codebase will be published in a future update.
