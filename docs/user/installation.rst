============
Installation
============


.. contents:: Table of Contents

Spinning Up requires Python3, Gymnasium, PyTorch, and OpenMPI.

Spinning Up is currently only supported on Linux and OSX. It may be possible to install on Windows (e.g. via WSL2), though this hasn't been extensively tested.

.. admonition:: You Should Know

    This fork of Spinning Up uses **PyTorch** (not TensorFlow) and **Gymnasium** (not the old Gym). MuJoCo is now **completely free and open source** — no license required.

    Many examples and benchmarks use the `MuJoCo`_ physics engine. MuJoCo is the de facto standard for benchmarking deep RL algorithms in continuous control.

    If you prefer not to install MuJoCo, you can still run RL algorithms on `Classic Control`_ and `Box2d`_ environments, which are free to use.

.. _`Classic Control`: https://gymnasium.farama.org/environments/classic_control/
.. _`Box2d`: https://gymnasium.farama.org/environments/box2d/
.. _`MuJoCo`: https://mujoco.org/


Installing Python
=================

We recommend installing Python through Miniforge_ (a lightweight alternative to Anaconda). Create a conda environment with Python 3.10+:

.. parsed-literal::

    conda create -n spinningup python=3.10
    conda activate spinningup

.. _Miniforge: https://github.com/conda-forge/miniforge


Installing OpenMPI
==================

OpenMPI enables parallel training for on-policy algorithms (VPG, TRPO, PPO).

Ubuntu
------

.. parsed-literal::

    sudo apt-get update && sudo apt-get install libopenmpi-dev


Mac OS X
--------

Installation of system packages on Mac requires Homebrew_. With Homebrew installed, run:

.. parsed-literal::

    brew install openmpi

.. _Homebrew: https://brew.sh


Installing Spinning Up
======================

.. parsed-literal::

    git clone https://github.com/flower418/spinningup.git
    cd spinningup
    pip install -e .

This installs Spinning Up with its core dependencies (Gymnasium, PyTorch, NumPy, etc.).

.. admonition:: You Should Know

    This fork is PyTorch-only. It does not require TensorFlow at all.


Check Your Install
==================

To verify installation, first install Box2D (required by LunarLander), then run PPO:

.. parsed-literal::

    pip install "gymnasium[box2d]"
    python -m spinup.run ppo --hid "[32,32]" --env LunarLander-v3 --exp_name installtest --gamma 0.999

This will train for about 10 minutes — enough to see some learning progress.

After training, watch the trained policy:

.. parsed-literal::

    python -m spinup.run test_policy data/installtest/installtest_s0

And plot the results:

.. parsed-literal::

    python -m spinup.run plot data/installtest/installtest_s0


Installing MuJoCo (Recommended)
===============================

MuJoCo has been **free and open source** (Apache 2.0) since 2021. No license required.

Install the modern MuJoCo Python bindings and Gymnasium MuJoCo environments:

.. parsed-literal::

    pip install mujoco "gymnasium[mujoco]"

Verify MuJoCo works:

.. parsed-literal::

    python -c "import mujoco; print(mujoco.__version__)"
    python -c "import gymnasium; gymnasium.make('HalfCheetah-v4')"

Then test by training PPO on a MuJoCo environment:

.. parsed-literal::

    python -m spinup.run ppo --hid "[64,64]" --env HalfCheetah-v4 --exp_name mujocotest

.. note::

    MuJoCo environments in Gymnasium use version **v4** or **v5**. The old v2 environments require the deprecated ``mujoco-py`` package and are not supported.

    Available MuJoCo environments: ``Ant-v4``, ``HalfCheetah-v4``, ``Hopper-v4``, ``Walker2d-v4``, ``Swimmer-v4``, ``Humanoid-v4``, ``HumanoidStandup-v4``, ``Pusher-v4``, ``Reacher-v4``, ``InvertedPendulum-v4``, ``InvertedDoublePendulum-v4``.
