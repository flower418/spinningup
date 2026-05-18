============
Introduction
============

.. contents:: Table of Contents

What This Is
============

Welcome to Spinning Up in Deep RL! This is an educational resource designed to make it easier to learn about deep reinforcement learning (deep RL).

For the unfamiliar: `reinforcement learning`_ (RL) is a machine learning approach for teaching agents how to solve tasks by trial and error. Deep RL refers to the combination of RL with `deep learning`_.

This module contains a variety of helpful resources, including:

- a short `introduction`_ to RL terminology, kinds of algorithms, and basic theory,
- a `curated list`_ of important papers organized by topic,
- a well-documented `code repo`_ of short, standalone implementations of key algorithms,
- and a few `exercises`_ to serve as warm-ups.

.. _`reinforcement learning`: https://en.wikipedia.org/wiki/Reinforcement_learning
.. _`deep learning`: http://ufldl.stanford.edu/tutorial/


Modern Stack
============

This version of Spinning Up uses a modern deep RL stack:

- **PyTorch** only — all algorithms implemented with clean, readable PyTorch code
- **Gymnasium** — the actively maintained RL environment library (successor to OpenAI Gym)
- **MuJoCo** — fully free and open source physics engine (no license required)
- **MPI** via ``mpi4py`` — for parallel training of on-policy algorithms


Code Design Philosophy
======================

The algorithm implementations are designed to be:

    - as simple as possible while still being reasonably good,
    - and highly-consistent with each other to expose fundamental similarities between algorithms.

They are almost completely self-contained, with virtually no common code shared between them (except for logging, saving, loading, and `MPI <https://en.wikipedia.org/wiki/Message_Passing_Interface>`_ utilities), so that an interested person can study each algorithm separately without having to dig through an endless chain of dependencies to see how something is done. The implementations are patterned so that they come as close to pseudocode as possible, to minimize the gap between theory and code.

Importantly, they're all structured similarly, so if you clearly understand one, jumping into the next is painless.

All algorithms are "reasonably good" in the sense that they achieve roughly the intended performance, but don't necessarily match the best reported results in the literature on every task. Details on each implementation's specific performance level can be found on our `benchmarks`_ page.


.. _`introduction`: ../spinningup/rl_intro.html
.. _`curated list`: ../spinningup/keypapers.html
.. _`code repo`: https://github.com/flower418/spinningup
.. _`exercises`: ../spinningup/exercises.html
.. _`benchmarks`: ../spinningup/bench.html
.. _`PyTorch`: https://pytorch.org/
.. _`Gymnasium`: https://gymnasium.farama.org/
.. _`MuJoCo`: https://mujoco.org/
