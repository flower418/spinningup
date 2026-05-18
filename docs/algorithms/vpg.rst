=======================
Vanilla Policy Gradient
=======================

.. contents:: Table of Contents


Background
==========

(Previously: `Introduction to RL, Part 3`_)

.. _`Introduction to RL, Part 3`: ../spinningup/rl_intro3.html

The key idea underlying policy gradients is to push up the probabilities of actions that lead to higher return, and push down the probabilities of actions that lead to lower return, until you arrive at the optimal policy.

Quick Facts
-----------

* VPG is an on-policy algorithm.
* VPG can be used for environments with either discrete or continuous action spaces.
* The Spinning Up implementation of VPG supports parallelization with MPI.

Key Equations
-------------

Let :math:`\pi_{\theta}` denote a policy with parameters :math:`\theta`, and :math:`J(\pi_{\theta})` denote the expected finite-horizon undiscounted return of the policy. The gradient of :math:`J(\pi_{\theta})` is

.. math:: 
    
    \nabla_{\theta} J(\pi_{\theta}) = \underE{\tau \sim \pi_{\theta}}{
        \sum_{t=0}^{T} \nabla_{\theta} \log \pi_{\theta}(a_t|s_t) A^{\pi_{\theta}}(s_t,a_t)
        },

where :math:`\tau` is a trajectory and :math:`A^{\pi_{\theta}}` is the advantage function for the current policy. 

The policy gradient algorithm works by updating policy parameters via stochastic gradient ascent on policy performance:

.. math::

    \theta_{k+1} = \theta_k + \alpha \nabla_{\theta} J(\pi_{\theta_k})

Policy gradient implementations typically compute advantage function estimates based on the infinite-horizon discounted return, despite otherwise using the finite-horizon undiscounted policy gradient formula. 

Exploration vs. Exploitation
----------------------------

VPG trains a stochastic policy in an on-policy way. This means that it explores by sampling actions according to the latest version of its stochastic policy. The amount of randomness in action selection depends on both initial conditions and the training procedure. Over the course of training, the policy typically becomes progressively less random, as the update rule encourages it to exploit rewards that it has already found. This may cause the policy to get trapped in local optima.


Pseudocode
----------

**Pseudocode**

#. Input: initial policy parameters :math:`\theta_0`, initial value function parameters :math:`\phi_0`
#. **for** :math:`k = 0,1,2,...` **do**
#.     Collect set of trajectories :math:`{\mathcal D}_k = \{\tau_i\}` by running policy :math:`\pi_k = \pi(\theta_k)` in the environment.
#.     Compute rewards-to-go :math:`\hat{R}_t`.
#.     Compute advantage estimates, :math:`\hat{A}_t` (using any method of advantage estimation) based on the current value function :math:`V_{\phi_k}`.
#.     Estimate policy gradient as

       .. math::
           \hat{g}_k = \frac{1}{|{\mathcal D}_k|} \sum_{\tau \in {\mathcal D}_k} \sum_{t=0}^T \left. \nabla_{\theta} \log\pi_{\theta}(a_t|s_t)\right|_{\theta_k} \hat{A}_t.

#.     Compute policy update, either using standard gradient ascent,

       .. math::
           \theta_{k+1} = \theta_k + \alpha_k \hat{g}_k,

       or via another gradient ascent algorithm like Adam.
#.     Fit value function by regression on mean-squared error:

       .. math::
           \phi_{k+1} = \arg \min_{\phi} \frac{1}{|{\mathcal D}_k| T} \sum_{\tau \in {\mathcal D}_k} \sum_{t=0}^T\left( V_{\phi} (s_t) - \hat{R}_t \right)^2,

       typically via some gradient descent algorithm.
#. **end for**


Documentation
=============

Below is the documentation for the PyTorch implementation.


Documentation: PyTorch Version
------------------------------

.. autofunction:: spinup.vpg

Saved Model Contents: PyTorch Version
-------------------------------------

The PyTorch saved model can be loaded with ``ac = torch.load('path/to/model.pt')``, yielding an actor-critic object (``ac``) that has the properties described in the docstring for ``vpg_pytorch``. 

You can get actions from this model with

.. code-block:: python

    actions = ac.act(torch.as_tensor(obs, dtype=torch.float32))



References
==========

Relevant Papers
---------------

- `Policy Gradient Methods for Reinforcement Learning with Function Approximation`_, Sutton et al. 2000
- `Optimizing Expectations: From Deep Reinforcement Learning to Stochastic Computation Graphs`_, Schulman 2016(a)
- `Benchmarking Deep Reinforcement Learning for Continuous Control`_, Duan et al. 2016
- `High Dimensional Continuous Control Using Generalized Advantage Estimation`_, Schulman et al. 2016(b)

.. _`Policy Gradient Methods for Reinforcement Learning with Function Approximation`: https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf
.. _`Optimizing Expectations: From Deep Reinforcement Learning to Stochastic Computation Graphs`: http://joschu.net/docs/thesis.pdf
.. _`Benchmarking Deep Reinforcement Learning for Continuous Control`: https://arxiv.org/abs/1604.06778
.. _`High Dimensional Continuous Control Using Generalized Advantage Estimation`: https://arxiv.org/abs/1506.02438

Why These Papers?
-----------------

Sutton 2000 is included because it is a timeless classic of reinforcement learning theory, and contains references to the earlier work which led to modern policy gradients. Schulman 2016(a) is included because Chapter 2 contains a lucid introduction to the theory of policy gradient algorithms, including pseudocode. Duan 2016 is a clear, recent benchmark paper that shows how vanilla policy gradient in the deep RL setting (eg with neural network policies and Adam as the optimizer) compares with other deep RL algorithms. Schulman 2016(b) is included because our implementation of VPG makes use of Generalized Advantage Estimation for computing the policy gradient.


Other Public Implementations
----------------------------

- rllab_
- `rllib (Ray)`_

.. _rllab: https://github.com/rll/rllab/blob/master/rllab/algos/vpg.py
.. _`rllib (Ray)`: https://github.com/ray-project/ray/blob/master/python/ray/rllib/agents/pg
