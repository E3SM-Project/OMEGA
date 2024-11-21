# Tridiagonal Solver

<!--- use table of contents if desired for longer documents  -->
**Table of Contents**
1. [Overview](#1-overview)
2. [Requirements](#2-requirements)
3. [Algorithmic Formulation](#3-algorithmic-formulation)
4. [Design](#4-design)
5. [Verification and Testing](#5-verification-and-testing)

## 1 Overview

In geophysical models, implicit time integration of vertical terms often requires solution to tridiagonal systems of equations.
Typically, a separate system needs to be solved in each vertical column, which requires an efficient batched tridiagonal solver.
One common situation where a tridiagonal system arises is implicit treatment of vertical diffusion/mixing.
In principle, this problem results in a symmetric diagonally-dominant tridiagonal system.
However, in ocean modeling, the coefficients of vertical mixing can vary by orders of magnitude and
can become very large in some layers.
This requires specialized algorithms that can handle this situation stably.

## 2 Requirements

### 2.1 Requirement: Modularity

There should be one implementation of batched tridiagonal solver that can be called every time a solution to such a system is needed.

### 2.2 Requirement: Stability for vertical mixing problems

The solver must be stable for vertical mixing problems with large variations in mixing coefficients.

### 2.3 Requirement: Bottom boundary conditions

When applied to the implicit vertical mixing of momentum, the solver should be
sufficiently general to be able to incorporate various bottom drag formulations.

### 2.4 Requirement: Performance

The solver must be performant on CPU and GPU architectures.

### 2.5 Desired: Ability to fuse kernels

Implicit vertical mixing will require some pre-processing (e.g. setup of the system) and post-processing work.
It is desirable to handle all of that in one computational kernel. This requires the ability to call the
solver inside a `parallelFor`.

## 3 Algorithmic Formulation
A general tridiagonal system has the form:
$$
a_i x_{i - 1} + b_i x_i + c_i x_{i + 1} = y_i.
$$
The standard second-order discretization of a diffusion problem leads to a system of the form
$$
-g_{i - 1} x_{i - 1} + (g_{i - 1} + g_i + h_i) x_i - g_{i} x_{i + 1} = y_i,
$$
where $g_i$, $h_i$ are positive and $g_i$ is proportional to the mixing coefficient
and can be much greater than $h_i$, which is the layer thickness.


### 3.1 Thomas algorithm
The Thomas algorithm is a simplified form of Gaussian elimination for tridiagonal systems of equations.
In its typical implementation, the forward elimination phase proceeds as follows
$$
\begin{aligned}
c'_i &= \frac{c_i}{b_i - a_i c'_{i - 1}}, \\
y'_i &= \frac{y_i - a_i y'_{i - 1}}{b_i - a_i c'_{i - 1}}.
\end{aligned}
$$
When applied to the diffusion system this leads to
$$
\begin{aligned}
c'_i &= \frac{-g_i}{h_i + g_{i - 1} (1 + c'_{i - 1}) + g_{i}}, \\
y'_i &= \frac{y_i + g_{i - 1} y'_{i - 1}}{h_i + g_{i - 1} (1 + c'_{i - 1}) + g_i}.
\end{aligned}
$$
Let's consider what happens when the mixing coefficient, and hence $g_i$, abruptly changes.
Suppose that $h_i$ is small, $g_{i - 1}$ is small, but $g_i$ is very large.
Then $c'_i \approx -1$ and in the next iteration the term $g_i (1 + c'_i)$ in
the denominator of both expression will multiply a very small number by a very large number.

Following [Appendix E in Schopf and Loughe](https://journals.ametsoc.org/view/journals/mwre/123/9/1520-0493_1995_123_2839_argiom_2_0_co_2.xml), to remedy that we can introduce
$$
\alpha'_i = g_i (1 + c'_i),
$$
which satisfies the following recursion relation
$$
\alpha'_i = \frac{g_i (h_i + \alpha'_{i - 1})}{h_i + \alpha'_{i - 1} + g_i}.
$$
The above equation together with
$$
\begin{aligned}
c'_i &= \frac{-g_i}{h_i + \alpha'_{i - 1} + g_{i}}, \\
y'_i &= \frac{y_i + g_{i - 1} y'_{i - 1}}{h_i + \alpha'_{i - 1} + g_i},
\end{aligned}
$$
forms the modifed stable algorithm.

### 3.2 (Parallel) cyclic reduction

The Thomas algorithm is work-efficient, but inherently serial.
While systems in different columns can
be solved in parallel, this might not expose enough parallelism on modern GPUs.
There are parallel tridiagonal algorithms that perform better on modern GPUs,
see [Zhang, Cohen, and Owens](https://doi.org/10.1145/1837853.1693472).
The two algorithms best suited for small systems are cyclic reduction and parallel cyclic reduction.

The basic idea of both cyclic reduction algorithms is as follows. Let's consider three consecutive equations corresponding to $y_{i- 1}$, $y_i$, and $y_{i+1}$
$$
\begin{aligned}
a_{i - 1} x_{i - 2} + b_{i - 1} x_{i - 1} + c_{i - 1} x_i &= y_{i - 1}, \\
a_i x_{i - 1} + b_i x_i + c_i x_{i + 1} &= y_i, \\
a_{i + 1} x_{i} + b_{i + 1} x_{i + 1} + c_{i + 1} x_{i + 2} &= y_{i + 1}.
\end{aligned}
$$
We can eliminate $x_{i - 1}$ and $x_{i + 1}$ to obtain a system of the form
$$
\hat{a}_i x_{i - 2} + \hat{b}_i x_i + \hat{c}_i x_{i + 2} = \hat{y}_i,
$$
where the modified coefficients $\hat{a}_i$, $\hat{b}_i$, $\hat{c}_i$ and the modified rhs $\hat{y}_i$ are
$$
\begin{aligned}
\hat{a}_i &= -\frac{a_{i - 1} a_i}{b_{i - 1}}, \\
\hat{b}_i &= b_i - \frac{c_{i - 1} a_i}{b_{i - 1}} - \frac{c_{i} a_{i + 1}}{b_{i + 1}}, \\
\hat{c}_i &= -\frac{c_i c_{i + 1}}{b_{i + 1}}, \\
\hat{y}_i &= y_i - \frac{a_i y_{i - 1}}{b_{i - 1}} - \frac{c_i y_{i + 1}}{b_{i + 1}}.
\end{aligned}
$$
The resulting system of equations for $x_{2j}$ is still tridiagonal and has roughly half the size of the original.

The cyclic reduction algorithm has two phases.
In the first phase, the above elimination step is iterated until the system is reduced to either one or two equations, which can then be directly solved.
In each iteration the computation of modified coefficients can be done in parallel.
The second phase involves finding the rest of the solution by using the final coefficients.
The second phase is also iterative, where at each iteration the number of know solution values increases by a factor of two. A drawback of this algorithm is that the amount of parallel computations available at each iteration is not constant.

The parallel cyclic reduction is based on the same idea, but has only one phase. In the first iteration it reduces the
original system to two systems of half the size. The second iteration reduces it to four systems of quarter the size
and so on. In the final iteration systems of size one or two are solved to obtain the whole solution at once.
In contrast to the cyclic reduction, this algorithm has constant amount of parallelism available at the cost of performing
more redundant work.

A naive application of the cyclic reduction to the diffusion system would result
in the following equation for the modified main diagonal
$$
\hat{b}_i = h_i
             + g_{i - 1} - \frac{g_{i-1}^2}{h_{i - 1} + g_{i - 2} + g_{i - 1}}
             + g_{i + 1} - \frac{g_{i+1}^2}{h_{i + 1} + g_{i} + g_{i + 1}}.
$$
Using this expression can potentially result in catastrophic cancellation errors and overflows if
$g_{i - 1}$ or $g_{i + 1}$ are very large.
To improve its stability, this expression can be rewritten as
$$
\hat{b}_i = h_i
             + h_{i - 1} \frac{g_{i - 1}}{h_{i - 1} + g_{i - 2} + g_{i - 1}}
             + h_{i + 1} \frac{g_{i + 1}}{h_{i + 1} + g_i + g_{i + 1}}
             + g_{i - 2} \frac{g_{i - 1}}{h_{i - 1} + g_{i - 2} + g_{i - 1}}
             + g_i \frac{g_{i + 1}}{h_{i + 1} + g_i + g_{i + 1}}.
$$
This equation can be shown to be in the form
$$
\hat{b}_i = \hat{h}_i + \hat{g}_{i - 2} + \hat{g}_i,
$$
where
$$
\begin{aligned}
\hat{h}_i &= h_i
             + h_{i - 1} \frac{g_{i - 1}}{h_{i - 1} + g_{i - 2} + g_{i - 1}}
             + h_{i + 1} \frac{g_{i + 1}}{h_{i + 1} + g_i + g_{i + 1}}, \\
\hat{g}_i &= g_i \frac{g_{i + 1}}{h_{i + 1} + g_i + g_{i + 1}}.
\end{aligned}
$$
These two equations form the basis of stable (parallel) cyclic reduction for diffusion problems.


## 4 Design

TODO

## 5 Verification and Testing

### 5.1 Test correctness

The solver will be compared against exact solution for a variety of (batch size, system size) combinations.

### 5.2 Test stability

The solver stability will be tested on an idealized vertical mixing problem with abrupt changes in
the diffusion coefficient.
