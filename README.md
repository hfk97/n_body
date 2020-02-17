# N-Body Problem (Barnes Hut)

Recently I stumbled across the Barnes-Hut algorithm which simplifies solving the n-body problem, i.e. predicting the individual motions of a group of objects interacting gravitationally.
Since I always wanted to write a program incorporating tree data structures, I decided to implement it.

This repository consists of a single file that enables you to compute the individual motions of interacting bodies in 3D.

Barnes Hut algorithms reduce the computational intensity of the N-body problem from U($n^2$) to U(n*log(n)). They do so by partitioning space into subspaces and, given sufficient distance, using the center of mass and mass of subspaces instead of each of the particles contained in them.

You can find more information on how Barnes Hut algorithms work and the N-body problem under the following links:

- [Barnes Hut] (http://arborjs.org/docs/barnes-hut)
- [Barnes Hut (interactive)] (https://jheer.github.io/barnes-hut/)
- [N-body problem] (https://en.wikipedia.org/wiki/N-body_problem) 

*Disclaimer: Normally algorithms like this would be used in astrophysics simulations or particle simulations. Since I did not optimize my code for efficiency, I do not recommend using for any sort of scientific work. If you want to use a barnes-hut algorithm for research you should find a GPU implementation, because those tend to be significantly faster* 