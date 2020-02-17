import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

global grav_const
# The real gravitational constant is about 6.67408e-11; 2 was chosen because the animation will be prettier
grav_const = 2


class OctNode:
    """Octtree node element, with id, center, size, mass, CenterOfMass, gravity, children"""

    def __init__(self, center, size, masses, points, ids, leaves=[]):
        self.center = center  # center of box
        self.size = size  # maximum side length of the box
        self.children = []  # will later be changed if node has children

        n_points = len(points)  # number of points/particles in a box

        # only one point in the box
        if n_points == 1:
            leaves.append(self)
            self.COM = points[0]  # set center of mass as point
            self.mass = masses[0]
            self.id = ids[0]
            self.g = np.zeros(3)  # the gravitational field will be inserted later

        # if two or more points subdivide box --> next level of octtree
        else:
            # generate children on the next level of the branch
            self.create_children(points, masses, ids, leaves)
            # initialize sum and center of mass variables
            com_total = np.zeros(3)
            m_total = 0.
            # calculate center of mass and sum of mass for parent node
            for c in self.children:
                m = c.mass
                m_total += m
                com_total += c.COM * m  # weigh center of masses by mass (use moments)
            self.mass = m_total
            self.COM = com_total / self.mass

    def create_children(self, points, masses, ids, leaves):
        octant_index = (points > self.center)  # for determining octants
        for i in range(2):  # looping over the 8 octants
            for j in range(2):
                for k in range(2):
                    # assign points to octants
                    in_octant = np.all(octant_index == np.bool_([i, j, k]), axis=1)
                    if not any(in_octant):
                        continue  # if no points in an octant no node

                    dx = 0.5 * self.size * (np.array([i, j, k]) - 0.5)  # calculate center of new child octant
                    # print(in_octant)
                    self.children.append(OctNode(self.center + dx,
                                                 self.size / 2,
                                                 masses[in_octant],
                                                 points[in_octant],
                                                 ids[in_octant],
                                                 leaves))  # store children info in node

    def __repr__(self):
        return "OctNodeObject"

    def __str__(self):
        return f"OctNodeObject: id({self.id}), center({self.center}), size({self.size}), mass({self.mass}), " \
            f"CenterOfMass({self.COM}), gravity({self.g}, children({self.children})"


def tree_walk(node, node0, thetamax=0.7, gc=grav_const):
    """
    Adds the contribution to the field at node0's point due to particles in node.
    Calling this with the topnode as node will walk the tree to calculate the total field at node0.
    """
    dx = node.COM - node0.COM  # vector between centres of mass
    r = np.sqrt(np.sum(dx ** 2))  # distance from vectors
    # make sure that points are not at the same position
    if r > 0:
        # if single node or theta larger than distance add the gravitational field contribution to node.g
        if (len(node.children) == 0) or (node.size / r < thetamax):
            node0.g += gc * node.mass * dx / r ** 3
        else:
            # otherwise split up the node and repeat
            for c in node.children:
                tree_walk(c, node0, thetamax, gc)


def grav_accel(points, masses, thetamax=.5, gc=grav_const):
    center = (np.max(points, axis=0) + np.min(points, axis=0)) / 2  # center of particle cluster
    topsize = np.max(np.max(points, axis=0) - np.min(points, axis=0))  # size of total bounding box

    leaves = []  # want to keep track of leaf nodes
    topnode = OctNode(center, topsize, masses, points, np.arange(len(masses)), leaves)  # build the tree

    accel = np.empty_like(points)
    for i, leaf in enumerate(leaves):
        tree_walk(topnode, leaf, thetamax, gc)  # do field summation
        accel[leaf.id] = leaf.g  # get the stored acceleration

    return accel


def gen_rand_prtcls(n_particles, n_iterations):
    global masses
    positions = np.random.normal(size=(n_particles-1, 3))
    masses = abs(np.random.normal(size=(n_particles-1, 1)))
    # add gravitational center
    positions = np.vstack((positions, (np.max(positions, axis=0) + np.min(positions, axis=0)) / 2))
    masses = np.append(masses, max(masses)*5)
    positions = [positions]
    # Computing trajectory
    for iteration in range(n_iterations):
        positions.append(positions[-1] + grav_accel(positions[-1], masses))
    return positions


def main(n_prtcls=50, n_frames=15000, scale=200):

    data = gen_rand_prtcls(n_particles=n_prtcls, n_iterations=n_frames-1)
    data = np.array(data)

    fig = plt.figure()
    ax = p3.Axes3D(fig)

    # Plot the first position for all particles
    h_particles = ax.plot(*data[0].T, marker='.', ls='', markersize=3)[0]

    # Setting the axes properties
    ax.set_xlim3d([-1*scale, 1*scale])

    ax.set_ylim3d([-1*scale, 1*scale])

    ax.set_zlim3d([-1*scale, 1*scale])
    ax.set_title(f'Particle sim (n={n_prtcls})', color='white')

    ax.set_axis_off()
    ax.set_facecolor('black')

    def update_particles(num):
        # Plot the iterations up to num for all particles
        h_particles.set_xdata(data[num, :, 0].ravel())
        h_particles.set_ydata(data[num, :, 1].ravel())
        h_particles.set_3d_properties(data[num, :, 2].ravel())
        return h_particles

    prtcl_ani = animation.FuncAnimation(fig, update_particles, frames=n_frames, interval=1)

    plt.show()


if __name__ == '__main__':
    main()
