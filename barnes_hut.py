import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

global grav_const
grav_const = 0.5

class OctNode:
    """Octtree node element, with id, center, size, mass, CenterOfMass, gravity, children"""

    def __init__(self, center, size, masses, points, ids, leaves=[]):
        self.center = center  # center of box
        self.size = size  # maximum side length of the box
        self.children = []  # will later be changed if node has children

        n_points = len(points)

        # only one point in the box
        if n_points == 1:
            leaves.append(self)
            self.COM = points[0] # set center of mass as point
            self.mass = masses[0]
            self.id = ids[0]
            self.g = np.zeros(3)  # the gravitational field is what we will insert later
        # two or more points
        else:
            # generate children on the next level of the branch
            self.SpawnChildren(points, masses, ids, leaves)
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

    def SpawnChildren(self, points, masses, ids, leaves):
        octant_index = (points > self.center)  # for determining octants
        for i in range(2):  # looping over the 8 octants
            for j in range(2):
                for k in range(2):
                    # assign points to octants
                    in_octant = np.all(octant_index == np.bool_([i, j, k]), axis=1)
                    # ToDo simplify? if not in_octant:
                    if not np.any(in_octant):
                        continue  # if no points in an octant no node

                    dx = 0.5 * self.size * (np.array([i, j, k]) - 0.5)  # calculate center of new child octant
                    # ToDo how does masses[in_octant], points[in_octant]... work
                    # print(in_octant)
                    self.children.append(OctNode(self.center + dx,
                                                 self.size / 2,
                                                 masses[in_octant],
                                                 points[in_octant],
                                                 ids[in_octant],
                                                 leaves))

    def __repr__(self):
        return "OctNodeObject"

    def __str__(self):
        return f"OctNodeObject: id({self.id}), center({self.center}), size({self.size}), mass({self.mass}), " \
            f"CenterOfMass({self.COM}), gravity({self.g}, children({self.children})"


def TreeWalk(node, node0, thetamax=0.7, G=grav_const):
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
            node0.g += G * node.mass * dx / r ** 3
        else:
            # otherwise split up the node and repeat
            for c in node.children:
                TreeWalk(c, node0, thetamax, G)


def GravAccel(points, masses, thetamax=0.7, G=grav_const):
    center = (np.max(points, axis=0) + np.min(points, axis=0)) / 2  # center of particle cluster
    topsize = np.max(np.max(points, axis=0) - np.min(points, axis=0))  # size of total bounding box
    leaves = []  # want to keep track of leaf nodes
    topnode = OctNode(center, topsize, masses, points, np.arange(len(masses)), leaves)  # build the tree

    accel = np.empty_like(points)
    for i, leaf in enumerate(leaves):
        TreeWalk(topnode, leaf, thetamax, G)  # do field summation
        accel[leaf.id] = leaf.g  # get the stored acceleration

    return accel


def Gen_RandPrtcls(n_particles, n_iterations):
    #ToDo implement proper mass and maybe collision
    global m
    x = np.random.normal(size=(n_particles, 3))
    m = np.repeat(np.random.normal()*10, n_particles)

    # Computing trajectory
    data = [x]
    for iteration in range(n_iterations):
        data.append(data[-1] + GravAccel(data[-1], m))
    return data


def main():
    while True:
        n_frames = 1000
        n_prtcls = 80

        # data = Gen_RandPrtcls(n_particles=5, n_iterations=100)
        data = Gen_RandPrtcls(n_particles=n_prtcls, n_iterations=n_frames-1)
        data = np.array(data)  # (n_iterations, n_particles, 3)


        fig = plt.figure()
        ax = p3.Axes3D(fig)

        # Plot the first position for all particles
        h = ax.plot(*data[0].T, marker='.', linestyle='None', markersize=3, color="y")[0]

        # Setting the axes properties

        #ax.set_xlim3d([min(data[-1][:,0])*1.1, max(data[-1][:,0])*1.1])
        ax.set_xlim3d([-125, 125])
        ax.set_xlabel('X')

        #ax.set_ylim3d([min(data[-1][:,1])*1.1, max(data[-1][:,1])*1.1])
        ax.set_ylim3d([-125, 125])
        ax.set_ylabel('Y')

        #ax.set_zlim3d([min(data[-1][:,2])*1.1, max(data[-1][:,1])*1.1])
        ax.set_zlim3d([-125, 125])
        ax.set_zlabel('Z')
        ax.set_title(f'Particle sim (n={n_prtcls})', color='white')

        ax.set_axis_off()
        ax.set_facecolor('black')

        def update_particles(num):
            # Plot the iterations up to num for all particles
            h.set_xdata(data[num, :, 0].ravel())
            h.set_ydata(data[num, :, 1].ravel())
            h.set_3d_properties(data[num, :, 2].ravel())
            return h

        prtcl_ani = animation.FuncAnimation(fig, update_particles, frames=n_frames,
                                            interval=33)

        plt.show()

        if 'q' in input("Enter q to quit: "):
            break

    return


if __name__ == '__main__':
    main()
