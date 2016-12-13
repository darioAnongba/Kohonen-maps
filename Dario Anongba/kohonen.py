"""Python script for Exercise set 6 of the Unsupervised and 
Reinforcement Learning.
"""

import numpy as np
import matplotlib.pylab as plb


def kohonen(targetdigits, size_k=6, sigma=10.0, eta=0.9, tmax=5000, threshold=1000, plot_errors=False):
    """Example for using create_data, plot_data and som_step.

    :param targetdigits: Set of labels to take into account for this algorithm
    :param size_k: Size of the Kohonen map. In this case it will be 6 X 6 by default
    :param sigma: Width of the neighborhood via the width of the gaussian that describes it, 10.0 by default
    :param eta: Learning rate, 0.9 by default
    :param tmax: Maximal iteration count. 5000 by default
    :param threshold: threshold of the error for convergence criteria. 1000 by default
    :param plot_errors: Plot the errors
    """
    plb.close('all')

    dim = 28 * 28
    data_range = 255.0
    unit_of_mean = 400

    # load in data and labels
    data = np.array(np.loadtxt('data.txt'))
    labels = np.loadtxt('labels.txt')

    # this selects all data vectors that corresponds to one of the four digits
    data = data[np.logical_or.reduce([labels == x for x in targetdigits]), :]
    # filter the labels
    labels = labels[np.logical_or.reduce([labels == x for x in targetdigits])]

    dy, dx = data.shape

    # initialise the centers randomly
    centers = np.random.rand(size_k ** 2, dim) * data_range

    # build a neighborhood matrix
    neighbor = np.arange(size_k ** 2).reshape((size_k, size_k))

    # set the random order in which the datapoints should be presented
    i_random = np.arange(tmax) % dy
    np.random.shuffle(i_random)

    # Converge step
    last_centers = np.copy(centers)
    errors = []
    mean_errors = []
    last_errors = [0.0]

    etas = [eta]

    for t, i in enumerate(i_random):
        sigma = som_step(centers, data[i, :], neighbor, eta, sigma)
        eta = max(0.9999 * eta, 0.1)
        etas.append(eta)

        err = np.sum(np.sum((last_centers - centers) ** 2, 1)) * 0.01

        if t > unit_of_mean:
            if len(last_errors) >= unit_of_mean:
                last_errors.pop(0)
            last_errors.append(err)

            # Update the mean error term
            tmp_error = np.mean(last_errors)
            mean_errors.append(tmp_error)

            if tmp_error < threshold:
                print('The algorithm converges after', t, 'iterations')
                break

        errors.append(err)
        last_centers = np.copy(centers)

    # Digit assignment given labels.txt
    digit_assignment = []
    for i in range(0, size_k ** 2):
        index = np.argmin(np.sum((data[:] - centers[i, :]) ** 2, 1))
        digit_assignment.append(labels[index])

    print('Digit assignment: \n')
    print(np.resize(digit_assignment, (size_k, size_k)))

    # for visualization, you can use this:
    for i in range(size_k ** 2):
        plb.subplot(size_k, size_k, i + 1)

        plb.imshow(np.reshape(centers[i, :], [28, 28]), interpolation='bilinear')
        plb.axis('off')

    # leave the window open at the end of the loop
    plb.show()
    plb.draw()

    if(plot_errors):
        plb.plot(errors)
        plb.title('Square of the errors over iterations', fontsize=20)
        plb.ylabel('Squared errors', fontsize=18)
        plb.xlabel('Iterations', fontsize=18)
        plb.ylim([0, 600000])
        plb.show()

        plb.plot(mean_errors)
        plb.title('Mean squared errors of the last 400 terms iterations', fontsize=20)
        plb.ylabel('Mean squared error', fontsize=18)
        plb.xlabel('Iterations', fontsize=18)
        plb.show()

        plb.plot(etas)
        plb.title('Learning rates decrease over iterations', fontsize=20)
        plb.ylabel('Learning rates', fontsize=18)
        plb.xlabel('Iterations', fontsize=18)
        plb.show()


def som_step(centers, data, neighbor, eta, sigma):
    """Performs one step of the sequential learning for a 
    self-organized map (SOM).
    
      centers = som_step(centers,data,neighbor,eta,sigma)
    
      Input and output arguments: 
       centers  (matrix) cluster centres. Have to be in format:
                         center X dimension
       data     (vector) the actually presented datapoint to be presented in
                         this timestep
       neighbor (matrix) the coordinates of the centers in the desired
                         neighborhood.
       eta      (scalar) a learning rate
       sigma    (scalar) the width of the gaussian neighborhood function.
                         Effectively describing the width of the neighborhood
    """

    size_k = int(np.sqrt(len(centers)))

    # find the best matching unit via the minimal distance to the datapoint
    b = np.argmin(np.sum((centers - np.resize(data, (size_k ** 2, data.size))) ** 2, 1))

    # find coordinates of the winner
    a, b = np.nonzero(neighbor == b)

    # update all units
    for j in range(size_k ** 2):
        # find coordinates of this unit
        a1, b1 = np.nonzero(neighbor == j)
        # calculate the distance and discounting factor
        disc = gauss(np.sqrt((a - a1) ** 2 + (b - b1) ** 2), [0, sigma])
        # update weights        
        centers[j, :] += disc * eta * (data - centers[j, :])

    # decrease the width of the neighborhood function
    return max(0.9999 * sigma, 1.0)


def gauss(x, p):
    """Return the gauss function N(x), with mean p[0] and std p[1].
    Normalized such that N(x=p[0]) = 1.
    """
    return np.exp((-(x - p[0]) ** 2) / (2 * p[1] ** 2))


def name2digits(name):
    """ takes a string NAME and converts it into a pseudo-random selection of 4
     digits from 0-9.
     
     Example:
     name2digits('Felipe Gerhard')
     returns: [0 4 5 7]
     """

    name = name.lower()

    if len(name) > 25:
        name = name[0:25]

    primenumbers = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

    n = len(name)

    s = 0.0

    for i in range(n):
        s += primenumbers[i] * ord(name[i]) * 2.0 ** (i + 1)

    import scipy.io.matlab
    data = scipy.io.matlab.loadmat('hash.mat', struct_as_record=True)
    x = data['x']
    t = np.mod(s, x.shape[0])

    return np.sort(x[int(t), :])


if __name__ == "__main__":
    target_digits = name2digits("Dario Anongba Varela")
    kohonen(targetdigits=target_digits, eta=0.7, tmax=20000, threshold=1000)
