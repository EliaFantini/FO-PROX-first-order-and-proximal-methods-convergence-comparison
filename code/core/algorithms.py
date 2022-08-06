import time
import numpy as np
import scipy.sparse.linalg as spla
from numpy.random import randint
from scipy.sparse.linalg.dsolve import linsolve

from core.utils import print_end_message,print_start_message,print_progress

n = np.load('dataset/training.npz')['b'].shape[0] # number of data points


##########################################################################
# Unconstrained methods
##########################################################################

def GD(fx, gradf, parameter):
    """
    Function:  [x, info] = GD(fx, gradf, parameter)
    Purpose:   Implementation of the gradient descent algorithm.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               Lips       - Lipschitz constant for gradient.
               strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradf:
    :param parameter:
    :return:
    """
    method_name = 'Gradient Descent'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize x and alpha.
    #### YOUR CODE GOES HERE
    x = parameter['x0']
    alpha = 1 / parameter['Lips']
    maxit = parameter['maxit']

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit, 'x': np.zeros([maxit, x.shape[0]])}

    # Main loop.

    for iter in range(maxit):
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        #### YOUR CODE GOES HERE
        x_next = x - alpha * gradf(x)

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)
        info['x'][iter] = x

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next
    info['iter'] = maxit

    print_end_message(method_name, time.time() - tic_start)
    return x, info


# gradient with strong convexity
def GDstr(fx, gradf, parameter):
    """
    Function:  GDstr(fx, gradf, parameter)
    Purpose:   Implementation of the gradient descent algorithm.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               Lips       - Lipschitz constant for gradient.
                strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradf:
    :param parameter:
    :return: x, info
    """

    method_name = 'Gradient Descent with strong convexity'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize x and alpha.
    #### YOUR CODE GOES HERE
    x = parameter['x0']
    alpha = 2 / (parameter['Lips'] + parameter['strcnvx'])
    maxit = parameter['maxit']

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit, 'x': np.zeros([maxit, x.shape[0]])}

    # Main loop.
    for iter in range(maxit):
        # Start timer
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        #### YOUR CODE GOES HERE
        x_next = x - alpha * gradf(x)

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)
        info['x'][iter] = x

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next

    print_end_message(method_name, time.time() - tic_start)
    return x, info


# accelerated gradient
def AGD(fx, gradf, parameter):
    """
    Function:  AGD (fx, gradf, parameter)
    Purpose:   Implementation of the accelerated gradient descent algorithm.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               Lips       - Lipschitz constant for gradient.
               strcnvx	  - strong convexity parameter
    :param fx:
    :param gradf:
    :param parameter:
    :return:
    """
    method_name = 'Accelerated Gradient'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize x, y and t.
    #### YOUR CODE GOES HERE
    x = parameter['x0']
    t = 1
    y = x
    alpha = 1 / parameter['Lips']
    maxit = parameter['maxit']

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):

        # Start the clock.
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        #### YOUR CODE GOES HERE
        t_next = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
        x_next = y - alpha * gradf(y)
        y_next = x_next + ((t - 1) / t_next) * (x_next - x)
        y = y_next

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare next iteration
        x = x_next
        t = t_next

    print_end_message(method_name, time.time() - tic_start)
    return x, info


# accelerated gradient with strong convexity
def AGDstr(fx, gradf, parameter):
    """
    Function:  AGDstr(fx, gradf, parameter)
    Purpose:   Implementation of the accelerated gradient descent algorithm.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               Lips       - Lipschitz constant for gradient.
               strcnvx	  - strong convexity parameter
    :param fx:
    :param gradf:
    :param parameter:
    :return: x, info
    """
    method_name = 'Accelerated Gradient with strong convexity'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize x, y and t.
    #### YOUR CODE GOES HERE
    x = parameter['x0']
    y = x
    sqrt_lips = np.sqrt(parameter['Lips'])
    sqrt_strcnvx = np.sqrt(parameter['strcnvx'])
    alpha = 1 / parameter['Lips']
    maxit = parameter['maxit']

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):

        # Start the clock.
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        #### YOUR CODE GOES HERE
        x_next = y - alpha * gradf(y)
        y_next = x_next + ((sqrt_lips - sqrt_strcnvx) / (sqrt_lips + sqrt_strcnvx)) * (x_next - x)
        y = y_next

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare next iteration
        x = x_next

    print_end_message(method_name, time.time() - tic_start)
    return x, info


# LSGD
def LSGD(fx, gradf, parameter):
    """
    Function:  [x, info] = LSGD(fx, gradf, parameter)
    Purpose:   Implementation of the gradient descent with line-search.
    Parameter: x0         - Initial estimate.
           maxit      - Maximum number of iterations.
           Lips       - Lipschitz constant for gradient.
           strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradf:
    :param parameter:
    :return: x, info
    """
    method_name = 'Gradient Descent with line search'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize x, y and t.
    #### YOUR CODE GOES HERE

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        # Start the clock.
        tic = time.time()
        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        #### YOUR CODE GOES HERE

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare next iteration
        x = x_next

    print_end_message(method_name, time.time() - tic_start)
    return x, info


# LSAGD
def LSAGD(fx, gradf, parameter):
    """
    Function:  [x, info] = LSAGD (fx, gradf, parameter)
    Purpose:   Implementation of AGD with line search.
    Parameter: x0         - Initial estimate.
           maxit      - Maximum number of iterations.
           Lips       - Lipschitz constant for gradient.
           strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradf:
    :param parameter:
    :return:
    """
    method_name = 'Accelerated Gradient with line search'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize x, y and t.
    #### YOUR CODE GOES HERE

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        # Start the clock.
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        #### YOUR CODE GOES HERE

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next
        t = t_next

    print_end_message(method_name, time.time() - tic_start)
    return x, info


# AGDR
def AGDR(fx, gradf, parameter):
    """
    Function:  [x, info] = AGDR (fx, gradf, parameter)
    Purpose:   Implementation of the AGD with adaptive restart.
    Parameter: x0         - Initial estimate.
           maxit      - Maximum number of iterations.
           Lips       - Lipschitz constant for gradient.
           strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradf:
    :param parameter:
    :return:
    """
    method_name = 'Accelerated Gradient with restart'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize x, y, t and find the initial function value (fval).
    #### YOUR CODE GOES HERE
    x = parameter['x0']
    t = 1
    y = x
    alpha = 1 / parameter['Lips']
    maxit = parameter['maxit']

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        # Start the clock.
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        #### YOUR CODE GOES HERE
        while True:
            t_next = (1 + np.sqrt(1 + 4 * t ** 2)) / 2
            x_next = y - alpha * gradf(y)
            y_next = x_next + ((t - 1) / t_next) * (x_next - x)
            y = y_next
            if fx(x_next) <= fx(x):
                break
            else:
                t = 1
                y = x

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next
        t = t_next

    print_end_message(method_name, time.time() - tic_start)
    return x, info


# LSAGDR
def LSAGDR(fx, gradf, parameter):
    """
    Function:  [x, info] = LSAGDR (fx, gradf, parameter)
    Purpose:   Implementation of AGD with line search and adaptive restart.
    Parameter: x0         - Initial estimate.
           maxit      - Maximum number of iterations.
           Lips       - Lipschitz constant for gradient.
           strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradf:
    :param parameter:
    :return:
    """
    method_name = 'Accelerated Gradient with line search + restart'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize x, y, t and find the initial function value (fval).
    #### YOUR CODE GOES HERE

    # Main loop.
    for iter in range(maxit):
        # Start the clock.
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        #### YOUR CODE GOES HERE

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next
        t = t_next

    print_end_message(method_name, time.time() - tic_start)
    return x, info


def AdaGrad(fx, gradf, parameter):
    """
    Function:  [x, info] = AdaGrad (fx, gradf, hessf, parameter)
    Purpose:   Implementation of the adaptive gradient method with scalar step-size.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               Lips       - Lipschitz constant for gradient.
               strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradf:
    :param parameter:
    :return:
    """
    method_name = 'Adaptive Gradient method'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize x, B0, alpha, grad (and any other)
    #### YOUR CODE GOES HERE
    x = parameter['x0']
    alpha = 1
    delta = 1 / 10E5
    q = 0
    maxit = parameter['maxit']

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        # Start the clock.
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        #### YOUR CODE GOES HERE
        q_prev = q
        gradf_x = gradf(x)
        q = q_prev + np.linalg.norm(gradf_x, 2) ** 2
        h = (np.sqrt(q) + delta) * np.identity(gradf_x.shape[0])
        x_next = x - alpha * np.dot(np.linalg.inv(h), gradf_x)

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next

    print_end_message(method_name, time.time() - tic_start)
    return x, info


# Newton
def ADAM(fx, gradf, parameter):
    """
    Function:  [x, info] = ADAM (fx, gradf, parameter)
    Purpose:   Implementation of ADAM.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               Lips       - Lipschitz constant for gradient.
               strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradf:
    :param parameter:
    :return:
    """

    method_name = 'ADAM'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize x, beta1, beta2, alphs, epsilon (and any other)
    #### YOUR CODE GOES HERE

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.
    for iter in range(maxit):
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        #### YOUR CODE GOES HERE

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next

    print_end_message(method_name, time.time() - tic_start)
    return x, info


def SGD(fx, gradfsto, parameter):
    """
    Function:  [x, info] = GD(fx, gradf, parameter)
    Purpose:   Implementation of the gradient descent algorithm.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               Lips       - Lipschitz constant for gradient.
               strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradfsto:
    :param parameter:
    :return:
    """
    method_name = 'Stochastic Gradient Descent'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize x and alpha.
    #### YOUR CODE GOES HERE
    x = parameter['x0']
    maxit = parameter['maxit']

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.

    for iter in range(maxit):
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        #### YOUR CODE GOES HERE
        i = randint(1, n)  # n is a global variable, see code line 11
        x_next = x - (1 / (iter + 1)) * gradfsto(x, i)

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next

    print_end_message(method_name, time.time() - tic_start)
    return x, info


def SAG(fx, gradfsto, parameter):
    """
    Function:  [x, info] = SAG(fx, gradfsto, parameter)
    Purpose:   Implementation of the gradient descent algorithm.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               Lips       - Lipschitz constant for gradient.
    :param fx:
    :param gradfsto:
    :param parameter:
    :return:
    """
    method_name = 'Stochastic Gradient Descent with averaging'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize x and alpha.
    #### YOUR CODE GOES HERE
    x = parameter['x0']
    maxit = parameter['maxit']
    alpha = 1 / (16 * parameter['Lmax'])
    v = np.zeros([n, x.shape[0]])

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}
    # Main loop.

    for iter in range(maxit):
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        #### YOUR CODE GOES HERE
        i = randint(1, n)

        v[i, :] = gradfsto(x, i)
        x_next = x - (alpha / n) * v.sum(axis=0)

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next

    print_end_message(method_name, time.time() - tic_start)
    return x, info


def SVR(fx, gradf, gradfsto, parameter):
    """
    Function:  [x, info] = GD(fx, gradf, parameter)
    Purpose:   Implementation of the gradient descent algorithm.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               Lips       - Lipschitz constant for gradient.
               strcnvx    - Strong convexity parameter of f(x).
    :param fx:
    :param gradf:
    :param gradfsto:
    :param parameter:
    :return:
    """
    method_name = 'Stochastic Gradient Descent with variance reduction'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize x and alpha.
    #### YOUR CODE GOES HERE
    x = parameter['x0']
    maxit = parameter['maxit']
    q = int(1000 * parameter['Lmax'])
    gamma = 0.01 / parameter['Lmax']
    n = x.shape[0]

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop.

    for iter in range(maxit):
        tic = time.time()

        # Update the next iteration. (main algorithmic steps here!)
        # Use the notation x_next for x_{k+1}, and x for x_{k}, and similar for other variables.
        #### YOUR CODE GOES HERE
        x_tilde = x
        x_tilde_l = x_tilde
        x_tilde_sum = np.zeros(x.shape[0])
        v_k = gradf(x_tilde)
        for l in range(q):
            i_l = randint(1, n)
            v_l = gradfsto(x_tilde_l, i_l) - gradfsto(x_tilde, i_l) + v_k
            x_tilde_l_next = x_tilde_l - gamma * v_l
            x_tilde_sum = x_tilde_sum + x_tilde_l_next
            x_tilde_l = x_tilde_l_next
        x_next = (1 / q) * x_tilde_sum

        # Compute error and save data to be plotted later on.
        info['itertime'][iter] = time.time() - tic
        info['fx'][iter] = fx(x)

        # Print the information.
        if (iter % 5 == 0) or (iter == 0):
            print('Iter = {:4d},  f(x) = {:0.9f}'.format(iter, info['fx'][iter]))

        # Prepare the next iteration
        x = x_next

    print_end_message(method_name, time.time() - tic_start)
    return x, info


##########################################################################
# Prox
##########################################################################

def SubG(fx, gx, gradfx, parameter):
    """
    Function:  [x, info] = subgrad(fx, gx, gradfx, parameter)
    Purpose:   Implementation of the gradient descent algorithm.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               Lips       - Lipschitz constant for gradient.
    :param fx:
    :param gx:
    :param gradfx:
    :param parameter:
    :return:
    """
    method_name = 'Subgradient'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize lambda, maxit, x0, alpha and subgrad function
    G = 54  # pre computed with np.linalg.norm(A)
    R = 0.41529129  # pre computed with np.linalg.norm(x0 - xstar)
    #### YOUR CODE GOES HERE
    x0 = parameter['x0']
    maxit = parameter['maxit']
    lmbd = parameter['lambda']

    subgrad = lambda x: gradfx(x) + lmbd * np.sign(x)

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    # Main loop
    x_k = x0
    for k in range(maxit):
        tic = time.time()

        # Update the next iteration (x and alpha)
        #### YOUR CODE GOES HERE
        x_k = x_k - (R / (G * np.sqrt(k + 1))) * subgrad(x_k)

        # Compute error and save data to be plotted later on.
        info['itertime'][k] = time.time() - tic
        info['fx'][k] = fx(x_k) + lmbd * gx(x_k)
        if k % parameter['iter_print'] == 0:
            print_progress(k, maxit, info['fx'][k], fx(x_k), gx(x_k))

    print_end_message(method_name, time.time() - tic_start)
    return x_k, info


def ista(fx, gx, gradf, proxg, params):
    """
    Function:  [x, info] = ista(fx, gx, gradf, proxg, parameter)
    Purpose:   Implementation of ISTA.
    Parameter: x0         - Initial estimate.
               maxit      - Maximum number of iterations.
               prox_Lips  - Lipschitz constant for gradient.
               lambda     - regularization factor in F(x)=f(x)+lambda*g(x).
    :param fx:
    :param gx:
    :param gradf:
    :param proxg:
    :param parameter:
    :return:
    """

    method_name = 'ISTA'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize parameters.
    #### YOUR CODE GOES HERE
    x0 = params['x0']
    maxit = params['maxit']
    lmbd = params['lambda']
    alpha = 1 / params['prox_Lips']

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    x_k = x0
    for k in range(maxit):
        tic = time.time()

        # Update the iterate
        #### YOUR CODE GOES HERE
        y = x_k - alpha * gradf(x_k)
        x_k_next = proxg(y, alpha * lmbd)
        x_k = x_k_next

        # Compute error and save data to be plotted later on.
        info['itertime'][k] = time.time() - tic
        info['fx'][k] = fx(x_k) + lmbd * gx(x_k)
        if k % params['iter_print'] == 0:
            print_progress(k, maxit, info['fx'][k], fx(x_k), gx(x_k))

    print_end_message(method_name, time.time() - tic_start)
    return x_k, info


def fista(fx, gx, gradf, proxg, params):
    """
    Function:  [x, info] = fista(fx, gx, gradf, proxg, parameter)
    Purpose:   Implementation of FISTA (with optional restart).
    Parameter: x0            - Initial estimate.
               maxit         - Maximum number of iterations.
               prox_Lips     - Lipschitz constant for gradient.
               lambda        - regularization factor in F(x)=f(x)+lambda*g(x).
               restart_fista - enable restart.
    :param fx:
    :param gx:
    :param gradf:
    :param proxg:
    :param parameter:
    :return:
    """

    if params['restart_fista']:
        method_name = 'FISTAR'
    else:
        method_name = 'FISTA'
    print_start_message(method_name)
    tic_start = time.time()

    # Initialize parameters
    #### YOUR CODE GOES HERE
    x0 = params['x0']
    maxit = params['maxit']
    lmbd = params['lambda']
    alpha = 1 / params['prox_Lips']
    y_k = x0
    t_k = 1
    restart_fista = params['restart_fista']

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}

    x_k = x0
    for k in range(maxit):
        tic = time.time()

        # Update iterate
        #### YOUR CODE GOES HERE
        prox_argument = y_k - alpha * gradf(y_k)
        x_k_next = proxg(prox_argument, lmbd * alpha)
        t_k_next = (1 + np.sqrt(4 * (t_k ** 2) + 1)) / 2
        y_k_next = x_k_next + ((t_k - 1) / t_k_next) * (x_k_next - x_k)
        if restart_fista and gradient_scheme_restart_condition(x_k, x_k_next, y_k):
            y_k = x_k
            # t_k = 1 I'm not sure if t_k needs to be reloaded too, but without this the code converges slightly faster
        else:
            y_k = y_k_next
            t_k = t_k_next
            x_k = x_k_next

        # Compute error and save data to be plotted later on.
        info['itertime'][k] = time.time() - tic
        info['fx'][k] = fx(x_k) + lmbd * gx(x_k)
        if k % params['iter_print'] == 0:
            print_progress(k, maxit, info['fx'][k], fx(x_k), gx(x_k))

    print_end_message(method_name, time.time() - tic_start)
    return x_k, info


def gradient_scheme_restart_condition(x_k, x_k_next, y_k):
    """
    Whether to restart
    """
    #### YOUR CODE GOES HERE
    return (y_k - x_k_next) @ (x_k_next - x_k) > 0


def prox_sg(fx, gx, gradfsto, proxg, params):
    """
    Function:  [x, info] = prox_sg(fx, gx, gradfsto, proxg, parameter)
    Purpose:   Implementation of ISTA.
    Parameter: x0                - Initial estimate.
               maxit             - Maximum number of iterations.
               prox_Lips         - Lipschitz constant for gradient.
               lambda            - regularization factor in F(x)=f(x)+lambda*g(x).
               no0functions      - number of elements in the finite sum in the objective.
               stoch_rate_regime - step size as a function of the iterate k.
    :param fx:
    :param gx:
    :param gradfsto:
    :param proxg:
    :param parameter:
    :return:
    """

    method_name = 'PROXSG'
    print_start_message(method_name)

    tic_start = time.time()

    # Initialize parameters
    #### YOUR CODE GOES HERE
    x0 = params['x0']
    maxit = params['maxit']
    lmbd = params['lambda']
    stoch_rate_regime = params['stoch_rate_regime']
    sum_gamma = 0.0
    sum_gamma_x = 0.0
    X_avg = None

    info = {'itertime': np.zeros(maxit), 'fx': np.zeros(maxit), 'iter': maxit}
    x_k = x0
    for k in range(maxit):
        tic = time.time()

        # Update the average iterate
        #### YOUR CODE GOES HERE
        i = randint(1, n)
        gamma_k = stoch_rate_regime(k)
        prox_argument = x_k - gamma_k * gradfsto(x_k, i)
        x_k_next = proxg(prox_argument, gamma_k * lmbd)
        sum_gamma_x += gamma_k * x_k
        sum_gamma += gamma_k
        X_avg = sum_gamma_x / sum_gamma
        x_k = x_k_next
        # Compute error and save data to be plotted later on.
        info['itertime'][k] = time.time() - tic
        info['fx'][k] = fx(X_avg) + lmbd * gx(X_avg)
        if k % params['iter_print'] == 0:
            print_progress(k, maxit, info['fx'][k], fx(X_avg), gx(X_avg))

    print_end_message(method_name, time.time() - tic_start)
    return X_avg, info
