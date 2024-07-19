def euler(f, t_eval, y0):
    '''
    Uses Euler's method to approximate the 
    solution to the IVP y' = f(t,y) with
    y(t_eval[0]) = y0.

    Returns an array with the same length 
    as t_eval with entries corresponding to
    the approximated values of y(t_eval[i]).
    '''
    res = [y0]
    for t_prev, t_current in zip(t_eval, t_eval[1:]):
        h = t_current - t_prev
        res.append(h * f(t_prev, res[-1]) + res[-1])
    return res


def midpoint(f, t_eval, y0):
    '''
    Uses the midpoint method to approximate 
    the solution to the IVP y' = f(t,y) with
    y(t_eval[0]) = y0.

    Returns an array with the same length 
    as t_eval with entries corresponding to
    the approximated values of y(t_eval[i]).
    '''
    res = [y0]
    for t_prev, t_current in zip(t_eval, t_eval[1:]):
        h = t_current - t_prev

        # compute the n + 1/2 term.
        y_half = res[-1] + h * f(t_current, res[-1])
        res.append(h * f(t_current + h / 2, y_half) + res[-1])
    return res


def leapfrog(f, t_eval, y0, initialize='midpoint'):
    '''
    Uses the midpoint method to approximate 
    the solution to the IVP y' = f(t,y) with
    y(t_eval[0]) = y0.

    initialize is in {'constant',
                      'euler',
                      'midpoint'}
    These correspond to how the approximation
    to y(t_eval[i]) is computed.  

    'constant' : y0 is used.
    'euler'    : Euler's method is used for
                 one step.
    'midpoint' : midpoint method is used for 
                 one step. 

    Returns an array with the same length 
    as t_eval with entries corresponding to
    the approximated values of y(t_eval[i]).
    '''
    res = [y0]

    match initialize:
        case 'constant':
            res.append(y0)
        case 'euler':
            res = euler(f, t_eval[:2], y0)
        case 'midpoint':
            res = midpoint(f, t_eval[:2], y0)

    for t_prev, t_current in zip(t_eval[1:], t_eval[2:]):
        h = t_current - t_prev
        res.append(2*h * f(t_current, res[-1]) + res[-2])
    return res


def find_zero(f, x0=1.0, h=10e-5,
              eps=10e-20,
              maxiter=1e6,
              fprime=None):
    '''
    Uses Newton's method starting at 
    x0 to try to find the root of f.  
    When the value changes by less than
    eps, the current value is returned.  
    The parameter h is used to 
    approximate f' if fprime is not 
    given as an argument.

    At most maxiter iterations are run.
    If all of these are run and the 
    stopping condition is not met, the
    last computed value is returned.  
    '''
    current = x0
    index = 0
    while index < maxiter:
        if fprime is not None:
            deriv = fprime(current)
        else:
            deriv = (f(current+h) - f(current)) / h
        new = current - (f(current) / deriv)
        if abs(new-current) < eps:
            return new
        current = new
        index += 1
    return current


def trapazoidal(f, t_eval, y0, modified=False):
    '''
    Uses the trapazoidal method to approximate 
    the solution to the IVP y' = f(t,y) with
    y(t_eval[0]) = y0.

    Returns an array with the same length 
    as t_eval with entries corresponding to
    the approximated values of y(t_eval[i]).

    If modified is True, then the modified 
    trapazoidal method (aka Heun's method or 
    the improved Euler method) is used.
    '''
    res = [y0]
    for t_prev, t_current in zip(t_eval, t_eval[1:]):
        h = t_current - t_prev
        if modified:
            y_next_bar = res[-1] + h * f(t_current, res[-1])
            res.append(res[-1] + h/2 *
                       (f(t_prev, res[-1]) + f(t_current, y_next_bar)))
        else:
            def trap(x):
                return res[-1] + h/2 * (f(t_prev, res[-1]) + f(t_current, x)) - x
            res.append(find_zero(trap))
    return res


def backward_euler(f, t_eval, y0):
    '''
    Uses the implicit backward Euler 
    method to approximate the solution 
    to the IVP y' = f(t,y) with
    y(t_eval[0]) = y0.

    Returns an array with the same length 
    as t_eval with entries corresponding to
    the approximated values of y(t_eval[i]).
    '''
    res = [y0]
    for t_prev, t_current in zip(t_eval, t_eval[1:]):
        h = t_current - t_prev

        def back_euler(x):
            return h*f(t_current, x) + res[-1] - x
        res.append(find_zero(back_euler))
    return res


def y_midpoint(f, t_eval, y0):
    '''
    Uses the y_midpoint method to 
    approximate the solution to the IVP 
    y' = f(t,y) with y(t_eval[0]) = y0.

    Returns an array with the same length 
    as t_eval with entries corresponding to
    the approximated values of y(t_eval[i]).
    '''
    res = [y0]
    for t_prev, t_current in zip(t_eval, t_eval[1:]):
        h = t_current - t_prev

        def back_euler(x):
            return h*f(t_current + h/2, (res[-1] + x) / 2) + res[-1] - x
        res.append(find_zero(back_euler))
    return res


def partial(f, index, h=1e-6):
    '''
    Returns the partial derivative of f
    with the derivative with respect to 
    the index variable.  Uses h to take
    the derivative.  
    '''
    def res(*x):
        return (f(*(xi if i != index else xi+h
                    for i, xi in enumerate(x)))
                - f(*x)) / h
    return res


def second_order_taylor(f, t_eval, y0):
    '''
    Uses the second-order Taylor method 
    to approximate the solution to the IVP 
    y' = f(t,y) with y(t_eval[0]) = y_0.

    Returns an array with the same length 
    as t_eval with entries corresponding to
    the approximated values of y(t_eval[i]).
    '''
    res = [y0]
    for t_prev, t_current in zip(t_eval, t_eval[1:]):
        h = t_current - t_prev
        res.append(res[-1]
                   + h * f(t_current, res[-1])
                   + ((h**2) / 2) * (partial(f, 0)(t_current, res[-1])
                                     + partial(f, 1)(t_current, res[-1])*f(t_current, res[-1])))
    return res
