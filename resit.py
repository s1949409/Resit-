#!/usr/bin/env python
# coding: utf-8

# # Resit assignment [50 marks]
# 
# The assignment consists of 5 exercises. Each exercise may contain coding and/or discussion questions.
# - Type your **code** in the **code cells** provided below each question.
# - For **discussion** questions, use the **Markdown cells** provided below each question, indicated by 📝. Double-click these cells to edit them, and run them to display your Markdown-formatted text. Please refer to the Week 1 tutorial notebook for Markdown syntax.

# ---
# ## Question 1: Sum of reciprocal squares [4 marks]
# 
# In 1741, Euler proved that the sum of the series of the reciprocals of squares of the natural numbers converged:
# 
# $$
# \sum_{n=1}^{\infty} \frac{1}{n^2} = \frac{\pi^2}{6}.
# $$
# 
# Write code to compute how many terms are needed to add, starting from $n=1$, to get the value of the sum correctly to within $10^{-4}$. Display the result in a clear and readable manner.
# 
# **[4 marks]**

# In[4]:


# Code for question 1

import math 
import numpy as np

tot = 1
n = 1

# Loop as long as tot is not close enough to the exact value
while not math.isclose((np.pi**2)/6, tot, rel_tol = 1e-4):
    n += 1              #Increment by 1
    tot += 1/(n**2)     # add the nth term of the series

# Displaying the number of iterations required depending on the chosen tolerance 
# Displaying a comparison between the approximate value and the exact value 
print(f'{n} iterations are needed.')
print(f'Approximate value: {tot:.6f}\n')
print(f'Exact value of pi^2/6: {(np.pi**2)/6:.6f}')


# ---
# ## Question 2: Linear algebra [10 marks]
# 
# Consider an arbitrary matrix $A \in \mathbb{R}^{n\times n}$, which we want to reduce to **row echelon form**. As seen in the Week 4 tutorial, in Section 4, we can apply elementary row operations to $A$ to do so.
# 
# The code below is a modified version of the `REF()` function from the Week 4 tutorial, which operates on a square matrix (not an augmented matrix). The main difference beyond this, is that this new function implements **partial pivoting**, allowing us to avoid the possibility of division by zero.
# 
# **2.1** Study the code carefully, and add detailed code comments in the modified section indicated below, to explain in detail how partial pivoting is implemented. Your comments must demonstrate your **understanding** of what the new code achieves and how it does it, not merely "rephrase" the code as a sentence.
# 
# **[3 marks]**

# In[5]:


import numpy as np

def row_op(A, alpha, i, beta, j):
    '''
    Applies row operation beta*A_j + alpha*A_i to A_j,
    the jth row of the matrix A.
    Changes A in place.
    '''
    # Allows us to choose a row 'j' and change it based on,
    # Scaling the row itself by beta and adding a scalar multiple of another chosen row 'i'
    
    # Apply the row operation on the jth row of A
    A[j, :] = beta * A[j, :] + alpha * A[i, :]


    
def REF_pivot(A):
    '''
    Reduces the square matrix A into row echelon form, in-place.
    '''
    # Extract the size of the matrix
    # -- We are dealing with a square matrix so the size of the column will indicate it's form as an n x n matrix.
    N = A.shape[0]
    # d scalar for for 2.2
    d = 1
    
    # Loop over the columns
    for col in range(N-1):
        
       ### NEW SECTION STARTS HERE
    
        '''
        If a diagonal entry is very small (almost zero), we acknowledge that this is a problem for computation as it can lead to division by zero.
        To avoid this problem, we implement a process to switch the almost zero row on the diagonal with another row.
        Consequently, the almost zero diagonal entry will no longer sit on the diagonal and thus reduce any complicating factors.

        To do this, first we check if a certain diagonal element is (almost zero/zero).
        If it is then we proceed with our remedial process

        We determine which row to switch our problem row with.
        The best choice is the row with the largest element in the column directly below our entry
        So, we generate the elements below our entry with A[col + 1: col].
        Taking the absolute value since we are concerned with position. For simplicity this will only be considered in the positive sense.
        Next, we determine the position of the larger value with 'argmax' -
        And situate it (the row) within the rest of the matrix by 'max_location += col + 1'
        Finally, we switch the initial position, with the newly desired row position by 'A[[col, max_location], :] = A[[max_location, col], :]'
        '''
        if abs(A[col, col]) < 1e-12:
        # for 2.2 where swapping the rows multiplies the determinant by -1
            d *= -1
            max_location = np.argmax(np.abs(A[col+1:, col]))
            # In order to position the max location relative to the rest of the matrix  
            max_location += col+1
            A[[col, max_location], :] = A[[max_location, col], :]

        '''
        Why 'for col in range(N-1):' and 'for row in range(col+1, N):'?

        For any square matrix imputed, a REF form means zero terms below the leading diagonal.
        Since the diagonal element for the last column in any case is not relevant to the REF format, our row of interest will be of range(N-1).

        It is convenient to consider columns first because we can use this value to determine how many times, we need to perform row operations, bellow the leading diagonal
        For each col range(N-1), row(col+ 1, N) takes us exactly bellow the diagonal as a staring place for iterating through the necessary row operations
        
        '''
        
        # In each column, loop over the rows below the diagonal
        for row in range(col+1, N):

            '''
            The final process to consider is how do we choose good alpha values which will allow us through row operations to step by step, produce our REF format?
            If we choose to fix beta with beta = 1, this means we are fixing the values in a chosen row.
            Thus, we want an alpha which will produce a row which is the additive inverse of the first element in our chosen row.
            This is easily achieved by taking the reciprocal of the first element from our row i, (resulting in 1) and multiplying this by the additive inverse.
            So overall through the row operations algorithm, we will have a zero entry for the first element of a given row acted upon by 'row_op'.

            '''
            
            # Calculate alpha as -(leading element / diagonal element)
            alpha = -A[row, col] / A[col, col]
            
            # * adjustments for 2.3
            if any([int(max(abs(A[col+1:, col]))) for col in range(N-1)]):
                pass
            else:
                break


            # Perform the row operation in-place (beta is always 1 here)
            row_op(A, alpha, col, 1, row)
            '''
            
            This process will continue until all possibilities within the range are iterated through. 
            Resulting in our matrix being changed through row operations into its REF.
           
            Overall, this code can be characterised as using process of alpha computation and row operations to reach a REF,
            Built into this is the safeguard that any leading diagonal element too close to zero will not cause the process to break down.
            '''
    
        # * Adjustments for 2.3
        if any([int(max(abs(A[col+1:, col]))) for col in range(N-1)]):
            pass
        else:
            break
    
    # * scalar for 2.2
    return d

    # We don't need to return anything as A is modified in-place.
    
 
#Testing with an example

# Testing with an example
A = np.array([[1, 1, 1],
              [2, 2, -1],
              [1, -1, 2]], dtype=float)

# Call the function to change A in-place
REF_pivot(A)

# The result should be:
# A = [[ 1.  1.  1.]
#      [ 0. -2.  1.]
#      [ 0.  0. -3.]]
print(A)


# **2.2** Let $B$ be the row echelon form of a matrix $A\in \mathbb{R}^{n\times n}$. The determinant of a triangular matrix (e.g. a matrix in row echelon form) is given by the product of its diagonal elements. Therefore, the determinant of $B$ is
# 
# $$
# \det(B) = \prod_{i=1}^N b_{ii}.
# $$
# 
# We also know that certain row operations modify the determinant of a matrix, in particular:
# - Adding a multiple of a row to another row does not change the determinant.
# - Multiplying a row by a scalar multiplies the determinant by that same scalar.
# - Swapping two rows multiplies the determinant by $-1$.
# 
# Let $d$ be the product of all scalars by which the determinant of $A$ is multiplied in the process of reducing it to its row echelon form $B$. We therefore have
# 
# $$
# \det(B) = d \det(A).
# $$
# 
# For example, if reducing $A$ to its REF $B$ necessitates two row swaps in total, and the multiplication of the first row by 3, then $d = (-1) \times (-1) \times 3 = 3$.
# 
# Write a function `REF_pivot_det()`, based on `REF_pivot()`, but modified to also return the value of $\det(A)$, computed using this method.
# 
# Test your function with the matrix $A$ from question **2.1**, and also by creating a $10 \times 10$ matrix of random floating point numbers between 0 and 1, and comparing your result with the determinant obtained using the built-in function `np.linalg.det()`.
# 
# **[4 marks]**

# In[6]:


# Code for question 2.2

def REF_pivot_det(A):
    '''
    The function is defined such that using the REF form from REF_pivot, the product of all its diagonal elements can 
    be determined to indicate det(B) i.e. REF of A. 
    We can find det(A) by dividing det(B) by any scalars accumulated along the way from row exchange.
    The only determinant scalar considerations taken into account are row swaps, which occur when a zero lies on 
    the diagonal in the previous function.
    There are multiples of rows added to another row as in row_op, however this does not affect the det(A) as it contributes no 
    excess scalars. 
    Since any row 'j' under row_op always has beta =1, there is no individual row/scalar multiplication that needs to be taken
    into account in terms of the determinant. 
    
    '''
    # size n of the square matrix, allows us to calculate how many products of diagonal elements are necessary
    N = A.shape[0]
    # reduce matrix A to it's REF B
    REF_pivot(A)
    # note scalars from row swaps
    d = REF_pivot(A)
    # set multiplicative identiy 
    detB = 1

    # loop across main diagonal - product of diagonal elements as detB
    for col in range(N):
        detB *= int(A[col, col])

    return  detB / d

# example matrix from 2.1
A = np.array([[1, 1, 1],
              [2, 2, -1],
              [1, -1, 2]], dtype=float)

       
print(REF_pivot_det(A))
# Error of REF_pivot_det() compared with built-in function:
# this is very effective for our example matrix
abs(np.linalg.det(A) - REF_pivot_det(A))

# random 10 x 10 matrix
#A = np.random.rand(10,10)
# Error of REF_pivot_det() compared with built-in function:
#abs(np.linalg.det(A) - REF_pivot_det(A))


# **2.3** Consider the following matrix, with corresponding row echelon form
# 
# $$
# A = \begin{pmatrix}
#           1 &  1 & 2 \\ 
#           0 &  0 & 1 \\ 
#           -1 & -1 & -2
#         \end{pmatrix},
# \qquad \text{with REF} \quad
# \begin{pmatrix}
#           1 &  1 & 2 \\ 
#           0 &  0 & 1 \\ 
#           0 & 0 & 0
#         \end{pmatrix}.
# $$
# 
# However, running the cell below shows that the function `REF_pivot()` from question **2.1** gives the wrong answer for this matrix. Find out why and modify `REF_pivot()` to ensure it produces the correct result. In a Markdown cell, briefly explain the process by which you found and solved the problem.
# 
# **[3 marks]**

# In[50]:


import numpy as np
# Testing with the new matrix
A = np.array([[1, 1, 2],
              [0, 0, 1],
              [-1, -1, -2]], dtype=float)

REF_pivot(A)
print(A)


# In[7]:


# Code for question 2.3

import numpy as np


def row_op(A, alpha, i, beta, j):
    '''
    Applies row operation beta*A_j + alpha*A_i to A_j,
    the jth row of the matrix A.
    Changes A in place.
    '''
    # Apply the row operation on the jth row of A
    A[j, :] = beta * A[j, :] + alpha * A[i, :]

    
def REF_pivot(A):
    '''
    Reduces the square matrix A into row echelon form, in-place.
    '''
    # Extract the size of the matrix
    N = A.shape[0]

        # Loop over the columns
    for col in range(N-1):
             
        #print(f'col = {col}')

       ### NEW SECTION STARTS HERE
        if abs(A[col, col]) < 1e-12:
            max_location = np.argmax(np.abs(A[col+1:, col]))         
            max_location += col+1
            A[[col, max_location], :] = A[[max_location, col], :]

        ### NEW SECTION ENDS HERE

        # In each column, loop over the rows below the diagonal
        for row in range(col+1, N):

            # Calculate alpha as -(leading element / diagonal element)
            alpha = -A[row, col] / A[col, col]

            #list comprehension and an all function to check what the maximum of all the elements below the diagonal is for each given column.
            #If we have that the maximum element in every column below the diagonal is zero then the loop is broken.
            # First time to prevent unnecessary row ops
            if any([int(max(abs(A[col+1:, col]))) for col in range(N-1)]):
                pass
            else:
                break

            # Perform the row operation in-place (beta is always 1 here)
            row_op(A, alpha, col, 1, row)
 
        # similarly as earlier, this time to break the overarching loop
        if any([int(max(abs(A[col+1:, col]))) for col in range(N-1)]):
            pass
        else:
            break

    # We don't need to return anything as A is modified in-place.
    

# Testing with an example
A = np.array([[1, 1, 2],
              [0, 0, 1],
              [-1, -1, -2]], dtype=float)

# Call the function to change A in-place
REF_pivot(A)

# The result should be:
# A = [[ 1.  1.  1.]
#      [ 0. -2.  1.]
#      [ 0.  0. -3.]]
print(A)


# ***📝 Discussion for question 2.3***

# To check where the code first started to break down, prints were made to indicate which stage in the iteration certain changes occurred. So the following were printed, column, then any row swaps if there were any zeros on the diagonal, then row, then row operations performed. From the output of our new test matrix, it became clear that for col = 0 and row = 2, row operations left the matrix in REF at an earlier stage in the iteration process. Since the code does not account for the cases where the matrix reaches its desired form early, it continues as if it were not complete and thus complicates the REF form and even in this case leads to division by zero. Thus from this example it becomes evident that we need to check at appropriate points if the matrix has completed early within the iterations. The method chosen is the use of list comprehension and an all function to check what the maximum of all the elements below the diagonal is for each given column. If we have that the maximum element in every column below the diagonal is zero then it is desirable to break the loop. This is implemented twice due to the structure of the REF code, the first time prevents any row operations from interfering with the desired form and the second time allows the overarching loop to be broken overall.

# ---
# ## Question 3: Quadrature [8 marks]
# 
# Recall that the integral $I$ of a function $f(x)$ over the interval $[-1, 1]$ can be approximated as a weighted sum, using an $N$-point quadrature rule,
# 
# $$
# I = \int_{-1}^1 f(x) \ dx \approx \sum_{k=0}^{N-1} w_k f(x_k),
# $$
# 
# where $x_k$ are the nodes and $w_k$ the weights of the quadrature rule.
# 
# Consider the quadrature rule with nodes
# 
# $$
# x_0 = -1, \quad
# x_1 = \frac{-1}{\sqrt{5}}, \quad
# x_2 = \frac{1}{\sqrt{5}}, \quad
# x_3 = 1,
# $$
# 
# and weights
# 
# $$
# w_0 = \frac{1}{6}, \quad
# w_1 = \frac{5}{6}, \quad
# w_2 = \frac{5}{6}, \quad
# w_3 = \frac{1}{6}.
# $$
# 
# The degree of precision of this quadrature rule is $5$; either prove this analytically, or perform an appropriate numerical investigation to provide evidence. You may reuse code from tutorial sheets.
# 
# *You may gain full marks with either approach, you do not need to do both.*
# 
# **[8 marks]**

# ***📝 Discussion for question 3***

# In[8]:


# Code for question 3

import numpy as np

def quadrature(f, xk, wk, a, b):
    '''
    Approximates the integral of f over [a, b],
    using the quadrature rule with weights wk
    and nodes xk.
    
    Input:
    f (function): function to integrate (as a Python function object)
    xk (Numpy array): vector containing all nodes
    wk (Numpy array): vector containing all weights
    a (float): left boundary of the interval
    b (float): right boundary of the interval
    
    Returns:
    I_approx (float): the approximate value of the integral
        of f over [a, b], using the quadrature rule.
    '''
    # Define the shifted and scaled nodes
    yk = (b - a)/2 * (xk + 1) + a
    
    # Compute the weighted sum
    I_approx = (b - a)/2 * np.sum(wk * f(yk))
    
    return I_approx


# Define interval, nodes and weights 
a, b = -1, 1
xk = np.array([-1, -1/np.sqrt(5), 1/np.sqrt(5), 1])
wk = np.array([1/6, 5/6, 5/6, 1/6])

# Calculate the integral of polynomials of degree up to 6
I_exact = [2, 0, 2/3, 0, 2/5, 0, 2/7]
for n in range(7):
    # Define a python function for the polynomial 
    def p(x):
        return x**n
    
    # Compute the integral and error 
    I_approx = quadrature(p, xk, wk, a, b)
    err = abs(I_exact[n] - I_approx)
    print(f'Error for degree {n}: {err:.6f}')


# As we can see, this quaderature rule does not integrate polynomials of degree 6 without error. Therefore the degree of precision is 5.

# ---
# ## Question 4: Numerical Integration of an ODE [14 marks]
# 
# The Brusselator is a system of ODEs describing chemical reactions. Here we consider the following variation thereof 
# 
# \begin{align}
#         \dot u & = f(u,v) =  1 - (b+1)u + u^3 v \ , \\
#         \dot v & = g(u,v) = b u - u^2 v \ ,
# \end{align}
# 
# where $b$ is a known parameter.
# 
# 
# **4.1** The fixed points of the system verify $\dot u = f(u, v) = 0$ and $\dot v = g(u, v) = 0$. Write a function which computes a fixed point $(u_b^*, v_b^*)$  for a given value of $b$ and a given initial guess $(u_{b, 0}, v_{b, 0})$, using Newton's method.
# 
# Use your function to compute a fixed point of the system for values of $b \in \{0.5, 0.75, 2.0, 2.5\}$, with initial guesses $(u_{b, 0}, v_{b, 0}) = (b, b)$. For each value of $b$, display the result in a clear and readable manner.
# 
# 
# **[6 marks]**

# In[9]:


# Code for question 4.1

import numpy as np
import matplotlib.pyplot as plt

#
def Ff(u, v, b):
    return 1 - (b + 1) * u + u**3 * v

def Gg(u, v, b):
    return b * u - u**2* v

b = [0.5, 0.75, 2.0, 2.5]

# Create a grid of values of u and v to evaluate both functions
xmin, xmax = -5, 5
ymin, ymax = -5, 5
U, V = np.meshgrid(np.linspace(xmin, xmax, 500), np.linspace(ymin, ymax, 500))


# Loop so we can find the fixed points using Newton's method for different b constants 
for i in range(4):
    # Create a contour plot to display the intersection of
    # z = Ff(u, v, b) and z = F2(u, v, b) with the plane z = 0
    
    fig, ax = plt.subplots(figsize=(15, 5))
    fig.subplots_adjust(wspace = 0.5, hspace = 0.3) 
    ax = fig.add_subplot(1,2,1)
    ax.contour(U, V, Ff(U, V, b[i]), 0, colors='r')
    ax.contour(U, V, Gg(U, V, b[i]), 0, colors='b')
    ax.set_title('Plot to Determine Initial Guesses')
    ax.grid()
    
    # Jacobain matrix - b dependant 
    def Jac(u, v, b):
        J = np.zeros([2, 2])
        J[0, 0] = - (b + 1) + 3 * u**2 * v
        J[0, 1] = u**3
        J[1, 0] = b - 2 * u * v
        J[1, 1] = - u**2
        return J
    
    
    def FG(u, v, b):
        return np.array([Ff(u, v, b), Gg(u, v, b)])

    # Initial guesses
    x0 = np.array([[1.7, 0.1], [1.5, 0.2], [1.5, 0.2], [1.6, 0.3]])

    # Tolerance
    tol = 1e-8

    # Initialise an array to store all the roots
    roots = []

    # Loop over initial guesses to find all the roots
    for x in x0:

        # Newton's method
        while np.linalg.norm(FG(x[0], x[1], b[i])) >= tol:

            # Newton iteration
            e = -np.linalg.solve(Jac(x[0], x[1], b[i]), FG(x[0], x[1], b[i]))
            x += e

        # Store the results
        roots.append(x)

    # Plot the roots on the same graph
    roots = np.array(roots)

    ax = fig.add_subplot(1,2,2)
    ax.contour(U, V, Ff(U, V, b[i]), 0, colors='y')
    ax.contour(U, V, Gg(U, V, b[i]), 0, colors='g')
    ax.set_title(f'Fixed Point After Newton Iteration, b = {b[i]} ')
    ax.grid()

    # Plot the fixed points 
    ax.plot(roots[:, 0], roots[:, 1], 'go', markersize=10)
    # Set the scale
    ax.set(xlabel='u', ylabel='v', xlim=[xmin, xmax], ylim=[ymin, ymax])
    plt.show()
    print(f"Fixed Point, for b = {b[i]}: x = {x}")




# **4.2** Using the method of your choice **\***, compute the numerical solution $(u_n, v_n) \approx (u(n\Delta t), v(n\Delta t)), n=0, 1, 2, \dots$ for the modified Brusselator.
# 
# You should compute the solution for $b \in \{0.5, 0.75, 2.0, 2.5\}$  starting at time $t = 0$ until at least $t = 100$, with $(u_0 = 0.8, v_0 = 0.8)$ as the initial condition.
# 
# Present your results graphically by plotting
# 
# (a) $u_n$ and $v_n$ with **time** (not time step) on the x-axis,  
# (b) $v_n$ as a function of $u_n$. This will show what we call the solution trajectories in *phase space*.
# 
# This is a total of 8 graphs. You should format the plots so that the data presentation is clear and easy to understand.
# 
# Compare the final values of the solution (at the end $t_\max$ of the simulation) for $b \in \{0.5, 0.75, 2.0, 2.5\}$ with the results from 3.1. Describe your observations in less than 150 words. Does the solution come back to the same fixed point?
# 
# 
# **\*** You may use e.g. the forward Euler method seen in Week 7 **with a small enough time step**, or use one of the functions provided by the `scipy.integrate` module, as seen in Quiz Q4.
# 
# 
# **[8 marks]**

# In[10]:


# import necessary libraries
import matplotlib.pyplot as plt
# show plots in notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# define the differential equations of the phase space
def f(u , v,  b):
    return 1 - (b + 1) * u + u**3 * v
def g(u , v,  b):
    return b * u - u**2 * v


# u, v = initial values, N = number of time steps,  dt = timestep, b = constant, t = time (not timestep)

for j in range(4):
    N = 1000
    dt = 0.1
    b = [0.5, 0.75, 2, 2.5]
    t = [n * dt for n in range(N + 1)]

    # initialize lists containing values
    u = [0.8]
    v = [0.8]
    
    # perform Forward Euler method
    for i in range(N):
        u.append(u[i] + (f(u[i],v[i], b[j])) * dt)
        v.append(v[i] + (g(u[i],v[i], b[j])) * dt)
    

    #plot
    fig = plt.figure(figsize=(15,5))
    fig.subplots_adjust(wspace = 0.5, hspace = 0.3)
    ax1 = fig.add_subplot(1,2,1)
    ax2 = fig.add_subplot(1,2,2)

    ax1.plot(t, u, 'r-', label='u')
    ax1.plot(t, v, 'b-', label='v')

    ax1.set_title(f"Time Series b = {b[j]}")
    ax1.set_xlabel("time")
    ax1.grid()
    ax1.legend(loc='best')

    ax2.plot(u, v, color="blue")
    ax2.plot(u[-1], v[-1], 'o', label = 'end point')
    
    ax2.set_xlabel("u")
    ax2.set_ylabel("v")  
    ax2.set_title(f"Phase Space b = {b[j]}")
    # print end points 
    print(f' End Point for b = {b[j]} is {[u[-1], v[-1]]}')
    ax2.grid()


# ***📝 Discussion for question 4.2***

# In[22]:


import plotly.graph_objects as go

fig = go.Figure(data=[go.Table(header=dict(values=['b value', 'fixed points', 'tmax']),
                 cells=dict(values=[['0.5', '0.75', '2', '2.5'], ['[1, 0.5]','[1, 0.75]', '[1, 2]','[1, 2.5]'] , ['[1, 0.5]','[1, 0.75]','[0.5, 4]','[0.4, 6.25]']]))])
fig.show()


# We can see from the table below that the fixed points for b values 0.5, 0.75 are essentially identical.
# For the b values 2, 2.5 the fixed points come back as significantly different to the end values of the phase spaces.
# For b = 2, 2.5 , 2(tmax) = fixed point for b = 2, 2.5(tmax) = fixed point for b = 2.5
# This result indicates that the tmax/end values after b is greater than a certain value, is related to the fixed points by 
# the scalar b.

# ---
# ## Question 5: Fractals [14 marks]
# 
# Halley's method is a root finding method, related to Newton's method, which uses the first and second derivatives of the function. To find a root of a function $f(z)$, starting with an initial guess $z_0$, Halley's method updates the guess until convergence is achieved. The Halley update is given by
# 
# $$
# z_{k+1} = z_k - \frac{2 f(z_k) f'(z_k)}{2\left(f'(z_k)\right)^2 - f(z_k) f''(z_k)}.
# $$
# 
# **5.1** Perform a numerical investigation to determine the rate of convergence of Halley's method.
# 
# **[4 marks]**

# In[11]:


# Code for 5.1

# Initialise a, the initial guess x0, and the number of iterations
import numpy as np
import matplotlib.pyplot as plt

# x = initial guess, its = iterations
x = 3.2
its = 0

# functions defined for Halley's method
def f(x):
    return np.sin(x) + 1
def ff(x):
    return np.cos(x)
def fff(x):
    return - np.sin(x)

# Start a list to store the error
err = [abs(x - (3 * np.pi/2))]

# Loop until convergence
while True:
    its += 1
    x_new = x - (2 * f(x) * ff(x)) / (2 * (ff(x))**2 - (f(x)) * (fff(x)))
    
    err.append(abs(x_new - (3 * np.pi/2)))
    
    # Convergence achieved
    if abs(x_new - x) < 1.0e-20:
        break
    
    # Update for next iteration
    x = x_new
    
print(x_new)

# Convert to Numpy array and remove last value (it's computed as zero)
err = np.array(err[:-1])


# One way to find the order of convergence
fig, ax = plt.subplots()
ax.plot(np.log(err[:-1]), np.log(err[1:]), 'bx')
plt.show()

slope, _ = np.polyfit(np.log(err[:-1]), np.log(err[1:]), 1)
print(f'The order of convergence is p = {slope:.5f}')


# In[12]:


# Code for 5.1

# Initialise a, the initial guess x0, and the number of iterations
import matplotlib.pyplot as plt

# x = initial value, iterations
x = 3.36
its = 0

# Functions defined for Halley's method
def f(x):
    return 1 + np.sin(x)
def ff(x): 
    return np.cos(x)
def fff(x):
    return -np.sin(x)

# Start a list to store the error
err = [abs(x - (3 * np.pi)/2)]

# Loop until convergence
while True:
    its += 1
    x_new = x - (2 * f(x) * ff(x)) / (2 * (ff(x))**2 - (f(x)) * (fff(x)))
    
    err.append(abs(x_new - (3 * np.pi)/2))
    
    # Convergence achieved
    if abs(x_new - x) < 1.0e-12:
        break
    
    # Update for next iteration
    x = x_new
print(x_new)
# Convert to Numpy array and remove last value (it's computed as zero)
err = np.array(err[:-1])


# One way to find the order of convergence
fig, ax = plt.subplots()
ax.plot(np.log(err[:-1]), np.log(err[1:]), 'bx')
plt.show()

slope, _ = np.polyfit(np.log(err[:-1]), np.log(err[1:]), 1)
print(f'The order of convergence is p = {slope:.5f}')


# In[32]:


# Code for 5.1

# Initialise a, the initial guess x0, and the number of iterations

# x = starting point, iterations
x = 7
its = 0

# Functions defined for Halley's method
def f(x):
    return np.arctan(x)
def ff(x):
    return 1/(x**2 + 1)
def fff(x):
    return (-2 * x)/(x**2 + 1)**2

# Start a list to store the error
err = [abs(x - 0)]

# Loop until convergence
while True:
    its += 1
    x_new = x - (2 * f(x) * ff(x)) / (2 * (ff(x))**2 - (f(x)) * (fff(x)))
    
    err.append(abs(x_new - 0))
    
    # Convergence achieved
    if abs(x_new - x) < 1.0e-15:
        break
    
    # Update for next iteration
    x = x_new

# Convert to Numpy array and remove last value (it's computed as zero)
err = np.array(err[:-2])


# One way to find the order of convergence
fig, ax = plt.subplots()
ax.plot(np.log(err[:-2]), np.log(err[2:]), 'bx')
plt.show()

slope, _ = np.polyfit(np.log(err[:-2]), np.log(err[2:]), 1)
print(f'The order of convergence is p = {slope:.5f}')


# **5.2** Consider the following polynomial
# 
# $$
# p(z) = z^4 - (c+1)z^3 + c,
# $$
# where $c \in \mathbb{C}$.
# 
# This polynomial is complex differentiable, and we can apply a root finding method to find a complex root $z_\ast$, using a complex initial guess $z_0 = a_0 + ib_0$. In this problem, we seek to map the values of $z_0$ which lead to convergence to a root of $p$.
# 
# Write a function `complex_halley(amin, amax, bmin, bmax, c, N, eps)` which implements Halley's method to find roots of $p(z)$ using $N^2$ initial guesses $z_0 = a_0 + ib_0$. The input arguments are as follows:
# 
# - The real part $a_0$ of the initial guess should take `N` linearly spaced values between `amin` and `amax` (inclusive).
# - The imaginary part $b_0$ of the initial guess should take `N` linearly spaced values between `bmin` and `bmax` (inclusive).
# - `c` is the parameter $c \in \mathbb{C}$ in $p(z)$.
# - `eps` is the tolerance $\varepsilon > 0$.
# 
# Your function should return an array `kmax` of size $N \times N$, containing the **total number of iterations required for convergence, for each value of $z_0$**. Note that you should not return the roots, only the number of iterations required until convergence. You should decide what to do in case a particular value of $z_0$ doesn't lead to convergence.
# 
# **[6 marks]**

# In[52]:


# Code for question 5.2

import numpy as np
import matplotlib.pyplot as plt

def complex_halley(amin, amax, bmin, bmax, c, N, eps):
    #define the polynomial and it's derivatives
    def p(z):
        return z**4 - (c + 1) * z**3 + c
    def pp(z):
        return 4 * z**3 - 3 * (c + 1) * z**2 
    def ppp(z):
        return 12 * z**2 - 6 * (c + 1)*z
    
    roots = np.roots([1, -(c + 1), 0, 0, c])
    
        #Halley's method 
        #creating a list for holding the number of iterations until convergence
    
    kmax = []
    
        # Create a complex plane
    Z = np.linspace(amin, amax, N)  + np.linspace(bmin, bmax, N) * 1j
    X, Y = np.meshgrid(Z.real, Z.imag)
    Z = X + Y * 1j
    Z = np.array(Z).reshape(N, N)

    # iterating through the complex plane
    for z in np.nditer(Z):
        n = 0
        # Using Halley's method to test how close initial guesses are to the target
        while np.all(abs(roots - z) > eps) and n < 100:
            z = z - (2 * p(z) * pp(z))/(2 * (pp(z))**2 - p(z) * ppp(z))
            n += 1
        kmax.append(n)
        # If conditions for convergence are not met then the number 1000 is used to indicate the place where z0 does not converge
        if np.all(abs(roots - z) <= eps) and n>=100:
            kmax.append(1000)
            
        
        
    return np.array(kmax)
        

complex_halley(-30, 30, -30, 30, 1 - 1 * i, 10, 1e-8)


# **5.3** For $c = -2.4 - 1.2i$, $a_0 \in [-5,5]$ and $b_0 \in [-5,5]$, with at least $N = 200$ values for each (you can increase $N$ if your computer allows it), use your function `complex_halley()` to calculate, for each $z_0 = a_0 + ib_0$, the total number of iterates needed to converge to any root (within some small tolerance). Present your results in a heatmap plot, with $a_0$ on the abscissa, $b_0$ on the ordinate and a colour map showing the total number of iterates. 
# 
# **[4 marks]**

# In[51]:


# Code for question 5.3

def plot_halley(amin, amax, bmin, bmax, c, N, eps):
    '''
    Defining a function for plotting iterates and convergence 
    '''
    #reshaping the array kmax
    kmax = complex_halley(amin, amax, bmin, bmax, c, N, eps)
    kmax = np.reshape(kmax, newshape = (N, N))
    # complex plane
    Z = np.linspace(amin, amax, N)  + np.linspace(bmin, bmax, N) * 1j
    X, Y = np.meshgrid(Z.real, Z.imag)

    #plot
    plt.figure(figsize = (15,15))
    plt.title("Halley's Method")
    plt.xlabel("re(z)")
    plt.ylabel("im(z)")


    plt.contourf(X, Y, kmax, cmap= 'rainbow')
    plt.colorbar()
    plt.show()

plot_halley(-5, 5, -5, 5, -2.4 - 1.2 * i, 500, 1e-8)

