'''

In the Letters and Numbers (L&N) game,
One contestant chooses how many "small" and "large" numbers they would like 
to make up six randomly chosen numbers. Small numbers are between 
1 and 10 inclusive, and large numbers are 25, 50, 75, or 100. 
All large numbers will be different, 
so at most four large numbers may be chosen. 


How to represent a computation?

Let Q = [q0, q1, q2, q3, q4, q5] be the list of drawn numbers

The building blocks of the expression trees are
 the arithmetic operators  +,-,*
 the numbers  q0, q1, q2, q3, q4, q5

We can encode arithmetic expressions with Polish notation
    op arg1 arg2
where op is one of the operators  +,-,*

or with expression trees:
    (op, left_tree, right_tree)
    
Recursive definition of an Expression Tree:
 an expression tree is either a 
 - a scalar   or
 - a binary tree (op, left_tree, right_tree)
   where op is in  {+,-,*}  and  
   the two subtrees left_tree, right_tree are expressions trees.

When an expression tree is reduced to a scalar, we call it trivial.


Author: f.maire@qut.edu.au

Created on April 1 , 2021
    

This module contains functions to manipulate expression trees occuring in the
L&N game.

'''
from multiprocessing import Pool
import multiprocessing
from itertools import product
import numpy as np
import random

import copy # for deepcopy

import time

import collections


SMALL_NUMBERS = tuple(range(1,11))
LARGE_NUMBERS = (25, 50, 75, 100)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    
    '''
    return [(10467858, 'Ethan', 'Griffiths'), (10467874, 'Mattias', 'Winsen'), (10486925, 'Connor', 'Browne')]


# ----------------------------------------------------------------------------

def pick_numbers():
    '''    
    Create a random list of numbers according to the L&N game rules.
    
    Returns
    -------
    Q : int list
        list of numbers drawn randomly for one round of the game
    '''
    LN = set(LARGE_NUMBERS)
    Q = []
    for i in range(6):
        x = random.choice(list(SMALL_NUMBERS)+list(LN))
        Q.append(x)
        if x in LN:
            LN.remove(x)
    return Q


# ----------------------------------------------------------------------------

def bottom_up_creator(Q):
    '''
    Create a random algebraic expression tree
    that respects the L&N rules.
    
    Warning: Q is shuffled during the process

    Parameters
    ----------
    Q : non empty list of available numbers
        

    Returns  T, U
    -------
    T : expression tree 
    U : values used in the tree

    '''
    n = random.randint(1,6) # number of values we are going to use
    
    random.shuffle(Q)
    # Q[:n]  # list of the numbers we should use
    U = Q[:n].copy()
    
    if n==1:
        # return [U[0], None, None], [U[0]] # T, U
        return U[0], [U[0]] # T, U
        
    F = [u for u in U]  # F is initially a forest of values
    # we start with at least two trees in the forest
    while len(F)>1:
        # pick two trees and connect then with an arithmetic operator
        random.shuffle(F)
        op = random.choice(['-','+','*'])
        T = [op,F[-2],F[-1]]  # combine the last two trees
        F[-2:] = [] # remove the last two trees from the forest
        # insert the new tree in the forest
        F.append(T)
    # assert len(F)==1
    return F[0], U
  
# ---------------------------------------------------------------------------- 

def display_tree(T, indent=0):
    '''
    
    Eval the algebraic expression represented by T
    
    Parameters
    ----------
    T : Expression Tree
    indent: indentation for the recursive call

    Returns None

    '''
    # if T is a scalar, then we return it directly
    if isinstance(T, int):
        print('|'*indent,T, sep='')
        return
    # T is non trivial
    root_item = T[0]
    print('|'*indent, root_item, sep='')
    display_tree(T[1], indent+1)
    print('|'*indent)
    display_tree(T[2], indent+1)
   
# ---------------------------------------------------------------------------- 

def eval_tree(T):
    '''
    
    Eval the algebraic expression represented by T
    
    Parameters
    ----------
    T : Expression Tree

    Returns
    -------
    value of the algebraic expression represented by the T

    '''
    # if T is a scalar, then we return it directly
    if isinstance(T, int):
        return T
    # T is non trivial
    root_item = T[0]
    # assert root_item in ('-','+','*')
    left_value = eval_tree(T[1])
    right_value = eval_tree(T[2])
    return eval( str(left_value) +root_item + str(right_value) )
    # return eval(root_item.join([str(left_value), str(right_value)]))
   
     
# ---------------------------------------------------------------------------- 

def expr_tree_2_polish_str(T):
    '''
    Convert the Expression Tree into Polish notation

    Parameters
    ----------
    T : expression tree

    Returns
    -------
    string in Polish notation represention the expression tree T

    '''
    if isinstance(T, int):
        return str(T)
    root_item = T[0]
    # assert root_item in ('-','+','*')
    left_str = expr_tree_2_polish_str(T[1])
    right_str = expr_tree_2_polish_str(T[2])
    return '[' + ','.join([root_item,left_str,right_str]) + ']'
    

# ----------------------------------------------------------------------------

def polish_str_2_expr_tree(pn_str):
    '''
    
    Convert a polish notation string of an expression tree
    into an expression tree T.

    Parameters
    ----------
    pn_str : string representing an L&N algebraic expression

    Returns
    -------
    T

    '''
    if pn_str.isnumeric():
        return int(pn_str)

    def find_match(i):
        '''
        Starting at position i where pn_str[i] == '['
        Return the index j of the matching ']'
        That is, pn_str[j] == ']' and the substring pn_str[i:j+1]
        is balanced
        '''
        ######Connor's Code######
        j = i+1
        open = 0
        if (pn_str[i] == '['):
            for j in range(i+1, len(pn_str)):
                if (pn_str[j] == ']' and open == 0):
                    return j
                elif (pn_str[j] == '['):
                    open = open+1
                elif (pn_str[j] == ']'):
                    open = open-1

     # .................................................................  

    # pn_str = pn_str.replace(',','')

    tree = []
    left_p = pn_str.find('[')
    operator = pn_str[left_p+1]
    tree.append(operator)

    #if there is still another lower branch
    if (left_p != -1):

        right_p = find_match(left_p)
        pn_str = pn_str[left_p + 1:right_p]

        #if the left side is a single num
        num = pn_str.find(',')+1
        if (pn_str[num].isnumeric()):
            tree.append(int(pn_str[num]))
            pn_str = pn_str[num:]
        else:
            new_left_p = pn_str.find('[')
            new_right_p = find_match(new_left_p)
            tree.append(polish_str_2_expr_tree(pn_str[new_left_p:new_right_p+1]))
            pn_str = pn_str[new_right_p:]

        # if the right side is a single num
        num = pn_str.find(',') + 1
        if (pn_str[num].isnumeric()):
            tree.append(int(pn_str[num]))
        else:
            new_left_p = pn_str.find('[')
            new_right_p = find_match(new_left_p)
            tree.append(polish_str_2_expr_tree(pn_str[new_left_p:new_right_p+1]))

    else:
        pn_str_arr = pn_str.split(',')
        tree.append(int(pn_str_arr[-2]))
        tree.append(int(pn_str_arr[-1]))

    return tree


 
   
# ----------------------------------------------------------------------------

def op_address_list(T, prefix = None):
    '''
    Return the address list L of the internal nodes of the expresssion tree T
    
    If T is a scalar, then L = []

    Note that the function 'decompose' is more general.

    Parameters
    ----------
    T : expression tree
    prefix: prefix to prepend to the addresses returned in L

    Returns
    -------
    L
    '''
    if isinstance(T, int):
        return []
    
    if prefix is None:
        prefix = []

    L = [prefix.copy()+[0]] # first adddress is the op of the root of T
    left_al = op_address_list(T[1], prefix.copy()+[1])
    L.extend(left_al)
    right_al = op_address_list(T[2], prefix.copy()+[2])
    L.extend(right_al)
    
    return L


# ----------------------------------------------------------------------------

def decompose(T, prefix = None):
    '''
    Compute
        Aop : address list of the operators
        Lop : list of the operators
        Anum : address of the numbers
        Lnum : list of the numbers
    
    For example, if 
    
    T =  ['-', ['+', ['-', 75, ['-', 10, 3]], ['-', 100, 50]], 3]
    
    then, 
    
     Aop is  [[0], [1, 0], [1, 1, 0], [1, 1, 2, 0], [1, 2, 0]] 
    
     Lop is ['-', '+', '-', '-', '-'] 
    
     Anum is [[1, 1, 1], [1, 1, 2, 1], [1, 1, 2, 2], [1, 2, 1], [1, 2, 2], [2]] 
    
     Lnum is [75, 10, 3, 100, 50, 3]    
        
    
    Parameters
    ----------
    T : expression tree 
    
    prefix : address to preprend 

    Returns
    -------
    Aop, Lop, Anum, Lnum

    '''
    if prefix is None:
        prefix = []

    if isinstance(T, int):
        Aop = []
        Lop = [] 
        Anum = [prefix]
        Lnum = [T]
        return Aop, Lop, Anum, Lnum
    #########Connor's code##################
    else:
        Aop = [prefix.copy() + [0]]  # first adddress is the op of the root of T
        Lop = [T[0]]
        Anum = []
        Lnum = []

        #left side
        left_aop, left_lop, left_anum, left_lnum = decompose(T[1], prefix.copy() + [1])
        Aop.extend(left_aop)
        Lop.extend(left_lop)
        Anum.extend(left_anum)
        Lnum.extend(left_lnum)

        #right side
        right_aop, right_lop, right_anum, right_lnum = decompose(T[2], prefix.copy() + [2])
        Aop.extend(right_aop)
        Lop.extend(right_lop)
        Anum.extend(right_anum)
        Lnum.extend(right_lnum)

        return Aop, Lop, Anum, Lnum

    #
    # assert isinstance(T, list)
    # raise NotImplementedError()


# ----------------------------------------------------------------------------

def get_item(T, a):
    '''
    Get the item at address a in the expression tree T

    Parameters
    ----------
    T : expression tree
    a : valid address of an item in the tree

    Returns
    -------
    the item at address a

    '''
    if len(a)==0:
        return T
    # else
    return get_item(T[a[0]], a[1:])
        
# ----------------------------------------------------------------------------

def replace_subtree(T, a, S):
    '''
    Replace the subtree at address a
    with the subtree S in the expression tree T
    
    The address a is a sequence of integers in {0,1,2}.
    
    If a == [] , then we return S
    If a == [1], we replace the left subtree of T with S
    If a == [2], we replace the right subtree of T with S

    Returns
    ------- 
    The modified tree

    Warning: the original tree T is modified. 
             Use copy.deepcopy()  if you want to preserve the original tree.
    '''    
    
    # base case, address empty
    if len(a)==0:
        return S
    
    # recursive case
    T[a[0]] = replace_subtree(T[a[0]], a[1:], S)
    return T


# ----------------------------------------------------------------------------

def mutate_num(T, Q):
    '''
    Mutate one of the numbers of the expression tree T
    
    Parameters
    ----------
    T : expression tree
    Q : list of numbers initially available in the game

    Returns
    -------
    A mutated copy of T

    '''
    
    Aop, Lop, Anum, Lnum = decompose(T)

    #get a list of Q numbers not yet in T
    avalibleNum = []
    counter_Q = collections.Counter(Q) # some small numbers can be repeated
    in_counter = collections.Counter(Lnum)

    for number in Lnum:
        diff = counter_Q[number] - in_counter[number]
        if diff > 0:
            for num in range(1,diff+1):
                avalibleNum.append(num)

    T_mutate = T
    #pick random new num from avalible and insert it
    if (len(avalibleNum) > 0):
        address = random.choice(Anum)  # random address of an number in T
        new_num = random.choice(avalibleNum)
        T_mutate = replace_subtree(T, address, new_num)
    return T_mutate

# ----------------------------------------------------------------------------

def mutate_op(T):
    '''
    Mutate an operator of the expression tree T
    If T is a scalar, return T

    Parameters
    ----------
    T : non trivial expression tree

    Returns
    -------
    A mutated copy of T

    '''
    if isinstance(T, int):
        return T

    op_list = ["*","+","-"]
    mutant_T = copy.deepcopy(T)
    La = op_address_list(mutant_T)
    a = random.choice(La)  # random address of an op in T
    op_c = get_item(T, a)       # the char of the op
    # mutant_c : a different op #########Connor's Code###############
    new_op = op_c[0]
    while (new_op == op_c[0]):
        new_op = random.choice(op_list)
    T_mutate = replace_subtree(mutant_T, a, new_op)
    return T_mutate
    

# ----------------------------------------------------------------------------

def cross_over(P1, P2, Q):    
    '''
    Perform crossover on two non trivial parents
    
    Parameters
    ----------
    P1 : parent 1, non trivial expression tree  (root is an op)
    P2 : parent 2, non trivial expression tree  (root is an op)
        DESCRIPTION
        
    Q : list of the available numbers
        Q may contain repeated small numbers    
        

    Returns
    -------
    C1, C2 : two children obtained by crossover
    '''
    
    def get_num_ind(aop, Anum):
        '''
        Return the indices [a,b) of the range of numbers
        in Anum and Lum that are in the sub-tree 
        rooted at address aop

        Parameters
        ----------
        aop : address of an operator (considered as the root of a subtree).
              The address aop is an element of Aop
        Anum : the list of addresses of the numbers

        Returns
        -------
        a, b : endpoints of the semi-open interval
        
        '''
        d = len(aop)-1  # depth of the operator. 
                        # Root of the expression tree is a depth 0
        # K: list of the indices of the numbers in the subtrees
        # These numbers must have the same address prefix as aop
        p = aop[:d] # prefix common to the elements of the subtrees
        K = [k for k in range(len(Anum)) if Anum[k][:d]==p ]
        return K[0], K[-1]+1
        # .........................................................
        
    Aop_1, Lop_1, Anum_1, Lnum_1 = decompose(P1)
    Aop_2, Lop_2, Anum_2, Lnum_2 = decompose(P2)

    C1 = copy.deepcopy(P1)
    C2 = copy.deepcopy(P2)
    
    i1 = np.random.randint(0,len(Lop_1)) # pick a subtree in C1 by selecting the index
                                         # of an op
    i2 = np.random.randint(0,len(Lop_2)) # Select a subtree in C2 in a similar way
 
    # i1, i2 = 4, 0 # DEBUG    
 
    # Try to swap in C1 and C2 the sub-trees S1 and S2 
    # at addresses Lop_1[i1] and Lop_2[i2].
    # That's our crossover operation!
    
    # Compute some auxiliary number lists
    
    # Endpoints of the intervals of the subtrees
    a1, b1 = get_num_ind(Aop_1[i1], Anum_1)     # indices of the numbers in S1 
                                                # wrt C1 number list Lnum_1
    a2, b2 = get_num_ind(Aop_2[i2], Anum_2)   # same for S2 wrt C2
    
    # Lnum_1[a1:b1] is the list of numbers in S1
    # Lnum_2[a2:b2] is the list of numbers in S2
    
    # numbers is C1 not used in S1
    nums_C1mS1 = Lnum_1[:a1]+Lnum_1[b1:]
    # numbers is C2-S2
    nums_C2mS2 = Lnum_2[:a2]+Lnum_2[b2:]
    #nums_S2 = Lnum_2[a2:b2]
    
    # S2 is a fine replacement of S1 in C1
    # if nums_S2 + nums_C1mS1 is contained in Q
    # if not we can bottom up a subtree with  Q-nums_C1mS1

    counter_Q = collections.Counter(Q) # some small numbers can be repeated
    
    d1 = len(Aop_1[i1])-1
    aS1 = Aop_1[i1][:d1] # address of the subtree S1 
    S1 = get_item(C1, aS1)

    # ABOUT 3 LINES DELETED #######connor's code
    d2 = len(Aop_2[i2]) - 1
    aS2 = Aop_2[i2][:d2]  # address of the subtree S1
    S2 = get_item(C2, aS2)
    ##############

    # print(' DEBUG -------- S1 and S2 ----------') # DEBUG
    # print(S1)
    # print(S2)


    # count the numbers (their occurences) in the candidate child C1
    counter_1 = collections.Counter(Lnum_2[a2:b2]+nums_C1mS1)
    
    # Test whether child C1 is ok
    if all(counter_Q[v]>=counter_1[v]  for v in counter_Q):
        # candidate is fine!  :-)
        C1 = replace_subtree(C1, aS1, S2)
    else:
        available_nums = counter_Q.copy()
        available_nums.subtract(
            collections.Counter(nums_C1mS1)
            )
        R1, _ = bottom_up_creator(list(available_nums.elements()))
        C1 = replace_subtree(C1, aS1, R1)
        
    # count the numbers (their occurences) in the candidate child C2
    counter_2 = collections.Counter(Lnum_1[a1:b1]+nums_C2mS2)
    
    # Test whether child C2 is ok
    # ########Connor's Code############
    if all(counter_Q[v] >= counter_2[v] for v in counter_Q):
        # candidate is fine!  :-)
        C2 = replace_subtree(C2, aS2, S1)
    else:
        available_nums = counter_Q.copy()
        available_nums.subtract(
            collections.Counter(nums_C2mS2)
        )
        R2, _ = bottom_up_creator(list(available_nums.elements()))
        C2 = replace_subtree(C2, aS2, R2)
    ###############
    
    
    return C1, C2

# #------------------------------------------------------
# #Experiment functions (ver 1)
# def Test_System():
#     a=1

# def Find_max_iteration():
#     pop_sizes = range(25,501,25) #25 -> 500 with 20 steps of size 25

#     ##compute max iteration possible for each pop size using multiproccessing pool, takes ~1min
#     try:
#         pool = multiprocessing.Pool()  # on 8 processors
#         max_iterations_possible = pool.map(run_func_1, pop_sizes)
#         pool.close()
#         pool.join()
#     finally:  # To make sure processes are closed in the end, even if errors happen
#         pool.close()
#         pool.join()

#     # max_iterations_possible = [260, 150, 55, 35, 55, 25, 25, 35, 13, 12, 9, 8, 5, 4, 4, 4, 3, 3, 3, 3]

#     return pop_sizes, max_iterations_possible


# def run_func_1(size):
#     Q = [100, 25, 7, 5, 3, 1]
#     target = 7280  # unachivable with Q given, will run to max iterations
#     best_iteration_size = 0
#     iterator = 0
#     while (True):
#         if (iterator < 20):
#             iterator += 1
#         else:
#             iterator += 5
#         time1 = time.perf_counter()
#         from genetic_algorithm import evolve_pop
#         v, T = evolve_pop(Q, target,
#                           max_num_iteration=iterator,
#                           population_size=size)
#         time2 = time.perf_counter()
#         timeToRun = time2 - time1
#         if (timeToRun <= 2):
#             best_iteration_size = iterator
#         elif(iterator > 200):
#             break
#         else:
#             break
#     return best_iteration_size

# def test_performance(iterator, pop_sizes):
#     # Q = []
#     # target = []
#     # for i in range(1,31):
#     #     Q.append(pick_numbers())
#     #     target.append(np.random.randint(1, 1000))


#     # performance = [15.933333333333334, 3.1666666666666665, 2.566666666666667, 2.2, 1.9, 2.066666666666667,
#     #                1.1333333333333333, 1.9333333333333333, 1.1666666666666667, 1.4666666666666666, 0.9333333333333333,
#     #                1.2, 1.6, 2.6333333333333333, 2.3666666666666667, 1.9, 1.3333333333333333, 2.2, 2.6666666666666665, 2.3]
#     # uncomment to do manually again
#     try:
#         pool = multiprocessing.Pool()  # on 8 processors
#         performance = pool.map(run_func_2, pop_sizes)
#         pool.close()
#         pool.join()
#     finally:  # To make sure processes are closed in the end, even if errors happen
#         pool.close()
#         pool.join()

#     return performance

# def run_func_2(size):
#     max_iterations_possible = [260, 150, 55, 35, 55, 25, 25, 35, 13, 12, 9, 8, 5, 4, 4, 4, 3, 3, 3, 3]
#     iterator = max_iterations_possible[int((size/25)-1)]

#     Q = [[9, 1, 8, 25, 75, 4], [100, 4, 75, 50, 5, 1], [1, 50, 9, 2, 10, 1], [8, 7, 7, 100, 6, 3], [75, 5, 6, 9, 7, 4],
#          [9, 9, 6, 100, 8, 75], [2, 75, 1, 3, 10, 8], [5, 1, 8, 7, 25, 6], [75, 7, 50, 8, 1, 4], [3, 50, 2, 75, 4, 7],
#          [50, 10, 8, 6, 2, 75], [2, 7, 3, 5, 6, 9], [3, 2, 5, 7, 6, 3], [3, 2, 75, 8, 7, 1], [25, 7, 5, 75, 6, 9],
#          [6, 3, 50, 7, 3, 25], [100, 1, 50, 25, 5, 2], [2, 4, 8, 3, 4, 2], [75, 25, 4, 9, 5, 6],
#          [100, 2, 25, 75, 50, 6], [50, 75, 25, 3, 5, 9], [50, 25, 3, 9, 3, 1], [50, 10, 10, 8, 8, 5],
#          [10, 5, 10, 8, 7, 25], [100, 25, 9, 8, 1, 8], [1, 25, 50, 9, 9, 3], [10, 1, 100, 8, 3, 7],
#          [6, 25, 5, 9, 4, 100], [50, 6, 3, 75, 2, 100], [9, 100, 8, 10, 10, 1]]

#     target = [969, 126, 58, 398, 626, 688, 517, 95, 934, 125, 556, 669, 211, 424, 446, 720, 63, 806, 206, 75, 553, 485,
#               904, 926, 155, 299, 736, 751, 1, 334]
#     perfomance = 0
#     for i in range(0,30):
#         from genetic_algorithm import evolve_pop
#         v, T = evolve_pop(Q[i], target[i],
#                           max_num_iteration=iterator,
#                           population_size=size)
#         perfomance += v
#     return perfomance/30



#------------------------------------------------------------------------------------------------------
#Experiment functions (ver 2)

def test_system():
    # Test the system to find the most efficient pair of pop size and max iterations

    # # Find the max pop size (11000 is max pop size for Ethan's pc)
    # max_pop_size = find_max_pop()    
    # print('Max Pop Size: ', max_pop_size)
    
    # Defined list of population sizes in range 10, max_pop_size
    pop_sizes = [10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 5000, 7500, 10000, 11000] # 11000 is max pop size for Ethan's pc
    # pop_sizes = [10, 15, 20, 30, 40, 50, 75, 100, 125, 150, 200, 300, 400, 500, 750, 1000, 2000, 3000, 4000, 5000]  # max size 5000 because thing broke

    max_iterations_possible = find_max_iterations(pop_sizes)

    # max_iterations_possible = [110, 100, 90, 80, 70, 67, 62, 65, 68, 80, 70, 60, 50, 30, 10, 7, 5, 3, 1, 1] # debugging
    performance = test_performance(pop_sizes, max_iterations_possible)
    
    print('Pop Sizes: ', pop_sizes)
    print('Max Iterations Possible: ', max_iterations_possible)
    print('Performance: ', performance)

def find_max_pop():
    # Find the largest population value that can be computed for one iteration in under 2 seconds
    from genetic_algorithm import evolve_pop
    Q = [100, 25, 7, 5, 3, 1]
    target = 7280  # unachievable with Q given, will run to max iterations
    max_pop_size = 0
    size = 1000 # start with 1000
    while (True):
        # from genetic_algorithm import evolve_pop
        time1 = time.perf_counter()
        v, T = evolve_pop(Q, target,
                          max_num_iteration=1, # run for 1 iteration only
                          population_size=size)
        time2 = time.perf_counter()
        timeToRun = time2 - time1
        if (timeToRun <= 2): # Check if it takes longer than 2 seconds
            max_pop_size = size
        else:
            break
        size += 1000 # increase pop size
    return max_pop_size # For my (Ethan's) pc with debugging, max size is 11000

def find_max_iterations(pop_sizes): # 11000 is max pop size for Ethan's pc
    # Find the max num of iterations for the range of 20 pop sizes

    ##compute max iteration possible for each pop size using multiproccessing pool, takes ~1min
    try:
        pool = multiprocessing.Pool() # multiprocessing
        max_iterations_possible = pool.map(max_iteration, pop_sizes)
        pool.close()
        pool.join()
    finally:  # To make sure processes are closed in the end, even if errors happen
        pool.close()
        pool.join()

    # # Compute without multiprocessing
    # max_iterations_possible = list(map(max_iteration, pop_sizes))

    # max_iterations_possible = [260, 150, 55, 35, 55, 25, 25, 35, 13, 12, 9, 8, 5, 4, 4, 4, 3, 3, 3, 3]

    return max_iterations_possible

def max_iteration(size):
    # Find the max iterations for the given population size
    from genetic_algorithm import evolve_pop
    Q = [100, 25, 7, 5, 3, 1]
    target = 7280  # unachievable with Q given, will run to max iterations
    best_iteration_size = 0
    iterations = 1
    prev_iterations = 1
    while (True):
        # from genetic_algorithm import evolve_pop
        time1 = time.perf_counter()
        v, T = evolve_pop(Q, target,
                          max_num_iteration=iterations,
                          population_size=size)
        time2 = time.perf_counter()
        timeToRun = time2 - time1
        # if (timeToRun <= 2 and iterations <= 200):
        if (timeToRun <= 2):
            best_iteration_size = prev_iterations
            prev_iterations = iterations
        # elif(iterations > 200):
        #     break
        else:
            break        

        # Increase max iterations and run again 
        if (iterations < 30):
            iterations += 1
        elif(iterations < 100):
            iterations += 5
        elif(iterations < 500):
            iterations += 10
        elif(iterations < 1000):
            iterations += 20
        else:
            iterations += 50

    return best_iteration_size

def test_performance(pop_sizes, max_iterations_possible):
    # test the performance of each population size 
    try:
        # Combine pop sizes and max iterations into one array for starmap
        sizes_iterations_list = list(zip(pop_sizes, max_iterations_possible))
        pool = multiprocessing.Pool() # multiprocessing
        performance = pool.starmap(run_func_2, sizes_iterations_list)
        pool.close()
        pool.join()
    finally:  # To make sure processes are closed in the end, even if errors happen
        pool.close()
        pool.join()

    return performance

def run_func_2(pop_size, max_iteration_possible):
    # test the average performance of the population size and max iterations pair for a predetermined set of tests
    from genetic_algorithm import evolve_pop

    Q = [[9, 1, 8, 25, 75, 4], [100, 4, 75, 50, 5, 1], [1, 50, 9, 2, 10, 1], [8, 7, 7, 100, 6, 3], [75, 5, 6, 9, 7, 4],
         [9, 9, 6, 100, 8, 75], [2, 75, 1, 3, 10, 8], [5, 1, 8, 7, 25, 6], [75, 7, 50, 8, 1, 4], [3, 50, 2, 75, 4, 7],
         [50, 10, 8, 6, 2, 75], [2, 7, 3, 5, 6, 9], [3, 2, 5, 7, 6, 3], [3, 2, 75, 8, 7, 1], [25, 7, 5, 75, 6, 9],
         [6, 3, 50, 7, 3, 25], [100, 1, 50, 25, 5, 2], [2, 4, 8, 3, 4, 2], [75, 25, 4, 9, 5, 6],
         [100, 2, 25, 75, 50, 6], [50, 75, 25, 3, 5, 9], [50, 25, 3, 9, 3, 1], [50, 10, 10, 8, 8, 5],
         [10, 5, 10, 8, 7, 25], [100, 25, 9, 8, 1, 8], [1, 25, 50, 9, 9, 3], [10, 1, 100, 8, 3, 7],
         [6, 25, 5, 9, 4, 100], [50, 6, 3, 75, 2, 100], [9, 100, 8, 10, 10, 1]]

    target = [969, 126, 58, 398, 626, 688, 517, 95, 934, 125, 556, 669, 211, 424, 446, 720, 63, 806, 206, 75, 553, 485,
              904, 926, 155, 299, 736, 751, 1, 334]
    performance = 0

    # Test pair 30 times
    for i in range(0,30):
        v, T = evolve_pop(Q[i], target[i],
                          max_num_iteration=max_iteration_possible,
                          population_size=pop_size)
        performance += v
    return performance/30 # get average performance for the 30 runs


if __name__ == '__main__':
    # UNCOMMENT THE FOLLOWING LINE TO RUN TESTS
    test_system()

