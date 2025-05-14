import numpy as np
from src.utils import candidates, ind_constraint, cum31, cum22


def TestCase(T, O, Z):
    if ind_constraint(T, O, Z)[0]:
        return 'a'
    
    root1, root2 = candidates(T, O)
    flag1, _ = ind_constraint(Z, T, O - root1 * T)
    flag2, _ = ind_constraint(Z, T, O - root2 * T)
    if flag1 ^ flag2:
        return 'b'
    
    root1, root2 = candidates(T, O)
    flag1, _ = ind_constraint(O - root1 * T, Z, T)
    flag2, _ = ind_constraint(O - root2 * T, Z, T)
    if flag1 or flag2:
        return 'c'
    
    root1, root2 = candidates(O, Z)
    flag1, _ = ind_constraint(T, O, Z - root1 * O)
    flag2, _ = ind_constraint(T, O, Z - root2 * O)
    if flag1 or flag2:
        return 'd'
    
    root_t1, root_t2 = candidates(Z, T)
    root_o1, root_o2 = candidates(Z, O)
    flag1, _ = ind_constraint(T - root_t1 * Z, O - root_o1 * Z, Z)
    flag2, _ = ind_constraint(T - root_t1 * Z, O - root_o2 * Z, Z)
    flag3, _ = ind_constraint(T - root_t2 * Z, O - root_o1 * Z, Z)
    flag4, _ = ind_constraint(T - root_t2 * Z, O - root_o2 * Z, Z)
    if flag1 or flag2 or flag3 or flag4:
        return 'e'

    return 'f,g,h'


def CalCase(T, O, Z, case):
    
    def perfect(T, O, Z):
        if cum31(T, Z) / cum31(Z, T) > 0:
            ratio = np.sign(np.cov(T,Z)[0,1]) * np.sqrt(cum31(T, Z) / cum31(Z, T))
        else:
            ratio1 = cum31(T, Z) / cum22(T, Z)
            ratio2 = cum22(T, Z) / cum31(Z, T)
            ratio = (ratio1 + ratio2) / 2
        result = (np.cov(T,O)[0,1] - ratio * np.cov(O,Z)[0,1]) / (np.var(T) - ratio * np.cov(T,Z)[0,1])
        return result

    if case == 'a':
        return perfect(T, O, Z)
    if case == 'b':
        root1, root2 = candidates(T, O)
        _, p1 = ind_constraint(Z, T, O - root1 * T)
        _, p2 = ind_constraint(Z, T, O - root2 * T)
        if p1 > p2:
            return root1
        else:
            return root2
    if case == 'c':
        root1, root2 = candidates(T, O)
        _, p1 = ind_constraint(O - root1 * T, Z, T)
        _, p2 = ind_constraint(O - root2 * T, Z, T)
        if p1 > p2:
            return root1
        else:
            return root2
    if case == 'd':
        root1, root2 = candidates(O, Z)
        _, p1 = ind_constraint(T, O, Z - root1 * O)
        _, p2 = ind_constraint(T, O, Z - root2 * O)
        if p1 > p2:
            return perfect(T, O, Z - root1 * O)
        else:
            return perfect(T, O, Z - root2 * O)
    if case == 'e':
        root_t1, root_t2 = candidates(Z, T)
        root_o1, root_o2 = candidates(Z, O)
        _, p1 = ind_constraint(T - root_t1 * Z, O - root_o1 * Z, Z)
        _, p2 = ind_constraint(T - root_t1 * Z, O - root_o2 * Z, Z)
        _, p3 = ind_constraint(T - root_t2 * Z, O - root_o1 * Z, Z)
        _, p4 = ind_constraint(T - root_t2 * Z, O - root_o2 * Z, Z)
        if p1 > max([p2, p3, p4]):
            return perfect(T - root_t1 * Z, O - root_o1 * Z, Z)
        elif p2 > max([p1, p3, p4]):
            return perfect(T - root_t1 * Z, O - root_o2 * Z, Z)
        elif p3 > max([p1, p2, p4]):
            return perfect(T - root_t2 * Z, O - root_o1 * Z, Z)
        else:
            return perfect(T - root_t2 * Z, O - root_o2 * Z, Z)
    else:
        raise ValueError


def CalComplexCase(T, O, Z1, Z2, case):
    
    def double_perfect(T, O, Z1, Z2):
        if cum31(T, Z1) / cum31(Z1, T) > 0:
            ratio1 = np.sign(np.cov(T,Z1)[0,1]) * np.sqrt(cum31(T, Z1) / cum31(Z1, T))
        else:
            ratio11 = cum31(T, Z1) / cum22(T, Z1)
            ratio12 = cum22(T, Z1) / cum31(Z1, T)
            ratio1 = (ratio11 + ratio12) / 2

        if cum31(T, Z2) / cum31(Z2, T) > 0:
            ratio2 = np.sign(np.cov(T,Z2)[0,1]) * np.sqrt(cum31(T, Z2) / cum31(Z2, T))
        else:
            ratio21 = cum31(T, Z2) / cum22(T, Z2)
            ratio22 = cum22(T, Z2) / cum31(Z2, T)
            ratio2 = (ratio21 + ratio22) / 2

        result = (np.cov(T,O)[0,1] - ratio1 * np.cov(O,Z1)[0,1] - ratio2 * np.cov(O,Z2)[0,1]) / (np.var(T) - ratio1 * np.cov(T,Z1)[0,1] - ratio2 * np.cov(T,Z2)[0,1])
        return result
    
    if case == 'a':
        return double_perfect(T, O, Z1, Z2)
    
    if case == 'b':
        root_t1, root_t2 = candidates(Z1, T)
        root_o1, root_o2 = candidates(Z1, O)
        _, p1 = ind_constraint(T - root_t1 * Z1, O - root_o1 * Z1, Z1)
        _, p2 = ind_constraint(T - root_t1 * Z1, O - root_o2 * Z1, Z1)
        _, p3 = ind_constraint(T - root_t2 * Z1, O - root_o1 * Z1, Z1)
        _, p4 = ind_constraint(T - root_t2 * Z1, O - root_o2 * Z1, Z1)
        if p1 > max([p2, p3, p4]):
            lamda1, lamda2 = root_t1, root_o1
        elif p2 > max([p1, p3, p4]):
            lamda1, lamda2 = root_t1, root_o2
        elif p3 > max([p1, p2, p4]):
            lamda1, lamda2 = root_t2, root_o1
        else:
            lamda1, lamda2 = root_t2, root_o2
        root1, root2 = candidates(O, Z2)
        _, p1 = ind_constraint(T, O, Z2 - root1 * O)
        _, p2 = ind_constraint(T, O, Z2 - root2 * O)
        if p1 > p2:
            lamda = root1
        else:
            lamda = root2

        Z2 = Z2 - lamda * O
        T = T - lamda1 * Z1
        O = O - lamda2 * Z1
        return double_perfect(T, O, Z1, Z2)

    if case == 'c':
        root1, root2 = candidates(T, Z1)
        _, p1 = ind_constraint(T, O, Z1 - root1 * T)
        _, p2 = ind_constraint(T, O, Z1 - root2 * T)
        if p1 > p2:
            lamda = root1
        else:
            lamda = root2
        root_t1, root_t2 = candidates(Z2, T)
        root_o1, root_o2 = candidates(Z2, O)
        _, p1 = ind_constraint(T - root_t1 * Z2, O - root_o1 * Z2, Z2)
        _, p2 = ind_constraint(T - root_t1 * Z2, O - root_o2 * Z2, Z2)
        _, p3 = ind_constraint(T - root_t2 * Z2, O - root_o1 * Z2, Z2)
        _, p4 = ind_constraint(T - root_t2 * Z2, O - root_o2 * Z2, Z2)
        if p1 > max([p2, p3, p4]):
            lamda1, lamda2 = root_t1, root_o1
        elif p2 > max([p1, p3, p4]):
            lamda1, lamda2 = root_t1, root_o2
        elif p3 > max([p1, p2, p4]):
            lamda1, lamda2 = root_t2, root_o1
        else:
            lamda1, lamda2 = root_t2, root_o2

        Z1 = Z1 - lamda * T
        T = T - lamda1 * Z2
        O = O - lamda2 * Z2
        return double_perfect(T, O, Z1, Z2)
    
    else:
        raise ValueError