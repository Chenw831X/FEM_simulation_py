#
# fem3d_tet_linear.py
#
# Created by Wei Chen on 8/12/21
#

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import math

'''
fem3d_tet_linear: linear fem of 3d tetrahedron model
@input:
    nvet: number of vertices
    nele: number of tetrahedron
    input/cor.txt: corrdinates of vertices
    input/elem.txt: indices of tetrahedron elements
@output:
    output/u.txt: displacement of vertices
    output/cor1.txt: result corrdinates of vertices
'''
def fem3d_tet_linear(nvet, nele):
    print('fem simulation start')

    eeps = 1e-8
    # user-defined material properities
    E = 1.0         # Young's modulus
    nu = 0.3        # Poisson's ratio
    # constitutive matrix
    D = E / (1. + nu) / (1. - 2. * nu) * np.array(
        [[1-nu, nu, nu, 0., 0., 0.],
         [nu, 1-nu, nu, 0., 0., 0.],
         [nu, nu, 1-nu, 0., 0., 0.],
         [0., 0., 0., (1.0-2.0*nu)/2.0, 0., 0.],
         [0., 0., 0., 0., (1.0-2.0*nu)/2.0, 0.],
         [0., 0., 0., 0., 0., (1.0-2.0*nu)/2.0]], dtype=np.float64)
    
    # read coordinates of vertices, tetrahedron information
    cor = readCor(nvet, 'input/cor.txt')
    elem = readELem(nele, 'input/elem.txt')

    # prepare finite element analysis
    ndof = nvet * 3
    U = np.zeros((ndof), dtype=np.float64)
    F = np.zeros((ndof), dtype=np.float64)

    # user-defined load dofs
    for vI in range(nvet):
        if abs(cor[vI, 2]-1.0) < eeps and abs(cor[vI, 0]) < eeps:
            F[3*vI] = -0.01
    print('compute load')

    # user-defined fixed dofs
    fixednids = np.asarray(cor[:, 2]==0.0).nonzero()[0]
    fixeddofs = np.concatenate((fixednids*3, fixednids*3+1, fixednids*3+2), axis=0)
    freedofs = np.setdiff1d(np.arange(ndof), fixeddofs)
    print('compute DBC')

    # assembly the stiffness matrix
    edofMat = np.kron(3*elem, np.ones((1, 3))) + np.kron(np.tile(np.arange(3), 4), np.ones((nele, 1)))
    edofMat = edofMat.astype(np.int32)
    iK = np.kron(edofMat, np.ones((12, 1))).flatten()
    jK = np.kron(edofMat, np.ones((1, 12))).flatten()
    iK = iK.astype(np.int32)
    jK = jK.astype(np.int32)

    sK = np.empty((nele, 12*12), dtype=np.float64)
    for eleI in range(nele):
        X = cor[[elem[eleI, 0], elem[eleI, 1], elem[eleI, 2], elem[eleI, 3]], :]
        KE = initKE(D, X)
        sK[eleI, :] = KE.flatten('F')
    sK = sK.flatten()

    K = coo_matrix((sK, (iK, jK)), shape=(ndof, ndof)).tocsc()

    # FE-analysis
    print('solve')
    K = K[freedofs, :][:, freedofs]
    U[freedofs] = spsolve(K, F[freedofs])

    # write result
    print('write displacement U.txt and new corrdinates cor1.txt')
    with open('output/U.txt', 'w') as f:
        for i in range(ndof):
            print(U[i], file=f)
    cor1 = cor + U.reshape(nvet, 3)
    with open('output/cor1.txt', 'w') as f:
        for vI in range(nvet):
            print(cor1[vI, 0], cor1[vI, 1], cor1[vI, 2], file=f)

    

def readCor(nvet, filePath):
    cor = np.empty((nvet, 3), dtype=np.float64)
    with open(filePath, 'r') as f:
        for vI in range(nvet):
            line = f.readline()
            line_split = line.split()
            cor[vI, 0] = float(line_split[0])
            cor[vI, 1] = float(line_split[1])
            cor[vI, 2] = float(line_split[2])
    return cor


def readELem(nele, filePath):
    elem = np.empty((nele, 4), dtype=np.int32)
    with open(filePath, 'r') as f:
        for eleI in range(nele):
            line = f.readline()
            line_split = line.split()
            elem[eleI, 0] = int(line_split[0])
            elem[eleI, 1] = int(line_split[1])
            elem[eleI, 2] = int(line_split[2])
            elem[eleI, 3] = int(line_split[3])
    elem = elem - 1
    return elem


# generate element stiffness matrix
# @input
# D: constitutive matrix
# X: 4*3 matrix, which is corridinates of element vertices
# @output
# KE: element stiffness matrix
def initKE(D, X):
    H = np.array([[1., X[0, 0], X[0, 1], X[0, 2]],
                  [1., X[1, 0], X[1, 1], X[1, 2]],
                  [1., X[2, 0], X[2, 1], X[2, 2]],
                  [1., X[3, 0], X[3, 1], X[3, 2]]])
    V6 = abs(np.linalg.det(H))

    a1 =  np.linalg.det(H[[1, 2, 3], :][:, [1, 2, 3]])    
    a2 = -np.linalg.det(H[[0, 2, 3], :][:, [1, 2, 3]])    
    a3 =  np.linalg.det(H[[0, 1, 3], :][:, [1, 2, 3]])    
    a4 = -np.linalg.det(H[[0, 1, 2], :][:, [1, 2, 3]])    

    b1 = -np.linalg.det(H[[1, 2, 3], :][:, [0, 2, 3]])    
    b2 =  np.linalg.det(H[[0, 2, 3], :][:, [0, 2, 3]])    
    b3 = -np.linalg.det(H[[0, 1, 3], :][:, [0, 2, 3]])    
    b4 =  np.linalg.det(H[[0, 1, 2], :][:, [0, 2, 3]])    

    c1 =  np.linalg.det(H[[1, 2, 3], :][:, [0, 1, 3]])    
    c2 = -np.linalg.det(H[[0, 2, 3], :][:, [0, 1, 3]])    
    c3 =  np.linalg.det(H[[0, 1, 3], :][:, [0, 1, 3]])    
    c4 = -np.linalg.det(H[[0, 1, 2], :][:, [0, 1, 3]])    

    d1 = -np.linalg.det(H[[1, 2, 3], :][:, [0, 1, 2]])    
    d2 =  np.linalg.det(H[[0, 2, 3], :][:, [0, 1, 2]])    
    d3 = -np.linalg.det(H[[0, 1, 3], :][:, [0, 1, 2]])    
    d4 =  np.linalg.det(H[[0, 1, 2], :][:, [0, 1, 2]])    

    B = np.empty((6, 12), dtype=np.float64)

    B[:, 0:3] = np.array([[b1, 0., 0.],
                          [0., c1, 0.],
                          [0., 0., d1],
                          [c1, b1, 0.],
                          [0., d1, c1],
                          [d1, 0., b1]])

    B[:, 3:6] = np.array([[b2, 0., 0.],
                          [0., c2, 0.],
                          [0., 0., d2],
                          [c2, b2, 0.],
                          [0., d2, c2],
                          [d2, 0., b2]])

    B[:, 6:9] = np.array([[b3, 0., 0.],
                          [0., c3, 0.],
                          [0., 0., d3],
                          [c3, b3, 0.],
                          [0., d3, c3],
                          [d3, 0., b3]])

    B[:, 9:12] = np.array([[b4, 0., 0.],
                           [0., c4, 0.],
                           [0., 0., d4],
                           [c4, b4, 0.],
                           [0., d4, c4],
                           [d4, 0., b4]])
    
    B = B / V6
    KE = V6 / 6.0 * (B.T @ D @ B)
    return KE


if __name__ == '__main__':
    fem3d_tet_linear(3168, 16563)