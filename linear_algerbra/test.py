#encoding=utf-8

import unittest
import numpy as np

from decimal import *

# TODO 返回矩阵的行数和列数
def shape(M):
    l_row = len(M)
    l_col = 0
    if l_row != 0:
        l_col = len(M[0])
    return l_row, l_col

# TODO 每个元素四舍五入到特定小数数位
# 直接修改参数矩阵，无返回值
def matxRound(M, decPts=4):
    for l_row, l_valueList in enumerate(M):
        M[l_row] = [round(l_item, decPts) for l_item in l_valueList]

# TODO 计算矩阵的转置
def transpose(M):
    (l_orgRow, l_orgCol) = shape(M)
    l_MT = []
    for l_row in xrange(l_orgCol):
        l_rowList = []
        for l_index in xrange(l_orgRow):
            l_rowList.append(M[l_index][l_row])
        l_MT.append(l_rowList)
    return l_MT

# TODO 计算矩阵乘法 AB，如果无法相乘则raise ValueError
def matxMultiply(A, B):
    l_rowA, l_colA = shape(A)
    l_rowB, l_colB = shape(B)
    
    if l_colA != l_rowB:
        raise ValueError
    
    l_result = []
    for l_row in xrange(l_rowA):
        l_rowList = []
        for l_col in xrange(l_colB):
            l_sum = 0
            for l_index in xrange(l_colA):
                l_sum += A[l_row][l_index] * B[l_index][l_col]
            l_rowList.append(l_sum)
            
        l_result.append(l_rowList)
        
    return l_result

# TODO 构造增广矩阵，假设A，b行数相同
def augmentMatrix(A, b):
    if len(A) != len(b):
        raise ValueError
    
    l_augMatrix = []
    for l_row in xrange(len(A)):
        l_augMatrix.append(A[l_row] + b[l_row])
    
    return l_augMatrix

# TODO r1 <---> r2
# 直接修改参数矩阵，无返回值
def swapRows(M, r1, r2):
    M[r1], M[r2] = M[r2], M[r1]

# TODO r1 <--- r1 * scale
# scale为0是非法输入，要求 raise ValueError
# 直接修改参数矩阵，无返回值
def scaleRow(M, r, scale):
    if abs(scale) < 1.0e-16:
        raise ValueError

    M[r] = [l_item * scale for l_item in M[r]]

# TODO r1 <--- r1 + r2*scale
# 直接修改参数矩阵，无返回值
def addScaledRow(M, r1, r2, scale):
    l_addList = [l_item * scale for l_item in M[r2]]
    M[r1] = [l_r1 + l_r2 * scale for l_r1, l_r2 in zip(M[r1], M[r2])]

# TODO 实现 Gaussain Jordan 方法求解 Ax = b

""" Gaussian Jordan 方法求解 Ax = b.
    参数
        A: 方阵 
        b: 列向量
        decPts: 四舍五入位数，默认为4
        epsilon: 判读是否为0的阈值，默认 1.0e-16
        
    返回列向量 x 使得 Ax = b 
    返回None，如果 A，b 高度不同
    返回None，如果 A 为奇异矩阵
"""
def gj_Solve(A, b, decPts=4, epsilon = 1.0e-16):
    l_row, l_col = shape(A)
    if l_row != l_col:
        raise ValueError
    # 判断高度是否一致
    if len(A) != len(b):
        return None

    l_augMatrix = augmentMatrix(A, b)
    for l_colIndex in xrange(l_col):
        # 寻找列c中 对角线以及对角线以下所有元素（行 c~N）的绝对值的最大值
        # 注意此处为方阵，行列相同
        l_max = abs(l_augMatrix[l_colIndex][l_colIndex])
        l_maxRow = l_colIndex
        for l_rowIndex in xrange(l_colIndex + 1, l_row):
            l_tempValue = abs(l_augMatrix[l_rowIndex][l_colIndex])
            if l_max < l_tempValue:
                l_max = l_tempValue
                l_maxRow = l_rowIndex
        if l_max < epsilon:
            # 奇异矩阵
            return None

        # 绝对值最大值所在行交换到对角线元素所在行
        swapRows(l_augMatrix, l_colIndex, l_maxRow)
        # 将列c的对角线元素缩放为1
        scaleRow(l_augMatrix, l_colIndex, 1.0 / l_augMatrix[l_colIndex][l_colIndex])
        # 将列c的其他元素消为0
        for l_rowIndex in xrange(l_row):
            if l_rowIndex == l_colIndex:
                continue
            if not (abs(l_augMatrix[l_rowIndex][l_colIndex]) < epsilon):
                # 消为0
                addScaledRow(l_augMatrix, l_rowIndex, l_colIndex, - l_augMatrix[l_rowIndex][l_colIndex])
            else:
                l_augMatrix[l_rowIndex][l_colIndex] = 0

    l_result = [[l_item[-1]] for l_item in l_augMatrix]

    return l_result

class LinearRegressionTestCase(unittest.TestCase):
    """Test for linear regression project"""

    def test_shape(self):

        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.randint(low=-10,high=10,size=(r,c))
            self.assertEqual(shape(matrix.tolist()),(r,c),'Wrong answer')


    def test_matxRound(self):

        for decpts in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()
            dec_true = [[Decimal(str(round(num,decpts))) for num in row] for row in mat]

            matxRound(mat,decpts)
            dec_test = [[Decimal(str(num)) for num in row] for row in mat]

            res = Decimal('0')
            for i in range(len(mat)):
                for j in range(len(mat[0])):
                    res += dec_test[i][j].compare_total(dec_true[i][j])

            self.assertEqual(res,Decimal('0'),'Wrong answer')


    def test_transpose(self):
        for _ in range(100):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()
            t = np.array(transpose(mat))

            self.assertEqual(t.shape,(c,r),"Expected shape{}, but got shape{}".format((c,r),t.shape))
            self.assertTrue((matrix.T == t).all(),'Wrong answer')


    def test_matxMultiply(self):

        for _ in range(100):
            r,d,c = np.random.randint(low=1,high=25,size=3)
            mat1 = np.random.randint(low=-10,high=10,size=(r,d)) 
            mat2 = np.random.randint(low=-5,high=5,size=(d,c)) 
            dotProduct = np.dot(mat1,mat2)

            dp = np.array(matxMultiply(mat1.tolist(),mat2.tolist()))
            self.assertEqual(dotProduct.shape, dp.shape,
                             'Wrong answer, expected shape{}, but got shape{}'.format(dotProduct.shape, dp.shape))
            self.assertTrue((dotProduct == dp).all(),'Wrong answer')

        mat1 = np.random.randint(low=-10,high=10,size=(r,5)) 
        mat2 = np.random.randint(low=-5,high=5,size=(4,c)) 
        mat3 = np.random.randint(low=-5,high=5,size=(6,c)) 
        with self.assertRaises(ValueError,msg="Matrix A\'s column number doesn\'t equal to Matrix b\'s row number"):
        	matxMultiply(mat1.tolist(),mat2.tolist())
        with self.assertRaises(ValueError,msg="Matrix A\'s column number doesn\'t equal to Matrix b\'s row number"):
        	matxMultiply(mat1.tolist(),mat3.tolist())


    def test_augmentMatrix(self):

        for _ in range(50):
            r,c = np.random.randint(low=1,high=25,size=2)
            A = np.random.randint(low=-10,high=10,size=(r,c))
            b = np.random.randint(low=-10,high=10,size=(r,1))
            Amat = A.tolist()
            bmat = b.tolist()

            Ab = np.array(augmentMatrix(Amat,bmat))
            ab = np.hstack((A,b))

            self.assertTrue(A.tolist() == Amat,"Matrix A shouldn't be modified")
            self.assertEqual(Ab.shape, ab.shape,
                             'Wrong answer, expected shape{}, but got shape{}'.format(ab.shape, Ab.shape))
            self.assertTrue((Ab == ab).all(),'Wrong answer')

    def test_swapRows(self):
        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()

            r1, r2 = np.random.randint(0,r, size = 2)
            swapRows(mat,r1,r2)

            matrix[[r1,r2]] = matrix[[r2,r1]]

            self.assertTrue((matrix == np.array(mat)).all(),'Wrong answer')

    def test_scaleRow(self):

        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()

            rr = np.random.randint(0,r)
            with self.assertRaises(ValueError):
                scaleRow(mat,rr,0)

            scale = np.random.randint(low=1,high=10)
            scaleRow(mat,rr,scale)
            matrix[rr] *= scale

            self.assertTrue((matrix == np.array(mat)).all(),'Wrong answer')
    
    def test_addScaledRow(self):

        for _ in range(10):
            r,c = np.random.randint(low=1,high=25,size=2)
            matrix = np.random.random((r,c))

            mat = matrix.tolist()

            r1,r2 = np.random.randint(0,r,size=2)

            scale = np.random.randint(low=1,high=10)
            addScaledRow(mat,r1,r2,scale)
            matrix[r1] += scale * matrix[r2]

            self.assertTrue((matrix == np.array(mat)).all(),'Wrong answer')


    def test_gj_Solve(self):

        for _ in range(9999):
            r = np.random.randint(low=3,high=10)
            A = np.random.randint(low=-10,high=10,size=(r,r))
            b = np.arange(r).reshape((r,1))

            x = gj_Solve(A.tolist(),b.tolist(),epsilon=1.0e-8)

            if np.linalg.matrix_rank(A) < r:
                self.assertEqual(x,None,"Matrix A is singular")
            else:
                self.assertNotEqual(x,None,"Matrix A is not singular")
                self.assertEqual(np.array(x).shape,(r,1),"Expected shape({},1), but got shape{}".format(r,np.array(x).shape))
                Ax = np.dot(A,np.array(x))
                loss = np.mean((Ax - b)**2)
                self.assertTrue(loss<0.1,"Bad result.")

if __name__ == '__main__':
    unittest.main()
