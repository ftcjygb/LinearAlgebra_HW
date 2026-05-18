import torch
from matrices_vectors import matrix_vector_product
from gauss import gauss_elimination

def in_column_space(matrix_M, vector_b):
    """
    Checks if vector_b lies within the column space of matrix_M.
    """
    # TODO: Replace the following line
    # Hint: use test_span.
    return test_span(matrix_M, vector_b)

def in_null_space(matrix_M, vector_b):
    """
    Checks if vector_b is in the null space of matrix_M.
    """
    zero_threshold = 1e-6
    
    # TODO: Replace the following line
    # Hint: Use matrix_vector_product
    # Note: When the absolute value is less than zero_threshold, we treat it as zero.
    product = matrix_vector_product(matrix_M, vector_b)
    return torch.all(torch.abs(product) < zero_threshold).item()

def test_invertibility_by_determinants(matrix_M):
    """
    Checks if a square matrix is invertible using its determinant.
    Logic: A matrix is invertible if and only if its determinant is non-zero.
    """
    zero_threshold = 1e-6

    rows, cols = matrix_M.shape

    # 1. Invertibility is only defined for square matrices
    if rows != cols:
        return False

    # TODO: Replace the following line
    # Hint: Use calculuate_determinants
    # Note: When the absolute value is larger than zero_threshold, we treat it as nonzero.
    from determinants import calculate_determinant # 確保有載入此函式
    det_value = calculate_determinant(matrix_M)
    return abs(det_value) > zero_threshold


def test_invertibility(matrix_M):
    """
    Checks if a square matrix is invertible.
    """
    rows, cols = matrix_M.shape

    # 1. Invertibility is only defined for square matrices
    if rows != cols:
        return False

    # 2. TODO: Impelement the invertibility check by using the property:
    # A square matrix is invertible if and only if its Reduced Row Echelon Form (RREF)
    # is the Identity Matrix.
    # (Should done in HW3, not report)
    rref_M = gauss_elimination(matrix_M)
    zero_thresh = 1e-6
    for i in range(rows):
        for j in range(cols):
            val = rref_M[i, j].item()
            if i == j:
                if abs(val - 1.0) > zero_thresh:
                    return False
            else:
                if abs(val) > zero_thresh:
                    return False
    return True

def test_span(matrix_M, vector_b):
    """
    TODO: Determine if vector_b is in the span of the columns of matrix_M.
    Hint: Does Mx = b have a solution?
    Hint: Use solve_linear_equations
    # (Should done in HW2, not report)
    """
    solution = solve_linear_equations(matrix_M, vector_b)
    return solution is not None

def test_linear_dependence(matrix_M):
    """
    TODO: Determine if the columns of matrix_M are linearly dependent.
    
    Logic: The columns are linearly dependent if there exists a NON-ZERO 
           vector 'x' such that Mx = 0.
    
    Challenge: Our 'solve_linear_equations' might return the trivial x=0.
    Hint: If the system has a unique solution (only x=0), they are Independent.
          If the system has infinite solutions (free variables), they are Dependent.
          Think about how your solver handles singular matrices.
    Hint: Nonzero: Euclidean norm > small epsilon (ex: 1e-6)
    Hint: Use solve_linear_equations
    # (Should done in HW2, not report)
    """
    b = torch.zeros((matrix_M.shape[0], 1), dtype=torch.float32)
    solution = solve_linear_equations(matrix_M, b)
    if solution is None:
        return False
    return torch.norm(solution).item() > 1e-6

def test_consistency(augmented_RREF):
    """
    Checks if the system Ax = b is consistent.
    """
    rows, cols = augmented_RREF.shape
    # TODO: Use the following property to determine consistency:
    # A system is inconsistent if and only if a row in the RREF looks like [0 0 ... 0 | b_i] 
    # where b_i != 0. 
    # Return False if inconsistent 
    # (Should done in HW1, not report)
    for r in range(rows):
            left_side = augmented_RREF[r, :-1]
            b_i = augmented_RREF[r, -1]
            if torch.all(torch.abs(left_side) < 1e-5).item() and torch.abs(b_i).item() >= 1e-5:
                return False
    return True # Consistent

def generate_column_space_basis(matrix_M):
    """
    Extracts the basis of the Column Space of matrix_M.
    Logic: 
    1. Perform Gaussian Elimination to get RREF.
    2. Identify the pivot columns (columns with leading 1s).
    3. The corresponding columns in the ORIGINAL matrix form the basis.
    """
    rows, cols = matrix_M.shape
    
    # 1. Get the RREF of the matrix
    rref = gauss_elimination(matrix_M)
    
    pivot_column_indices = []
    
    # 2. TODO: Identify pivot columns
    # We look for the first non-zero entry in each row of the RREF
    for i in range(rows):
        for j in range(cols):
            if abs(rref[i, j].item()) > 1e-6:  # 找到每一行第一個非零元素
                pivot_column_indices.append(j)
                break # 找到該行的 pivot 後就跳出換下一行
            
    # If no pivots found (zero matrix), return an empty tensor or handled case
    if not pivot_column_indices:
        return torch.empty((rows, 0))
        
    # 3. Extract those columns from the ORIGINAL matrix M
    basis_matrix = matrix_M[:, pivot_column_indices]
    
    return basis_matrix

def generate_null_space_basis(matrix_M):
    """
    Generates a basis for the null space of matrix_M.
    Logic: 
    1. Transform matrix M to Reduced Row Echelon Form (rref).
    2. Identify pivot and free variable columns.
    3. Express basic variables in terms of free variables to form basis vectors.
    """
    # 1. Get the Reduced Row Echelon Form
    rref = gauss_elimination(matrix_M)
    rows, cols = rref.shape
    
    # 2. TODO: Identify pivot columns (where leading 1s are located)
    pivot_cols = []
    for r in range(rows):
        for c in range(cols):
            if abs(rref[r, c].item()) > 1e-6:
                pivot_cols.append(c)
                break
            
    # 3. TODO: Identify free variable columns
    free_cols = []
    for c in range(cols):
        if c not in pivot_cols: # 只要不是 pivot column，就是 free variable column
            free_cols.append(c)
    
    # If no free variables exist, the null space contains only the zero vector
    if not free_cols:
        return torch.zeros((cols, 1))
    
    # 4. TODO: Construct one basis vector for each free variable, append the vector into basis_vectors
    basis_vectors = []
    for free_col in free_cols:
        # 初始化一個全為 0 的向量
        vec = torch.zeros((cols, 1), dtype=torch.float32)
        # 將對應的自由變數設為 1
        vec[free_col, 0] = 1.0 
        
        # 根據 RREF 回推基本變數的值 (基本變數 = - (自由變數的係數))
        for r in range(rows):
            p_col = -1
            for c in range(cols):
                if abs(rref[r, c].item()) > 1e-6:
                    p_col = c
                    break
            if p_col != -1: # 若這列有 pivot
                vec[p_col, 0] = -rref[r, free_col]
                
        basis_vectors.append(vec)
    
    # Combine all basis vectors into a single basis matrix
    return torch.cat(basis_vectors, dim=1)

def generate_solution(augmented_RREF):
    """
    Extracts a solution vector. Basic variables are solved, Free variables are set to 1.
    """
    rows, cols_aug = augmented_RREF.shape
    num_vars = cols_aug - 1
    solution = torch.ones((num_vars, 1), dtype=torch.float32) # Default free variables to 1
    
    # TODO: Identify pivot columns
    # Hint: Find the first nonzero in each row
    # (Should done in HW1, not report)
    pivot_cols = []
    for r in range(rows):
        for c in range(num_vars):
            if torch.abs(augmented_RREF[r, c]).item() > 1e-5:
                pivot_cols.append((r, c))
                break
    # TODO: Calculate basic variables
    # Hint: Note that we have assume all free variables are set to 1
    # (Should done in HW1, not report)
    for r, c in pivot_cols:
        sum_others = 0.0
        for j in range(num_vars):
            if j != c:
                sum_others += augmented_RREF[r, j] * solution[j, 0]
        solution[c, 0] = (augmented_RREF[r, -1] - sum_others) / augmented_RREF[r, c]        
    return solution

def solve_linear_equations_by_inverse(A, b):
    """
    Solves Ax = b using the inverse matrix method: x = A^(-1)b.
    The inverse is calculated via Gaussian elimination on [A | I].
    """
    # 1. First, check if the matrix is invertible
    if not test_invertibility(A):
        return None

    num_vars = A.shape[2]
    solution = torch.zeros((num_vars, 1), dtype=torch.float32) # Solution is default to zero
    #TODO: Implement the matrix inversion method to get the inverse of A 
    # (Should done in HW3, not report)
    # 列數為 num_vars，行數為 num_vars * 2
    augmented_matrix = torch.zeros((num_vars, num_vars * 2), dtype=torch.float32)
    
    # 填入 A 矩陣與單位矩陣 I 的數值
    for i in range(num_vars):
        for j in range(num_vars):
            # 左半邊填入原矩陣 A
            augmented_matrix[i, j] = A[i, j]
        # 右半邊對角線填入 1，形成單位矩陣
        augmented_matrix[i, num_vars + i] = 1.0
        
    # 對增廣矩陣執行高斯消去法取得 RREF
    augmented_rref = gauss_elimination(augmented_matrix)
    
    # 右半部作為反矩陣
    A_inverse = augmented_rref[:, num_vars:]
    #TODO: Solve the solution by using inverse of A and return the result
    # (Should done in HW3, not report)
    solution = matrix_vector_product(A_inverse, b)
    return solution


def solve_linear_equations(A, b):
    """
    Wrapper function to solve Ax = b.
    """

    # Create Augmented Matrix [A | b]
    augmented_matrix = torch.cat((A, b), dim=1)

    rows, cols_aug = augmented_matrix.shape
    num_vars = cols_aug - 1
    solution = torch.zeros((num_vars, 1), dtype=torch.float32) # Solution is default to zero
    
    # TODO: Use our custom gauss_elimination to obtain the Reduced Row Echelon Form.
    # Then, use test_consistency to inspect the reduced matrix.
    # Return None if the system is inconsistent. Otherwise, return the results of generate_solution.
    # (Should done in HW1, not report)
    augmented_RREF = gauss_elimination(augmented_matrix)
    # Then, use test_consistency to inspect the reduced matrix.
    if not test_consistency(augmented_RREF):
        return None
    # Return None if the system is inconsistent. Otherwise, return the results of generate_solution.
    return generate_solution(augmented_RREF)
