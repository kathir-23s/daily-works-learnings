'''
        This function will return a multiplied matrix of dimension 2

        Input: Tensor1 - 2d - MxN; Tensor1 - 2d - NxK 
        Output: TensorT - 2d - MxK

        Conditions:
        Number of columns in matrix A should be equal to number of rows in matrix B
        '''

'''
        Blocked matrix multiplication (cache-friendly optimization).

        Input:
            self  -> TensorT of shape (N, N)
            other -> TensorT of shape (N, N)
            block_size -> integer block dimension (e.g., 32, 64)
        Output:
            TensorT of shape (N, N)

        Conditions:
            - Both operands must be 2D square matrices of the same shape
            - block_size must divide the dimension or be smaller than N
        '''