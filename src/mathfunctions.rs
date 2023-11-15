#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(unused_imports)]

use crate::*;



pub trait MathFunctions<T:MatrixValues> {
    fn linspace(lower_bound: T, higher_bound: T, steps: usize) -> Vec<T>;
    fn abs(value: T) -> T ;
    fn vec_dot_vec(vec_1: Vec<T>, vec_2: Vec<T>) -> T ;
    fn vec_summed(vec_1: Vec<T>) -> T ;
    fn row_dot_collumn(row: & Row<T>, collumn: & Collumn<T>) -> T ;
    fn matrix_dot_collumn(matrix: & Matrix<T>, collumn: & Collumn<T>) -> Collumn<T> ;
    fn matrix_add_matrix(mut A_matrix: Matrix<T>, B_matrix:& Matrix<T>) -> Matrix<T> {todo!()}
    fn matrix_dot_matrix(A_matrix:& Matrix<T>, B_matrix:& Matrix<T>) -> Matrix<T> ;
}

macro_rules! impl_math_functions_per_type {
    ($T: ident) => {
    impl MathFunctions<$T> for $T {
        fn linspace(lower_bound: $T, higher_bound: $T, steps: usize) -> Vec<$T> {
            let step_size = (higher_bound - lower_bound) / (steps-1) as $T;
            let mut lin_spaced_vec = Vec::<$T>::with_capacity(steps);
            for i in 0..steps {
                lin_spaced_vec.push(lower_bound + step_size * i as $T);
            }
            lin_spaced_vec
        }

        fn abs(value: $T) -> $T {
            if value >= 0 as $T {
                return value;
            } else {
                let min_one: $T = NumCast::from(-1).unwrap();
                let return_value:$T = value * min_one;
                return return_value;
            }
        }

        fn vec_dot_vec(vec_1: Vec<$T>, vec_2: Vec<$T>) ->$T {
            if !(vec_1.len()==vec_2.len()) {
                panic!("Vectors not of same length!");
            }

            let vec_len = vec_1.len();

            let piecewice_mut = vec_1.iter().zip(vec_2).map(|(u, v)| 
                // if !(*u == NumCast::from(0).unwrap() || v == NumCast::from(0).unwrap()) {
                    u.clone() * v.clone()).collect();
                // } else {
                    // NumCast::from(0).unwrap()
                // }).collect();

            $T::vec_summed(piecewice_mut)
        }

        fn vec_summed(vec_1: Vec<$T>) ->$T {
            let mut total: $T = NumCast::from(0).unwrap();
            for j in 0..vec_1.len() {
                // if !(vec_1[j] == NumCast::from(0).unwrap()) {
                    total = total + vec_1[j];
                // }
            }
            total
        }

        fn row_dot_collumn(row: & Row<$T>, collumn: & Collumn<$T>) -> $T {
            if !(row.len() == collumn.n_rows()) {
                panic!("Dimension of Row length must match collumn height. Row has dimension [1x{}], Collumn has [{}x1]", row.len(), collumn.n_rows());
            }
            let n_elements = row.len();

            let piecewice_mut = (0..n_elements).into_iter().map(|j| 
                // if !(row[j] == NumCast::from(0).unwrap() || collumn[j] == NumCast::from(0).unwrap()) {
                    row[j]*collumn[j]).collect();
                // } else {
                    // NumCast::from(0).unwrap()
                // }).collect();

            $T::vec_summed(piecewice_mut)

        }

        fn matrix_dot_collumn(matrix: & Matrix<$T>, collumn: & Collumn<$T>) -> Collumn<$T> {
            if !(matrix.width() == collumn.n_rows()) {
                panic!("Dimension of matrix width must match collumn height. Matrix has dimension [{}x{}], Collumn has [{}x1]", matrix.height(), matrix.width(), collumn.n_rows());
            }

            let mut result_collumn: Collumn<$T> = Collumn::new_with_constant_values(matrix.height(), NumCast::from(0).unwrap());

            for row_number in 0..matrix.height() {
                result_collumn[row_number] = $T::row_dot_collumn(& matrix[row_number], & collumn);
            }

            result_collumn
        }

        // pub fn row_dot_collumn<T: MatrixValues>(row:& Row<T>, collumn:& Collumn<T>) -> $T {
        //     if 
        // }

        fn matrix_add_matrix(mut A_matrix: Matrix<$T>, B_matrix:& Matrix<$T>) -> Matrix<$T> {
            // this function consumes the A_matrix and takes a reference to the B_matrix (no need to consume it) 
            // returns the altered A_matrix
            if !(A_matrix.height() == B_matrix.height() || A_matrix.width() == B_matrix.width()) {
                panic!("Matrices dimensions do not match for adition! given: [{}x{}] + [{}x{}]", A_matrix.height(), A_matrix.width(), B_matrix.height(), B_matrix.width());
            }

            let N = A_matrix.height();
            let M = A_matrix.width();

            for row_index in 0..N {
                A_matrix[row_index].addition_row_with_external_row(&B_matrix[row_index]);
            }

            A_matrix
        }

        fn matrix_dot_matrix(A_matrix:& Matrix<$T>, B_matrix:& Matrix<$T>) -> Matrix<$T> {
            // this function takes 2 references to the matrices to multiply, returns a new matrix.
            // As in place matrix multiplication is practically impossible, there should be the space for all 3 in memory.

            // this is a slow function because of: many operations and collumn operations in B_matrix.
            // Ideally A_matrix would be row major, B_matrix collumn major
            if !(A_matrix.width() == B_matrix.height()) {
                panic!("Matrices dimensions do not match for dot operation! given: [{}x{}] + [{}x{}]", A_matrix.height(), A_matrix.width(), B_matrix.height(), B_matrix.width());
            }

            let mut C_matrix: Matrix<$T> = Matrix::new_with_constant_values(A_matrix.height(), B_matrix.width(), NumCast::from(0).unwrap());

            for row_index in 0..C_matrix.height() {
                for collumn_index in 0..C_matrix.width() {
                    C_matrix[row_index][collumn_index] = $T::row_dot_collumn(&A_matrix[row_index], &B_matrix.get_collumn(collumn_index))
                }
            }

            C_matrix
        }
    }
    };
}

// impl_math_functions_per_type!(i8);
// impl_math_functions_per_type!(i16);
// impl_math_functions_per_type!(i32);
impl_math_functions_per_type!(i64);

// impl_math_functions_per_type!(u8);
// impl_math_functions_per_type!(u16);
// impl_math_functions_per_type!(u32);
// impl_math_functions_per_type!(u64);

// impl_math_functions_per_type!(isize);
// impl_math_functions_per_type!(usize);

// impl_math_functions_per_type!(f32);
// impl_math_functions_per_type!(f64);
