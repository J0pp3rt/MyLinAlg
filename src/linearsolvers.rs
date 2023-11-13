#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(unused_imports)]

use crate::*;

pub struct Solver2D<T: MatrixValues> {
    pub A_matrix: Matrix<T>,
    pub B_matrix: Matrix<T>,
    pub solver: Solver2DStrategy
}

pub struct Solved2D<T: MatrixValues> {
    pub A_matrix: Matrix<T>,
    pub B_matrix: Matrix<T>
}

pub enum Solver2DStrategy {
    Guass,
    LUDecomposition,
    CholeskyDecomposition,
    CholeskySquareRootFreeDecomposition,
}

pub trait LinearSolverFunctions<T: MatrixValues> {
    fn solve_with_guass(system_of_equations: Solver2D<T>) -> Solved2D<T> ;    
    fn solve_with_cholesky_quare_root_free_decomposition(mut system_of_equations: Solver2D<T>) -> Solved2D<T>{
        todo!()
    }
    
}

macro_rules! impl_linear_solve_for_type {
    ($T: ident) => {
    impl LinearSolverFunctions<$T> for $T {        
        fn solve_with_guass(system_of_equations: Solver2D<$T>) -> Solved2D<$T> {
            let start_copy = Instant::now();
            let mut A_matrix = system_of_equations.A_matrix.clone();
            let mut B_matrix = system_of_equations.B_matrix.clone();
            let time_copy = start_copy.elapsed();

            let mut working_collumn :usize = 0; // A better name for this one probably exists
            let mut working_row :usize = 0; // A better name for this one probably exists
            let mut keep_itterating: bool = true;

            while keep_itterating {
                // start by finding the first row that has a value for the first collumn
                let to_work_on_collumn = A_matrix.get_collumn(working_collumn);
                let mut first_nonzero_row: usize = 0;
                let mut non_zero_row_value_found: bool = false;
                for value_index in working_row..to_work_on_collumn.n_rows(){ // do something when first collumn is empty everywhere
                    if !(to_work_on_collumn[value_index] == NumCast::from(0).unwrap()) {
                        first_nonzero_row = value_index;
                        non_zero_row_value_found = true;
                        break
                    }
                }

                // if the current remaining collumn does not contain non zero values, a ... (? non pivot ?) is found
                // continue to the next collumn
                if !(non_zero_row_value_found) { 
                    working_collumn += 1;
                    if working_collumn == A_matrix.row_length() {
                        keep_itterating = false;
                    }
                    continue;
                }

                if !(first_nonzero_row == working_row) { // the current pivot may not be 0, make sure it has a value (prevent divide by zero)
                    A_matrix.swap_rows(working_row, first_nonzero_row);
                    B_matrix.swap_rows(working_row, first_nonzero_row);
                }

                let normalize_factor = A_matrix[working_row][working_collumn];
                B_matrix[working_row].divide_all_elements_by(normalize_factor);
                A_matrix[working_row].normalize_all_elements_to_element(working_collumn); // normalize all values in the row to the pivot so in the end Identity matrix is reached

                // substract current row from all other rows to eliminate variable
                for row_number_of_other_row in (0..A_matrix.rows.len()).filter(|index| !(*index == working_row)) {
                    // skip substraction of value in this row is already zero, save computational time
                    
                    if A_matrix[row_number_of_other_row][working_collumn] == NumCast::from(0).unwrap() {
                        continue;
                    }
                    let factor_of_substraction = A_matrix[row_number_of_other_row][working_collumn] / A_matrix[working_row][working_collumn];
                    A_matrix.substract_multiplied_internal_row_from_row_by_index(working_row, factor_of_substraction, row_number_of_other_row);
                    B_matrix.substract_multiplied_internal_row_from_row_by_index(working_row, factor_of_substraction, row_number_of_other_row);
                }

                // in all other rows the variable should have been eliminated, continue to the next row and collumn
                working_collumn += 1;
                working_row += 1;

                // check if the calculation is done by comparing the new rows to the bounds

                if working_collumn == A_matrix.row_length() {
                    keep_itterating = false;
                }
                if working_row == A_matrix.coll_length() {
                    keep_itterating = false;
                }

                // println!("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
                // println!("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
                // println!("Completed row {:}", working_row-1);
                // println!("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
                // println!("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");

                // A_matrix.printer();
                // B_matrix.printer();
            }
            

            Solved2D {A_matrix, B_matrix}
        }

        fn solve_with_cholesky_quare_root_free_decomposition(mut system_of_equations: Solver2D<$T>) -> Solved2D<$T> {
            // this function consumes A and B
            let start_assign = Instant::now();
            let mut A_matrix = system_of_equations.A_matrix;
            let mut B_matrix = system_of_equations.B_matrix;
            let time_assign = start_assign.elapsed();

            if !(A_matrix.is_square()) {
                panic!("If a matrix aint square it cant be symmetric -> given A matrix not square")
            }

            let matrix_size = A_matrix.len();

            let start_Lmake = Instant::now();

            let mut L: Matrix<$T> = Matrix::new_square_eye(matrix_size, NumCast::from(1).unwrap());
            let time_Lmake_1 = start_Lmake.elapsed();

            let test1 = Instant::now();
            let x = Matrix::new_square_with_constant_values(matrix_size, 0.1 as f64);
            let time_test1 = test1.elapsed();
            println!("time test 1 {:?}", time_test1);

            for diagonal_index in 0..matrix_size {
                // if A_matrix[diagonal_index][diagonal_index] == NumCast::from(0).unwrap() {
                //     panic!("ERROR! found zero value at diagonal");
                // }

                // create values under the diagonal of the L matrix
                // similtaniously substract the row if we are here anyway
                for row_number in diagonal_index+1..matrix_size {
                    // if !(A_matrix[row_number][diagonal_index] == NumCast::from(0).unwrap()) { // checking for zero values will significantly reduce time for sparse matrices, add 1/2 * n comparisons for dense.
                        let value_under_diagonal = A_matrix[row_number][diagonal_index] / A_matrix[diagonal_index][diagonal_index];
                        L[row_number][diagonal_index] = value_under_diagonal;
                        A_matrix.substract_multiplied_internal_row_from_row_by_index(diagonal_index, value_under_diagonal, row_number) // upto row_number becuase this is the same as the diagonal, range should also include the diagonal so plus 1
                    // }
                }

                // all values under the pivot should be zero: collumn operations is trivial -> just set values to 0
                // Hypothesis: this is redundadnt
                // for collumn_number in diagonal_index+1..matrix_size {
                //     A_matrix[diagonal_index][collumn_number] = NumCast::from(0).unwrap();
                // }
            }

            let time_Lmake = start_Lmake.elapsed() - time_Lmake_1;

            let start_Ltrans = Instant::now();
            // adding transpose of L onto itself so no second matrix needs to be created
            for row_index in 0..matrix_size-1{
                for column_index in row_index+1..matrix_size {
                    // if !(L[column_index][row_index] == NumCast::from(0).unwrap()){
                    L[row_index][column_index] = L[column_index][row_index];
                    // }
                }
            }
            let time_Ltrans = start_Ltrans.elapsed();
            
            let start_mat1 = Instant::now();
            for collumn_index in 0..matrix_size-1 { // last downwards sweep is trivial
                for row_index in collumn_index+1..matrix_size{
                    let factor = L[collumn_index][row_index];
                    // if !(factor == NumCast::from(0).unwrap()){
                        L.substract_multiplied_internal_row_from_row_by_index_with_collumn_range(collumn_index, factor, row_index, 0..collumn_index+1);
                        B_matrix.substract_multiplied_internal_row_from_row_by_index(collumn_index, factor, row_index);
                    // }

                }
            }
            let time_mat1 = start_mat1.elapsed();

            let start_mat2 = Instant::now();
            // do diagonal part of D (called A here)
            for diagonal_index in 0..matrix_size {
                if !(A_matrix[diagonal_index][diagonal_index] == NumCast::from(0).unwrap()) {
                    B_matrix[diagonal_index].divide_all_elements_by(A_matrix[diagonal_index][diagonal_index]);
                }

            }
            let time_mat2 = start_mat2.elapsed();
            // do upward sweep of L_t

            let start_mat3 = Instant::now();
            let mut coll_range = (1..matrix_size).collect::<Vec<usize>>();
            coll_range.reverse();
                for collumn_index in coll_range {
                    let mut row_range = (0..=collumn_index-1).collect::<Vec<usize>>();
                    row_range.reverse();
                    for row_index in row_range {
                        let factor = L[row_index][collumn_index];                
                        // if !(factor == NumCast::from(0).unwrap()){
                            L.substract_multiplied_internal_row_from_row_by_index_with_collumn_range(collumn_index, factor, row_index, (row_index+1..collumn_index).rev().collect::<Vec<usize>>());
                            B_matrix.substract_multiplied_internal_row_from_row_by_index(collumn_index, factor, row_index);
                        // }

                    }
            }
            let time_mat3 = start_mat3.elapsed();

            println!("time assign: {:?}, time Lmake1: {:?}, time Lmake: {:?},time Ltrans: {:?}, time mat1: {:?}, time mat2: {:?}, time mat3: {:?}", time_assign, time_Lmake_1,time_Lmake, time_Ltrans, time_mat1, time_mat2, time_mat3);

            Solved2D { A_matrix: A_matrix, B_matrix: B_matrix}

            }

    }
        };
}

impl_linear_solve_for_type!(i8);
impl_linear_solve_for_type!(i16);
impl_linear_solve_for_type!(i32);
impl_linear_solve_for_type!(i64);

impl_linear_solve_for_type!(u8);
impl_linear_solve_for_type!(u16);
impl_linear_solve_for_type!(u32);
impl_linear_solve_for_type!(u64);

impl_linear_solve_for_type!(isize);
impl_linear_solve_for_type!(usize);

impl_linear_solve_for_type!(f32);
impl_linear_solve_for_type!(f64);