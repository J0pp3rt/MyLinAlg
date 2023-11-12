#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(unused_imports)]

use core::{f32, panic};
use std::arch::x86_64::{_mm256_set_pd, _mm256_loadu_pd};
use std::ops::{Index, IndexMut, Deref, Add};
use std::any::{TypeId};
use std::sync::Arc;
use std::ops::Range;
use std::fmt::Display;
use std::fmt::Debug;
use std::time::Instant;
use std::vec;
use cudarc::cublas::{safe, result, sys, Gemv, CudaBlas, GemvConfig, Gemm};

use cudarc::driver::{CudaDevice, DevicePtr, CudaSlice, DeviceSlice};
use num::traits::NumOps;
use num::{Num, NumCast};
use rand::Rng;
use rand::distributions::uniform::SampleUniform;

lazy_static!{
    static ref IS_AVX2: bool = is_x86_feature_detected!("avx2");
}

pub fn abs<T: PartialOrd + NumCast + std::ops::Mul<Output = T>>(value: T) -> T {
    if value >= NumCast::from(0).unwrap() {
        return value;
    } else {
        let min_one:T = NumCast::from(-1).unwrap();
        let return_value: T = value * min_one;
        return return_value;
    }
}

pub fn vec_dot_vec<T: MatrixValues>(vec_1: Vec<T>, vec_2: Vec<T>) -> T {
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

    vec_summed(piecewice_mut)
}

pub fn vec_summed<T: MatrixValues>(vec_1: Vec<T>) -> T {
    let mut total: T = NumCast::from(0).unwrap();
    for j in 0..vec_1.len() {
        // if !(vec_1[j] == NumCast::from(0).unwrap()) {
            total = total + vec_1[j];
        // }
    }
    total
}

pub fn row_dot_collumn<T: MatrixValues>(row: & Row<T>, collumn: & Collumn<T>) -> T {
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

    vec_summed(piecewice_mut)

}

pub fn matrix_dot_collumn<T: MatrixValues>(matrix: & Matrix<T>, collumn: & Collumn<T>) -> Collumn<T> {
    if !(matrix.width() == collumn.n_rows()) {
        panic!("Dimension of matrix width must match collumn height. Matrix has dimension [{}x{}], Collumn has [{}x1]", matrix.height(), matrix.width(), collumn.n_rows());
    }

    let mut result_collumn: Collumn<T> = Collumn::new_with_constant_values(matrix.height(), NumCast::from(0).unwrap());

    for row_number in 0..matrix.height() {
        result_collumn[row_number] = row_dot_collumn(& matrix[row_number], & collumn);
    }

    result_collumn
}

// pub fn row_dot_collumn<T: MatrixValues>(row:& Row<T>, collumn:& Collumn<T>) -> T {
//     if 
// }

pub fn matrix_add_matrix<T: MatrixValues>(mut A_matrix: Matrix<T>, B_matrix:& Matrix<T>) -> Matrix<T> {
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

pub fn matrix_dot_matrix<T: MatrixValues>(A_matrix:& Matrix<T>, B_matrix:& Matrix<T>) -> Matrix<T> {
    // this function takes 2 references to the matrices to multiply, returns a new matrix.
    // As in place matrix multiplication is practically impossible, there should be the space for all 3 in memory.

    // this is a slow function because of: many operations and collumn operations in B_matrix.
    // Ideally A_matrix would be row major, B_matrix collumn major
    if !(A_matrix.width() == B_matrix.height()) {
        panic!("Matrices dimensions do not match for dot operation! given: [{}x{}] + [{}x{}]", A_matrix.height(), A_matrix.width(), B_matrix.height(), B_matrix.width());
    }

    let mut C_matrix: Matrix<T> = Matrix::new_with_constant_values(A_matrix.height(), B_matrix.width(), NumCast::from(0).unwrap());

    for row_index in 0..C_matrix.height() {
        for collumn_index in 0..C_matrix.width() {
            C_matrix[row_index][collumn_index] = row_dot_collumn(&A_matrix[row_index], &B_matrix.get_collumn(collumn_index))
        }
    }

    C_matrix
}

pub trait GPUValues: std::marker::Unpin + cudarc::driver::DeviceRepr +cudarc::driver::ValidAsZeroBits + std::default::Default{}
impl<G> GPUValues for G where G: std::marker::Unpin + cudarc::driver::DeviceRepr +cudarc::driver::ValidAsZeroBits + std::default::Default{}

pub fn matrix_dot_collumn_gpu<T: MatrixValues + GPUValues >(matrix: & Matrix<T>, collumn: & Collumn<T>) -> Collumn<T>
    where CudaBlas: Gemv<T>{
        //this function is cool and all, but is always slower then cpu due to massive overheads. Dont use it.
        //rather spend time paralysing the BLAS solver.
    if !(matrix.width() == collumn.n_rows()) {
        panic!("Dimension of matrix width must match collumn height. Matrix has dimension [{}x{}], Collumn has [{}x1]", matrix.height(), matrix.width(), collumn.n_rows());
    }

    let dev: Arc<CudaDevice> = CudaDevice::new(0).unwrap();
    let cublas_handle = CudaBlas::new(dev.clone()).unwrap();

    let result_collumn_length = matrix.height(); 
    
    let matrix_transformed = matrix.to_vec_row_major();
    let collumn_in_vector = collumn.to_vec();

    let gpu_matrix= dev.htod_copy(matrix_transformed).unwrap();
    let gpu_input_collumn = dev.htod_copy(collumn_in_vector).unwrap();
    let mut gpu_result_collumn = dev.alloc_zeros::<T>(result_collumn_length).unwrap();

    unsafe {
        cublas_handle.gemv(
            safe::GemvConfig {
                trans: sys::cublasOperation_t::CUBLAS_OP_T, // by using transpose less time consuming copy can be applied
                m: matrix.width() as i32, 
                n: matrix.height() as i32,
                alpha: NumCast::from(1).unwrap(),
                lda: matrix.width() as i32, // leading dimension: next collumn starts at this position
                incx: 1,
                beta: NumCast::from(0).unwrap(),
                incy: 1,
            }, 
            &gpu_matrix, 
            &gpu_input_collumn, 
            &mut gpu_result_collumn).unwrap();
    } // it aint stopid if it works

    let result_collumn = Collumn::new_form_vec(dev.sync_reclaim(gpu_result_collumn).unwrap());

    result_collumn

}

pub fn matrix_dot_collumn_repeated_gpu_initiator<T: MatrixValues + GPUValues>(matrix: &Matrix<T>, collumn: & Collumn<T>) -> RepeatedMatrixDotVectorGPU<T>{
    RepeatedMatrixDotVectorGPU::new(matrix, collumn)
}

pub struct RepeatedMatrixDotVectorGPU<T: MatrixValues + GPUValues> {
    // computes A x = B, where A the matrix, x the input collumn and B the resulting collumn
    // output is assumed to be the input of the next. This is only if X_(n+1) = B_n. or x_(n+1) = {b_1, b_2, b_3}_n
    // so x_(n+1) = {f_1{b_1}, f_2{b_2}, f_3{b_3}}_n is NOT possible, but with a manually written cuda function running cocurrently on the gpu this should be doable.
    dev: Arc<CudaDevice>,
    cublas: CudaBlas,
    gpu_matrix: CudaSlice<T>,
    gpu_input_collumn: CudaSlice<T>,
    gpu_result_collumn: CudaSlice<T>,
    m: usize,
    n: usize,
}

impl<T: MatrixValues + GPUValues> RepeatedMatrixDotVectorGPU<T> {
    pub fn new(matrix: &Matrix<T>, collumn: & Collumn<T>) -> Self {
        let dev: Arc<CudaDevice> = CudaDevice::new(0).unwrap();
        let cublas = CudaBlas::new(dev.clone()).unwrap();

        let n = matrix.height();
        let m = matrix.width();
    
        let result_collumn_length = matrix.height(); 
        
        let matrix_transformed = matrix.to_vec_row_major();
        let collumn_in_vector = collumn.to_vec();
    
        let gpu_matrix= dev.htod_copy(matrix_transformed).unwrap();
        let gpu_input_collumn = dev.htod_copy(collumn_in_vector).unwrap();
        let mut gpu_result_collumn = dev.alloc_zeros::<T>(result_collumn_length).unwrap();

        RepeatedMatrixDotVectorGPU {
            dev,
            cublas,
            gpu_matrix,
            gpu_input_collumn,
            gpu_result_collumn,
            m,
            n,
        }
    }

    pub fn result_vec_length(&self) -> usize {
        self.n
    }

    pub fn next(&mut self) -> &mut Self 
    where CudaBlas: Gemv<T>{
        // does one itteration
        unsafe {
            self.cublas.gemv(
                safe::GemvConfig {
                    trans: sys::cublasOperation_t::CUBLAS_OP_T, // by using transpose less time consuming copy can be applied
                    m: self.m as i32, 
                    n: self.n as i32,
                    alpha: NumCast::from(1).unwrap(),
                    lda: self.m as i32, // leading dimension: next collumn starts at this position
                    incx: 1,
                    beta: NumCast::from(0).unwrap(),
                    incy: 1,
                }, 
                &self.gpu_matrix, 
                &self.gpu_input_collumn, 
                &mut self.gpu_result_collumn).unwrap();
        } // it aint stopid if it works

        self
    }

    pub fn itterate(&mut self, n_itterations: usize) -> &mut Self 
    where CudaBlas: Gemv<T>{
        // does multiple itterations withouth passing results

        // TODO! creation of repeated Gemv probably slows things down -> put more time into getting this thing out.
        // TODO! CPU at 100% while evaluation (also gpu). CPU not expected, probably that after every (very quick) itteration, slow GPU communication needs to be called.
        // -> therefore better to write the forloop in CUDA directly. It will take handles to the data and just do its own thing. GOOD TO HAVE: make (an alternate) function able 
        // to take functions for translating the previous results to the inputs of the current simulation.
        for itter in 0..n_itterations {
            unsafe {
                self.cublas.gemv(
                    safe::GemvConfig {
                        trans: sys::cublasOperation_t::CUBLAS_OP_T, // by using transpose less time consuming copy can be applied
                        m: self.m as i32, 
                        n: self.n as i32,
                        alpha: NumCast::from(1).unwrap(),
                        lda: self.m as i32, // leading dimension: next collumn starts at this position
                        incx: 1,
                        beta: NumCast::from(0).unwrap(),
                        incy: 1,
                    }, 
                    &self.gpu_matrix, 
                    &self.gpu_input_collumn, 
                    &mut self.gpu_result_collumn).unwrap();
            } // it aint stopid if it works
            self.dev.dtod_copy(&self.gpu_result_collumn, &mut self.gpu_input_collumn).unwrap();
        }
        self
    }

    pub fn take_result(&mut self) -> Collumn<T> {
        Collumn::new_form_vec(self.dev.sync_reclaim(self.gpu_result_collumn.to_owned()).unwrap())
    }

    pub fn copy_result(& self) -> Collumn<T> {
        Collumn::new_form_vec(self.dev.dtoh_sync_copy(&self.gpu_result_collumn).unwrap())
    }
}

pub fn matrix_dot_matrix_gpu<T: MatrixValues + GPUValues>(A_matrix:& Matrix<T>, B_matrix:& Matrix<T>) -> Matrix<T>
where CudaBlas: Gemm<T>{
    // this function takes 2 references to the matrices to multiply, returns a new matrix.
    // As in place matrix multiplication is practically impossible, there should be the space for all 3 in memory.

    // this is a slow function because of: many operations and collumn operations in B_matrix.
    // Ideally A_matrix would be row major, B_matrix collumn major
    if !(A_matrix.width() == B_matrix.height()) {
        panic!("Matrices dimensions do not match for dot operation! given: [{}x{}] + [{}x{}]", A_matrix.height(), A_matrix.width(), B_matrix.height(), B_matrix.width());
    }

    let start_prep = Instant::now();
    let dev: Arc<CudaDevice> = CudaDevice::new(0).unwrap();
    let cublas_handle = CudaBlas::new(dev.clone()).unwrap();

    let result_matrix_height = A_matrix.height();
    let result_matrix_width = B_matrix.width();

    // transforming and transferring matricises in turn so less peak memory is needed.
    let A_matrix_vec = A_matrix.to_vec_row_major();
    let gpu_A_matrix = dev.htod_copy(A_matrix_vec).unwrap();
    let B_matrix_vec = B_matrix.to_vec_row_major();
    let gpu_B_matrix = dev.htod_copy(B_matrix_vec).unwrap();
    let mut gpu_result_matrix = dev.alloc_zeros::<T>(result_matrix_height * result_matrix_width).unwrap();
    let time_prep = start_prep.elapsed();
    let start_calc = Instant::now();
    unsafe{
        cublas_handle.gemm(
            safe::GemmConfig { 
                transa: sys::cublasOperation_t::CUBLAS_OP_T, 
                transb: sys::cublasOperation_t::CUBLAS_OP_T, 
                m: A_matrix.height() as i32, 
                n: B_matrix.width() as i32, 
                k: A_matrix.width() as i32, 
                alpha: NumCast::from(1).unwrap(), 
                lda: A_matrix.width() as i32, 
                ldb: B_matrix.width() as i32, 
                beta: NumCast::from(0).unwrap(), 
                ldc: A_matrix.height()  as i32}, 
            &gpu_A_matrix, 
            &gpu_B_matrix, 
            &mut gpu_result_matrix).unwrap();
    }
    // dev.synchronize();

    let time_calc = start_calc.elapsed();

    let start_reclaim = Instant::now();
    let reclaim = dev.sync_reclaim(gpu_result_matrix).unwrap();
    let time_reclaim = start_reclaim.elapsed();

    let start_matrix = Instant::now();

    let mut result_matrix = Matrix::new_from_row_major_vector(
        reclaim, 
        result_matrix_width, 
        result_matrix_height
    );

    let time_matrix = start_matrix.elapsed();
    let start_transpose = Instant::now();

    result_matrix.transpose_non_skinny();

    let time_transpose = start_transpose.elapsed();

    println!("prep: {:?}, calc: {:?}, reclaim: {:?}, Matrix: {:?}, transpose: {:?}", time_prep, time_calc, time_reclaim,time_matrix,  time_transpose);
    
    result_matrix
}


struct MatrixDotCollumnSystem<T> {
    gemv_config: GemvConfig<T>, 
    gpu_matrix: CudaSlice<T>, 
    gpu_input_collumn: CudaSlice<T>, 
    gpu_result_collumn: CudaSlice<T>
}

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
// Gaus seems better for dense matrices, CholeskySquare root better for sparser / diagnal matrices.
// some further optimizations possible by making the cholesky more for row for collum then other way around (use row major and all), then likely faster then gaus in the dense ones.
// maybe be smarter around the whole transpose part -> not actually making a transpose but just playing with indices to make it work. 
// otherwise making Matrix either collumn major or row major could help.
// Larger performance gain likely when mutli threading is introduced, especially for larger matrices.
// not sure what LU decomposition would add (maybe support for generic matrices), 
// normal Cholesky deomposition might be advantages as only 2 matrices are involved instead of 1.
// also find moments when Cholesky(root free) is applicable and maybe add in some tests/checks (maybe row swapping needed?: https://math.stackexchange.com/questions/4504209/when-does-a-real-symmetric-matrix-have-ldlt-decomposition-and-when-is-the
// look around in gaus whether there are more optimizations.

impl<T:MatrixValues> Add for Matrix<T> {
    type Output = Self;

    fn add(mut self, other_matrix: Self) -> Self::Output {
        if !(self.height() == other_matrix.height() || self.width() == other_matrix.width()) {
            panic!("Dimensions do not match for addition! Given [{}x{}] + [{}+{}]", self.height(), self.width(), other_matrix.height(), other_matrix.width());
        }

        self.addition_on_partial(0..self.height(), 0..self.width(), other_matrix);

        self
    }
}

pub struct Matrix<T: MatrixValues> {
    pub rows : Vec<Row<T>>
}

pub trait MatrixValues: Copy + Display + PartialEq + Num + NumCast{
}
impl<T> MatrixValues for T where T: Copy + Display + PartialEq + Num + NumCast {
}

impl<T: MatrixValues> Matrix<T> {
    pub fn new_square_with_constant_values(n_rows:usize, value: T) -> Matrix<T> {
        let mut rows = Vec::<Row<T>>::with_capacity(n_rows);
        for _ in 0..n_rows {
            rows.push(Row::new_row_with_value(n_rows, value));
        } 
        Matrix {
             rows,
            }
    }

    pub fn new_with_constant_values(n_rows:usize, n_collumns: usize, value: T) -> Matrix<T> {
        let mut rows = Vec::<Row<T>>::with_capacity(n_rows);
        for _ in 0..n_rows {
            rows.push(Row::new_row_with_value(n_collumns, value));
        } 
        Matrix {
             rows,
            }
    }

    pub fn new_from_vector_rows(input: Vec<Vec<T>>) -> Matrix<T> {
        let mut rows = Vec::<Row<T>>::with_capacity(input.len());
        for dimension in input{
            rows.push( Row { cells : dimension});
        }
        Matrix { rows }
    }

    pub fn new_from_collumn(input_collumn: Collumn<T>) -> Matrix<T>{
        let mut rows = Vec::<Row<T>>::with_capacity(input_collumn.n_rows());
        for row_number in 0..input_collumn.n_rows(){
            rows.push(Row { cells: vec![input_collumn[row_number]] })
        }

        Matrix { rows }
    }

    pub fn new_square_eye(size: usize, values_of_diagonal: T) -> Matrix<T> {
        println!("creating fixed ");
        let mut new_matrix: Matrix<T> = Matrix::new_square_with_constant_values(size, NumCast::from(0).unwrap());
        for j in 0..size {
            new_matrix[j][j] = values_of_diagonal;
        }
        new_matrix
    }

    pub fn new_eye(n_rows: usize, n_collumns: usize, values_of_diagonal: T) -> Matrix<T> {
        let mut new_matrix: Matrix<T> = Matrix::new_with_constant_values(n_rows, n_collumns, NumCast::from(0).unwrap());
        
        let smallest_value: usize;
        if n_rows < n_collumns{
            smallest_value = n_rows;
        } else {
            smallest_value = n_collumns;
        }

        for j in 0..smallest_value {
            new_matrix[j][j] = values_of_diagonal;
        }
        new_matrix
    }
    
    pub fn new_from_row_major_vector(mut vector: Vec<T>, height: usize, width: usize) -> Self {
        if !(vector.len() == height*width) {
            panic!("Given dimensions do not match! Vec length: {}, height x width = {} x {} = {}", vector.len(), height, width, height*width);
        }

        let mut result_matrix: Matrix<T> = Matrix::new_with_constant_values(height, width, NumCast::from(0).unwrap());

        for row_index in 0..height {
            let row = Row::new_row_from_vec(
                vector[row_index*width..(row_index+1)*width].to_owned()
            );
            result_matrix[row_index] = row;
        }

        result_matrix
    }

    pub fn clone(&self) -> Matrix<T> {
        let mut rows = Vec::<Row<T>>::with_capacity(self.rows.len());
        for row in &self.rows{
            rows.push( row.clone());
        }
        Matrix { rows }
    }

    pub fn len(&self) -> usize {
        if self.row_length() >= self.coll_length() {
            self.row_length()
        } else {
            self.coll_length()
        }
    }

    pub fn row_length(&self) -> usize {
        let mut largest_length = self.rows[0].len();
        for row in &self.rows {
            if row.len() > largest_length {
                largest_length = row.len();
                println!("Row lengths of matrix not consistant!")
            }
        }
        largest_length
    }

    pub fn diagonal_contain_zeros(&self) -> bool {
        let smallest_dimension: usize;
        if self.height() < self.width() {
            smallest_dimension = self.height();
        } else {
            smallest_dimension = self.width();
        }

        let mut zero_values_found = false;
        for j in 0..smallest_dimension {
            if self[j][j] == NumCast::from(0).unwrap() {
                zero_values_found = true
            }
        }

        zero_values_found
    }

    pub fn new_with_random_ones_chance(height: usize, width: usize, chance: f64, base_value: T) -> Self {
        let mut result_matrix: Matrix<T> = Matrix::new_with_constant_values(height, width, base_value);
        let mut rng = rand::thread_rng();
        for row_index in 0..height {
            for collumn_index in 0..width {
                let random_value  = rng.gen_range(0.0..1.0);
                
                if random_value < chance {
                    result_matrix[row_index][collumn_index] = NumCast::from(1).unwrap();
                } else {
                    result_matrix[row_index][collumn_index] = NumCast::from(0).unwrap();
                }
            }
        }

        result_matrix
    }

    pub fn remove_last_row(&mut self) -> &Self {
        self.rows.pop().unwrap();

        self
    }

    pub fn remove_last_collumn(&mut self) -> &Self {
        for row_index in 0..self.height() {
            self[row_index].cells.pop();
        }

        self
    }

    pub fn width(&self) -> usize {
        self.row_length()
    }

    pub fn coll_length(&self) -> usize {
        self.rows.len()
    }

    pub fn height(&self) -> usize {
        self.coll_length()
    }

    pub fn is_square(&self) -> bool {
        self.height() == self.width()
    }

    pub fn get_collumn(&self, coll_number : usize) -> Collumn<T> {
        let mut cells = Vec::<T>::with_capacity(self.coll_length());
        for row_number in 0..self.coll_length(){
            cells.push( self[row_number][coll_number])
        }
        Collumn {cells}
    }

    pub fn to_vec_collumn_major(&self) -> Vec<T> {
        let N = self.width()*self.height();
        let mut output_vec = Vec::<T>::with_capacity(N);

        for column_number in 0..self.width() {
            for row_number in 0..self.height() {
                output_vec.push(self[row_number][column_number]);
            }
        }

        output_vec
    }

    pub fn to_vec_row_major(&self) -> Vec<T> {
        let N = self.width()*self.height();
        let mut output_vec = Vec::<T>::with_capacity(N);

        for row_number in 0..self.height() {
            output_vec.extend(self[row_number].cells.clone());
        }

        output_vec
    }

    pub fn swap_rows(&mut self, row_1: usize, row_2: usize){
        let row_1_copy = self[row_1].clone();
        
        self[row_1] = self[row_2].clone();
        self[row_2] = row_1_copy;
    }

    pub fn substract_internal_row_from_row_by_index(&mut self, row_number_to_substract: usize, from_row_number: usize) {
        let row_to_substract = self[row_number_to_substract].clone();
        self[from_row_number].substract_row(row_to_substract)
    }

    pub fn substract_multiplied_internal_row_from_row_by_index(&mut self, row_number_to_substract_with: usize, factor: T , from_row_number: usize) {
        let mut mutliplied_row_to_substract = self[row_number_to_substract_with].clone();
        mutliplied_row_to_substract.multiply_all_elements_by(factor);
        self[from_row_number].substract_row(mutliplied_row_to_substract)
    }

    pub fn substract_multiplied_internal_row_from_row_by_index_with_collumn_range<U>(&mut self, row_number_to_substract_with: usize, factor: T , from_row_number: usize, collumn_range: U)
    where  U: InputTraitRowCol<U> {
        let colls_input:Vec<usize> = parse_dimension_input(collumn_range);

        for collumn_index in colls_input.iter(){
            self[from_row_number][*collumn_index] = self[from_row_number][*collumn_index] - self[row_number_to_substract_with][*collumn_index] * factor
        }  
    }

    pub fn new_from_solver(system_of_equations : Solver2D<T>) -> Solved2D<T>{
        match system_of_equations.solver {
            Solver2DStrategy::Guass => solve_with_guass(system_of_equations),
            Solver2DStrategy::CholeskySquareRootFreeDecomposition => solve_with_cholesky_quare_root_free_decomposition(system_of_equations),
            // Solver2DStrategy::LUDecomposition => todo!(),
            _ => (panic!("Error: Solver not yet implemented!"))
        }
    }

    pub fn transpose_square(&mut self) -> &Self {
        for row_index in 0..self.height()-1 {
            for collumn_index in row_index+1..self.width() {
                let buffer = self[collumn_index][row_index];
                self[collumn_index][row_index] = self[row_index][collumn_index];
                self[row_index][collumn_index] = buffer;
            }
        }
        self
    }

    pub fn transpose_non_skinny(&mut self) -> &Self {
        let initial_height = self.height();
        let initial_width = self.width();
        if initial_height == initial_width {
            return self.transpose_square();
        }

        let smallest_dimension_was_height: bool;
        let dimension_difference: usize = abs(initial_height as isize- initial_width as isize) as usize;
        if initial_height < initial_width {
            smallest_dimension_was_height = true;

        } else {
            smallest_dimension_was_height = false;
        }


        self.make_square();
        self.transpose_square();


        if smallest_dimension_was_height {
            for _ in 0..dimension_difference {
                self.remove_last_collumn();
            }
        } else {
            for _ in 0..dimension_difference {
                self.remove_last_row();
            }
        }

        self
    }

    fn make_square(&mut self) -> &Self {
        let initial_height = self.height();
        let initial_width = self.width();

        let dimension_difference = abs(initial_height as isize - initial_width as isize) as usize;
        if initial_height == initial_width {
            return  self;
        } else if initial_height < initial_width {
            for _ in 0..dimension_difference {
                self.append_row_zeros();
            }
        } else {
            for _ in 0..dimension_difference {
                self.append_collumn_zeros();
            }
        }

        self
    }

    pub fn add_row(&mut self, insert_row_at: usize, new_row: Row<T>) {
        if insert_row_at == self.rows.len() {
            self.append_row(new_row);
        } else {
            self.rows.insert(insert_row_at, new_row)
        }
    }

    pub fn add_row_from_vec(&mut self, insert_row_at: usize, new_row_vec: Vec::<T>) {
        let new_row = Row::new_row_from_vec(new_row_vec);
        self.add_row(insert_row_at, new_row);
    }

    pub fn append_row(&mut self, new_row: Row<T>) {
            self.rows.push(new_row);
    }

    pub fn append_row_zeros(&mut self) {
        let width = self.width();
        self.rows.push( Row::new_row_with_constant_values(width, NumCast::from(0).unwrap()));
    }

    pub fn append_collumn_zeros(&mut self) {
        let height = self.height();
        let zero: T = NumCast::from(0).unwrap();
        let collumn= Collumn::new_form_vec((0..height).into_iter().map(|x| zero).collect::<Vec<T>>());
        self.append_collumn(collumn);
    }

    pub fn append_collumn_zeros_n_times(&mut self, n: usize) {
        let height = self.height();
        let zero: T = NumCast::from(0).unwrap();
        let collumn= Collumn::new_form_vec((0..height).into_iter().map(|x| zero).collect::<Vec<T>>());
        for _ in 0..n{
            self.append_collumn(collumn.clone());
        }
    }

    pub fn append_collumn(&mut self, collumn: Collumn<T>) -> &Self {
        if !(self.height() == collumn.height()) {
            panic!("Collumn dimensions do not match, given collumn is {} long, current matrix is {} long.", collumn.height(), self.height())
        }

        for row_index in 0..self.height() {
            self[row_index].cells.push(collumn[row_index]);
        }

        self
    }

    pub fn append_row_from_vec(&mut self, new_row_vec: Vec<T>) {
        let new_row = Row::new_row_from_vec(new_row_vec);
        self.append_row(new_row);
    }

    pub fn multiply_all_elements_by(&mut self, factor: T) -> &Self {
        for row_number in 0..self.rows.len() {
            self.rows[row_number].multiply_all_elements_by(factor);
        }

        self
    }

    pub fn divide_all_elements_by(&mut self, factor: T) {
        for row_number in 0..self.rows.len() {
            self.rows[row_number].divide_all_elements_by(factor)
        }
    }

}

fn solve_with_guass<T: MatrixValues>(system_of_equations: Solver2D<T>) -> Solved2D<T> {
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

fn solve_with_cholesky_quare_root_free_decomposition<T: MatrixValues>(mut system_of_equations: Solver2D<T>) -> Solved2D<T> {
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

    let mut L: Matrix<T> = Matrix::new_square_eye(matrix_size, NumCast::from(1).unwrap());
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
    // let mut Lt = L.clone();
    // let Lt: () = L.transpose_square();

    // let sol_1 = Matrix::new_from_solver(Solver2D { A_matrix: L.clone() , B_matrix: B_matrix.clone(), solver: Solver2DStrategy::Guass });

    // let mut D: Matrix<T> = Matrix::new_square_eye(matrix_size, NumCast::from(1).unwrap());
    // for diag_index in 0..matrix_size {
    //     D[diag_index][diag_index] = A_matrix[diag_index][diag_index];
    // }

    // let sol_2 = Matrix::new_from_solver(Solver2D { A_matrix: D, B_matrix: sol_1.B_matrix, solver: Solver2DStrategy::Guass });

    // L.transpose_square();

    // let sol_3 = Matrix::new_from_solver(Solver2D { A_matrix: L, B_matrix: sol_2.B_matrix, solver: Solver2DStrategy::Guass });
    // sol_3
    
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

impl<T: MatrixValues> Matrix<T>{
    pub fn printer(&self) {
        println!("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
        println!("Printing the Matrix");
        println!("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
        for row_number in 0..self.rows.len(){
            print!("row {} :", row_number);
            for coll in 0..self.rows[row_number].cells.len(){
                print!(" {:.5} ", self.rows[row_number].cells[coll])
            }
            println!("\n")
        }
    }
}

pub trait InputTraitRowCol<T> {
    fn parse_input(&self) -> Vec<usize> {
        panic!()
    }
}

impl<T> InputTraitRowCol<T> for Range<i32> {
    fn parse_input(&self) -> Vec<usize> {
        let mut output_vec = Vec::<usize>::with_capacity(self.len());
        for index in self.clone() {
            output_vec.push(index as usize)
        }
        output_vec
    }
}

impl<T> InputTraitRowCol<T> for Range<usize> {
    fn parse_input(&self) -> Vec<usize> {
        let mut output_vec = Vec::<usize>::with_capacity(self.len());
        for index in self.clone() {
            output_vec.push(index)
        }
        output_vec
    }
}

impl<T> InputTraitRowCol<T> for Vec<usize> {
    fn parse_input(&self) -> Vec<usize> 
    where 
    {
        self.clone()
    }
}

impl<T> InputTraitRowCol<T> for i32 {
    fn parse_input(&self) -> Vec<usize> {
        vec![self.clone() as usize]
    }
}

impl<T> InputTraitRowCol<T> for usize {
    fn parse_input(&self) -> Vec<usize> {
        vec![self.clone()]
    }
}

fn parse_dimension_input<T: InputTraitRowCol<T>>(input: T) -> Vec<usize> {
    input.parse_input()
}

pub trait InputMatrix<W, T: MatrixValues> { // can not be realisticly implemeted for lists because length of list must be hardcoded -> just use vecs :(
    fn parse_input(&self) -> Matrix<T> {
        panic!()
    }
}

impl<W, T: MatrixValues> InputMatrix<W, T> for Vec<Vec<T>> {
    fn parse_input(&self) -> Matrix<T> {
        let mut rows = Vec::<Row<T>>::with_capacity(self.len());
        for row in self{
            rows.push( Row { cells : row.to_vec()});
        }
        Matrix { rows}
    }
}

impl<W, T: MatrixValues> InputMatrix<W, T> for Vec<T> {
    fn parse_input(&self) -> Matrix<T> {
            let vac = self.to_vec();
            let row = vec!{Row{ cells : vac}};

            Matrix { rows: row}
        
    }
}

impl<W, T: MatrixValues> InputMatrix<W, T> for &Vec<T> {
    fn parse_input(&self) -> Matrix<T> {
            let row = vec!{Row{ cells : self.to_vec()}};

            Matrix { rows: row}
        
    }
}

impl<W, T: MatrixValues> InputMatrix<W, T> for Matrix<T> {
    fn parse_input(&self) -> Matrix<T> {
            self.clone()
    }
}

impl<W, T: MatrixValues> InputMatrix<W, T> for Collumn<T> {
    fn parse_input(&self) -> Matrix<T> {
            let contents_clone = Collumn::new_form_vec(self.to_vec());
            Matrix::new_from_collumn(contents_clone)
            // TODO: THIS IS A GROSS SOLUTION BUT THE COMPILER IS HAPPY IG
        
    }
}

fn parse_matrix_input<W: InputMatrix<W, T>,T: MatrixValues>(input: W) -> Matrix<T> {
    input.parse_input()
}

impl<T: MatrixValues> Matrix<T> {
    pub fn update<U, V, W>(&mut self, rows: U, colls: V, new_values: W) 
    where  U: InputTraitRowCol<U>,
    V: InputTraitRowCol<V>,
    W: InputMatrix<W, T>
    {
        let rows_input:Vec<usize> = parse_dimension_input(rows);
        let colls_input:Vec<usize> = parse_dimension_input(colls);
        let matrix_input:Matrix<T> = parse_matrix_input(new_values);

        // checking whether input matrix matches given dimensions:

        if !(rows_input.len() == matrix_input.coll_length()) {
            println!("Error: Given row range does not match given values ");
            println!("Specified length = {:?}, given matrix is = {:?}", rows_input.len(), matrix_input.coll_length());
            panic!()
        } else if !(colls_input.len() == matrix_input.row_length()) {
            println!("Error: Given collumn range does not match given values ");
            panic!()
        }

        let mut i : usize = 0;
        let mut j : usize = 0;
        for row in &rows_input {
            j = 0; // seems like i missed this, not 100% sure.
            for coll in &colls_input {
                self[*row][*coll] = matrix_input[i][j];
                j += 1;
            }
            i += 1;
        }
    }

    pub fn addition_on_partial<U, V, W>(&mut self, rows: U, colls: V, new_values: W) 
    where  U: InputTraitRowCol<U>,
    V: InputTraitRowCol<V>,
    W: InputMatrix<W, T>
    {
        let rows_input:Vec<usize> = parse_dimension_input(rows);
        let colls_input:Vec<usize> = parse_dimension_input(colls);
        let matrix_input:Matrix<T> = parse_matrix_input(new_values);

        // checking whether input matrix matches given dimensions:

        if !(rows_input.len() == matrix_input.coll_length()) {
            println!("Error: Given row range does not match given values ");
            println!("Specified length = {:?}, given matrix is = {:?}", rows_input.len(), matrix_input.coll_length());
            panic!()
        } else if !(colls_input.len() == matrix_input.row_length()) {
            println!("Error: Given collumn range does not match given values ");
            panic!()
        }

        let mut i : usize = 0;
        let mut j : usize = 0;
        for row in &rows_input {
            j = 0; // seems like i missed this, not 100% sure.
            for coll in &colls_input {
                self[*row][*coll] = self[*row][*coll] + matrix_input[i][j];
                j += 1;
            }
            i += 1;
        }
    }

    pub fn addition_total(&mut self, other_matrix: &Matrix<T>) -> &Self {
        if !(self.height() == other_matrix.height() || self.width() == other_matrix.width()) {
            panic!("Dimensions do not match for addition! Given [{}x{}] + [{}+{}]", self.height(), self.width(), other_matrix.height(), other_matrix.width());
        }

        for row_index in 0..self.height() {
            self[row_index].addition_row_with_external_row(&other_matrix[row_index]);
        }

        self
    }

    pub fn multiply_selected_values_with_factor<U, V>(&mut self, rows: U, colls: V, factor: T)
    where  U: InputTraitRowCol<U>,
    V: InputTraitRowCol<V>, 
    {
        let rows_input:Vec<usize> = parse_dimension_input(rows);
        let colls_input:Vec<usize> = parse_dimension_input(colls);

        for row in &rows_input {
            for coll in &colls_input {
                self[*row][*coll] = self[*row][*coll] * factor;
            }
        }
    }
}

impl<T: MatrixValues> Index<usize> for Matrix< T> {
    type Output = Row<T>;
    fn index(&self, index: usize) -> &Self::Output {
        &self.rows[index]
    }
}

impl<T: MatrixValues> IndexMut<usize> for Matrix< T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.rows[index]
    }
}

#[derive(Debug)]
pub struct Row<T: MatrixValues> {
    pub cells : Vec<T>,
}

impl<T: MatrixValues> Row<T> {
    pub fn new_row_with_value(size: usize, value: T) -> Row<T> {
        let mut cells = Vec::<T>::with_capacity(size);
        for _ in 0..size{
            cells.push(value);
        }
        Row { cells}
    }

    pub fn to_vec(&self) -> Vec<T> {
        self.cells.clone()
    }
    
    pub fn new_row_from_vec(input_vec: Vec<T>) -> Row<T> {
        Row { cells: input_vec }
    }

    pub fn new_row_with_constant_values(width: usize, value: T) -> Row<T> {
        return Row::new_row_with_value(width, value)
    }

    pub fn len(&self) -> usize {
        self.cells.len()
    }

    pub fn export(self) -> Row<T> {
        self
    }

    pub fn clone(&self) -> Row<T> {
        let mut cells = Vec::<T>::with_capacity(self.len());
        for cell in &self.cells{
            cells.push(cell.clone());
        }
        Row { cells}
    }

    pub fn divide_all_elements_by(&mut self, value: T) {
        for n in 0..self.cells.len() {
            // if !(self.cells[n] == NumCast::from(0).unwrap()) {// quickly tested on some sparse matrices but seem to really boost performance. In some more filled ones: around 50x improvemnt, ful matrix not tested yet
                self.cells[n] = self.cells[n] / value;
            // }
        }
    }

    pub fn multiply_all_elements_by(&mut self, value: T) -> &Self{
        for n in 0..self.cells.len() {
            // if !(self.cells[n] == NumCast::from(0).unwrap()) {// quickly tested on some sparse matrices but seem to really boost performance. In some more filled ones: around 50x improvemnt, ful matrix not tested yet
            self.cells[n] = self.cells[n] * value;
            // }
        }

        self
    }

    pub fn addition_row_with_external_row(&mut self, row_to_add_to_this_one:& Row<T>) {
        for n in 0..self.cells.len() {
            // if !(self.cells[n] == NumCast::from(0).unwrap() && row_to_add_to_this_one.cells[n] == NumCast::from(0).unwrap()) {
                self.cells[n] = self.cells[n] + row_to_add_to_this_one[n];
            // }
        }
    }

    

    pub fn normalize_all_elements_to_element(&mut self, index: usize) {
        self.divide_all_elements_by(self[index]);
    }

    pub fn normalize_all_elements_to_first(&mut self) {
        self.normalize_all_elements_to_element(0);
    }

    pub fn substract_row(&mut self, substraction_row: Row<T>) {
        if !(self.cells.len() == substraction_row.cells.len()) {
            panic!("Error: Length of substracting row is not equal to row length")
        }
        if *IS_AVX2{
            unsafe {self.substract_avx2(substraction_row)}
        } else {
            self.substract_all(substraction_row)
        }


    }

    // #[target_feature(enable = "avx2")]
    // unsafe fn substract_avx2(&mut self, substraction_row: Row<T>) {
    //     for cell_number in 0..self.cells.len() {
    //         // if !(self[cell_number] == NumCast::from(0).unwrap() || substraction_row[cell_number] == NumCast::from(0).unwrap()) { // quickly tested on some sparse matrices but seem to really boost performance . In some more filled ones: around 50x improvemnt, ful matrix not tested yet
    //         self[cell_number] = self[cell_number] - substraction_row[cell_number];
    //         // } 
    //         assert!(substraction_row.len() % 4 == 0);
    //         let mut A_row_base = self.cells.as_ptr();
    //         let mut B_row_base = substraction_row.cells.as_ptr();
    //         for _ in 0..self.len() /4{
    //             let A_row = _mm256_loadu_pd(A_row_base);
    //         }
    //     }
    // }

    fn substract_all(&mut self, substraction_row: Row<T>) {
        for cell_number in 0..self.cells.len() {
            // if !(self[cell_number] == NumCast::from(0).unwrap() || substraction_row[cell_number] == NumCast::from(0).unwrap()) { // quickly tested on some sparse matrices but seem to really boost performance . In some more filled ones: around 50x improvemnt, ful matrix not tested yet
            self[cell_number] = self[cell_number] - substraction_row[cell_number];
            // }
        }
    }

    pub fn replace_values(&mut self, index_range: Range<usize>, values: Vec<T>) {
        // maybe add a check if not of same size TODO
        // instead of range use rangebounds apperantly
        for (val_index, row_index) in index_range.enumerate() {
            self.cells[row_index] = values[val_index];
        }
    }

}

fn substract_row_with_row<U: AVX2Row<T>, T:MatrixValues>(mut row_a: U, row_b: U){
    row_a.substraction_f64(row_b)
}

pub trait AVX2Row<U: AVX2Row<U, T>, T:MatrixValues> {
    fn substraction_f64(&mut self, substraction_row: Row<f64>) {
        panic!("this should not be able to run")
    }
}

impl<T:MatrixValues> AVX2Row<T> for Row<f64> {
    fn substraction_f64(&mut self, substraction_row: Row<f64>) {
        unsafe {substract_avx2_f64(self, substraction_row)}
    }
}

#[target_feature(enable = "avx2")]
unsafe fn substract_avx2_f64(a_row: &mut Row<f64>, substraction_row: Row<f64>) {
    for cell_number in 0..a_row.cells.len() {
        // if !(self[cell_number] == NumCast::from(0).unwrap() || substraction_row[cell_number] == NumCast::from(0).unwrap()) { // quickly tested on some sparse matrices but seem to really boost performance . In some more filled ones: around 50x improvemnt, ful matrix not tested yet
        a_row[cell_number] = a_row[cell_number] - substraction_row[cell_number];
        // } 
        assert!(substraction_row.len() % 4 == 0);
        let mut A_row_base = a_row.cells.as_ptr();
        let mut B_row_base = substraction_row.cells.as_ptr();
        for _ in 0..a_row.len() /4{
            let A_row = _mm256_loadu_pd(A_row_base);
        }
    }
}

impl<T: MatrixValues> Index<usize> for Row<T> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        &self.cells[index]
    }
}

impl<T: MatrixValues> IndexMut<usize> for Row<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.cells[index]
    }
}

impl<T: MatrixValues> Index<Range<usize>> for Row<T> {
    type Output = [T];
    fn index(&self, index: Range<usize>) -> &Self::Output {
        // index.into_iter().map(|i| &self.cells[i]).collect()
        &self.cells[index]
    }
}

impl<T: MatrixValues> IndexMut<Range<usize>> for Row<T> {
    fn index_mut(&mut self, index: Range<usize>) -> &mut Self::Output {
        &mut self.cells[index]
    }
}

pub struct Collumn<T: MatrixValues> {
    pub cells : Vec<T>
}

impl<T: MatrixValues> Index<usize> for Collumn<T> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        &self.cells[index]
    }
}

impl<T: MatrixValues> IndexMut<usize> for Collumn<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.cells[index]
    }
}

impl<T: MatrixValues> Collumn<T> {
    pub fn n_rows(&self) -> usize {
        self.cells.len()
    }

    pub fn len(&self) -> usize {
        self.cells.len()
    }

    pub fn height(&self) -> usize {
        self.cells.len()
    }

    pub fn export(self) -> Collumn<T> {
        self
    }

    pub fn clone(&self) -> Collumn<T> {
        let mut cells = Vec::<T>::with_capacity(self.len());
        for cell in &self.cells{
            cells.push(cell.clone());
        }
        Collumn { cells}
    }

    pub fn new_with_constant_values(length: usize, value: T) -> Collumn<T> {
        let cells: Vec<T> = (0..length).into_iter().map(|_| value).collect();

        Collumn { cells }
    }

    pub fn new_form_vec(input_vector: Vec<T>) -> Collumn<T>{
        let mut cells = Vec::<T>::with_capacity(input_vector.len());
        for index in 0..input_vector.len() {
            cells.push(input_vector[index])
        }

        Collumn {cells}
    }

    pub fn to_vec(&self) -> Vec<T> {
        let mut output_vec = Vec::<T>::with_capacity(self.cells.len());
        for i in 0..self.cells.len() {
            output_vec.push(self.cells[i]);
        }
        output_vec
    }

    pub fn extend_with_collumn(&mut self, other_collumn: Collumn<T>) {
        self.cells.extend(other_collumn.cells);
    }

}


pub fn linspace<T: MatrixValues>(lower_bound: T, higher_bound: T, steps: usize) -> Vec<T> {
    let step_size = (higher_bound - lower_bound) / (NumCast::from(steps-1).unwrap());
    let mut lin_spaced_vec = Vec::<T>::with_capacity(steps);
    for i in 0..steps {
        lin_spaced_vec.push(lower_bound + step_size * NumCast::from(i).unwrap());
    }
    lin_spaced_vec
}

// a first attempt to do a matrix that can hold any struct that implements Copy. Usefull as Matrix struct is not cabable of holding a 
// non standard generics. Very limited in higher level functionality 
// name subject to change
#[derive(Debug)]
pub struct SuperMatrix<U: Clone + Debug> {
    matrix_rows: Vec<Vec<U>>
}

impl<U: Clone + Debug> SuperMatrix<U> {
    pub fn new_with_cloned_values(n_rows: usize, n_collumns: usize, struct_to_clone: U) -> SuperMatrix<U>{
        let mut matrix_rows: Vec<Vec<U>> = Vec::with_capacity(n_rows);
        for _ in 0..n_rows {
            let mut row = Vec::<U>::with_capacity(n_collumns);
            for _ in 0..n_collumns{
                row.push(struct_to_clone.clone());
            }
            matrix_rows.push(row)
        } 
        SuperMatrix { matrix_rows }
    }

    pub fn row_length(&self) -> usize {
        let mut largest_length = self.matrix_rows[0].len();
        for row in &self.matrix_rows {
            if row.len() > largest_length {
                largest_length = row.len();
                println!("Row lengths of matrix not consistant!")
            }
        }
        largest_length
    }

    pub fn width(&self) -> usize {
        self.row_length()
    }

    pub fn coll_length(&self) -> usize {
        self.matrix_rows.len()
    }

    pub fn height(&self) -> usize {
        self.coll_length()
    }

    pub fn clone(&self) -> SuperMatrix<U> {
        let mut matrix_rows = Vec::<Vec<U>>::with_capacity(self.matrix_rows.len());
        for index in 0..self.matrix_rows.len() {
            matrix_rows.push(self.matrix_rows[index].clone())
        }
        SuperMatrix { matrix_rows: matrix_rows }
    }

    pub fn output_to_single_vector(&self) -> Vec<U> {
        let mut output_vector = Vec::<U>::with_capacity(self.width()*self.height());
        for row_index in 0..self.height() {
            for collumn_index in 0..self.width() {
                output_vector.push(self[row_index][collumn_index].clone())
            }
        }
        output_vector
    }
}

impl<U:Clone + Debug> Index<usize> for SuperMatrix<U> {
    type Output = Vec<U>;
    fn index(&self, index: usize) -> &Self::Output {
        &self.matrix_rows[index]
    }
}

impl<U: Clone + Debug> IndexMut<usize> for SuperMatrix<U> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.matrix_rows[index]
    }
}
