#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(unused_imports)]

use crate::*;

pub trait GPUValues: std::marker::Unpin + cudarc::driver::DeviceRepr +cudarc::driver::ValidAsZeroBits + std::default::Default{}
impl<G> GPUValues for G where G: std::marker::Unpin + cudarc::driver::DeviceRepr +cudarc::driver::ValidAsZeroBits + std::default::Default{}

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

pub trait RepeatedMatrixDotVectorGPUFunctions<T: MatrixValues + GPUValues> {
    fn new(matrix: &Matrix<T>, collumn: & Collumn<T>) -> Self ;
    fn result_vec_length(&self) -> usize ;
    fn next(&mut self) -> &mut Self where CudaBlas: Gemv<T>;
    fn itterate(&mut self, n_itterations: usize) -> &mut Self where CudaBlas: Gemv<T>;
    fn take_result(&mut self) -> Collumn<T> ;
    fn copy_result(& self) -> Collumn<T> ;
    fn matrix_dot_collumn_repeated_gpu_initiator(matrix: &Matrix<T>, collumn: & Collumn<T>) -> RepeatedMatrixDotVectorGPU<T>;
}


struct MatrixDotCollumnSystem<T> {
    gemv_config: GemvConfig<T>, 
    gpu_matrix: CudaSlice<T>, 
    gpu_input_collumn: CudaSlice<T>, 
    gpu_result_collumn: CudaSlice<T>
}

pub trait GPUFunctions<T: MatrixValues+ GPUValues> {
    fn matrix_dot_collumn_gpu(matrix: & Matrix<T>, collumn: & Collumn<T>) -> Collumn<T> where CudaBlas: Gemv<T>;
    fn matrix_dot_matrix_gpu(A_matrix:& Matrix<T>, B_matrix:& Matrix<T>) -> Matrix<T> where CudaBlas: Gemm<T>;
}

macro_rules! impl_gpu_functions_per_type {
    ($T: ident) => {
        impl GPUFunctions<$T> for $T { 
            fn matrix_dot_collumn_gpu(matrix: & Matrix<$T>, collumn: & Collumn<$T>) -> Collumn<$T>
            where CudaBlas: Gemv<$T>{
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
                let mut gpu_result_collumn = dev.alloc_zeros::<$T>(result_collumn_length).unwrap();

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
                
            fn matrix_dot_matrix_gpu(A_matrix:& Matrix<$T>, B_matrix:& Matrix<$T>) -> Matrix<$T>
            where CudaBlas: Gemm<$T>{
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
                let mut gpu_result_matrix = dev.alloc_zeros::<$T>(result_matrix_height * result_matrix_width).unwrap();
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
    }

    impl RepeatedMatrixDotVectorGPUFunctions<$T> for RepeatedMatrixDotVectorGPU<$T> {
        fn new(matrix: &Matrix<$T>, collumn: & Collumn<$T>) -> Self {
            let dev: Arc<CudaDevice> = CudaDevice::new(0).unwrap();
            let cublas = CudaBlas::new(dev.clone()).unwrap();
    
            let n = matrix.height();
            let m = matrix.width();
        
            let result_collumn_length = matrix.height(); 
            
            let matrix_transformed = matrix.to_vec_row_major();
            let collumn_in_vector = collumn.to_vec();
        
            let gpu_matrix= dev.htod_copy(matrix_transformed).unwrap();
            let gpu_input_collumn = dev.htod_copy(collumn_in_vector).unwrap();
            let mut gpu_result_collumn = dev.alloc_zeros::<$T>(result_collumn_length).unwrap();
    
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
    
        fn result_vec_length(&self) -> usize {
            self.n
        }
    
        fn next(&mut self) -> &mut Self 
        where CudaBlas: Gemv<$T>{
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
    
        fn itterate(&mut self, n_itterations: usize) -> &mut Self 
        where CudaBlas: Gemv<$T>{
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
    
        fn take_result(&mut self) -> Collumn<$T> {
            Collumn::new_form_vec(self.dev.sync_reclaim(self.gpu_result_collumn.to_owned()).unwrap())
        }
    
        fn copy_result(& self) -> Collumn<$T> {
            Collumn::new_form_vec(self.dev.dtoh_sync_copy(&self.gpu_result_collumn).unwrap())
        }

        fn matrix_dot_collumn_repeated_gpu_initiator(matrix: &Matrix<$T>, collumn: & Collumn<$T>) -> RepeatedMatrixDotVectorGPU<$T>{
            RepeatedMatrixDotVectorGPU::new(matrix, collumn)
        }
    }
    };
}

impl_gpu_functions_per_type!(f64);





// pub struct RepeatedMatrixDotVectorGPU<T: MatrixValues + GPUValues> {
//     // computes A x = B, where A the matrix, x the input collumn and B the resulting collumn
//     // output is assumed to be the input of the next. This is only if X_(n+1) = B_n. or x_(n+1) = {b_1, b_2, b_3}_n
//     // so x_(n+1) = {f_1{b_1}, f_2{b_2}, f_3{b_3}}_n is NOT possible, but with a manually written cuda function running cocurrently on the gpu this should be doable.
//     dev: Arc<CudaDevice>,
//     cublas: CudaBlas,
//     gpu_matrix: CudaSlice<T>,
//     gpu_input_collumn: CudaSlice<T>,
//     gpu_result_collumn: CudaSlice<T>,
//     m: usize,
//     n: usize,
// }

// impl<T: MatrixValues + GPUValues> RepeatedMatrixDotVectorGPU<T> {
//     pub fn new(matrix: &Matrix<T>, collumn: & Collumn<T>) -> Self {
//         let dev: Arc<CudaDevice> = CudaDevice::new(0).unwrap();
//         let cublas = CudaBlas::new(dev.clone()).unwrap();

//         let n = matrix.height();
//         let m = matrix.width();
    
//         let result_collumn_length = matrix.height(); 
        
//         let matrix_transformed = matrix.to_vec_row_major();
//         let collumn_in_vector = collumn.to_vec();
    
//         let gpu_matrix= dev.htod_copy(matrix_transformed).unwrap();
//         let gpu_input_collumn = dev.htod_copy(collumn_in_vector).unwrap();
//         let mut gpu_result_collumn = dev.alloc_zeros::<T>(result_collumn_length).unwrap();

//         RepeatedMatrixDotVectorGPU {
//             dev,
//             cublas,
//             gpu_matrix,
//             gpu_input_collumn,
//             gpu_result_collumn,
//             m,
//             n,
//         }
//     }

//     pub fn result_vec_length(&self) -> usize {
//         self.n
//     }

//     pub fn next(&mut self) -> &mut Self 
//     where CudaBlas: Gemv<T>{
//         // does one itteration
//         unsafe {
//             self.cublas.gemv(
//                 safe::GemvConfig {
//                     trans: sys::cublasOperation_t::CUBLAS_OP_T, // by using transpose less time consuming copy can be applied
//                     m: self.m as i32, 
//                     n: self.n as i32,
//                     alpha: NumCast::from(1).unwrap(),
//                     lda: self.m as i32, // leading dimension: next collumn starts at this position
//                     incx: 1,
//                     beta: NumCast::from(0).unwrap(),
//                     incy: 1,
//                 }, 
//                 &self.gpu_matrix, 
//                 &self.gpu_input_collumn, 
//                 &mut self.gpu_result_collumn).unwrap();
//         } // it aint stopid if it works

//         self
//     }

//     pub fn itterate(&mut self, n_itterations: usize) -> &mut Self 
//     where CudaBlas: Gemv<T>{
//         // does multiple itterations withouth passing results

//         // TODO! creation of repeated Gemv probably slows things down -> put more time into getting this thing out.
//         // TODO! CPU at 100% while evaluation (also gpu). CPU not expected, probably that after every (very quick) itteration, slow GPU communication needs to be called.
//         // -> therefore better to write the forloop in CUDA directly. It will take handles to the data and just do its own thing. GOOD TO HAVE: make (an alternate) function able 
//         // to take functions for translating the previous results to the inputs of the current simulation.
//         for itter in 0..n_itterations {
//             unsafe {
//                 self.cublas.gemv(
//                     safe::GemvConfig {
//                         trans: sys::cublasOperation_t::CUBLAS_OP_T, // by using transpose less time consuming copy can be applied
//                         m: self.m as i32, 
//                         n: self.n as i32,
//                         alpha: NumCast::from(1).unwrap(),
//                         lda: self.m as i32, // leading dimension: next collumn starts at this position
//                         incx: 1,
//                         beta: NumCast::from(0).unwrap(),
//                         incy: 1,
//                     }, 
//                     &self.gpu_matrix, 
//                     &self.gpu_input_collumn, 
//                     &mut self.gpu_result_collumn).unwrap();
//             } // it aint stopid if it works
//             self.dev.dtod_copy(&self.gpu_result_collumn, &mut self.gpu_input_collumn).unwrap();
//         }
//         self
//     }

//     pub fn take_result(&mut self) -> Collumn<T> {
//         Collumn::new_form_vec(self.dev.sync_reclaim(self.gpu_result_collumn.to_owned()).unwrap())
//     }

//     pub fn copy_result(& self) -> Collumn<T> {
//         Collumn::new_form_vec(self.dev.dtoh_sync_copy(&self.gpu_result_collumn).unwrap())
//     }
// }




// struct MatrixDotCollumnSystem<T> {
//     gemv_config: GemvConfig<T>, 
//     gpu_matrix: CudaSlice<T>, 
//     gpu_input_collumn: CudaSlice<T>, 
//     gpu_result_collumn: CudaSlice<T>
// }


