#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(unused_imports)]

use std::ops::Mul;
use std::ops::Sub;
use std::ops::Add;
use std::ops::Div;
use std::marker::PhantomData;

use crate::*;

#[derive(Debug, Clone)]
pub struct Row<T: MatrixValues> {
    pub cells : Vec<T>,
}

impl<T: MatrixValues> From<SpatialVectorNDof<T, IsColl>> for Row<T> {
    fn from(spatial_vector: SpatialVectorNDof<T, IsColl>) -> Self {
        Row { cells: spatial_vector.vector }
    }
}

impl<T: MatrixValues> From<SpatialVectorWithBasePointNDof<T, IsColl>> for Row<T> {
    fn from(spatial_vector: SpatialVectorWithBasePointNDof<T, IsColl>) -> Self {
        Row { cells: spatial_vector.vector }
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

pub trait RowFunctions<T:MatrixValues> {
    fn new_row_with_value(size: usize, value: T) -> Row<T>;
    fn to_vec(&self) -> Vec<T>;
    fn new_row_from_vec(input_vec: Vec<T>) -> Row<T>;
    fn new_row_with_constant_values(width: usize, value: T) -> Row<T>;
    fn len(&self) -> usize;
    fn export(self) -> Row<T>;
    fn transpose(&self) -> Collumn<T>;
    // fn clone(&self) -> Row<T>;
    fn all_values_equal(&self, value: T) ->bool;
    fn divide_all_elements_by(&mut self, value: T)-> &mut Self;
    fn multiply_all_elements_by(&mut self, value: T)-> &mut Self;
    fn addition_row_with_external_row(&mut self, row_to_add_to_this_one:& Row<T>)-> &mut Self;
    fn normalize_all_elements_to_element(&mut self, index: usize)-> &mut Self;
    fn normalize_all_elements_to_first(&mut self) -> &mut Self;
    fn substract_row(&mut self, substraction_row: Row<T>)-> &mut Self ;
    fn substract_all(&mut self, substraction_row: Row<T>)-> &mut Self;
    fn replace_values(&mut self, index_range: Range<usize>, values: Vec<T>)-> &mut Self;
    fn subtract_scalar(&mut self, value: T) -> &mut Self;

}

// impl<T: MatrixValues + std::ops::Add> Add for Row<T> 
// where std::vec::Vec<T>: FromIterator<<T as std::ops::Add>::Output>{
//     type Output = Self;

//     fn add(self, rhs: Self) -> Self::Output {
//         if (self.cells.len() == rhs.cells.len()).not() {
//             panic!("added rows not equal length!")
//         }
//         let cells = self.cells.iter().zip(rhs.cells.iter()).map(|(lhs, rhs)| *lhs + *rhs).collect();
//         Row {cells}
//     }
// }

// impl<T: MatrixValues + std::ops::Mul + std::iter::Sum<<T as std::ops::Mul>::Output>> Mul for Row<T> {
//     type Output = T;

//     fn mul(self, rhs: Self) -> Self::Output {
//         if (self.cells.len() == rhs.cells.len()).not() {
//             panic!("multiplied rows not equal length!")
//         }

//         self.cells.iter().zip(rhs.cells.iter()).map(|(lhs, rhs)| *lhs * *rhs).sum::<T>()
//     }
// }

macro_rules! impl_row_type {
    ($T: ident) => {
        impl RowFunctions<$T> for Row<$T> {
            fn new_row_with_value(size: usize, value: $T) -> Row<$T> {
                let mut cells = Vec::<$T>::with_capacity(size);
                for _ in 0..size{
                    cells.push(value);
                }
                Row { cells}
            }
        
            fn to_vec(&self) -> Vec<$T> {
                self.cells.clone()
            }
            
            fn new_row_from_vec(input_vec: Vec<$T>) -> Row<$T> {
                Row { cells: input_vec }
            }
        
            fn new_row_with_constant_values(width: usize, value: $T) -> Row<$T> {
                return Row::new_row_with_value(width, value)
            }
        
            fn len(&self) -> usize {
                self.cells.len()
            }
        
            fn export(self) -> Row<$T> {
                self
            }
        
            fn transpose(&self) -> Collumn<$T> {
                Collumn { cells: self.cells.clone() }
            }

            // fn clone(&self) -> Row<$T> {
            //     let mut cells = Vec::<$T>::with_capacity(self.len());
            //     for cell in &self.cells{
            //         cells.push(cell.clone());
            //     }
            //     Row { cells}
            // }

        
            fn all_values_equal(&self, value: $T) ->bool {
                let mut no_inequality_found = true;
                for row_value in self.cells.iter() {
                    if (*row_value == value).not() {
                        no_inequality_found = false;
                    }
                }
                no_inequality_found
            }

            fn divide_all_elements_by(&mut self, value: $T) -> &mut Self {

                // if *IS_AVX512 && USE_OPIMIZERS{
                //     unsafe {self.const_multiply_avx512_row(1 as $T/value)}
                // } else {

                    for n in 0..self.cells.len() {
                        // if !(self.cells[n] == NumCast::from(0).unwrap()) {// quickly tested on some sparse matrices but seem to really boost performance. In some more filled ones: around 50x improvemnt, ful matrix not tested yet
                            self.cells[n] = self.cells[n] / value;
                        // }
                    }
                // }

                self
            }
        
            fn multiply_all_elements_by(&mut self, value: $T) -> &mut Self{

                // if *IS_AVX512 && USE_OPIMIZERS{
                //     unsafe {self.const_multiply_avx512_row(value)}
                // } else {
                    for n in 0..self.cells.len() {
                        // if !(self.cells[n] == NumCast::from(0).unwrap()) {// quickly tested on some sparse matrices but seem to really boost performance. In some more filled ones: around 50x improvemnt, ful matrix not tested yet
                        self.cells[n] = self.cells[n] * value;
                        // }
                    }
                // }
                self
            }
        
            fn addition_row_with_external_row(&mut self, row_to_add_to_this_one:& Row<$T>) -> &mut Self {
                for n in 0..self.cells.len() {
                    // if !(self.cells[n] == NumCast::from(0).unwrap() && row_to_add_to_this_one.cells[n] == NumCast::from(0).unwrap()) {
                        self.cells[n] = self.cells[n] + row_to_add_to_this_one[n];
                    // }
                }
                self
            }
        
            
        
            fn normalize_all_elements_to_element(&mut self, index: usize) -> &mut Self {
                self.divide_all_elements_by(self[index]);
                self
            }
        
            fn normalize_all_elements_to_first(&mut self) -> &mut Self {
                self.normalize_all_elements_to_element(0);
                self
            }
        
            fn substract_row(&mut self, substraction_row: Row<$T>)-> &mut Self {
                if !(self.cells.len() == substraction_row.cells.len()) {
                    panic!("Error: Length of substracting row is not equal to row length")
                }
                // if *IS_AVX512 && USE_OPIMIZERS{
                //     unsafe {self.substract_avx512_row(substraction_row)}
                // } else if *IS_AVX2 && USE_OPIMIZERS{
                //     unsafe {self.substract_avx2_row(substraction_row)}
                // } else {
                    self.substract_all(substraction_row);
                // }
                self
            }
        
        
            fn substract_all(&mut self, substraction_row: Row<$T>)-> &mut Self {
                for cell_number in 0..self.cells.len() {
                    // if !(self[cell_number] == NumCast::from(0).unwrap() || substraction_row[cell_number] == NumCast::from(0).unwrap()) { // quickly tested on some sparse matrices but seem to really boost performance . In some more filled ones: around 50x improvemnt, ful matrix not tested yet
                    self[cell_number] = self[cell_number] - substraction_row[cell_number];
                    // }
                }
                self
            }
        
            fn replace_values(&mut self, index_range: Range<usize>, values: Vec<$T>)-> &mut Self {
                // maybe add a check if not of same size TODO
                // instead of range use rangebounds apperantly
                for (val_index, row_index) in index_range.enumerate() {
                    self.cells[row_index] = values[val_index];
                }
                self
            }

            fn subtract_scalar(&mut self, value: $T) -> &mut Self {
                for cell in self.cells.iter_mut() {
                    *cell = *cell - value;
                }
                self
            }
        }
    }
}


pub trait IntoRow<T: MatrixValues> {
    fn to_row(self) -> Row<T>;
}

impl<T: MatrixValues> IntoRow<T> for Row<T> {
    fn to_row(self) -> Row<T> {
        self
    }
}

impl<T: MatrixValues> IntoRow<T> for SpatialVectorNDof<T, IsRow> {
    fn to_row(self) -> Row<T> {
        Row { cells: self.vector }
    }
}

pub trait RowStdOps<T: MatrixValues>: IntoRow<T> {}
impl<T:MatrixValues> RowStdOps<T> for Row<T> {}
impl<T:MatrixValues> RowStdOps<T> for SpatialVectorNDof<T, IsRow> {}

macro_rules! impl_std_ops_row_per_type {
    ($T: ident) => {
        impl Add<Row<$T>> for Row<$T> {
            type Output = Row<$T>;

            fn add(mut self, rhs: Row<$T>) -> Self::Output {
                assert!(self.cells.len() == rhs.cells.len(), "Provided Rows are not of same length");
                self.cells = self.cells.iter().zip(rhs.cells.iter()).map(|(x_l, x_r)| x_l + x_r).collect::<Vec<$T>>();
                self
            }
        }

        impl Add<SpatialVectorNDof<$T, IsRow>> for Row<$T> {
            type Output = Row<$T>;

            fn add(mut self, rhs: SpatialVectorNDof<$T, IsRow>) -> Self::Output {
                assert!(self.cells.len() == rhs.vector.len(), "Provided Rows are not of same length");
                self.cells = self.cells.iter().zip(rhs.vector.iter()).map(|(x_l, x_r)| x_l + x_r).collect::<Vec<$T>>();
                self
            }
        }

        impl Sub<Row<$T>> for Row<$T> {
            type Output = Row<$T>;

            fn sub(mut self, rhs: Row<$T>) -> Self::Output {
                assert!(self.cells.len() == rhs.cells.len(), "Provided Rows are not of same length");
                self.cells = self.cells.iter().zip(rhs.cells.iter()).map(|(x_l, x_r)| x_l - x_r).collect::<Vec<$T>>();
                self
            }
        }

        impl Sub<SpatialVectorNDof<$T, IsRow>> for Row<$T> {
            type Output = Row<$T>;

            fn sub(mut self, rhs: SpatialVectorNDof<$T, IsRow>) -> Self::Output {
                assert!(self.cells.len() == rhs.vector.len(), "Provided Rows are not of same length");
                self.cells = self.cells.iter().zip(rhs.vector.iter()).map(|(x_l, x_r)| x_l - x_r).collect::<Vec<$T>>();
                self
            }
        }

        impl Mul<Collumn<$T>> for Row<$T> {
            type Output = $T;

            fn mul(mut self, rhs: Collumn<$T>) -> Self::Output {
                $T::row_dot_collumn(&self, &rhs)
            }
        }

        impl Mul<SpatialVectorNDof<$T, IsColl>> for Row<$T> {
            type Output = $T;

            fn mul(mut self, rhs: SpatialVectorNDof<$T, IsColl>) -> Self::Output {
                let rhs = rhs.clone().to_collumn();
                $T::row_dot_collumn(&self, &rhs)
            }
        }

        impl Mul<Matrix<$T>> for Row<$T> {
            type Output = Row<$T>;

            fn mul(mut self, rhs: Matrix<$T>) -> Self::Output {
                $T::row_dot_matrix(&self, &rhs)
            }
        }

        impl Mul<$T> for Row<$T> {
            type Output = Row<$T>;

            fn mul(mut self, rhs: $T) -> Self::Output {
                self.multiply_all_elements_by(rhs);
                self
            }
        }

        impl Div<$T> for Row<$T> {
            type Output = Row<$T>;

            fn div(mut self, rhs: $T) -> Self::Output {
                self.divide_all_elements_by(rhs);
                self
            }
        }

        //////////////////////////////////////
        /// left hand side borrowed
        /// //////////////////////////////////
        
        impl Add<Row<$T>> for &Row<$T> {
            type Output = Row<$T>;

            fn add(mut self, mut rhs: Row<$T>) -> Self::Output {
                assert!(self.cells.len() == rhs.cells.len(), "Provided Rows are not of same length");
                rhs.cells = self.cells.iter().zip(rhs.cells.iter()).map(|(x_l, x_r)| x_l + x_r).collect::<Vec<$T>>();
                rhs
            }
        }

        impl Add<SpatialVectorNDof<$T, IsRow>> for &Row<$T> {
            type Output = Row<$T>;

            fn add(mut self, mut rhs: SpatialVectorNDof<$T, IsRow>) -> Self::Output {
                assert!(self.cells.len() == rhs.vector.len(), "Provided Rows are not of same length");
                rhs.vector = self.cells.iter().zip(rhs.vector.iter()).map(|(x_l, x_r)| x_l + x_r).collect::<Vec<$T>>();
                Row{ cells: rhs.vector }
            }
        }

        impl Sub<Row<$T>> for &Row<$T> {
            type Output = Row<$T>;

            fn sub(mut self, mut rhs: Row<$T>) -> Self::Output {
                assert!(self.cells.len() == rhs.cells.len(), "Provided Rows are not of same length");
                rhs.cells = self.cells.iter().zip(rhs.cells.iter()).map(|(x_l, x_r)| x_l - x_r).collect::<Vec<$T>>();
                rhs
            }
        }

        impl Sub<SpatialVectorNDof<$T, IsRow>> for &Row<$T> {
            type Output = Row<$T>;

            fn sub(self, mut rhs: SpatialVectorNDof<$T, IsRow>) -> Self::Output {
                assert!(self.cells.len() == rhs.vector.len(), "Provided Rows are not of same length");
                rhs.vector = self.cells.iter().zip(rhs.vector.iter()).map(|(x_l, x_r)| x_l - x_r).collect::<Vec<$T>>();
                Row{ cells: rhs.vector }
            }
        }

        impl Mul<Collumn<$T>> for &Row<$T> {
            type Output = $T;

            fn mul(self, rhs: Collumn<$T>) -> Self::Output {
                $T::row_dot_collumn(&self, &rhs)
            }
        }

        impl Mul<SpatialVectorNDof<$T, IsColl>> for &Row<$T> {
            type Output = $T;

            fn mul(self, rhs: SpatialVectorNDof<$T, IsColl>) -> Self::Output {
                let rhs = rhs.clone().to_collumn();
                $T::row_dot_collumn(&self, &rhs)
            }
        }

        impl Mul<Matrix<$T>> for &Row<$T> {
            type Output = Row<$T>;

            fn mul(self, rhs: Matrix<$T>) -> Self::Output {
                $T::row_dot_matrix(&self, &rhs)
            }
        }

        impl Mul<$T> for &Row<$T> {
            type Output = Row<$T>;

            fn mul(self, rhs: $T) -> Self::Output {
                let mut new_row = self.clone();
                new_row.multiply_all_elements_by(rhs);
                new_row
            }
        }

        impl Div<$T> for &Row<$T> {
            type Output = Row<$T>;

            fn div(self, rhs: $T) -> Self::Output {
                let mut new_row = self.clone();
                new_row.divide_all_elements_by(rhs);
                new_row
            }
        }

        ///////////////////////////////////////////
        /// RIGHT HAND SIDE BORROWED
        /// ///////////////////////////////////////
         
        impl Add<&Row<$T>> for Row<$T> {
            type Output = Row<$T>;

            fn add(mut self, rhs: &Row<$T>) -> Self::Output {
                assert!(self.cells.len() == rhs.cells.len(), "Provided Rows are not of same length");
                self.cells = self.cells.iter().zip(rhs.cells.iter()).map(|(x_l, x_r)| x_l + x_r).collect::<Vec<$T>>();
                self
            }
        }

        impl Add<&SpatialVectorNDof<$T, IsRow>> for Row<$T> {
            type Output = Row<$T>;

            fn add(mut self, rhs: &SpatialVectorNDof<$T, IsRow>) -> Self::Output {
                assert!(self.cells.len() == rhs.vector.len(), "Provided Rows are not of same length");
                self.cells = self.cells.iter().zip(rhs.vector.iter()).map(|(x_l, x_r)| x_l + x_r).collect::<Vec<$T>>();
                self
            }
        }

        impl Sub<&Row<$T>> for Row<$T> {
            type Output = Row<$T>;

            fn sub(mut self, rhs: &Row<$T>) -> Self::Output {
                assert!(self.cells.len() == rhs.cells.len(), "Provided Rows are not of same length");
                self.cells = self.cells.iter().zip(rhs.cells.iter()).map(|(x_l, x_r)| x_l - x_r).collect::<Vec<$T>>();
                self
            }
        }

        impl Sub<&SpatialVectorNDof<$T, IsRow>> for Row<$T> {
            type Output = Row<$T>;

            fn sub(mut self, rhs: &SpatialVectorNDof<$T, IsRow>) -> Self::Output {
                assert!(self.cells.len() == rhs.vector.len(), "Provided Rows are not of same length");
                self.cells = self.cells.iter().zip(rhs.vector.iter()).map(|(x_l, x_r)| x_l - x_r).collect::<Vec<$T>>();
                self
            }
        }

        impl Mul<&Collumn<$T>> for Row<$T> {
            type Output = $T;

            fn mul(mut self, rhs: &Collumn<$T>) -> Self::Output {
                $T::row_dot_collumn(&self, &rhs)
            }
        }

        impl Mul<&SpatialVectorNDof<$T, IsColl>> for Row<$T> {
            type Output = $T;

            fn mul(mut self, rhs: &SpatialVectorNDof<$T, IsColl>) -> Self::Output {
                let rhs = rhs.clone().to_collumn();
                $T::row_dot_collumn(&self, &rhs)
            }
        }

        impl Mul<&Matrix<$T>> for Row<$T> {
            type Output = Row<$T>;

            fn mul(mut self, rhs: &Matrix<$T>) -> Self::Output {
                $T::row_dot_matrix(&self, &rhs)
            }
        }

        ///////////////////////////////////////
        /// all borrowed
        /// //////////////////////////////////

        impl Add<&Row<$T>> for &Row<$T> {
            type Output = Row<$T>;

            fn add(self, rhs: &Row<$T>) -> Self::Output {
                assert!(self.cells.len() == rhs.cells.len(), "Provided Rows are not of same length");
                let cells = self.cells.iter().zip(rhs.cells.iter()).map(|(x_l, x_r)| x_l + x_r).collect::<Vec<$T>>();
                Row { cells }
            }
        }

        impl Add<&SpatialVectorNDof<$T, IsRow>> for &Row<$T> {
            type Output = Row<$T>;

            fn add(self, rhs: &SpatialVectorNDof<$T, IsRow>) -> Self::Output {
                let rhs = rhs.clone().to_row();
                assert!(self.cells.len() == rhs.cells.len(), "Provided Rows are not of same length");
                let cells = self.cells.iter().zip(rhs.cells.iter()).map(|(x_l, x_r)| x_l + x_r).collect::<Vec<$T>>();
                Row { cells }
            }
        }

        impl Sub<&Row<$T>> for &Row<$T> {
            type Output = Row<$T>;

            fn sub(self, rhs: &Row<$T>) -> Self::Output {
                let rhs = rhs.clone().to_row();
                assert!(self.cells.len() == rhs.cells.len(), "Provided Rows are not of same length");
                let cells = self.cells.iter().zip(rhs.cells.iter()).map(|(x_l, x_r)| x_l - x_r).collect::<Vec<$T>>();
                Row { cells }
            }
        }

        impl Sub<&SpatialVectorNDof<$T, IsRow>> for &Row<$T> {
            type Output = Row<$T>;

            fn sub(self, rhs: &SpatialVectorNDof<$T, IsRow>) -> Self::Output {
                let rhs = rhs.clone().to_row();
                assert!(self.cells.len() == rhs.cells.len(), "Provided Rows are not of same length");
                let cells = self.cells.iter().zip(rhs.cells.iter()).map(|(x_l, x_r)| x_l - x_r).collect::<Vec<$T>>();
                Row { cells }
            }
        }

        impl Mul<&Collumn<$T>> for &Row<$T> {
            type Output = $T;

            fn mul(self, rhs: &Collumn<$T>) -> Self::Output {
                $T::row_dot_collumn(&self, &rhs)
            }
        }

        impl Mul<&SpatialVectorNDof<$T, IsColl>> for &Row<$T> {
            type Output = $T;

            fn mul(self, rhs: &SpatialVectorNDof<$T, IsColl>) -> Self::Output {
                let rhs = rhs.clone().to_collumn();
                $T::row_dot_collumn(&self, &rhs)
            }
        }

        impl Mul<&Matrix<$T>> for &Row<$T> {
            type Output = Row<$T>;

            fn mul(self, rhs: &Matrix<$T>) -> Self::Output {
                $T::row_dot_matrix(&self, &rhs)
            }
        }
    };
}

macro_rules! impl_std_ops_vector_row_per_type {
    ($T: ident) => {

        impl Add<Row<$T>> for SpatialVectorNDof<$T, IsRow> {
            type Output = SpatialVectorNDof<$T, IsRow>;

            fn add(mut self, rhs: Row<$T>) -> Self::Output {
                assert!(self.vector.len() == rhs.cells.len(), "Provided Rows are not of same length");
                self.vector = self.vector.iter().zip(rhs.cells.iter()).map(|(x_l, x_r)| x_l + x_r).collect::<Vec<$T>>();
                self
            }
        }

        impl Add<SpatialVectorNDof<$T, IsRow>> for SpatialVectorNDof<$T, IsRow> {
            type Output = SpatialVectorNDof<$T, IsRow>;

            fn add(mut self, rhs: SpatialVectorNDof<$T, IsRow>) -> Self::Output {
                let rhs = rhs.to_row();
                assert!(self.vector.len() == rhs.cells.len(), "Provided Rows are not of same length");
                self.vector = self.vector.iter().zip(rhs.cells.iter()).map(|(x_l, x_r)| x_l + x_r).collect::<Vec<$T>>();
                self
            }
        }

        impl Sub<Row<$T>> for SpatialVectorNDof<$T, IsRow> {
            type Output = SpatialVectorNDof<$T, IsRow>;

            fn sub(mut self, rhs: Row<$T>) -> Self::Output {
                let rhs = rhs.to_row();
                assert!(self.vector.len() == rhs.cells.len(), "Provided Rows are not of same length");
                self.vector = self.vector.iter().zip(rhs.cells.iter()).map(|(x_l, x_r)| x_l - x_r).collect::<Vec<$T>>();
                self
            }
        }

        impl Sub<SpatialVectorNDof<$T, IsRow>> for SpatialVectorNDof<$T, IsRow> {
            type Output = SpatialVectorNDof<$T, IsRow>;

            fn sub(mut self, rhs: SpatialVectorNDof<$T, IsRow>) -> Self::Output {
                let rhs = rhs.to_row();
                assert!(self.vector.len() == rhs.cells.len(), "Provided Rows are not of same length");
                self.vector = self.vector.iter().zip(rhs.cells.iter()).map(|(x_l, x_r)| x_l - x_r).collect::<Vec<$T>>();
                self
            }
        }

        impl Mul<Collumn<$T>> for SpatialVectorNDof<$T, IsRow> {
            type Output = $T;

            fn mul(mut self, rhs: Collumn<$T>) -> Self::Output {
                let self_converted = self.to_row();
                $T::row_dot_collumn(&self_converted, &rhs)
            }
        }

        impl Mul<SpatialVectorNDof<$T, IsColl>> for SpatialVectorNDof<$T, IsRow> {
            type Output = $T;

            fn mul(mut self, rhs: SpatialVectorNDof<$T, IsColl>) -> Self::Output {
                let self_converted = self.clone().to_row();
                let rhs = rhs.clone().to_collumn();
                $T::row_dot_collumn(&self_converted, &rhs)
            }
        }

        impl Mul<Matrix<$T>> for SpatialVectorNDof<$T, IsRow> {
            type Output = SpatialVectorNDof<$T, IsRow>;

            fn mul(mut self, rhs: Matrix<$T>) -> Self::Output {
                let self_converted = self.clone().to_row();
                let result_row = $T::row_dot_matrix(&self_converted, &rhs);
                SpatialVectorNDof {vector: result_row.cells, _orientation: PhantomData::<IsRow>}
            }
        }

        impl Mul<$T> for SpatialVectorNDof<$T, IsRow> {
            type Output = SpatialVectorNDof<$T, IsRow>;

            fn mul(mut self, rhs: $T) -> Self::Output {
                let mut self_converted = self.clone().to_row();
                self_converted.multiply_all_elements_by(rhs);
                SpatialVectorNDof {vector: self_converted.cells, _orientation: PhantomData::<IsRow>}
            }
        }

        impl Div<$T> for SpatialVectorNDof<$T, IsRow> {
            type Output = SpatialVectorNDof<$T, IsRow>;

            fn div(mut self, rhs: $T) -> Self::Output {
                let mut self_converted = self.clone().to_row();
                self_converted.divide_all_elements_by(rhs);
                SpatialVectorNDof {vector: self_converted.cells, _orientation: PhantomData::<IsRow>}
            }
        }

        ///////////////////////////////////////////
        /// left hand sdie borrewed
        /// ///////////////////////////////////////

        impl Add<Row<$T>> for &SpatialVectorNDof<$T, IsRow> {
            type Output = SpatialVectorNDof<$T, IsRow>;

            fn add(self, mut rhs: Row<$T>) -> Self::Output {
                assert!(self.vector.len() == rhs.cells.len(), "Provided Rows are not of same length");
                rhs.cells = self.vector.iter().zip(rhs.cells.iter()).map(|(x_l, x_r)| x_l + x_r).collect::<Vec<$T>>();
                SpatialVectorNDof {vector: rhs.cells, _orientation: PhantomData}
            }
        }

        impl Add<SpatialVectorNDof<$T, IsRow>> for &SpatialVectorNDof<$T, IsRow> {
            type Output = SpatialVectorNDof<$T, IsRow>;

            fn add(self, mut rhs: SpatialVectorNDof<$T, IsRow>) -> Self::Output {
                assert!(self.vector.len() == rhs.vector.len(), "Provided Rows are not of same length");
                rhs.vector = self.vector.iter().zip(rhs.vector.iter()).map(|(x_l, x_r)| x_l + x_r).collect::<Vec<$T>>();
                rhs
            }
        }

        impl Sub<Row<$T>> for &SpatialVectorNDof<$T, IsRow> {
            type Output = SpatialVectorNDof<$T, IsRow>;

            fn sub(self, rhs: Row<$T>) -> Self::Output {
                let mut rhs = rhs.to_row();
                assert!(self.vector.len() == rhs.cells.len(), "Provided Rows are not of same length");
                rhs.cells = self.vector.iter().zip(rhs.cells.iter()).map(|(x_l, x_r)| x_l - x_r).collect::<Vec<$T>>();
                SpatialVectorNDof {vector: rhs.cells, _orientation: PhantomData}
            }
        }

        impl Sub<SpatialVectorNDof<$T, IsRow>> for &SpatialVectorNDof<$T, IsRow> {
            type Output = SpatialVectorNDof<$T, IsRow>;

            fn sub(self, rhs: SpatialVectorNDof<$T, IsRow>) -> Self::Output {
                let mut rhs = rhs.to_row();
                assert!(self.vector.len() == rhs.cells.len(), "Provided Rows are not of same length");
                rhs.cells = self.vector.iter().zip(rhs.cells.iter()).map(|(x_l, x_r)| x_l - x_r).collect::<Vec<$T>>();
                SpatialVectorNDof {vector: rhs.cells, _orientation: PhantomData}
            }
        }

        impl Mul<Collumn<$T>> for &SpatialVectorNDof<$T, IsRow> {
            type Output = $T;

            fn mul(self, rhs: Collumn<$T>) -> Self::Output {
                let self_converted = self.clone().to_row();
                $T::row_dot_collumn(&self_converted, &rhs)
            }
        }

        impl Mul<SpatialVectorNDof<$T, IsColl>> for &SpatialVectorNDof<$T, IsRow> {
            type Output = $T;

            fn mul(self, rhs: SpatialVectorNDof<$T, IsColl>) -> Self::Output {
                let self_converted = self.clone().to_row();
                let rhs = rhs.clone().to_collumn();
                $T::row_dot_collumn(&self_converted, &rhs)
            }
        }

        impl Mul<Matrix<$T>> for &SpatialVectorNDof<$T, IsRow> {
            type Output = SpatialVectorNDof<$T, IsRow>;

            fn mul(self, rhs: Matrix<$T>) -> Self::Output {
                let self_converted = self.clone().to_row();
                let result_row = $T::row_dot_matrix(&self_converted, &rhs);
                SpatialVectorNDof {vector: result_row.cells, _orientation: PhantomData::<IsRow>}
            }
        }

        impl Mul<$T> for &SpatialVectorNDof<$T, IsRow> {
            type Output = SpatialVectorNDof<$T, IsRow>;

            fn mul(self, rhs: $T) -> Self::Output {
                let mut self_converted = self.clone().to_row();
                self_converted.multiply_all_elements_by(rhs);
                SpatialVectorNDof {vector: self_converted.cells, _orientation: PhantomData::<IsRow>}
            }
        }

        impl Div<$T> for &SpatialVectorNDof<$T, IsRow> {
            type Output = SpatialVectorNDof<$T, IsRow>;

            fn div(self, rhs: $T) -> Self::Output {
                let mut self_converted = self.clone().to_row();
                self_converted.divide_all_elements_by(rhs);
                SpatialVectorNDof {vector: self_converted.cells, _orientation: PhantomData::<IsRow>}
            }
        } 

        //////////////////////////////////////////
        /// IRHGT HAND SIDE MUTABLE
        /// ////////////////////////////////////////
        
        impl Add<&Row<$T>> for SpatialVectorNDof<$T, IsRow> {
            type Output = SpatialVectorNDof<$T, IsRow>;

            fn add(mut self, rhs: &Row<$T>) -> Self::Output {
                assert!(self.vector.len() == rhs.cells.len(), "Provided Rows are not of same length");
                self.vector = self.vector.iter().zip(rhs.cells.iter()).map(|(x_l, x_r)| x_l + x_r).collect::<Vec<$T>>();
                self
            }
        }

        impl Add<&SpatialVectorNDof<$T, IsRow>> for SpatialVectorNDof<$T, IsRow> {
            type Output = SpatialVectorNDof<$T, IsRow>;

            fn add(mut self, rhs: &SpatialVectorNDof<$T, IsRow>) -> Self::Output {
                assert!(self.vector.len() == rhs.vector.len(), "Provided Rows are not of same length");
                self.vector = self.vector.iter().zip(rhs.vector.iter()).map(|(x_l, x_r)| x_l + x_r).collect::<Vec<$T>>();
                self
            }
        }

        impl Sub<&Row<$T>> for SpatialVectorNDof<$T, IsRow> {
            type Output = SpatialVectorNDof<$T, IsRow>;

            fn sub(mut self, rhs: &Row<$T>) -> Self::Output {
                assert!(self.vector.len() == rhs.cells.len(), "Provided Rows are not of same length");
                self.vector = self.vector.iter().zip(rhs.cells.iter()).map(|(x_l, x_r)| x_l - x_r).collect::<Vec<$T>>();
                self
            }
        }

        impl Sub<&SpatialVectorNDof<$T, IsRow>> for SpatialVectorNDof<$T, IsRow> {
            type Output = SpatialVectorNDof<$T, IsRow>;

            fn sub(mut self, rhs: &SpatialVectorNDof<$T, IsRow>) -> Self::Output {
                assert!(self.vector.len() == rhs.vector.len(), "Provided Rows are not of same length");
                self.vector = self.vector.iter().zip(rhs.vector.iter()).map(|(x_l, x_r)| x_l - x_r).collect::<Vec<$T>>();
                self
            }
        }

        impl Mul<&Collumn<$T>> for SpatialVectorNDof<$T, IsRow> {
            type Output = $T;

            fn mul(mut self, rhs: &Collumn<$T>) -> Self::Output {
                let self_converted = self.to_row();
                $T::row_dot_collumn(&self_converted, &rhs)
            }
        }

        impl Mul<&SpatialVectorNDof<$T, IsColl>> for SpatialVectorNDof<$T, IsRow> {
            type Output = $T;

            fn mul(mut self, rhs: &SpatialVectorNDof<$T, IsColl>) -> Self::Output {
                let self_converted = self.clone().to_row();
                let rhs = rhs.clone().to_collumn();
                $T::row_dot_collumn(&self_converted, &rhs)
            }
        }

        impl Mul<&Matrix<$T>> for SpatialVectorNDof<$T, IsRow> {
            type Output = SpatialVectorNDof<$T, IsRow>;

            fn mul(mut self, rhs: &Matrix<$T>) -> Self::Output {
                let self_converted = self.clone().to_row();
                let result_row = $T::row_dot_matrix(&self_converted, &rhs);
                SpatialVectorNDof {vector: result_row.cells, _orientation: PhantomData::<IsRow>}
            }
        }
        
        ///////////////////////////////////////////
        /// ALL BORORWED
        /// ////////////////////////////////////////
        
        impl Add<&Row<$T>> for &SpatialVectorNDof<$T, IsRow> {
            type Output = SpatialVectorNDof<$T, IsRow>;

            fn add(self, rhs: &Row<$T>) -> Self::Output {
                let self_converted = self.clone().to_row();
                assert!(self_converted.cells.len() == rhs.cells.len(), "Provided Rows are not of same length");
                let cells = self_converted.cells.iter().zip(rhs.cells.iter()).map(|(x_l, x_r)| x_l + x_r).collect::<Vec<$T>>();
                SpatialVectorNDof {vector: cells, _orientation: PhantomData::<IsRow>}
            }
        }

        impl Add<&SpatialVectorNDof<$T, IsRow>> for &SpatialVectorNDof<$T, IsRow> {
            type Output = SpatialVectorNDof<$T, IsRow>;

            fn add(self, rhs: &SpatialVectorNDof<$T, IsRow>) -> Self::Output {
                let self_converted = self.clone().to_row();
                let rhs = rhs.clone().to_row();
                assert!(self_converted.cells.len() == rhs.cells.len(), "Provided Rows are not of same length");
                let cells = self_converted.cells.iter().zip(rhs.cells.iter()).map(|(x_l, x_r)| x_l + x_r).collect::<Vec<$T>>();
                SpatialVectorNDof {vector: cells, _orientation: PhantomData::<IsRow>}
            }
        }

        impl Sub<&Row<$T>> for &SpatialVectorNDof<$T, IsRow> {
            type Output = SpatialVectorNDof<$T, IsRow>;

            fn sub(self, rhs: &Row<$T>) -> Self::Output {
                let self_converted = self.clone().to_row();
                let rhs = rhs.clone().to_row();
                assert!(self_converted.cells.len() == rhs.cells.len(), "Provided Rows are not of same length");
                let cells = self_converted.cells.iter().zip(rhs.cells.iter()).map(|(x_l, x_r)| x_l - x_r).collect::<Vec<$T>>();
                SpatialVectorNDof {vector: cells, _orientation: PhantomData::<IsRow>}
            }
        }

        impl Sub<&SpatialVectorNDof<$T, IsRow>> for &SpatialVectorNDof<$T, IsRow> {
            type Output = SpatialVectorNDof<$T, IsRow>;

            fn sub(self, rhs: &SpatialVectorNDof<$T, IsRow>) -> Self::Output {
                let self_converted = self.clone().to_row();
                let rhs = rhs.clone().to_row();
                assert!(self_converted.cells.len() == rhs.cells.len(), "Provided Rows are not of same length");
                let cells = self_converted.cells.iter().zip(rhs.cells.iter()).map(|(x_l, x_r)| x_l - x_r).collect::<Vec<$T>>();
                SpatialVectorNDof {vector: cells, _orientation: PhantomData::<IsRow>}
            }
        }

        impl Mul<&Collumn<$T>> for &SpatialVectorNDof<$T, IsRow> {
            type Output = $T;

            fn mul(self, rhs: &Collumn<$T>) -> Self::Output {
                let self_converted = self.clone().to_row();
                $T::row_dot_collumn(&self_converted, &rhs)
            }
        }

        impl Mul<&SpatialVectorNDof<$T, IsColl>> for &SpatialVectorNDof<$T, IsRow> {
            type Output = $T;

            fn mul(self, rhs: &SpatialVectorNDof<$T, IsColl>) -> Self::Output {
                let self_converted = self.clone().to_row();
                let rhs = rhs.clone().to_collumn();
                $T::row_dot_collumn(&self_converted, &rhs)
            }
        }

        impl Mul<&Matrix<$T>> for &SpatialVectorNDof<$T, IsRow> {
            type Output = SpatialVectorNDof<$T, IsRow>;

            fn mul(self, rhs: &Matrix<$T>) -> Self::Output {
                let self_converted = self.clone().to_row();
                let result_row = $T::row_dot_matrix(&self_converted, &rhs);
                SpatialVectorNDof {vector: result_row.cells, _orientation: PhantomData::<IsRow>}
            }
        }
    };
}

impl_std_ops_row_per_type!(i8);
impl_std_ops_row_per_type!(i16);
impl_std_ops_row_per_type!(i32);
impl_std_ops_row_per_type!(i64);

impl_std_ops_row_per_type!(isize);

impl_std_ops_row_per_type!(f32);
impl_std_ops_row_per_type!(f64);

impl_std_ops_vector_row_per_type!(i8);
impl_std_ops_vector_row_per_type!(i16);
impl_std_ops_vector_row_per_type!(i32);
impl_std_ops_vector_row_per_type!(i64);

impl_std_ops_vector_row_per_type!(isize);

impl_std_ops_vector_row_per_type!(f32);
impl_std_ops_vector_row_per_type!(f64);

impl_row_type!(i8);
impl_row_type!(i16);
impl_row_type!(i32);
impl_row_type!(i64);

impl_row_type!(u8);
impl_row_type!(u16);
impl_row_type!(u32);
impl_row_type!(u64);

impl_row_type!(isize);
impl_row_type!(usize);

impl_row_type!(f32);
impl_row_type!(f64);