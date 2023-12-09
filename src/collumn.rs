#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(unused_imports)]

use crate::*;

use std::ops::Mul;
use std::ops::Sub;
use std::ops::Add;
use std::ops::Div;
use std::marker::PhantomData;

#[derive(Debug)]
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

impl<T: MatrixValues> From<SpatialVectorNDof<T, IsColl>> for Collumn<T> {
    fn from(spatial_vector: SpatialVectorNDof<T, IsColl>) -> Self {
        Collumn { cells: spatial_vector.vector }
    }
}

impl<T: MatrixValues> From<SpatialVectorWithBasePointNDof<T, IsColl>> for Collumn<T> {
    fn from(spatial_vector: SpatialVectorWithBasePointNDof<T, IsColl>) -> Self {
        Collumn { cells: spatial_vector.vector }
    }
}

macro_rules! impl_std_ops_coll_per_type {
    ($T: ident) => {
        impl Add<Collumn<$T>> for Collumn<$T> {
            type Output = Collumn<$T>;

            fn add(self, rhs: Collumn<$T>) -> Self::Output {
                assert!(self.cells.len() == rhs.cells.len(), "Provided Colls are not of same length");
                let cells = self.cells.iter().zip(rhs.cells.iter()).map(|(x_l, x_r)| x_l + x_r).collect::<Vec<$T>>();
                Collumn { cells }
            }
        }

        impl Add<SpatialVectorNDof<$T, IsColl>> for Collumn<$T> {
            type Output = Collumn<$T>;

            fn add(self, rhs: SpatialVectorNDof<$T, IsColl>) -> Self::Output {
                let rhs = rhs.to_collumn();
                assert!(self.cells.len() == rhs.cells.len(), "Provided Colls are not of same length");
                let cells = self.cells.iter().zip(rhs.cells.iter()).map(|(x_l, x_r)| x_l + x_r).collect::<Vec<$T>>();
                Collumn { cells }
            }
        }

        impl Sub<Collumn<$T>> for Collumn<$T> {
            type Output = Collumn<$T>;

            fn sub(self, rhs: Collumn<$T>) -> Self::Output {
                assert!(self.cells.len() == rhs.cells.len(), "Provided Colls are not of same length");
                let cells = self.cells.iter().zip(rhs.cells.iter()).map(|(x_l, x_r)| x_l + x_r).collect::<Vec<$T>>();
                Collumn { cells }
            }
        }

        impl Sub<SpatialVectorNDof<$T, IsColl>> for Collumn<$T> {
            type Output = Collumn<$T>;

            fn sub(self, rhs: SpatialVectorNDof<$T, IsColl>) -> Self::Output {
                let rhs = rhs.to_collumn();
                assert!(self.cells.len() == rhs.cells.len(), "Provided Colls are not of same length");
                let cells = self.cells.iter().zip(rhs.cells.iter()).map(|(x_l, x_r)| x_l + x_r).collect::<Vec<$T>>();
                Collumn { cells }
            }
        }

        impl Mul<Row<$T>> for Collumn<$T> {
            type Output = Matrix<$T>;

            fn mul(self, rhs: Row<$T>) -> Self::Output {
                $T::collumn_dot_row(&self, &rhs)
            }
        }

        impl Mul<SpatialVectorNDof<$T, IsRow>> for Collumn<$T> {
            type Output = Matrix<$T>;

            fn mul(self, rhs: SpatialVectorNDof<$T, IsRow>) -> Self::Output {
                let rhs = rhs.to_row();
                $T::collumn_dot_row(&self, &rhs)
            }
        }

        impl Mul<Matrix<$T>> for Collumn<$T> {
            type Output = Matrix<$T>;

            fn mul(self, rhs: Matrix<$T>) -> Self::Output {
                $T::collumn_dot_matrix(&self, &rhs)
            }
        }

        impl Mul<$T> for Collumn<$T> {
            type Output = Collumn<$T>;

            fn mul(mut self, rhs: $T) -> Self::Output {
                self.multiply_all_elements_by(rhs);
                self
            }
        }

        impl Div<$T> for Collumn<$T> {
            type Output = Collumn<$T>;

            fn div(mut self, rhs: $T) -> Self::Output {
                self.divide_all_elements_by(rhs);
                self
            }
        }
    }
}

macro_rules! impl_std_ops_vec_coll_per_type {
    ($T: ident) => {
        impl Add<Collumn<$T>> for SpatialVectorNDof<$T, IsColl> {
            type Output = SpatialVectorNDof<$T, IsColl>;

            fn add(self, rhs: Collumn<$T>) -> Self::Output {
                let mut converted_self = self.to_collumn();
                assert!(converted_self.cells.len() == rhs.cells.len(), "Provided Colls are not of same length");
                let cells = converted_self.cells.iter().zip(rhs.cells.iter()).map(|(x_l, x_r)| x_l + x_r).collect::<Vec<$T>>();
                SpatialVectorNDof {vector: cells, _orientation: PhantomData::<IsColl>}
            }
        }

        impl Add<SpatialVectorNDof<$T, IsColl>> for SpatialVectorNDof<$T, IsColl> {
            type Output = SpatialVectorNDof<$T, IsColl>;

            fn add(self, rhs: SpatialVectorNDof<$T, IsColl>) -> Self::Output {
                let mut converted_self = self.to_collumn();
                let rhs = rhs.to_collumn();
                assert!(converted_self.cells.len() == rhs.cells.len(), "Provided Colls are not of same length");
                let cells = converted_self.cells.iter().zip(rhs.cells.iter()).map(|(x_l, x_r)| x_l + x_r).collect::<Vec<$T>>();
                SpatialVectorNDof {vector: cells, _orientation: PhantomData::<IsColl>}
            }
        }

        impl Sub<Collumn<$T>> for SpatialVectorNDof<$T, IsColl> {
            type Output = SpatialVectorNDof<$T, IsColl>;

            fn sub(self, rhs: Collumn<$T>) -> Self::Output {
                let mut converted_self = self.to_collumn();
                assert!(converted_self.cells.len() == rhs.cells.len(), "Provided Colls are not of same length");
                let cells = converted_self.cells.iter().zip(rhs.cells.iter()).map(|(x_l, x_r)| x_l - x_r).collect::<Vec<$T>>();
                SpatialVectorNDof {vector: cells, _orientation: PhantomData::<IsColl>}
            }
        }

        impl Sub<SpatialVectorNDof<$T, IsColl>> for SpatialVectorNDof<$T, IsColl> {
            type Output = SpatialVectorNDof<$T, IsColl>;

            fn sub(self, rhs: SpatialVectorNDof<$T, IsColl>) -> Self::Output {
                let mut converted_self = self.to_collumn();
                let rhs = rhs.to_collumn();
                assert!(converted_self.cells.len() == rhs.cells.len(), "Provided Colls are not of same length");
                let cells = converted_self.cells.iter().zip(rhs.cells.iter()).map(|(x_l, x_r)| x_l - x_r).collect::<Vec<$T>>();
                SpatialVectorNDof {vector: cells, _orientation: PhantomData::<IsColl>}
            }
        }

        impl Mul<Row<$T>> for SpatialVectorNDof<$T, IsColl> {
            type Output = Matrix<$T>;

            fn mul(self, rhs: Row<$T>) -> Self::Output {
                let mut converted_self = self.to_collumn();
                $T::collumn_dot_row(&converted_self, &rhs)
            }
        }

        impl Mul<SpatialVectorNDof<$T, IsRow>> for SpatialVectorNDof<$T, IsColl> {
            type Output = Matrix<$T>;

            fn mul(self, rhs: SpatialVectorNDof<$T, IsRow>) -> Self::Output {
                let mut converted_self = self.to_collumn();
                let rhs = rhs.to_row();
                $T::collumn_dot_row(&converted_self, &rhs)
            }
        }

        impl Mul<Matrix<$T>> for SpatialVectorNDof<$T, IsColl> {
            type Output = Matrix<$T>;

            fn mul(self, rhs: Matrix<$T>) -> Self::Output {
                let mut converted_self = self.to_collumn();
                $T::collumn_dot_matrix(&converted_self, &rhs)
            }
        }

        impl Mul<$T> for SpatialVectorNDof<$T, IsColl> {
            type Output = SpatialVectorNDof<$T, IsColl>;

            fn mul(mut self, rhs: $T) -> Self::Output {
                let mut converted_self = self.to_collumn();
                converted_self.multiply_all_elements_by(rhs);
                SpatialVectorNDof {vector: converted_self.cells, _orientation: PhantomData::<IsColl>}
            }
        }

        impl Div<$T> for SpatialVectorNDof<$T, IsColl> {
            type Output = SpatialVectorNDof<$T, IsColl>;

            fn div(mut self, rhs: $T) -> Self::Output {
                let mut converted_self = self.to_collumn();
                converted_self.divide_all_elements_by(rhs);
                SpatialVectorNDof {vector: converted_self.cells, _orientation: PhantomData::<IsColl>}
            }
        }
    }
}

impl_std_ops_coll_per_type!(i8);
impl_std_ops_coll_per_type!(i16);
impl_std_ops_coll_per_type!(i32);
impl_std_ops_coll_per_type!(i64);

impl_std_ops_coll_per_type!(isize);

impl_std_ops_coll_per_type!(f32);
impl_std_ops_coll_per_type!(f64);

impl_std_ops_vec_coll_per_type!(i8);
impl_std_ops_vec_coll_per_type!(i16);
impl_std_ops_vec_coll_per_type!(i32);
impl_std_ops_vec_coll_per_type!(i64);

impl_std_ops_vec_coll_per_type!(isize);

impl_std_ops_vec_coll_per_type!(f32);
impl_std_ops_vec_coll_per_type!(f64);

pub trait CollumnFunctions<T: MatrixValues> {
    fn n_rows(&self) -> usize;
    fn len(&self) -> usize;
    fn height(&self) -> usize;
    fn export(self) -> Collumn<T>;
    fn transpose(&self) -> Row<T>;
    fn clone(&self) -> Collumn<T>;
    fn new_with_constant_values(length: usize, value: T) -> Collumn<T>;
    fn new_form_vec(input_vector: Vec<T>) -> Collumn<T>;
    fn to_vec(&self) -> Vec<T> ;
    fn extend_with_collumn(&mut self, other_collumn: Collumn<T>);
    fn multiply_all_elements_by(&mut self, factor: T);
    fn divide_all_elements_by(&mut self, factor: T);
}

macro_rules! impl_collumn_functions_per_type {
    ($T: ident) => {
        impl CollumnFunctions<$T> for Collumn<$T> {
            fn n_rows(&self) -> usize {
                self.cells.len()
            }
        
            fn len(&self) -> usize {
                self.cells.len()
            }
        
            fn height(&self) -> usize {
                self.cells.len()
            }
        
            fn export(self) -> Collumn<$T> {
                self
            }

            fn transpose(&self) -> Row<$T> {
                Row { cells: self.cells.clone() }
            }
        
            fn clone(&self) -> Collumn<$T> {
                let mut cells = Vec::<$T>::with_capacity(self.len());
                for cell in &self.cells{
                    cells.push(cell.clone());
                }
                Collumn { cells}
            }
        
            fn new_with_constant_values(length: usize, value: $T) -> Collumn<$T> {
                let cells: Vec<$T> = (0..length).into_iter().map(|_| value).collect();
        
                Collumn { cells }
            }
        
            fn new_form_vec(input_vector: Vec<$T>) -> Collumn<$T>{
                let mut cells = Vec::<$T>::with_capacity(input_vector.len());
                for index in 0..input_vector.len() {
                    cells.push(input_vector[index])
                }
        
                Collumn {cells}
            }
        
            fn to_vec(&self) -> Vec<$T> {
                let mut output_vec = Vec::<$T>::with_capacity(self.cells.len());
                for i in 0..self.cells.len() {
                    output_vec.push(self.cells[i]);
                }
                output_vec
            }
        
            fn extend_with_collumn(&mut self, other_collumn: Collumn<$T>) {
                self.cells.extend(other_collumn.cells);
            }

            fn multiply_all_elements_by(&mut self, factor: $T) {
                for cell in self.cells.iter_mut() {
                    *cell = *cell * factor;
                }
            }

            fn divide_all_elements_by(&mut self, factor: $T) {
                for cell in self.cells.iter_mut() {
                    *cell = *cell / factor;
                }
            }
    }
    };
}

pub trait IntoCollumn<T: MatrixValues> {
    fn to_collumn(self) -> Collumn<T>;
}

impl<T: MatrixValues> IntoCollumn<T> for Collumn<T> {
    fn to_collumn(self) -> Collumn<T> {
        self
    }
}

impl<T: MatrixValues> IntoCollumn<T> for SpatialVectorNDof<T, IsColl> {
    fn to_collumn(self) -> Collumn<T> {
        Collumn { cells: self.vector }
    }
}

impl_collumn_functions_per_type!(i8);
impl_collumn_functions_per_type!(i16);
impl_collumn_functions_per_type!(i32);
impl_collumn_functions_per_type!(i64);

impl_collumn_functions_per_type!(u8);
impl_collumn_functions_per_type!(u16);
impl_collumn_functions_per_type!(u32);
impl_collumn_functions_per_type!(u64);

impl_collumn_functions_per_type!(isize);
impl_collumn_functions_per_type!(usize);

impl_collumn_functions_per_type!(f32);
impl_collumn_functions_per_type!(f64);