#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(unused_imports)]

use crate::*;

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

pub trait CollumnFunctions<T: MatrixValues> {
    fn n_rows(&self) -> usize;
    fn len(&self) -> usize;
    fn height(&self) -> usize;
    fn export(self) -> Collumn<T>;
    fn clone(&self) -> Collumn<T>;
    fn new_with_constant_values(length: usize, value: T) -> Collumn<T>;
    fn new_form_vec(input_vector: Vec<T>) -> Collumn<T>;
    fn to_vec(&self) -> Vec<T> ;
    fn extend_with_collumn(&mut self, other_collumn: Collumn<T>) ;
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
    }
    };
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