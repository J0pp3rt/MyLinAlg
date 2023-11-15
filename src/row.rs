#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(unused_imports)]

use crate::*;

#[derive(Debug)]
pub struct Row<T: MatrixValues> {
    pub cells : Vec<T>,
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
    fn clone(&self) -> Row<T>;
    fn divide_all_elements_by(&mut self, value: T);
    fn multiply_all_elements_by(&mut self, value: T)-> &Self;
    fn addition_row_with_external_row(&mut self, row_to_add_to_this_one:& Row<T>);
    fn normalize_all_elements_to_element(&mut self, index: usize);
    fn normalize_all_elements_to_first(&mut self) ;
    fn substract_row(&mut self, substraction_row: Row<T>) ;
    fn substract_all(&mut self, substraction_row: Row<T>);
    fn replace_values(&mut self, index_range: Range<usize>, values: Vec<T>);

}

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
        
            fn clone(&self) -> Row<$T> {
                let mut cells = Vec::<$T>::with_capacity(self.len());
                for cell in &self.cells{
                    cells.push(cell.clone());
                }
                Row { cells}
            }
        
            fn divide_all_elements_by(&mut self, value: $T) {
                for n in 0..self.cells.len() {
                    // if !(self.cells[n] == NumCast::from(0).unwrap()) {// quickly tested on some sparse matrices but seem to really boost performance. In some more filled ones: around 50x improvemnt, ful matrix not tested yet
                        self.cells[n] = self.cells[n] / value;
                    // }
                }
            }
        
            fn multiply_all_elements_by(&mut self, value: $T) -> &Self{
                for n in 0..self.cells.len() {
                    // if !(self.cells[n] == NumCast::from(0).unwrap()) {// quickly tested on some sparse matrices but seem to really boost performance. In some more filled ones: around 50x improvemnt, ful matrix not tested yet
                    self.cells[n] = self.cells[n] * value;
                    // }
                }
        
                self
            }
        
            fn addition_row_with_external_row(&mut self, row_to_add_to_this_one:& Row<$T>) {
                for n in 0..self.cells.len() {
                    // if !(self.cells[n] == NumCast::from(0).unwrap() && row_to_add_to_this_one.cells[n] == NumCast::from(0).unwrap()) {
                        self.cells[n] = self.cells[n] + row_to_add_to_this_one[n];
                    // }
                }
            }
        
            
        
            fn normalize_all_elements_to_element(&mut self, index: usize) {
                self.divide_all_elements_by(self[index]);
            }
        
            fn normalize_all_elements_to_first(&mut self) {
                self.normalize_all_elements_to_element(0);
            }
        
            fn substract_row(&mut self, substraction_row: Row<$T>) {
                if !(self.cells.len() == substraction_row.cells.len()) {
                    panic!("Error: Length of substracting row is not equal to row length")
                }
                if *IS_AVX2{
                    unsafe {self.substract_avx2_row(substraction_row)}
                } else {
                    self.substract_all(substraction_row)
                }
            }
        
        
            fn substract_all(&mut self, substraction_row: Row<$T>) {
                for cell_number in 0..self.cells.len() {
                    // if !(self[cell_number] == NumCast::from(0).unwrap() || substraction_row[cell_number] == NumCast::from(0).unwrap()) { // quickly tested on some sparse matrices but seem to really boost performance . In some more filled ones: around 50x improvemnt, ful matrix not tested yet
                    self[cell_number] = self[cell_number] - substraction_row[cell_number];
                    // }
                }
            }
        
            fn replace_values(&mut self, index_range: Range<usize>, values: Vec<$T>) {
                // maybe add a check if not of same size TODO
                // instead of range use rangebounds apperantly
                for (val_index, row_index) in index_range.enumerate() {
                    self.cells[row_index] = values[val_index];
                }
            }
        }
    }
}


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