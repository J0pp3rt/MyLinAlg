#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(unused_imports)]

use crate::*;

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
