use std::ops::{Index, IndexMut};
use std::any::{TypeId};
use std::ops::Range;
use std::fmt::Display;
use std::vec;

use num::{Num, NumCast};

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
}

pub struct Matrix<T: MatrixValues> {
    pub rows : Vec<Row<T>>
}

pub trait MatrixValues: Copy + Display + PartialEq + Num + NumCast{}
impl<T> MatrixValues for T where T: Copy + Display + PartialEq + Num + NumCast {}

impl<T: MatrixValues> Matrix<T> {
    pub fn new_square_with_value(size:usize, value: T) -> Matrix<T> {
        let mut rows = Vec::<Row<T>>::with_capacity(size);
        let mut rowsr = Vec::<& Row<T>>::with_capacity(size);
        for index in 0..size {
            rows.push(Row::new_row_with_value(size, value));
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

    pub fn coll_length(&self) -> usize {
        self.rows.len()
    }

    pub fn get_collumn(&self, coll_number : usize) -> Collumn<T> {
        let mut cells = Vec::<T>::with_capacity(self.coll_length());
        for row_number in 0..self.coll_length(){
            cells.push( self[row_number][coll_number])
        }
        Collumn {cells}
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

    pub fn substract_multiplied_internal_row_from_row_by_index(&mut self, row_number_to_substract: usize, factor: T , from_row_number: usize) {
        let mut mutliplied_row_to_substract = self[row_number_to_substract].clone();
        mutliplied_row_to_substract.multiply_all_elements_by(factor);
        self[from_row_number].substract_row(mutliplied_row_to_substract)
    }

    pub fn new_from_solver(system_of_equations : Solver2D<T>) -> Solved2D<T>{
        match system_of_equations.solver {
            Solver2DStrategy::Guass => solve_with_guass(system_of_equations),
            _ => (panic!("Error: Solver not regocnized!"))
        }
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

    pub fn append_row_from_vec(&mut self, new_row_vec: Vec<T>) {
        let new_row = Row::new_row_from_vec(new_row_vec);
        self.append_row(new_row);
    }

    pub fn multiply_all_elements_by(&mut self, factor: T) {
        for row_number in 0..self.rows.len() {
            self.rows[row_number].multiply_all_elements_by(factor)
        }
    }

    pub fn divide_all_elements_by(&mut self, factor: T) {
        for row_number in 0..self.rows.len() {
            self.rows[row_number].divide_all_elements_by(factor)
        }
    }

}

fn solve_with_guass<T: MatrixValues>(system_of_equations: Solver2D<T>) -> Solved2D<T> {
    let mut A_matrix = system_of_equations.A_matrix.clone();
    let mut B_matrix = system_of_equations.B_matrix.clone();

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

impl<T: MatrixValues> Matrix<T>{
    pub fn printer(&self) {
        println!("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
        println!("Printing the Matrix");
        println!("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
        for row_number in 0..self.rows.len(){
            print!("row {} :", row_number);
            for coll in 0..self.rows[row_number].cells.len(){
                print!(" {} ", self.rows[row_number].cells[coll])
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
            panic!()
        } else if !(colls_input.len() == matrix_input.row_length()) {
            println!("Error: Given collumn range does not match given values ");
            panic!()
        }

        let mut i : usize = 0;
        let mut j : usize = 0;
        for row in &rows_input {
            for coll in &colls_input {
                self[*row][*coll] = matrix_input[i][j];
                j += 1;
            }
            i += 1;
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
    fn new_row_with_value(size: usize, value: T) -> Row<T> {
        let mut cells = Vec::<T>::with_capacity(size);
        for _ in 0..size{
            cells.push(value);
        }
        Row { cells}
    }
    
    fn new_row_from_vec(input_vec: Vec<T>) -> Row<T> {
        Row { cells: input_vec }
    }

    fn len(&self) -> usize {
        self.cells.len()
    }

    fn export(self) -> Row<T> {
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
            self.cells[n] = self.cells[n] / value;
        }
    }

    pub fn multiply_all_elements_by(&mut self, value: T) {
        for n in 0..self.cells.len() {
            self.cells[n] = self.cells[n] * value;
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
        for cell_number in 0..self.cells.len() {
            self[cell_number] = self[cell_number] - substraction_row[cell_number];
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

pub struct Collumn<T: MatrixValues> {
    cells : Vec<T>
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
}


pub fn linspace<T: MatrixValues>(lower_bound: T, higher_bound: T, steps: usize) -> Vec<T> {
    let step_size = (higher_bound - lower_bound) / (NumCast::from(steps-1).unwrap());
    let mut lin_spaced_vec = Vec::<T>::with_capacity(steps);
    for i in 0..steps {
        lin_spaced_vec.push(lower_bound + step_size * NumCast::from(i).unwrap());
    }
    lin_spaced_vec
}