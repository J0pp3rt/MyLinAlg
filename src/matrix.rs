#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(unused_imports)]

use crate::*;

pub struct Matrix<T: MatrixValues> {
    pub rows : Vec<Row<T>>
}

pub trait MatrixValues: Copy + Display + PartialEq + Num + NumCast + Copy + 'static{
}
impl<T> MatrixValues for T where T: Copy + Display + PartialEq + Num + NumCast + Copy + 'static{
}

// macro_rules! impl_add_matrix { // todo: reintegrate this
//     ($T: ident) => {
//         impl Add for Matrix<$T> {
//             type Output = Self;
        
//             fn add(mut self, other_matrix: Self) -> Self::Output {
//                 if !(self.height() == other_matrix.height() || self.width() == other_matrix.width()) {
//                     panic!("Dimensions do not match for addition! Given [{}x{}] + [{}+{}]", self.height(), self.width(), other_matrix.height(), other_matrix.width());
//                 }
        
//                 self.addition_on_partial(0..self.height(), 0..self.width(), other_matrix);

//                 self
//             }
//         }
//     };
// }

// impl_add_matrix!(f64);



pub trait MatrixFunctions<T:MatrixValues> {
    fn new_square_with_constant_values(n_rows:usize, value: T) -> Matrix<T> ;
    fn new_with_constant_values(n_rows:usize, n_collumns: usize, value: T) -> Matrix<T> ;
    fn new_from_vector_rows(input: Vec<Vec<T>>) -> Matrix<T> ;
    fn new_from_collumn(input_collumn: Collumn<T>) -> Matrix<T>;
    fn new_square_eye(size: usize, values_of_diagonal: T) -> Matrix<T> ;
    fn new_eye(n_rows: usize, n_collumns: usize, values_of_diagonal: T) -> Matrix<T> ;
    fn new_from_row_major_vector(vector: Vec<T>, height: usize, width: usize) -> Self ;
    fn clone(&self) -> Matrix<T> ;
    fn len(&self) -> usize ;
    fn row_length(&self) -> usize ;
    fn diagonal_contain_zeros(&self) -> bool ;
    fn new_with_random_ones_chance(height: usize, width: usize, chance: f64, base_value: T) -> Self ;
    fn remove_last_row(&mut self) -> &Self ;
    fn remove_last_collumn(&mut self) -> &Self ;
    fn width(&self) -> usize ;
    fn coll_length(&self) -> usize ;
    fn height(&self) -> usize ;
    fn is_square(&self) -> bool ;
    fn get_collumn(&self, coll_number : usize) -> Collumn<T> ;
    fn to_vec_collumn_major(&self) -> Vec<T> ;
    fn to_vec_row_major(&self) -> Vec<T> ;
    fn swap_rows(&mut self, row_1: usize, row_2: usize);
    fn substract_internal_row_from_row_by_index(&mut self, row_number_to_substract: usize, from_row_number: usize);
    fn substract_multiplied_internal_row_from_row_by_index(&mut self, row_number_to_substract_with: usize, factor: T , from_row_number: usize) ;
    fn substract_multiplied_internal_row_from_row_by_index_with_collumn_range<U>(&mut self, row_number_to_substract_with: usize, factor: T , from_row_number: usize, collumn_range: U) where  U: InputTraitRowCol<U> ;
    fn new_from_solver(system_of_equations : Solver2D<T>) -> Solved2D<T>;
    fn transpose_square(&mut self) -> &Self ;
    fn transpose_non_skinny(&mut self) -> &Self;
    fn make_square(&mut self) -> &Self;
    fn add_row(&mut self, insert_row_at: usize, new_row: Row<T>);
    fn add_row_from_vec(&mut self, insert_row_at: usize, new_row_vec: Vec::<T>) ;
    fn append_row(&mut self, new_row: Row<T>) ;
    fn append_row_zeros(&mut self);
    fn append_collumn_zeros(&mut self);
    fn append_collumn_zeros_n_times(&mut self, n: usize) ;
    fn append_collumn(&mut self, collumn: Collumn<T>) -> &Self ;
    fn append_row_from_vec(&mut self, new_row_vec: Vec<T>);
    fn multiply_all_elements_by(&mut self, factor: T) -> &Self;
    fn divide_all_elements_by(&mut self, factor: T);

}

macro_rules! impl_matrix_functions_for_type {
    ($T: ident) => {
        impl MatrixFunctions<$T> for Matrix<$T> {
        fn new_square_with_constant_values(n_rows:usize, value: $T) -> Matrix<$T> {
            let mut rows = Vec::<Row<$T>>::with_capacity(n_rows);
            for _ in 0..n_rows {
                rows.push(Row::new_row_with_value(n_rows, value));
            } 
            Matrix {
                 rows,
                }
        }
    
        fn new_with_constant_values(n_rows:usize, n_collumns: usize, value: $T) -> Matrix<$T> {
            let mut rows = Vec::<Row<$T>>::with_capacity(n_rows);
            for _ in 0..n_rows {
                rows.push(Row::new_row_with_value(n_collumns, value));
            } 
            Matrix {
                 rows,
                }
        }
    
        fn new_from_vector_rows(input: Vec<Vec<$T>>) -> Matrix<$T> {
            let mut rows = Vec::<Row<$T>>::with_capacity(input.len());
            for dimension in input{
                rows.push( Row { cells : dimension});
            }
            Matrix { rows }
        }
    
        fn new_from_collumn(input_collumn: Collumn<$T>) -> Matrix<$T>{
            let mut rows = Vec::<Row<$T>>::with_capacity(input_collumn.n_rows());
            for row_number in 0..input_collumn.n_rows(){
                rows.push(Row { cells: vec![input_collumn[row_number]] })
            }
    
            Matrix { rows }
        }
    
        fn new_square_eye(size: usize, values_of_diagonal: $T) -> Matrix<$T> {
            println!("creating fixed ");
            let mut new_matrix: Matrix<$T> = Matrix::new_square_with_constant_values(size, NumCast::from(0).unwrap());
            for j in 0..size {
                new_matrix[j][j] = values_of_diagonal;
            }
            new_matrix
        }
    
        fn new_eye(n_rows: usize, n_collumns: usize, values_of_diagonal: $T) -> Matrix<$T> {
            let mut new_matrix: Matrix<$T> = Matrix::new_with_constant_values(n_rows, n_collumns, NumCast::from(0).unwrap());
            
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
        
        fn new_from_row_major_vector(vector: Vec<$T>, height: usize, width: usize) -> Self {
            if !(vector.len() == height*width) {
                panic!("Given dimensions do not match! Vec length: {}, height x width = {} x {} = {}", vector.len(), height, width, height*width);
            }
    
            let mut result_matrix: Matrix<$T> = Matrix::new_with_constant_values(height, width, NumCast::from(0).unwrap());
    
            for row_index in 0..height {
                let row = Row::new_row_from_vec(
                    vector[row_index*width..(row_index+1)*width].to_owned()
                );
                result_matrix[row_index] = row;
            }
    
            result_matrix
        }
    
        fn clone(&self) -> Matrix<$T> {
            let mut rows = Vec::<Row<$T>>::with_capacity(self.rows.len());
            for row in &self.rows{
                rows.push( row.clone());
            }
            Matrix { rows }
        }
    
        fn len(&self) -> usize {
            if self.row_length() >= self.coll_length() {
                self.row_length()
            } else {
                self.coll_length()
            }
        }
    
        fn row_length(&self) -> usize {
            let mut largest_length = self.rows[0].len();
            for row in &self.rows {
                if row.len() > largest_length {
                    largest_length = row.len();
                    println!("Row lengths of matrix not consistant!")
                }
            }
            largest_length
        }
    
        fn diagonal_contain_zeros(&self) -> bool {
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
    
        fn new_with_random_ones_chance(height: usize, width: usize, chance: f64, base_value: $T) -> Self {
            let mut result_matrix: Matrix<$T> = Matrix::new_with_constant_values(height, width, base_value);
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
    
        fn remove_last_row(&mut self) -> &Self {
            self.rows.pop().unwrap();
    
            self
        }
    
        fn remove_last_collumn(&mut self) -> &Self {
            for row_index in 0..self.height() {
                self[row_index].cells.pop();
            }
    
            self
        }
    
        fn width(&self) -> usize {
            self.row_length()
        }
    
        fn coll_length(&self) -> usize {
            self.rows.len()
        }
    
        fn height(&self) -> usize {
            self.coll_length()
        }
    
        fn is_square(&self) -> bool {
            self.height() == self.width()
        }
    
        fn get_collumn(&self, coll_number : usize) -> Collumn<$T> {
            let mut cells = Vec::<$T>::with_capacity(self.coll_length());
            for row_number in 0..self.coll_length(){
                cells.push( self[row_number][coll_number])
            }
            Collumn {cells}
        }
    
        fn to_vec_collumn_major(&self) -> Vec<$T> {
            let N = self.width()*self.height();
            let mut output_vec = Vec::<$T>::with_capacity(N);
    
            for column_number in 0..self.width() {
                for row_number in 0..self.height() {
                    output_vec.push(self[row_number][column_number]);
                }
            }
    
            output_vec
        }
    
        fn to_vec_row_major(&self) -> Vec<$T> {
            let N = self.width()*self.height();
            let mut output_vec = Vec::<$T>::with_capacity(N);
    
            for row_number in 0..self.height() {
                output_vec.extend(self[row_number].cells.clone());
            }
    
            output_vec
        }
    
        fn swap_rows(&mut self, row_1: usize, row_2: usize){
            let row_1_copy = self[row_1].clone();
            
            self[row_1] = self[row_2].clone();
            self[row_2] = row_1_copy;
        }
    
        fn substract_internal_row_from_row_by_index(&mut self, row_number_to_substract: usize, from_row_number: usize) {
            let row_to_substract = self[row_number_to_substract].clone();
            self[from_row_number].substract_row(row_to_substract)
        }
    
        fn substract_multiplied_internal_row_from_row_by_index(&mut self, row_number_to_substract_with: usize, factor: $T , from_row_number: usize) {
            let mut mutliplied_row_to_substract = self[row_number_to_substract_with].clone();
            mutliplied_row_to_substract.multiply_all_elements_by(factor);
            self[from_row_number].substract_row(mutliplied_row_to_substract)
        }
    
        fn substract_multiplied_internal_row_from_row_by_index_with_collumn_range<U>(&mut self, row_number_to_substract_with: usize, factor: $T , from_row_number: usize, collumn_range: U)
        where  U: InputTraitRowCol<U> {
            let colls_input:Vec<usize> = parse_dimension_input(collumn_range);
    
            for collumn_index in colls_input.iter(){
                self[from_row_number][*collumn_index] = self[from_row_number][*collumn_index] - self[row_number_to_substract_with][*collumn_index] * factor
            }  
        }
    
        fn new_from_solver(system_of_equations : Solver2D<$T>) -> Solved2D<$T>{
            match system_of_equations.solver {
                Solver2DStrategy::Guass => $T::solve_with_guass(system_of_equations),
                Solver2DStrategy::CholeskySquareRootFreeDecomposition => $T::solve_with_cholesky_quare_root_free_decomposition(system_of_equations),
                // Solver2DStrategy::LUDecomposition => todo!(),
                _ => (panic!("Error: Solver not yet implemented!"))
            }
        }
    
        fn transpose_square(&mut self) -> &Self {
            for row_index in 0..self.height()-1 {
                for collumn_index in row_index+1..self.width() {
                    let buffer = self[collumn_index][row_index];
                    self[collumn_index][row_index] = self[row_index][collumn_index];
                    self[row_index][collumn_index] = buffer;
                }
            }
            self
        }
    
        fn transpose_non_skinny(&mut self) -> &Self {
            let initial_height = self.height();
            let initial_width = self.width();
            if initial_height == initial_width {
                return self.transpose_square();
            }
    
            let smallest_dimension_was_height: bool;
            let dimension_difference: usize = isize::abs(initial_height as isize- initial_width as isize) as usize;
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
    
            let dimension_difference = isize::abs(initial_height as isize - initial_width as isize) as usize;
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
    
        fn add_row(&mut self, insert_row_at: usize, new_row: Row<$T>) {
            if insert_row_at == self.rows.len() {
                self.append_row(new_row);
            } else {
                self.rows.insert(insert_row_at, new_row)
            }
        }
    
        fn add_row_from_vec(&mut self, insert_row_at: usize, new_row_vec: Vec::<$T>) {
            let new_row = Row::new_row_from_vec(new_row_vec);
            self.add_row(insert_row_at, new_row);
        }
    
        fn append_row(&mut self, new_row: Row<$T>) {
                self.rows.push(new_row);
        }
    
        fn append_row_zeros(&mut self) {
            let width = self.width();
            self.rows.push( Row::new_row_with_constant_values(width, NumCast::from(0).unwrap()));
        }
    
        fn append_collumn_zeros(&mut self) {
            let height = self.height();
            let zero: $T = NumCast::from(0).unwrap();
            let collumn= Collumn::new_form_vec((0..height).into_iter().map(|x| zero).collect::<Vec<$T>>());
            self.append_collumn(collumn);
        }
    
        fn append_collumn_zeros_n_times(&mut self, n: usize) {
            let height = self.height();
            let zero: $T = NumCast::from(0).unwrap();
            let collumn= Collumn::new_form_vec((0..height).into_iter().map(|x| zero).collect::<Vec<$T>>());
            for _ in 0..n{
                self.append_collumn(collumn.clone());
            }
        }
    
        fn append_collumn(&mut self, collumn: Collumn<$T>) -> &Self {
            if !(self.height() == collumn.height()) {
                panic!("Collumn dimensions do not match, given collumn is {} long, current matrix is {} long.", collumn.height(), self.height())
            }
    
            for row_index in 0..self.height() {
                self[row_index].cells.push(collumn[row_index]);
            }
    
            self
        }
    
        fn append_row_from_vec(&mut self, new_row_vec: Vec<$T>) {
            let new_row = Row::new_row_from_vec(new_row_vec);
            self.append_row(new_row);
        }
    
        fn multiply_all_elements_by(&mut self, factor: $T) -> &Self {
            for row_number in 0..self.rows.len() {
                self.rows[row_number].multiply_all_elements_by(factor);
            }
    
            self
        }
    
        fn divide_all_elements_by(&mut self, factor: $T) {
            for row_number in 0..self.rows.len() {
                self.rows[row_number].divide_all_elements_by(factor)
            }
        }
        }
    };
}

impl_matrix_functions_for_type!(i8);
impl_matrix_functions_for_type!(i16);
impl_matrix_functions_for_type!(i32);
impl_matrix_functions_for_type!(i64);

impl_matrix_functions_for_type!(u8);
impl_matrix_functions_for_type!(u16);
impl_matrix_functions_for_type!(u32);
impl_matrix_functions_for_type!(u64);

impl_matrix_functions_for_type!(isize);
impl_matrix_functions_for_type!(usize);

impl_matrix_functions_for_type!(f32);
impl_matrix_functions_for_type!(f64);

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