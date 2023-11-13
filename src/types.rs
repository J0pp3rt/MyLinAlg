






// Gaus seems better for dense matrices, CholeskySquare root better for sparser / diagnal matrices.
// some further optimizations possible by making the cholesky more for row for collum then other way around (use row major and all), then likely faster then gaus in the dense ones.
// maybe be smarter around the whole transpose part -> not actually making a transpose but just playing with indices to make it work. 
// otherwise making Matrix either collumn major or row major could help.
// Larger performance gain likely when mutli threading is introduced, especially for larger matrices.
// not sure what LU decomposition would add (maybe support for generic matrices), 
// normal Cholesky deomposition might be advantages as only 2 matrices are involved instead of 1.
// also find moments when Cholesky(root free) is applicable and maybe add in some tests/checks (maybe row swapping needed?: https://math.stackexchange.com/questions/4504209/when-does-a-real-symmetric-matrix-have-ldlt-decomposition-and-when-is-the
// look around in gaus whether there are more optimizations.




// impl<T: MatrixValues> Matrix<T> {
//     pub fn new_square_with_constant_values(n_rows:usize, value: T) -> Matrix<T> {
//         let mut rows = Vec::<Row<T>>::with_capacity(n_rows);
//         for _ in 0..n_rows {
//             rows.push(Row::new_row_with_value(n_rows, value));
//         } 
//         Matrix {
//              rows,
//             }
//     }

//     pub fn new_with_constant_values(n_rows:usize, n_collumns: usize, value: T) -> Matrix<T> {
//         let mut rows = Vec::<Row<T>>::with_capacity(n_rows);
//         for _ in 0..n_rows {
//             rows.push(Row::new_row_with_value(n_collumns, value));
//         } 
//         Matrix {
//              rows,
//             }
//     }

//     pub fn new_from_vector_rows(input: Vec<Vec<T>>) -> Matrix<T> {
//         let mut rows = Vec::<Row<T>>::with_capacity(input.len());
//         for dimension in input{
//             rows.push( Row { cells : dimension});
//         }
//         Matrix { rows }
//     }

//     pub fn new_from_collumn(input_collumn: Collumn<T>) -> Matrix<T>{
//         let mut rows = Vec::<Row<T>>::with_capacity(input_collumn.n_rows());
//         for row_number in 0..input_collumn.n_rows(){
//             rows.push(Row { cells: vec![input_collumn[row_number]] })
//         }

//         Matrix { rows }
//     }

//     pub fn new_square_eye(size: usize, values_of_diagonal: T) -> Matrix<T> {
//         println!("creating fixed ");
//         let mut new_matrix: Matrix<T> = Matrix::new_square_with_constant_values(size, NumCast::from(0).unwrap());
//         for j in 0..size {
//             new_matrix[j][j] = values_of_diagonal;
//         }
//         new_matrix
//     }

//     pub fn new_eye(n_rows: usize, n_collumns: usize, values_of_diagonal: T) -> Matrix<T> {
//         let mut new_matrix: Matrix<T> = Matrix::new_with_constant_values(n_rows, n_collumns, NumCast::from(0).unwrap());
        
//         let smallest_value: usize;
//         if n_rows < n_collumns{
//             smallest_value = n_rows;
//         } else {
//             smallest_value = n_collumns;
//         }

//         for j in 0..smallest_value {
//             new_matrix[j][j] = values_of_diagonal;
//         }
//         new_matrix
//     }
    
//     pub fn new_from_row_major_vector(mut vector: Vec<T>, height: usize, width: usize) -> Self {
//         if !(vector.len() == height*width) {
//             panic!("Given dimensions do not match! Vec length: {}, height x width = {} x {} = {}", vector.len(), height, width, height*width);
//         }

//         let mut result_matrix: Matrix<T> = Matrix::new_with_constant_values(height, width, NumCast::from(0).unwrap());

//         for row_index in 0..height {
//             let row = Row::new_row_from_vec(
//                 vector[row_index*width..(row_index+1)*width].to_owned()
//             );
//             result_matrix[row_index] = row;
//         }

//         result_matrix
//     }

//     pub fn clone(&self) -> Matrix<T> {
//         let mut rows = Vec::<Row<T>>::with_capacity(self.rows.len());
//         for row in &self.rows{
//             rows.push( row.clone());
//         }
//         Matrix { rows }
//     }

//     pub fn len(&self) -> usize {
//         if self.row_length() >= self.coll_length() {
//             self.row_length()
//         } else {
//             self.coll_length()
//         }
//     }

//     pub fn row_length(&self) -> usize {
//         let mut largest_length = self.rows[0].len();
//         for row in &self.rows {
//             if row.len() > largest_length {
//                 largest_length = row.len();
//                 println!("Row lengths of matrix not consistant!")
//             }
//         }
//         largest_length
//     }

//     pub fn diagonal_contain_zeros(&self) -> bool {
//         let smallest_dimension: usize;
//         if self.height() < self.width() {
//             smallest_dimension = self.height();
//         } else {
//             smallest_dimension = self.width();
//         }

//         let mut zero_values_found = false;
//         for j in 0..smallest_dimension {
//             if self[j][j] == NumCast::from(0).unwrap() {
//                 zero_values_found = true
//             }
//         }

//         zero_values_found
//     }

//     pub fn new_with_random_ones_chance(height: usize, width: usize, chance: f64, base_value: T) -> Self {
//         let mut result_matrix: Matrix<T> = Matrix::new_with_constant_values(height, width, base_value);
//         let mut rng = rand::thread_rng();
//         for row_index in 0..height {
//             for collumn_index in 0..width {
//                 let random_value  = rng.gen_range(0.0..1.0);
                
//                 if random_value < chance {
//                     result_matrix[row_index][collumn_index] = NumCast::from(1).unwrap();
//                 } else {
//                     result_matrix[row_index][collumn_index] = NumCast::from(0).unwrap();
//                 }
//             }
//         }

//         result_matrix
//     }

//     pub fn remove_last_row(&mut self) -> &Self {
//         self.rows.pop().unwrap();

//         self
//     }

//     pub fn remove_last_collumn(&mut self) -> &Self {
//         for row_index in 0..self.height() {
//             self[row_index].cells.pop();
//         }

//         self
//     }

//     pub fn width(&self) -> usize {
//         self.row_length()
//     }

//     pub fn coll_length(&self) -> usize {
//         self.rows.len()
//     }

//     pub fn height(&self) -> usize {
//         self.coll_length()
//     }

//     pub fn is_square(&self) -> bool {
//         self.height() == self.width()
//     }

//     pub fn get_collumn(&self, coll_number : usize) -> Collumn<T> {
//         let mut cells = Vec::<T>::with_capacity(self.coll_length());
//         for row_number in 0..self.coll_length(){
//             cells.push( self[row_number][coll_number])
//         }
//         Collumn {cells}
//     }

//     pub fn to_vec_collumn_major(&self) -> Vec<T> {
//         let N = self.width()*self.height();
//         let mut output_vec = Vec::<T>::with_capacity(N);

//         for column_number in 0..self.width() {
//             for row_number in 0..self.height() {
//                 output_vec.push(self[row_number][column_number]);
//             }
//         }

//         output_vec
//     }

//     pub fn to_vec_row_major(&self) -> Vec<T> {
//         let N = self.width()*self.height();
//         let mut output_vec = Vec::<T>::with_capacity(N);

//         for row_number in 0..self.height() {
//             output_vec.extend(self[row_number].cells.clone());
//         }

//         output_vec
//     }

//     pub fn swap_rows(&mut self, row_1: usize, row_2: usize){
//         let row_1_copy = self[row_1].clone();
        
//         self[row_1] = self[row_2].clone();
//         self[row_2] = row_1_copy;
//     }

//     pub fn substract_internal_row_from_row_by_index(&mut self, row_number_to_substract: usize, from_row_number: usize) {
//         let row_to_substract = self[row_number_to_substract].clone();
//         self[from_row_number].substract_row(row_to_substract)
//     }

//     pub fn substract_multiplied_internal_row_from_row_by_index(&mut self, row_number_to_substract_with: usize, factor: T , from_row_number: usize) {
//         let mut mutliplied_row_to_substract = self[row_number_to_substract_with].clone();
//         mutliplied_row_to_substract.multiply_all_elements_by(factor);
//         self[from_row_number].substract_row(mutliplied_row_to_substract)
//     }

//     pub fn substract_multiplied_internal_row_from_row_by_index_with_collumn_range<U>(&mut self, row_number_to_substract_with: usize, factor: T , from_row_number: usize, collumn_range: U)
//     where  U: InputTraitRowCol<U> {
//         let colls_input:Vec<usize> = parse_dimension_input(collumn_range);

//         for collumn_index in colls_input.iter(){
//             self[from_row_number][*collumn_index] = self[from_row_number][*collumn_index] - self[row_number_to_substract_with][*collumn_index] * factor
//         }  
//     }

//     pub fn new_from_solver(system_of_equations : Solver2D<T>) -> Solved2D<T>{
//         match system_of_equations.solver {
//             Solver2DStrategy::Guass => solve_with_guass(system_of_equations),
//             Solver2DStrategy::CholeskySquareRootFreeDecomposition => solve_with_cholesky_quare_root_free_decomposition(system_of_equations),
//             // Solver2DStrategy::LUDecomposition => todo!(),
//             _ => (panic!("Error: Solver not yet implemented!"))
//         }
//     }

//     pub fn transpose_square(&mut self) -> &Self {
//         for row_index in 0..self.height()-1 {
//             for collumn_index in row_index+1..self.width() {
//                 let buffer = self[collumn_index][row_index];
//                 self[collumn_index][row_index] = self[row_index][collumn_index];
//                 self[row_index][collumn_index] = buffer;
//             }
//         }
//         self
//     }

//     pub fn transpose_non_skinny(&mut self) -> &Self {
//         let initial_height = self.height();
//         let initial_width = self.width();
//         if initial_height == initial_width {
//             return self.transpose_square();
//         }

//         let smallest_dimension_was_height: bool;
//         let dimension_difference: usize = abs(initial_height as isize- initial_width as isize) as usize;
//         if initial_height < initial_width {
//             smallest_dimension_was_height = true;

//         } else {
//             smallest_dimension_was_height = false;
//         }


//         self.make_square();
//         self.transpose_square();


//         if smallest_dimension_was_height {
//             for _ in 0..dimension_difference {
//                 self.remove_last_collumn();
//             }
//         } else {
//             for _ in 0..dimension_difference {
//                 self.remove_last_row();
//             }
//         }

//         self
//     }

//     fn make_square(&mut self) -> &Self {
//         let initial_height = self.height();
//         let initial_width = self.width();

//         let dimension_difference = abs(initial_height as isize - initial_width as isize) as usize;
//         if initial_height == initial_width {
//             return  self;
//         } else if initial_height < initial_width {
//             for _ in 0..dimension_difference {
//                 self.append_row_zeros();
//             }
//         } else {
//             for _ in 0..dimension_difference {
//                 self.append_collumn_zeros();
//             }
//         }

//         self
//     }

//     pub fn add_row(&mut self, insert_row_at: usize, new_row: Row<T>) {
//         if insert_row_at == self.rows.len() {
//             self.append_row(new_row);
//         } else {
//             self.rows.insert(insert_row_at, new_row)
//         }
//     }

//     pub fn add_row_from_vec(&mut self, insert_row_at: usize, new_row_vec: Vec::<T>) {
//         let new_row = Row::new_row_from_vec(new_row_vec);
//         self.add_row(insert_row_at, new_row);
//     }

//     pub fn append_row(&mut self, new_row: Row<T>) {
//             self.rows.push(new_row);
//     }

//     pub fn append_row_zeros(&mut self) {
//         let width = self.width();
//         self.rows.push( Row::new_row_with_constant_values(width, NumCast::from(0).unwrap()));
//     }

//     pub fn append_collumn_zeros(&mut self) {
//         let height = self.height();
//         let zero: T = NumCast::from(0).unwrap();
//         let collumn= Collumn::new_form_vec((0..height).into_iter().map(|x| zero).collect::<Vec<T>>());
//         self.append_collumn(collumn);
//     }

//     pub fn append_collumn_zeros_n_times(&mut self, n: usize) {
//         let height = self.height();
//         let zero: T = NumCast::from(0).unwrap();
//         let collumn= Collumn::new_form_vec((0..height).into_iter().map(|x| zero).collect::<Vec<T>>());
//         for _ in 0..n{
//             self.append_collumn(collumn.clone());
//         }
//     }

//     pub fn append_collumn(&mut self, collumn: Collumn<T>) -> &Self {
//         if !(self.height() == collumn.height()) {
//             panic!("Collumn dimensions do not match, given collumn is {} long, current matrix is {} long.", collumn.height(), self.height())
//         }

//         for row_index in 0..self.height() {
//             self[row_index].cells.push(collumn[row_index]);
//         }

//         self
//     }

//     pub fn append_row_from_vec(&mut self, new_row_vec: Vec<T>) {
//         let new_row = Row::new_row_from_vec(new_row_vec);
//         self.append_row(new_row);
//     }

//     pub fn multiply_all_elements_by(&mut self, factor: T) -> &Self {
//         for row_number in 0..self.rows.len() {
//             self.rows[row_number].multiply_all_elements_by(factor);
//         }

//         self
//     }

//     pub fn divide_all_elements_by(&mut self, factor: T) {
//         for row_number in 0..self.rows.len() {
//             self.rows[row_number].divide_all_elements_by(factor)
//         }
//     }

// }








// impl<T: MatrixValues> Row<T> {
//     pub fn new_row_with_value(size: usize, value: T) -> Row<T> {
//         let mut cells = Vec::<T>::with_capacity(size);
//         for _ in 0..size{
//             cells.push(value);
//         }
//         Row { cells}
//     }

//     pub fn to_vec(&self) -> Vec<T> {
//         self.cells.clone()
//     }
    
//     pub fn new_row_from_vec(input_vec: Vec<T>) -> Row<T> {
//         Row { cells: input_vec }
//     }

//     pub fn new_row_with_constant_values(width: usize, value: T) -> Row<T> {
//         return Row::new_row_with_value(width, value)
//     }

//     pub fn len(&self) -> usize {
//         self.cells.len()
//     }

//     pub fn export(self) -> Row<T> {
//         self
//     }

//     pub fn clone(&self) -> Row<T> {
//         let mut cells = Vec::<T>::with_capacity(self.len());
//         for cell in &self.cells{
//             cells.push(cell.clone());
//         }
//         Row { cells}
//     }

//     pub fn divide_all_elements_by(&mut self, value: T) {
//         for n in 0..self.cells.len() {
//             // if !(self.cells[n] == NumCast::from(0).unwrap()) {// quickly tested on some sparse matrices but seem to really boost performance. In some more filled ones: around 50x improvemnt, ful matrix not tested yet
//                 self.cells[n] = self.cells[n] / value;
//             // }
//         }
//     }

//     pub fn multiply_all_elements_by(&mut self, value: T) -> &Self{
//         for n in 0..self.cells.len() {
//             // if !(self.cells[n] == NumCast::from(0).unwrap()) {// quickly tested on some sparse matrices but seem to really boost performance. In some more filled ones: around 50x improvemnt, ful matrix not tested yet
//             self.cells[n] = self.cells[n] * value;
//             // }
//         }

//         self
//     }

//     pub fn addition_row_with_external_row(&mut self, row_to_add_to_this_one:& Row<T>) {
//         for n in 0..self.cells.len() {
//             // if !(self.cells[n] == NumCast::from(0).unwrap() && row_to_add_to_this_one.cells[n] == NumCast::from(0).unwrap()) {
//                 self.cells[n] = self.cells[n] + row_to_add_to_this_one[n];
//             // }
//         }
//     }

    

//     pub fn normalize_all_elements_to_element(&mut self, index: usize) {
//         self.divide_all_elements_by(self[index]);
//     }

//     pub fn normalize_all_elements_to_first(&mut self) {
//         self.normalize_all_elements_to_element(0);
//     }

//     pub fn substract_row(&mut self, substraction_row: Row<T>) {
//         if !(self.cells.len() == substraction_row.cells.len()) {
//             panic!("Error: Length of substracting row is not equal to row length")
//         }
//         if *IS_AVX2{
//             unsafe {self.substract_avx2(substraction_row)}
//         } else {
//             self.substract_all(substraction_row)
//         }
//     }

//     #[target_feature(enable = "avx2")]
//     unsafe fn substract_avx2(&mut self, substraction_row: Row<T>) {
//         for cell_number in 0..self.cells.len() {
//             // if !(self[cell_number] == NumCast::from(0).unwrap() || substraction_row[cell_number] == NumCast::from(0).unwrap()) { // quickly tested on some sparse matrices but seem to really boost performance . In some more filled ones: around 50x improvemnt, ful matrix not tested yet
//             self[cell_number] = self[cell_number] - substraction_row[cell_number];
//             // } 

//             // let test_row = substraction_row.is_f64;
//             assert!(substraction_row.len() % 4 == 0);
//             let mut A_row_base = self.cells.as_ptr();
//             let mut B_row_base = substraction_row.cells.as_ptr();
//             for _ in 0..self.len() /4{
//                 // let A_row = _mm256_loadu_pd(A_row_base);
//             }
//         }
//     }

//     fn substract_all(&mut self, substraction_row: Row<T>) {
//         for cell_number in 0..self.cells.len() {
//             // if !(self[cell_number] == NumCast::from(0).unwrap() || substraction_row[cell_number] == NumCast::from(0).unwrap()) { // quickly tested on some sparse matrices but seem to really boost performance . In some more filled ones: around 50x improvemnt, ful matrix not tested yet
//             self[cell_number] = self[cell_number] - substraction_row[cell_number];
//             // }
//         }
//     }

//     pub fn replace_values(&mut self, index_range: Range<usize>, values: Vec<T>) {
//         // maybe add a check if not of same size TODO
//         // instead of range use rangebounds apperantly
//         for (val_index, row_index) in index_range.enumerate() {
//             self.cells[row_index] = values[val_index];
//         }
//     }

//     fn convert_to_f64(&self) -> Row<f64> where T: MatrixValues {
//         let mut new: Vec<f64> = Vec::<f64>::with_capacity(self.cells.len());
//         for value in self.cells.iter() {
//             new.push(NumCast::from(*value).unwrap())
//         }

//         Row { cells: new }
//     }
// }


// pub trait RowToTypeTest<T: MatrixValues>{
//     fn is_f64(&self) -> Option<Row<f64>> 
//     where 
//         Self: Sized,
//         Row<f64>: From<Row<T>>  {
//             Option::None
//         }
// }

// impl<T: MatrixValues> RowToTypeTest<T> for Row<T> {
//     fn is_f64(&self) -> Option<Row<f64>> 
//         where 
//             Self: Sized,
//             Row<f64>: From<Row<T>> {
//         if TypeId::of::<T>() == TypeId::of::<f64>() {
//             Option::Some(Row::<f64>::from(*self))
//         } else {
//             Option::None
//         }
//     }
// }

// impl<T:MatrixValues> RowToTypeTest<T> for Row<f64> {
//     fn is_f64(self) -> Option<Row<f64>> where Self: Sized {
        
//     }
// }




// fn substract_row_with_row<U: AVX2Row<T>, T:MatrixValues>(mut row_a: U, row_b: U){
//     row_a.substraction_f64(row_b)
// }

// pub trait AVX2Row<U: AVX2Row<U, T>, T:MatrixValues> {
//     fn substraction_f64(&mut self, substraction_row: Row<f64>) {
//         panic!("this should not be able to run")
//     }
// }

// impl<T:MatrixValues> AVX2Row<T> for Row<f64> {
//     fn substraction_f64(&mut self, substraction_row: Row<f64>) {
//         unsafe {substract_avx2_f64(self, substraction_row)}
//     }
// }

// #[target_feature(enable = "avx2")]
// unsafe fn substract_avx2_f64(a_row: &mut Row<f64>, substraction_row: Row<f64>) {
//     for cell_number in 0..a_row.cells.len() {
//         // if !(self[cell_number] == NumCast::from(0).unwrap() || substraction_row[cell_number] == NumCast::from(0).unwrap()) { // quickly tested on some sparse matrices but seem to really boost performance . In some more filled ones: around 50x improvemnt, ful matrix not tested yet
//         a_row[cell_number] = a_row[cell_number] - substraction_row[cell_number];
//         // } 
//         assert!(substraction_row.len() % 4 == 0);
//         let mut A_row_base = a_row.cells.as_ptr();
//         let mut B_row_base = substraction_row.cells.as_ptr();
//         for _ in 0..a_row.len() /4{
//             let A_row = _mm256_loadu_pd(A_row_base);
//         }
//     }
// }







