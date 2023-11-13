use crate::*;


pub trait SIMDFunctions<T:MatrixValues> {
    unsafe fn substract_avx2_row(&mut self, substracting_row: Row<T>);
}

impl SIMDFunctions<f64> for Row<f64> {
    #[target_feature(enable = "avx2")]
    unsafe fn substract_avx2_row(&mut self, substracting_row: Row<f64>) {
        for group_number in (0..self.cells.len()).step_by(4) {
            assert!(substracting_row.len() % 4 == 0);
            let mut A_row_base = self.cells[group_number..group_number+4].as_mut_ptr();
            let mut B_row_base = substracting_row.cells[group_number..group_number+4].as_ptr();
            let A_row = _mm256_load_pd(A_row_base);
            let B_row = _mm256_load_pd(B_row_base);

            let C_row = _mm256_mul_pd(A_row, B_row);

            _mm256_store_pd(A_row_base, C_row)
        }
    }
}


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