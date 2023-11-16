#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(unused_imports)]

use crate::*;

pub trait InputTraitRowCol {
    fn parse_input(&self) -> Vec<usize> {
        panic!()
    }
}

pub fn parse_dimension_input(input: impl InputTraitRowCol) -> Vec<usize> {
    input.parse_input()
}

// pub trait InputMatrix<W, T: MatrixValues> { // can not be realisticly implemeted for lists because length of list must be hardcoded -> just use vecs :(
//     fn parse_input_matrix(&self, example_value: T) -> Matrix<T> {
//         panic!()
//     }
// }

// fn parse_matrix_input<W: InputMatrix<W, T>,T: MatrixValues>(input: W) -> Matrix<T> {
//     input.parse_input()
// }

impl InputTraitRowCol for Range<i32> {
    fn parse_input(&self) -> Vec<usize> {
        let mut output_vec = Vec::<usize>::with_capacity(self.len());
        for index in self.clone() {
            output_vec.push(index as usize)
        }
        output_vec
    }
}

impl InputTraitRowCol for Range<usize> {
    fn parse_input(&self) -> Vec<usize> {
        let mut output_vec = Vec::<usize>::with_capacity(self.len());
        for index in self.clone() {
            output_vec.push(index)
        }
        output_vec
    }
}

impl InputTraitRowCol for Vec<usize> {
    fn parse_input(&self) -> Vec<usize> 
    {
        self.clone()
    }
}

impl InputTraitRowCol for i32 {
    fn parse_input(&self) -> Vec<usize> {
        vec![self.clone() as usize]
    }
}

impl InputTraitRowCol for usize {
    fn parse_input(&self) -> Vec<usize> {
        vec![self.clone()]
    }
}

macro_rules! impl_input_parse_for_type {
    ($P: ident) => {

        

             
        // impl<W, T: MatrixValues> InputMatrix<W, T> for Vec<Vec<$P>> {
        //     fn parse_input_matrix(&self, example_value: T) -> Matrix<$P> {
        //         let mut rows = Vec::<Row<T>>::with_capacity(self.len());
        //         for row in self{
        //             rows.push( Row { cells : row.to_vec()});
        //         }
        //         Matrix { rows}
        //     }
        // }
        
        // impl<W, T: MatrixValues> InputMatrix<W, T> for Vec<$P> {
        //     fn parse_input_matrix(&self) -> Matrix<$P> {
        //             let vac = self.to_vec();
        //             let row = vec!{Row{ cells : vac}};
        
        //             Matrix { rows: row}
                
        //     }
        // }
        
        // impl<W, T: MatrixValues> InputMatrix<W, T> for &Vec<$P> {
        //     fn parse_input_matrix(&self) -> Matrix<$P> {
        //             let row = vec!{Row{ cells : self.to_vec()}};
        
        //             Matrix { rows: row}
                
        //     }
        // }
        
        // impl<W, T: MatrixValues> InputMatrix<W, T> for Matrix<$P> {
        //     fn parse_input_matrix(&self) -> Matrix<$P> {
        //             self.clone()
        //     }
        // }
        
        // impl<W, T: MatrixValues> InputMatrix<W, T> for Collumn<$P> {
        //     fn parse_input_matrix(&self, example_value: $P) -> Matrix<$P> {
        //             let contents_clone = Collumn::new_form_vec(self.to_vec());
        //             Matrix::new_from_collumn(contents_clone)
        //             // TODO: THIS IS A GROSS SOLUTION BUT THE COMPILER IS HAPPY IG
                
        //     }
        // }
    };
}

impl_input_parse_for_type!(i64);

// impl<T> InputTraitRowCol<T> for Range<i32> {
//     fn parse_input(&self) -> Vec<usize> {
//         let mut output_vec = Vec::<usize>::with_capacity(self.len());
//         for index in self.clone() {
//             output_vec.push(index as usize)
//         }
//         output_vec
//     }
// }

// impl<T> InputTraitRowCol<T> for Range<usize> {
//     fn parse_input(&self) -> Vec<usize> {
//         let mut output_vec = Vec::<usize>::with_capacity(self.len());
//         for index in self.clone() {
//             output_vec.push(index)
//         }
//         output_vec
//     }
// }

// impl<T> InputTraitRowCol<T> for Vec<usize> {
//     fn parse_input(&self) -> Vec<usize> 
//     where 
//     {
//         self.clone()
//     }
// }

// impl<T> InputTraitRowCol<T> for i32 {
//     fn parse_input(&self) -> Vec<usize> {
//         vec![self.clone() as usize]
//     }
// }

// impl<T> InputTraitRowCol<T> for usize {
//     fn parse_input(&self) -> Vec<usize> {
//         vec![self.clone()]
//     }
// }

// fn parse_dimension_input<T: InputTraitRowCol<T>>(input: T) -> Vec<usize> {
//     input.parse_input()
// }

// pub trait InputMatrix<W, T: MatrixValues> { // can not be realisticly implemeted for lists because length of list must be hardcoded -> just use vecs :(
//     fn parse_input(&self) -> Matrix<T> {
//         panic!()
//     }
// }

// impl<W, T: MatrixValues> InputMatrix<W, T> for Vec<Vec<T>> {
//     fn parse_input(&self) -> Matrix<T> {
//         let mut rows = Vec::<Row<T>>::with_capacity(self.len());
//         for row in self{
//             rows.push( Row { cells : row.to_vec()});
//         }
//         Matrix { rows}
//     }
// }

// impl<W, T: MatrixValues> InputMatrix<W, T> for Vec<T> {
//     fn parse_input(&self) -> Matrix<T> {
//             let vac = self.to_vec();
//             let row = vec!{Row{ cells : vac}};

//             Matrix { rows: row}
        
//     }
// }

// impl<W, T: MatrixValues> InputMatrix<W, T> for &Vec<T> {
//     fn parse_input(&self) -> Matrix<T> {
//             let row = vec!{Row{ cells : self.to_vec()}};

//             Matrix { rows: row}
        
//     }
// }

// impl<W, T: MatrixValues> InputMatrix<W, T> for Matrix<T> {
//     fn parse_input(&self) -> Matrix<T> {
//             self.clone()
//     }
// }

// impl<W, T: MatrixValues> InputMatrix<W, T> for Collumn<T> {
//     fn parse_input(&self) -> Matrix<T> {
//             let contents_clone = Collumn::new_form_vec(self.to_vec());
//             Matrix::new_from_collumn(contents_clone)
//             // TODO: THIS IS A GROSS SOLUTION BUT THE COMPILER IS HAPPY IG
        
//     }
// }
