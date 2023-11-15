use crate::*;

use packed_simd::{*};

macro_rules! get_type_enum {
    ($T: ty) => {
        if TypeId::of::<$T>() == TypeId::of::<i8>() {
            SupportedTypes::I8
        } else if TypeId::of::<$T>() == TypeId::of::<i16>() {
            SupportedTypes::I16
        } else if TypeId::of::<$T>() == TypeId::of::<i32>() {
            SupportedTypes::I32
        } else if TypeId::of::<$T>() == TypeId::of::<i64>() {
            SupportedTypes::I64
        } else if TypeId::of::<$T>() == TypeId::of::<isize>() {
            SupportedTypes::Isize
        } else if TypeId::of::<$T>() == TypeId::of::<u8>() {
            SupportedTypes::U8
        } else if TypeId::of::<$T>() == TypeId::of::<u16>() {
            SupportedTypes::U16
        } else if TypeId::of::<$T>() == TypeId::of::<u32>() {
            SupportedTypes::U32
        } else if TypeId::of::<$T>() == TypeId::of::<u64>() {
            SupportedTypes::U64
        } else if TypeId::of::<$T>() == TypeId::of::<usize>() {
            SupportedTypes::Usize
        } else if TypeId::of::<$T>() == TypeId::of::<f32>() {
            SupportedTypes::F32
        } else if TypeId::of::<$T>() == TypeId::of::<f64>() {
            SupportedTypes::F64
        } else {
            panic!("Non supported type requested")
        }
    };
}

macro_rules! AVX2_type_amount {
    ($T: ident) => {
        match get_type_enum!($T) {
            SupportedTypes::I8 => {
                32 as usize
            },
            SupportedTypes::I16 => {
                16 as usize
            }
            SupportedTypes::I32 => {
                8 as usize
            },
            SupportedTypes::I64 => {
                4 as usize
            },
            SupportedTypes::U8 => {
                32 as usize
            },
            SupportedTypes::U16 => {
                16 as usize
            }
            SupportedTypes::U32 => {
                8 as usize
            },
            SupportedTypes::U64 => {
                4 as usize
            },
            SupportedTypes::Usize => {
                4 as usize
            },
            SupportedTypes::Isize => {
                4 as usize
            }
            SupportedTypes::F32 => {
                8 as usize
            },
            SupportedTypes::F64 => {
                4 as usize
            },
        }
    };
}

macro_rules! AVX2_mm_load {
    ($T: ident, $values_pointer: expr) => {
        match get_type_enum!($T) {
            SupportedTypes::I8 => {
                _mm256_loadu_epi8($values_pointer);
            },
            SupportedTypes::I16 => {
                _mm256_loadu_epi16($values_pointer);
            }
            SupportedTypes::I32 => {
                _mm256_load_epi32($values_pointer);
            },
            SupportedTypes::I64 => {
                _mm256_load_epi64($values_pointer);
            },
            SupportedTypes::U8 => {
                _mm256_loadu_epi8($values_pointer);
            },
            SupportedTypes::U16 => {
                _mm256_loadu_epi16($values_pointer);
            }
            SupportedTypes::U32 => {
                _mm256_load_epi32($values_pointer);
            },
            SupportedTypes::U64 => {
                _mm256_load_epi64($values_pointer);
            },
            SupportedTypes::Usize => {
                _mm256_load_epi64($values_pointer);
            },
            SupportedTypes::Isize => {
                _mm256_loadu_epi64($values_pointer);
            }
            SupportedTypes::F32 => {
                _mm256_load_ps($values_pointer);
            },
            SupportedTypes::F64 => {
                _mm256_load_pd($values_pointer);
            },
        }
    };
}

macro_rules! AVX2_mm_loadu {
    ($T: ident, $values_pointer: expr) => {
        match get_type_enum!($T) {
            SupportedTypes::I8 => {
                _mm256_loadu_epi8(values_pointer);
            },
            SupportedTypes::I16 => {
                _mm256_loadu_epi16(values_pointer);
            }
            SupportedTypes::I32 => {
                _mm256_loadu_epi32(values_pointer);
            },
            SupportedTypes::I64 => {
                _mm256_loadu_epi64(values_pointer);
            },
            SupportedTypes::U8 => {
                _mm256_loadu_epi8(values_pointer);
            },
            SupportedTypes::U16 => {
                _mm256_loadu_epi16(values_pointer);
            }
            SupportedTypes::U32 => {
                _mm256_loadu_epi32(values_pointer);
            },
            SupportedTypes::U64 => {
                _mm256_loadu_epi64(values_pointer);
            },
            SupportedTypes::Usize => {
                _mm256_loadu_epi64(values_pointer);
            },
            SupportedTypes::Isize => {
                _mm256_loadu_epi64(values_pointer);
            }
            SupportedTypes::F32 => {
                _mm256_loadu_ps(values_pointer);
            },
            SupportedTypes::F64 => {
                _mm256_loadu_pd(values_pointer);
            },
        }
    };
}



pub trait SpecializedFunctions<T:MatrixValues> {
    fn substract_row(&mut self, substraction_row: Row<T>) ;
    unsafe fn substract_avx2(&self, substraction_row: Row<T>);
}

impl SpecializedFunctions<f64> for Row<f64> {
    fn substract_row(&mut self, substraction_row: Row<f64>) {
        if !(self.cells.len() == substraction_row.cells.len()) {
            panic!("Error: Length of substracting row is not equal to row length")
        }
        if *IS_AVX2 && *IS_64X{
            unsafe {self.substract_avx2(substraction_row)}
        } else {
            self.substract_all(substraction_row)
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn substract_avx2(&self, substraction_row: Row<f64>) {
        let x = get_type_enum!(f64);
        // _mm256_load
        todo!()
    }
}

enum SupportedTypes {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    Isize,
    Usize,
    F32,
    F64
}


pub trait SIMDFunctions<T:MatrixValues> {
    unsafe fn substract_avx2_row(&mut self, substracting_row: Row<T>);
}

macro_rules! impl_SIMDFunctions_per_type {
    ($T: ident) => {
        impl SIMDFunctions<$T> for Row<$T> {
            #[target_feature(enable = "avx2")]
            unsafe fn substract_avx2_row(&mut self, substracting_row: Row<$T>) {
                let n_elements = avx2_type_amount!($T);
                for group_number in (0..self.cells.len()).step_by(4) {
                    assert!(substracting_row.len() % 4 == 0);
                    let mut A_row_base = self.cells[group_number..group_number+n_elements].as_mut_ptr();
                    let mut B_row_base = substracting_row.cells[group_number..group_number+n_elements].as_ptr();
                    let A_row = AVX2_mm_load!($T,A_row_base);
                    let B_row = _mm256_load_pd(B_row_base);
        
                    let C_row = _mm256_mul_pd(A_row, B_row);
        
                    _mm256_store_pd(A_row_base, C_row);
        
                // _mm
                }
            }
        }
    };
}

// impl SIMDFunctions<f64> for Row<f64> {
//     #[target_feature(enable = "avx2")]
//     unsafe fn substract_avx2_row(&mut self, substracting_row: Row<f64>) {
//         for group_number in (0..self.cells.len()).step_by(4) {
//             assert!(substracting_row.len() % 4 == 0);
//             let mut A_row_base = self.cells[group_number..group_number+4].as_mut_ptr();
//             let mut B_row_base = substracting_row.cells[group_number..group_number+4].as_ptr();
//             let A_row = _mm256_load_pd(A_row_base);
//             let B_row = _mm256_load_pd(B_row_base);

//             let C_row = _mm256_mul_pd(A_row, B_row);

//             _mm256_store_pd(A_row_base, C_row);

//         // _mm
//         }
//     }
// }

impl_SIMDFunctions_per_type!(i8);
impl_SIMDFunctions_per_type!(i16);
impl_SIMDFunctions_per_type!(i32);
impl_SIMDFunctions_per_type!(i64);

impl_SIMDFunctions_per_type!(u8);
impl_SIMDFunctions_per_type!(u16);
impl_SIMDFunctions_per_type!(u32);
impl_SIMDFunctions_per_type!(u64);

impl_SIMDFunctions_per_type!(isize);
impl_SIMDFunctions_per_type!(usize);

impl_SIMDFunctions_per_type!(f32);
impl_SIMDFunctions_per_type!(f64);

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