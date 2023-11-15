use crate::*;

use std::simd::{*};



// macro_rules! get_type_enum {
//     ($T: ty) => {
//         if TypeId::of::<$T>() == TypeId::of::<i8>() {
//             SupportedTypes::I8
//         } else if TypeId::of::<$T>() == TypeId::of::<i16>() {
//             SupportedTypes::I16
//         } else if TypeId::of::<$T>() == TypeId::of::<i32>() {
//             SupportedTypes::I32
//         } else if TypeId::of::<$T>() == TypeId::of::<i64>() {
//             SupportedTypes::I64
//         } else if TypeId::of::<$T>() == TypeId::of::<isize>() {
//             SupportedTypes::Isize
//         } else if TypeId::of::<$T>() == TypeId::of::<u8>() {
//             SupportedTypes::U8
//         } else if TypeId::of::<$T>() == TypeId::of::<u16>() {
//             SupportedTypes::U16
//         } else if TypeId::of::<$T>() == TypeId::of::<u32>() {
//             SupportedTypes::U32
//         } else if TypeId::of::<$T>() == TypeId::of::<u64>() {
//             SupportedTypes::U64
//         } else if TypeId::of::<$T>() == TypeId::of::<usize>() {
//             SupportedTypes::Usize
//         } else if TypeId::of::<$T>() == TypeId::of::<f32>() {
//             SupportedTypes::F32
//         } else if TypeId::of::<$T>() == TypeId::of::<f64>() {
//             SupportedTypes::F64
//         } else {
//             panic!("Non supported type requested")
//         }
//     };
// }

macro_rules! AVX2_type_amount {
    ($T: ty) => {
        match <$T>::get_type_enum() {
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

enum AVX2Stores {
    M256(__m256),
    M256d(__m256d),
    M256i(__m256i)
}

trait AVX2Functions {
    unsafe fn _mm_load(values_pointer: *const Self) -> AVX2Stores where Self: Sized;
    unsafe fn _mm_add(row_A: AVX2Stores, row_B: AVX2Stores) -> AVX2Stores;
    unsafe fn _mm_store(memory_pointer: *mut Self, values_to_store: AVX2Stores) ;
}

impl AVX2Functions for i8 {
    unsafe fn _mm_load(values_pointer: *const Self) -> AVX2Stores where Self: Sized {
        AVX2Stores::M256i(_mm256_loadu_epi8(values_pointer))
    }

    unsafe fn _mm_add(row_A: AVX2Stores, row_B: AVX2Stores) -> AVX2Stores {
        match (row_A,row_B) {
            (AVX2Stores::M256i(row_A_store), AVX2Stores::M256i(row_B_store) )=> {
                AVX2Stores::M256i(_mm256_add_epi8(row_A_store, row_B_store))
            },
            _ => {panic!("The stores are not of right kind!")}
        }
    }

    unsafe fn _mm_store(memory_pointer: *mut Self, values_to_store: AVX2Stores)  {
        match values_to_store {
            AVX2Stores::M256i(values_to_store_value) => {
                _mm256_storeu_epi8(memory_pointer, values_to_store_value)
            },
            _ => panic!("AVX2 store is of wrong kind!")
        }
    }
    
}
impl AVX2Functions for i16 {
    unsafe fn _mm_load(values_pointer: *const Self) -> AVX2Stores where Self: Sized {
        AVX2Stores::M256i(_mm256_loadu_epi16(values_pointer))
    }

    unsafe fn _mm_add(row_A: AVX2Stores, row_B: AVX2Stores) -> AVX2Stores {
        match (row_A,row_B) {
            (AVX2Stores::M256i(row_A_store), AVX2Stores::M256i(row_B_store) )=> {
                AVX2Stores::M256i(_mm256_add_epi16(row_A_store, row_B_store))
            },
            _ => {panic!("The stores are not of right kind!")}
        }
    }

    unsafe fn _mm_store(memory_pointer: *mut Self, values_to_store: AVX2Stores)  {
        match values_to_store {
            AVX2Stores::M256i(values_to_store_value) => {
                _mm256_storeu_epi16(memory_pointer, values_to_store_value)
            },
            _ => panic!("AVX2 store is of wrong kind!")
        }
    }
}

impl AVX2Functions for i32 {
    unsafe fn _mm_load(values_pointer: *const Self) -> AVX2Stores where Self: Sized {
        AVX2Stores::M256i(_mm256_load_epi32(values_pointer))
    }

    unsafe fn _mm_add(row_A: AVX2Stores, row_B: AVX2Stores) -> AVX2Stores {
        match (row_A,row_B) {
            (AVX2Stores::M256i(row_A_store), AVX2Stores::M256i(row_B_store) )=> {
                AVX2Stores::M256i(_mm256_add_epi32(row_A_store, row_B_store))
            },
            _ => {panic!("The stores are not of right kind!")}
        }
    }

    unsafe fn _mm_store(memory_pointer: *mut Self, values_to_store: AVX2Stores)  {
        match values_to_store {
            AVX2Stores::M256i(values_to_store_value) => {
                _mm256_store_epi32(memory_pointer, values_to_store_value)
            },
            _ => panic!("AVX2 store is of wrong kind!")
        }
    }
}
impl AVX2Functions for i64 {
    unsafe fn _mm_load(values_pointer: *const Self) -> AVX2Stores where Self: Sized {
        AVX2Stores::M256i(_mm256_load_epi64(values_pointer))
    }

    unsafe fn _mm_add(row_A: AVX2Stores, row_B: AVX2Stores) -> AVX2Stores {
        match (row_A,row_B) {
            (AVX2Stores::M256i(row_A_store), AVX2Stores::M256i(row_B_store) )=> {
                AVX2Stores::M256i(_mm256_add_epi64(row_A_store, row_B_store))
            },
            _ => {panic!("The stores are not of right kind!")}
        }
    }

    unsafe fn _mm_store(memory_pointer: *mut Self, values_to_store: AVX2Stores)  {
        match values_to_store {
            AVX2Stores::M256i(values_to_store_value) => {
                _mm256_store_epi64(memory_pointer, values_to_store_value)
            },
            _ => panic!("AVX2 store is of wrong kind!")
        }
    }
}

impl AVX2Functions for u8 {
    unsafe fn _mm_load(values_pointer: *const Self) -> AVX2Stores where Self: Sized {
        AVX2Stores::M256i(__m256i::from(u8x32::from_slice(&[*values_pointer])))
    }

    unsafe fn _mm_add(row_A: AVX2Stores, row_B: AVX2Stores) -> AVX2Stores {
        match (row_A,row_B) {
            (AVX2Stores::M256i(row_A_store), AVX2Stores::M256i(row_B_store) )=> {
                AVX2Stores::M256i(_mm256_adds_epu8(row_A_store, row_B_store))
            },
            _ => {panic!("The stores are not of right kind!")}
        }
    }

    unsafe fn _mm_store(memory_pointer: *mut Self, values_to_store: AVX2Stores)  {
        match values_to_store {
            AVX2Stores::M256i(values_to_store_value) => {
                // unsafe{_mm256_storeu_epi8(memory_pointer, values_to_store_value)}
                panic!("storing u8's is not a thing yet apparantly :(")
            },
            _ => panic!("AVX2 store is of wrong kind!")
        }
    }
}

impl AVX2Functions for u16 {
    unsafe fn _mm_load(values_pointer: *const Self) -> AVX2Stores where Self: Sized {
        AVX2Stores::M256i(__m256i::from(u16x16::from_slice(&[*values_pointer])))
    }

    unsafe fn _mm_add(row_A: AVX2Stores, row_B: AVX2Stores) -> AVX2Stores {
        match (row_A,row_B) {
            (AVX2Stores::M256i(row_A_store), AVX2Stores::M256i(row_B_store) )=> {
                AVX2Stores::M256i(_mm256_adds_epu16(row_A_store, row_B_store))
            },
            _ => {panic!("The stores are not of right kind!")}
        }
    }

    unsafe fn _mm_store(memory_pointer: *mut Self, values_to_store: AVX2Stores)  {
        match values_to_store {
            AVX2Stores::M256i(values_to_store_value) => {
                // unsafe{_mm256_storeu_epi8(memory_pointer, values_to_store_value)}
                panic!("storing u16's is not a thing yet apparantly :(")
            },
            _ => panic!("AVX2 store is of wrong kind!")
        }
    }
}

impl AVX2Functions for u32 {
    unsafe fn _mm_load(values_pointer: *const Self) -> AVX2Stores where Self: Sized {
        AVX2Stores::M256i(__m256i::from(u32x8::from_slice(&[*values_pointer])))
    }

    unsafe fn _mm_add(row_A: AVX2Stores, row_B: AVX2Stores) -> AVX2Stores {
        match (row_A,row_B) {
            (AVX2Stores::M256i(row_A_store), AVX2Stores::M256i(row_B_store) )=> {
                // AVX2Stores::M256i(_mm256_adds_epu32(row_A_store, row_B_store))
                panic!("Unsinged addition for u32 is apparangly not a thing :(")
            },
            _ => {panic!("The stores are not of right kind!")}
        }
    }

    unsafe fn _mm_store(memory_pointer: *mut Self, values_to_store: AVX2Stores)  {
        match values_to_store {
            AVX2Stores::M256i(values_to_store_value) => {
                // unsafe{_mm256_storeu_epi8(memory_pointer, values_to_store_value)}
                panic!("storing u32's is not a thing yet apparantly :(")
            },
            _ => panic!("AVX2 store is of wrong kind!")
        }
    }
}
impl AVX2Functions for u64 {
    unsafe fn _mm_load(values_pointer: *const Self) -> AVX2Stores where Self: Sized {
        AVX2Stores::M256i(__m256i::from(u64x4::from_slice(&[*values_pointer])))
    }

    unsafe fn _mm_add(row_A: AVX2Stores, row_B: AVX2Stores) -> AVX2Stores {
        match (row_A,row_B) {
            (AVX2Stores::M256i(row_A_store), AVX2Stores::M256i(row_B_store) )=> {
                // AVX2Stores::M256i(_mm256_adds_epu64(row_A_store, row_B_store))
                panic!("Unsinged addition for u64 is apparangly not a thing :(")
            },
            _ => {panic!("The stores are not of right kind!")}
        }
    }

    unsafe fn _mm_store(memory_pointer: *mut Self, values_to_store: AVX2Stores)  {
        match values_to_store {
            AVX2Stores::M256i(values_to_store_value) => {
                // unsafe{_mm256_storeu_epi8(memory_pointer, values_to_store_value)}
                panic!("storing u64's is not a thing yet apparantly :(")
            },
            _ => panic!("AVX2 store is of wrong kind!")
        }
    }
}
impl AVX2Functions for isize {
    unsafe fn _mm_load(values_pointer: *const Self) -> AVX2Stores where Self: Sized {
        AVX2Stores::M256i(__m256i::from(i64x4::from_slice(&[*values_pointer as i64])))
    }

    unsafe fn _mm_add(row_A: AVX2Stores, row_B: AVX2Stores) -> AVX2Stores {
        match (row_A,row_B) {
            (AVX2Stores::M256i(row_A_store), AVX2Stores::M256i(row_B_store) )=> {
                AVX2Stores::M256i(_mm256_add_epi64(row_A_store, row_B_store))
            },
            _ => {panic!("The stores are not of right kind!")}
        }
    }

    unsafe fn _mm_store(memory_pointer: *mut Self, values_to_store: AVX2Stores)  {
        match values_to_store {
            AVX2Stores::M256i(values_to_store_value) => {
                _mm256_store_epi64(memory_pointer as *mut i64, values_to_store_value)
            },
            _ => panic!("AVX2 store is of wrong kind!")
        }
    }
}
impl AVX2Functions for usize {
    unsafe fn _mm_load(values_pointer: *const Self) -> AVX2Stores where Self: Sized {
        AVX2Stores::M256i(__m256i::from(u64x4::from_slice(&[*values_pointer as u64])))
    }

    unsafe fn _mm_add(row_A: AVX2Stores, row_B: AVX2Stores) -> AVX2Stores {
        match (row_A,row_B) {
            (AVX2Stores::M256i(row_A_store), AVX2Stores::M256i(row_B_store) )=> {
                // AVX2Stores::M256i(_mm256_adds_epu64(row_A_store, row_B_store))
                panic!("Unsinged addition for usize (u64) is apparangly not a thing :(")
            },
            _ => {panic!("The stores are not of right kind!")}
        }
    }

    unsafe fn _mm_store(memory_pointer: *mut Self, values_to_store: AVX2Stores)  {
        match values_to_store {
            AVX2Stores::M256i(values_to_store_value) => {
                // unsafe{_mm256_storeu_epi8(memory_pointer, values_to_store_value)}
                panic!("storing usize's (u64) is not a thing yet apparantly :(")
            },
            _ => panic!("AVX2 store is of wrong kind!")
        }
    }
}
impl AVX2Functions for f32 {
    unsafe fn _mm_load(values_pointer: *const Self) -> AVX2Stores where Self: Sized {
        AVX2Stores::M256(_mm256_load_ps(values_pointer))
    }

    unsafe fn _mm_add(row_A: AVX2Stores, row_B: AVX2Stores) -> AVX2Stores {
        match (row_A,row_B) {
            (AVX2Stores::M256(row_A_store), AVX2Stores::M256(row_B_store) )=> {
                AVX2Stores::M256(_mm256_add_ps(row_A_store, row_B_store))
            },
            _ => {panic!("The stores are not of right kind!")}
        }
    }

    unsafe fn _mm_store(memory_pointer: *mut Self, values_to_store: AVX2Stores)  {
        match values_to_store {
            AVX2Stores::M256(values_to_store_value) => {
                _mm256_store_ps(memory_pointer, values_to_store_value)
            },
            _ => panic!("AVX2 store is of wrong kind!")
        }
    }
}
impl AVX2Functions for f64 {
    unsafe fn _mm_load(values_pointer: *const Self) -> AVX2Stores where Self: Sized {
        AVX2Stores::M256d(_mm256_load_pd(values_pointer))
    }

    unsafe fn _mm_add(row_A: AVX2Stores, row_B: AVX2Stores) -> AVX2Stores {
        match (row_A,row_B) {
            (AVX2Stores::M256d(row_A_store), AVX2Stores::M256d(row_B_store) )=> {
                AVX2Stores::M256d(_mm256_add_pd(row_A_store, row_B_store))
            },
            _ => {panic!("The stores are not of right kind!")}
        }
    }

    unsafe fn _mm_store(memory_pointer: *mut Self, values_to_store: AVX2Stores)  {
        match values_to_store {
            AVX2Stores::M256d(values_to_store_value) => {
                _mm256_store_pd(memory_pointer, values_to_store_value)
            },
            _ => panic!("AVX2 store is of wrong kind!")
        }
    }
}

// macro_rules! AVX2_mm_load {
//     ($T: ty, $values_pointer: expr) => {
//         match (<$T>::get_type_enum()) {
//             // SupportedTypes::I8 => {
//             //     _mm256_loadu_epi8($values_pointer);
//             // },
//             // SupportedTypes::I16 => {
//             //     _mm256_loadu_epi16($values_pointer);
//             // }
//             SupportedTypes::I32 => {
//                 _mm256_load_epi32($values_pointer);
//             },
//             SupportedTypes::I64 => {
//                 _mm256_load_epi64($values_pointer);
//             },
//             // SupportedTypes::U8 => {
//             //     _mm256_loadu_epi8($values_pointer);
//             // },
//             // SupportedTypes::U16 => {
//             //     _mm256_loadu_epi16($values_pointer);
//             // }
//             // SupportedTypes::U32 => {
//             //     _mm256_load_epi32($values_pointer);
//             // },
//             // SupportedTypes::U64 => {
//             //     _mm256_load_epi64($values_pointer);
//             // },
//             // SupportedTypes::Usize => {
//             //     _mm256_load_epi64($values_pointer);
//             // },
//             // SupportedTypes::Isize => {
//             //     _mm256_loadu_epi64($values_pointer);
//             // }
//             // SupportedTypes::F32 => {
//             //     _mm256_load_ps($values_pointer);
//             // },
//             // SupportedTypes::F64 => {
//             //     _mm256_load_pd($values_pointer);
//             // },
//             _ => {panic!()}
//         }
//     };
// }

// macro_rules! AVX2_mm_loadu {
//     ($T: ident, $values_pointer: expr) => {
//         match get_type_enum!($T) {
//             SupportedTypes::I8 => {
//                 _mm256_loadu_epi8(values_pointer);
//             },
//             SupportedTypes::I16 => {
//                 _mm256_loadu_epi16(values_pointer);
//             }
//             SupportedTypes::I32 => {
//                 _mm256_loadu_epi32(values_pointer);
//             },
//             SupportedTypes::I64 => {
//                 _mm256_loadu_epi64(values_pointer);
//             },
//             SupportedTypes::U8 => {
//                 _mm256_loadu_epi8(values_pointer);
//             },
//             SupportedTypes::U16 => {
//                 _mm256_loadu_epi16(values_pointer);
//             }
//             SupportedTypes::U32 => {
//                 _mm256_loadu_epi32(values_pointer);
//             },
//             SupportedTypes::U64 => {
//                 _mm256_loadu_epi64(values_pointer);
//             },
//             SupportedTypes::Usize => {
//                 _mm256_loadu_epi64(values_pointer);
//             },
//             SupportedTypes::Isize => {
//                 _mm256_loadu_epi64(values_pointer);
//             }
//             SupportedTypes::F32 => {
//                 _mm256_loadu_ps(values_pointer);
//             },
//             SupportedTypes::F64 => {
//                 _mm256_loadu_pd(values_pointer);
//             },
//         }
//     };
// }



// pub trait SpecializedFunctions<T:MatrixValues> {
//     fn substract_row(&mut self, substraction_row: Row<T>) ;
//     unsafe fn substract_avx2(&self, substraction_row: Row<T>);
// }

// impl SpecializedFunctions<i64> for Row<i64> {
//     fn substract_row(&mut self, substraction_row: Row<i64>) {
//         if !(self.cells.len() == substraction_row.cells.len()) {
//             panic!("Error: Length of substracting row is not equal to row length")
//         }
//         if *IS_AVX2 && *IS_64X{
//             unsafe {self.substract_avx2(substraction_row)}
//         } else {
//             self.substract_all(substraction_row)
//         }
//     }

//     #[target_feature(enable = "avx2")]
//     unsafe fn substract_avx2(&self, substraction_row: Row<i64>) {
//         // let x = get_type_enum!(f64);
//         let y = vec![1,2,3,4,5];
//         // let z = u8x32::from_slice_aligned(&y);
//         // let p = __m256i::from(z);
//         // let q = __m256i::from(u8x32::from_slice_unaligned(slice));
//         // let b = _mm256_set_
//         todo!()
//     }
// }

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
    ($T: ty) => {
        impl SIMDFunctions<$T> for Row<$T> {
            #[target_feature(enable = "avx2")]
            unsafe fn substract_avx2_row(&mut self, substracting_row: Row<$T>) {
                let n_elements = AVX2_type_amount!($T);
                for group_number in (0..self.cells.len()).step_by(4) {
                    assert!(substracting_row.cells.len() % 4 == 0);
                    let mut A_row_base = self.cells[group_number..group_number+n_elements].as_mut_ptr();
                    let mut B_row_base = substracting_row.cells[group_number..group_number+n_elements].as_ptr();
                    let A_row = <$T>::_mm_load(A_row_base);
                    let B_row = <$T>::_mm_load(B_row_base);
        
                    let result_row = <$T>::_mm_add(A_row, B_row);
        
                    <$T>::_mm_store(A_row_base, result_row);
        
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

pub trait GetTypeEnum {
    fn get_type_enum() -> SupportedTypes;
}

impl GetTypeEnum for i8 {
    fn get_type_enum() -> SupportedTypes {
        SupportedTypes::I8
    }
}
impl GetTypeEnum for i16 {
    fn get_type_enum() -> SupportedTypes {
        SupportedTypes::I16
    }
}
impl GetTypeEnum for i32 {
    fn get_type_enum() -> SupportedTypes {
        SupportedTypes::I32
    }
}
impl GetTypeEnum for i64 {
    fn get_type_enum() -> SupportedTypes {
        SupportedTypes::I64
    }
}
impl GetTypeEnum for u8 {
    fn get_type_enum() -> SupportedTypes {
        SupportedTypes::U8
    }
}
impl GetTypeEnum for u16 {
    fn get_type_enum() -> SupportedTypes {
        SupportedTypes::U16
    }
}
impl GetTypeEnum for u32 {
    fn get_type_enum() -> SupportedTypes {
        SupportedTypes::U32
    }
}
impl GetTypeEnum for u64 {
    fn get_type_enum() -> SupportedTypes {
        SupportedTypes::U64
    }
}
impl GetTypeEnum for isize {
    fn get_type_enum() -> SupportedTypes {
        SupportedTypes::Isize
    }
}
impl GetTypeEnum for usize {
    fn get_type_enum() -> SupportedTypes {
        SupportedTypes::Usize
    }
}
impl GetTypeEnum for f32 {
    fn get_type_enum() -> SupportedTypes {
        SupportedTypes::F32
    }
}
impl GetTypeEnum for f64 {
    fn get_type_enum() -> SupportedTypes {
        SupportedTypes::F64
    }
}