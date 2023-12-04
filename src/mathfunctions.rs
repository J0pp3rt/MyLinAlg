#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(unused_imports)]

use crate::*;

pub fn golden_ratio() -> f64 {
    (1. + (5. as f64).sqrt()) / 2.
}

pub fn golden_ratio_conjugate() -> f64 {
    1. / golden_ratio()
}

#[derive(Debug)]
pub struct Pos2<T>  {
    pub x: T,
    pub y: T
}

pub struct PosNDof<T>  {
    pub x: Vec<T>,
}

#[derive(Debug, Clone)]
pub struct SpatialVector2<T>  {
    pub x_direction: T,
    pub y_direction: T
}

#[derive(Debug, Clone)]
pub struct SpatialVectorWithBase2<T>  {
    pub x_base: T,
    pub y_base: T,
    pub x_direction: T,
    pub y_direction: T
}

#[derive(Clone)]
pub struct SpatialVectorNDof<T> {
    pub vector: Vec<T>,
}

#[derive(Clone)]
pub struct SpatialVectorWithBasePointNDof<T> {
    pub point: Vec<T>,
    pub vector: Vec<T>
}

// Impl

pub trait Pos2Functions<T> {
    fn new(x: T, y: T) -> Self;
    fn clone(&self) -> Self;
    fn x(&self) -> T ;
    fn y(&self) -> T;
    fn add_vector(&mut self, vector: SpatialVector2<T>) -> &mut Self;
}

pub trait PosNDofFunctions<T> {
    fn new(x: Vec<T>) -> Self;
    fn clone(&self) -> Self;
    fn x(&self)  -> Vec<T>;
    fn x_ndof(&self, dof: usize) -> T ;
    fn n_dof(&self) -> usize;
    fn add_vector(&mut self, vector: SpatialVectorNDof<T>) -> &mut Self;
    fn add_vector_vec(&mut self, vector: Vec<T>) -> &mut Self;
}

pub trait SpatialVector2Functions<T> {
    fn new_direction(x_directions: T, y_directions: T) -> Self;
    fn from_pos2(pos1: &Pos2<T>, pos2: &Pos2<T>) -> Self;
    fn x_direction(&self) -> T ;
    fn y_direction(&self) -> T;
    fn length(&self) -> T;
    fn normalize(&mut self) -> &mut Self;
    fn get_normals(&self) -> [Self; 2] where Self: Sized;
    fn scale(&mut self, scaling_factor: T) -> &mut Self;
}

pub trait SpatialVectorWithBase2Functions<T> {
    fn new_with_base(x_base: T, y_base: T, x_direction: T, y_direction: T) -> Self;
    fn base_pos2_direction_pos2(base_pos: &Pos2<T>, pos1: &Pos2<T>, pos2: &Pos2<T>) -> Self;
    fn base_at_first_direction_pos2(pos1: &Pos2<T>, pos2: &Pos2<T>) -> Self;
    fn end_position(&self) -> Pos2<T>;
}

pub trait SpatialVectorNdofFunctions<T> {
    fn new_from_direction(direction_per_dof: Vec<T>) -> Self;
    fn from_difference(posndof_1: &PosNDof<T>, posndof_2: &PosNDof<T>) -> Self;
    fn volatile(&self) -> Self;
    fn x_i_direction(&self, index: usize) -> T;
    fn set_vector_i_value(&mut self, index: usize, value: T) -> &mut Self;
    fn set_vector_values(&mut self, values: Vec<T>) -> &mut Self;
    fn length(&self) -> T;
    fn normalize(&mut self) -> &mut Self;
    // fn get_normals(&self) -> [Self; 2] where Self: Sized;
    fn scale(&mut self, scaling_factor: T) -> &mut Self;
    fn scale_to_length(&mut self, new_length: T) -> &mut Self;
    fn flip_direction(&mut self) -> &mut Self;
    fn n_dof(&self) -> usize;
}

pub trait SpatialVectorWithBaseNDofFunctions<T> {
    fn new_with_base(base_point: &Vec<T>, directions: &Vec<T>) -> Self;
    fn base_direction_from_difference(base_point: &PosNDof<T>, pos1: &PosNDof<T>, pos2: &PosNDof<T>) -> Self;
    fn base_at_first_direction_from_difference(pos1: &PosNDof<T>, pos2: &PosNDof<T>) -> Self;
    fn base_point(&self) -> PosNDof<T>;
    fn base_point_vec(&self) -> Vec<T>;
    fn end_position(&self) -> PosNDof<T>;
    fn end_position_vec(&self) -> Vec<T>;
    fn move_base_to_end_position(&mut self) -> &mut Self;
}

macro_rules! impl_spatial_types_per_type {
    ($T: ident) => {
        impl Pos2Functions<$T> for Pos2<$T> {
            fn new(x: $T, y: $T) -> Self {
                Pos2 {x, y}
            }

            fn clone(&self) -> Self {
                Self {x: self.x, y: self.y}
            }
        
            fn x(&self) -> $T {
                self.x
            }
        
            fn y(&self) -> $T {
                self.y
            }

            fn add_vector(&mut self, vector: SpatialVector2<$T>) -> &mut Self {
                self.x = self.x + vector.x_direction;
                self.y = self.y + vector.y_direction;
                self
            }
        }

        impl SpatialVector2Functions<$T> for SpatialVectorWithBase2<$T> {
            fn new_direction(x_direction: $T, y_direction: $T) -> Self {
                Self {x_base: 0 as $T, y_base: 0 as $T, x_direction, y_direction}
            }

            fn from_pos2(pos1: &Pos2<$T>, pos2: &Pos2<$T>) -> Self {
                let x_direction = pos2.x - pos1.x;
                let y_direction = pos2.y - pos1.x;
                Self {x_base: 0 as $T, y_base: 0 as $T, x_direction, y_direction}
            }
        
            fn x_direction(&self) -> $T {
                self.x_direction
            }
        
            fn y_direction(&self) -> $T {
                self.y_direction
            }

            fn length(&self) -> $T {
                ((self.x_direction as f64).powi(2) + (self.y_direction as f64).powi(2) ).sqrt() as $T
            }

            fn normalize(&mut self) -> &mut Self {
                let length = self.length();
                self.x_direction = self.x_direction / length;
                self.y_direction = self.y_direction / length;
                self
            }
        
            fn get_normals(&self) -> [Self; 2] {
                let normal_1 = Self::new_direction(-self.y_direction, self.x_direction);
                let normal_2 = Self::new_direction(self.y_direction, -self.x_direction);
                [normal_1, normal_2]
            }

            fn scale(&mut self, scaling_factor: $T) -> &mut Self {
                self.x_direction = self.x_direction * scaling_factor;
                self.y_direction = self.y_direction * scaling_factor;

                self
            }
        }

        impl SpatialVector2Functions<$T> for SpatialVector2<$T> {
            fn new_direction(x_direction: $T, y_direction: $T) -> Self {
                SpatialVector2 {x_direction, y_direction}
            }

            fn from_pos2(pos1: &Pos2<$T>, pos2: &Pos2<$T>) -> Self {
                let x_direction = pos2.x - pos1.x;
                let y_direction = pos2.y - pos1.y;
                SpatialVector2 {x_direction, y_direction}
            }
        
            fn x_direction(&self) -> $T {
                self.x_direction
            }
        
            fn y_direction(&self) -> $T {
                self.y_direction
            }

            fn length(&self) -> $T {
                ((self.x_direction as f64).powi(2) + (self.y_direction as f64).powi(2) ).sqrt() as $T
            }

            fn normalize(&mut self) -> &mut Self {
                let length = self.length();
                self.x_direction = self.x_direction / length;
                self.y_direction = self.y_direction / length;
                self
            }
        
            fn get_normals(&self) -> [Self; 2] {
                let normal_1 = Self::new_direction(-self.y_direction, self.x_direction);
                let normal_2 = Self::new_direction(self.y_direction, -self.x_direction);
                [normal_1, normal_2]
            }

            fn scale(&mut self, scaling_factor: $T) -> &mut Self {
                self.x_direction = self.x_direction * scaling_factor;
                self.y_direction = self.y_direction * scaling_factor;

                self
            }
        }

        impl SpatialVectorWithBase2Functions<$T> for SpatialVectorWithBase2<$T> {
            fn new_with_base(x_base: $T, y_base: $T, x_direction: $T, y_direction: $T) -> Self {

                Self {x_base, y_base, x_direction, y_direction}
            }

            fn base_pos2_direction_pos2(base_pos: &Pos2<$T>, pos1: &Pos2<$T>, pos2: &Pos2<$T>) -> Self {
                let vector_part = SpatialVector2::from_pos2(&pos1, &pos2);

                Self {x_base: base_pos.x, y_base: base_pos.y, x_direction: vector_part.x_direction, y_direction: vector_part.y_direction}
            }

            fn base_at_first_direction_pos2(pos1: &Pos2<$T>, pos2: &Pos2<$T>) -> Self {
                let vector_part = SpatialVector2::from_pos2(&pos1, &pos2);

                Self {x_base: pos1.x, y_base: pos1.y, x_direction: vector_part.x_direction, y_direction: vector_part.y_direction}
            }

            fn end_position(&self) -> Pos2<$T> {
                Pos2::new(self.x_base + self.x_direction, self.y_base + self.y_direction)
            }
        }

        impl PosNDofFunctions<$T> for PosNDof<$T> {
            fn new(x: Vec<$T>) -> Self {
                PosNDof {x}
            }

            fn clone(&self) -> Self {
                Self {x: self.x.clone()}
            }
        
            fn x(&self) -> Vec<$T> {
                self.x.clone()
            }
        
            fn x_ndof(&self, dof: usize) -> $T {
                self.x[dof]
            }

            fn n_dof(&self) -> usize {
                self.x.len()
            }

            fn add_vector(&mut self, vector: SpatialVectorNDof<$T>) -> &mut Self {
                self.add_vector_vec(vector.vector)
            }

            fn add_vector_vec(&mut self, vector: Vec<$T>) -> &mut Self {
                if (self.x.len() == vector.len()).not() {
                    panic!("Added vectors do not have equal number of DOF's!")
                }

                self.x = self.x.iter().zip(vector.iter()).map(|(point_xi, vector_xi)| point_xi + vector_xi).collect();
                self
            }
        }

        impl SpatialVectorNdofFunctions<$T> for SpatialVectorNDof<$T> {
            fn new_from_direction(direction_per_dof: Vec<$T>) -> Self {
                SpatialVectorNDof {vector: direction_per_dof}
            }

            fn from_difference(posndof_1: &PosNDof<$T>, posndof_2: &PosNDof<$T>) -> Self {
                if (posndof_1.x.len() == posndof_1.x.len()).not() {
                    panic!("Differentiated points do not have equal number of DOF's!")
                }
                let directions = posndof_1.x.iter().zip(posndof_2.x.iter()).map(|(point_1_x, point_2_x)| point_2_x - point_1_x).collect();
                SpatialVectorNDof {vector: directions}
            }

            fn volatile(&self) -> Self {
                self.clone()
            }
        
            fn x_i_direction(&self, index: usize) -> $T {
                self.vector[index]
            }

            fn length(&self) -> $T {
                self.vector.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt() as $T
            }

            fn normalize(&mut self) -> &mut Self {
                let length = self.length();
                self.vector = self.vector.iter().map(|x| x / length).collect();
                self
            }

            fn set_vector_values(&mut self, values: Vec<$T>) -> &mut Self {
                self.vector = values;

                self
            }

            fn set_vector_i_value(&mut self, index: usize, value: $T) -> &mut Self {
                self.vector[index] = value;

                self
            }
        
            // fn get_normals(&self) -> [Self; 2] {
            //     let normal_1 = Self::new_direction(-self.y_direction, self.x_direction);
            //     let normal_2 = Self::new_direction(self.y_direction, -self.x_direction);
            //     [normal_1, normal_2]
            // }

            fn scale(&mut self, scaling_factor: $T) -> &mut Self {
                self.vector = self.vector.iter().map(|x| x * scaling_factor).collect();
                self
            }

            fn scale_to_length(&mut self, new_length: $T) -> &mut Self {
                let current_length = self.length();
                self.vector = self.vector.iter().map(|x| x / current_length * new_length).collect();
                
                self
            }

            fn flip_direction(&mut self) -> &mut Self {
                self.vector = self.vector.iter().map(|x| -x).collect();
                
                self
            }

            fn n_dof(&self) -> usize {
                self.vector.len()
            }
        }

        impl SpatialVectorNdofFunctions<$T> for SpatialVectorWithBasePointNDof<$T> {
            fn new_from_direction(direction_per_dof: Vec<$T>) -> Self {
                let n_dof = direction_per_dof.len();
                let point = vec![0 as $T; n_dof];
                Self {point, vector: direction_per_dof}
            }

            fn from_difference(posndof_1: &PosNDof<$T>, posndof_2: &PosNDof<$T>) -> Self {
                if (posndof_1.x.len() == posndof_1.x.len()).not() {
                    panic!("Differentiated points do not have equal number of DOF's!")
                }
                let directions = posndof_1.x.iter().zip(posndof_2.x.iter()).map(|(point_1_x, point_2_x)| point_2_x - point_1_x).collect();
                let n_dof = posndof_1.x.len();
                let point = vec![0 as $T; n_dof];
                Self {point, vector: directions}
            }

            fn volatile(&self) -> Self {
                self.clone()
            }
        
            fn x_i_direction(&self, index: usize) -> $T {
                self.vector[index]
            }

            fn length(&self) -> $T {
                self.vector.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt() as $T
            }

            fn normalize(&mut self) -> &mut Self {
                let length = self.length();
                self.vector = self.vector.iter().map(|x| x / length).collect();
                self
            }
        
            fn set_vector_values(&mut self, values: Vec<$T>) -> &mut Self {
                self.vector = values;

                self
            }

            fn set_vector_i_value(&mut self, index: usize, value: $T) -> &mut Self {
                self.vector[index] = value;

                self
            }

            // fn get_normals(&self) -> [Self; 2] {
            //     let normal_1 = Self::new_direction(-self.y_direction, self.x_direction);
            //     let normal_2 = Self::new_direction(self.y_direction, -self.x_direction);
            //     [normal_1, normal_2]
            // }

            fn scale(&mut self, scaling_factor: $T) -> &mut Self {
                self.vector = self.vector.iter().map(|x| x * scaling_factor).collect();
                self
            }

            fn scale_to_length(&mut self, new_length: $T) -> &mut Self {
                let current_length = self.length();
                self.vector = self.vector.iter().map(|x| x / current_length * new_length).collect();
                self
            }

            fn flip_direction(&mut self) -> &mut Self {
                self.vector = self.vector.iter().map(|x| -x).collect();
                
                self
            }

            fn n_dof(&self) -> usize {
                self.vector.len()
            }
        }

        impl SpatialVectorWithBaseNDofFunctions<$T> for SpatialVectorWithBasePointNDof<$T> {
            fn new_with_base(base_point: &Vec<$T>, directions: &Vec<$T>) -> Self {

                Self {point: base_point.clone(), vector: directions.clone()}
            }

            fn base_direction_from_difference(base_pos: &PosNDof<$T>, pos1: &PosNDof<$T>, pos2: &PosNDof<$T>) -> Self {
                if (pos1.x.len() == pos2.x.len() && base_pos.x.len() == pos1.x.len()).not() {
                    panic!("Differentiated points do not have equal number of DOF's!")
                }
                let directions = pos1.x.iter().zip(pos2.x.iter()).map(|(point_1_x, point_2_x)| point_2_x - point_1_x).collect();
                SpatialVectorWithBasePointNDof {point: base_pos.x.clone(), vector: directions}
            }

            fn base_at_first_direction_from_difference(pos1: &PosNDof<$T>, pos2: &PosNDof<$T>) -> Self {
                let vector_part = SpatialVectorNDof::from_difference(&pos1, &pos2);

                Self {point: pos1.x.clone(), vector: vector_part.vector}
            }

            fn base_point(&self) -> PosNDof<$T> {
                PosNDof::new(self.point.clone())
            }

            fn base_point_vec(&self) -> Vec<$T> {
                self.point.clone()
            }

            fn end_position(&self) -> PosNDof<$T> {
                let mut point = PosNDof::new(self.point.clone());
                point.add_vector_vec(self.vector.clone());

                point
            }

            fn end_position_vec(&self) -> Vec<$T> {
                let mut point = PosNDof::new(self.point.clone());
                point.add_vector_vec(self.vector.clone());

                point.x
            }

            fn move_base_to_end_position(&mut self) -> &mut Self {
                let end_position = self.end_position_vec();
                self.point = end_position;

                self
            }
        }

    };
}






pub trait MathFunctions<T:MatrixValues> {
    fn linspace(lower_bound: T, higher_bound: T, steps: usize) -> Vec<T>;
    fn abs(value: T) -> T ;
    fn vec_dot_vec(vec_1: Vec<T>, vec_2: Vec<T>) -> T ;
    fn vec_summed(vec_1: Vec<T>) -> T ;
    fn row_dot_collumn(row: & Row<T>, collumn: & Collumn<T>) -> T ;
    fn matrix_dot_collumn(matrix: & Matrix<T>, collumn: & Collumn<T>) -> Collumn<T> ;
    fn matrix_add_matrix(mut A_matrix: Matrix<T>, B_matrix:& Matrix<T>) -> Matrix<T> {todo!()}
    fn matrix_dot_matrix(A_matrix:& Matrix<T>, B_matrix:& Matrix<T>) -> Matrix<T> ;
}

macro_rules! impl_math_functions_per_type {
    ($T: ident) => {
    impl MathFunctions<$T> for $T {
        fn linspace(lower_bound: $T, higher_bound: $T, steps: usize) -> Vec<$T> {
            let step_size = (higher_bound - lower_bound) / (steps-1) as $T;
            let mut lin_spaced_vec = Vec::<$T>::with_capacity(steps);
            for i in 0..steps {
                lin_spaced_vec.push(lower_bound + step_size * i as $T);
            }
            lin_spaced_vec
        }

        fn abs(value: $T) -> $T {
            if value >= 0 as $T {
                return value;
            } else {
                let min_one: $T = NumCast::from(-1).unwrap();
                let return_value:$T = value * min_one;
                return return_value;
            }
        }

        fn vec_dot_vec(vec_1: Vec<$T>, vec_2: Vec<$T>) ->$T {
            if !(vec_1.len()==vec_2.len()) {
                panic!("Vectors not of same length!");
            }

            let vec_len = vec_1.len();

            let piecewice_mut = vec_1.iter().zip(vec_2).map(|(u, v)| 
                // if !(*u == NumCast::from(0).unwrap() || v == NumCast::from(0).unwrap()) {
                    u.clone() * v.clone()).collect();
                // } else {
                    // NumCast::from(0).unwrap()
                // }).collect();

            $T::vec_summed(piecewice_mut)
        }

        fn vec_summed(vec_1: Vec<$T>) ->$T {
            let mut total: $T = NumCast::from(0).unwrap();
            for j in 0..vec_1.len() {
                // if !(vec_1[j] == NumCast::from(0).unwrap()) {
                    total = total + vec_1[j];
                // }
            }
            total
        }

        fn row_dot_collumn(row: & Row<$T>, collumn: & Collumn<$T>) -> $T {
            if !(row.len() == collumn.n_rows()) {
                panic!("Dimension of Row length must match collumn height. Row has dimension [1x{}], Collumn has [{}x1]", row.len(), collumn.n_rows());
            }
            let n_elements = row.len();

            let piecewice_mut = (0..n_elements).into_iter().map(|j| 
                // if !(row[j] == NumCast::from(0).unwrap() || collumn[j] == NumCast::from(0).unwrap()) {
                    row[j]*collumn[j]).collect();
                // } else {
                    // NumCast::from(0).unwrap()
                // }).collect();

            $T::vec_summed(piecewice_mut)

        }

        fn matrix_dot_collumn(matrix: & Matrix<$T>, collumn: & Collumn<$T>) -> Collumn<$T> {
            if !(matrix.width() == collumn.n_rows()) {
                panic!("Dimension of matrix width must match collumn height. Matrix has dimension [{}x{}], Collumn has [{}x1]", matrix.height(), matrix.width(), collumn.n_rows());
            }

            let mut result_collumn: Collumn<$T> = Collumn::new_with_constant_values(matrix.height(), NumCast::from(0).unwrap());

            for row_number in 0..matrix.height() {
                result_collumn[row_number] = $T::row_dot_collumn(& matrix[row_number], & collumn);
            }

            result_collumn
        }

        // pub fn row_dot_collumn<T: MatrixValues>(row:& Row<T>, collumn:& Collumn<T>) -> $T {
        //     if 
        // }

        fn matrix_add_matrix(mut A_matrix: Matrix<$T>, B_matrix:& Matrix<$T>) -> Matrix<$T> {
            // this function consumes the A_matrix and takes a reference to the B_matrix (no need to consume it) 
            // returns the altered A_matrix
            if !(A_matrix.height() == B_matrix.height() || A_matrix.width() == B_matrix.width()) {
                panic!("Matrices dimensions do not match for adition! given: [{}x{}] + [{}x{}]", A_matrix.height(), A_matrix.width(), B_matrix.height(), B_matrix.width());
            }

            let N = A_matrix.height();
            let M = A_matrix.width();

            for row_index in 0..N {
                A_matrix[row_index].addition_row_with_external_row(&B_matrix[row_index]);
            }

            A_matrix
        }

        fn matrix_dot_matrix(A_matrix:& Matrix<$T>, B_matrix:& Matrix<$T>) -> Matrix<$T> {
            // this function takes 2 references to the matrices to multiply, returns a new matrix.
            // As in place matrix multiplication is practically impossible, there should be the space for all 3 in memory.

            // this is a slow function because of: many operations and collumn operations in B_matrix.
            // Ideally A_matrix would be row major, B_matrix collumn major
            if !(A_matrix.width() == B_matrix.height()) {
                panic!("Matrices dimensions do not match for dot operation! given: [{}x{}] + [{}x{}]", A_matrix.height(), A_matrix.width(), B_matrix.height(), B_matrix.width());
            }

            let mut C_matrix: Matrix<$T> = Matrix::new_with_constant_values(A_matrix.height(), B_matrix.width(), NumCast::from(0).unwrap());

            for row_index in 0..C_matrix.height() {
                for collumn_index in 0..C_matrix.width() {
                    C_matrix[row_index][collumn_index] = $T::row_dot_collumn(&A_matrix[row_index], &B_matrix.get_collumn(collumn_index))
                }
            }

            C_matrix
        }
    }
    };
}

impl_math_functions_per_type!(i8);
impl_math_functions_per_type!(i16);
impl_math_functions_per_type!(i32);
impl_math_functions_per_type!(i64);

impl_math_functions_per_type!(u8);
impl_math_functions_per_type!(u16);
impl_math_functions_per_type!(u32);
impl_math_functions_per_type!(u64);

impl_math_functions_per_type!(isize);
impl_math_functions_per_type!(usize);

impl_math_functions_per_type!(f32);
impl_math_functions_per_type!(f64);



impl_spatial_types_per_type!(i8);
impl_spatial_types_per_type!(i16);
impl_spatial_types_per_type!(i32);
impl_spatial_types_per_type!(i64);

impl_spatial_types_per_type!(isize);

impl_spatial_types_per_type!(f32);
impl_spatial_types_per_type!(f64);