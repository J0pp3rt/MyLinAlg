#![allow(non_snake_case)]

lazy_static!{
    static ref IS_AVX2: bool = is_x86_feature_detected!("avx2");
}

#[macro_use]
extern crate lazy_static;

pub mod types;
pub use types::{*};

pub mod matrix;
pub use matrix::{*};
pub mod row;
pub use row::{*};
pub mod collumn;
pub use collumn::{*};
pub mod supermatrix;
pub use supermatrix::{*};
pub mod mathfunctions;
pub use mathfunctions::{*};
pub mod inputparse;
pub use inputparse::{*};
pub mod linearsolvers;
pub use linearsolvers::{*};
pub mod gpuaccalerators;
pub use gpuaccalerators::{*};
pub mod cpuaccalerators;
pub use cpuaccalerators::{*};

use core::{f32};
use std::arch::x86_64::{_mm256_set_pd, _mm256_load_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_store_pd};
use std::ops::{Index, IndexMut, Deref, Add};
use std::any::{TypeId};
use std::sync::Arc;
use std::ops::Range;
use std::fmt::Display;
use std::fmt::Debug;
use std::time::Instant;
use std::vec;
use cudarc::cublas::{safe, result, sys, Gemv, CudaBlas, GemvConfig, Gemm};

use cudarc::driver::{CudaDevice, DevicePtr, CudaSlice, DeviceSlice};
use num::traits::NumOps;
use num::{Num, NumCast};
use rand::Rng;
use rand::distributions::uniform::SampleUniform;
