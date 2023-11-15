#![allow(non_snake_case)]
#![feature(stdsimd)]
#![feature(portable_simd)]

lazy_static!{
    static ref IS_AVX2: bool = is_x86_feature_detected!("avx2");
}

lazy_static!{
    static ref IS_64X: bool = cfg!(target_pointer_width = "64");
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

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

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
