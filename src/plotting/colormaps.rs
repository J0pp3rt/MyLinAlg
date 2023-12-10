use std::marker::PhantomData;
use plotters::prelude::full_palette::GREY;

use crate::plotting::{*};

#[derive(Clone, Debug)]
pub enum PlotBuilderColorMaps {
    Palette99,
    Palette99ReversedOrder,
    Viridis,
    ViridisInverse,
    Copper,
    CopperInverse,
    Bone,
    BoneInverser,
    Mandelbrodt,
    MandelBrodtInverse,
    ConstantGray
}

pub struct ColorMaps {}

impl ColorMaps{
/// input variables are normalized value and value it is normalized to
/// if more are included these are ignored, values are also not always used
    pub fn get_color_from_map(input_values: Vec<f32>, specified_colormap: PlotBuilderColorMaps) -> RGBAColor {
        match specified_colormap {
            PlotBuilderColorMaps::Palette99 => Self::get_color_palette_index_of_total(input_values),
            PlotBuilderColorMaps::Palette99ReversedOrder => Self::get_color_palette_index_of_total_reversed(input_values),
            PlotBuilderColorMaps::Viridis => Self::get_color_viridis(input_values),
            PlotBuilderColorMaps::ViridisInverse => Self::get_color_viridis_inverse(input_values),
            PlotBuilderColorMaps::Copper => Self::get_color_copper(input_values),
            PlotBuilderColorMaps::CopperInverse => Self::get_color_copper_inverse(input_values),
            PlotBuilderColorMaps::Bone => Self::get_color_bone(input_values),
            PlotBuilderColorMaps::BoneInverser => Self::get_color_bone_inverse(input_values),
            PlotBuilderColorMaps::Mandelbrodt => Self::get_color_mandelbrodt(input_values),
            PlotBuilderColorMaps::MandelBrodtInverse => Self::get_color_mandelbrodt_inverse(input_values),
            PlotBuilderColorMaps::ConstantGray => Self::get_color_gray(input_values),
        }
    }

    fn get_color_palette_index_of_total(input_values: Vec<f32>) -> RGBAColor {
        let mut index = (input_values[0] * input_values[1]) as usize;
        let translation_vector = vec![3, 4, 1, 2, 0, 5, 6, 9, 10, 7, 8];
        /////////////////////////////////////vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        if index <= translation_vector.len()-1 {
            index = translation_vector[index]
        }
        Palette99::pick(index).to_rgba()

        
    }

    fn get_color_palette_index_of_total_reversed(input_values: Vec<f32>) -> RGBAColor {
        let index = 1 - (input_values[0] * input_values[1]) as usize;
        Palette99::pick(index).mix(0.9).to_rgba()
    }

    fn get_color_viridis(input_values: Vec<f32>) -> RGBAColor {
        ViridisRGBA::get_color(input_values[0])
    }

    fn get_color_viridis_inverse(input_values: Vec<f32>) -> RGBAColor {
        Self::get_color_viridis(vec![1.-input_values[0]])
    }

    fn get_color_copper(input_values: Vec<f32>) -> RGBAColor {
        Copper::get_color(input_values[0]).to_rgba()

    }

    fn get_color_copper_inverse(input_values: Vec<f32>) -> RGBAColor {
        Self::get_color_copper(vec![1.-input_values[0]])
    }

    fn get_color_bone(input_values: Vec<f32>) -> RGBAColor {
        Bone::get_color(input_values[0]).to_rgba()
    }

    fn get_color_bone_inverse(input_values: Vec<f32>) -> RGBAColor {
        Self::get_color_bone(vec![1.-input_values[0]])
    }

    fn get_color_mandelbrodt(input_values: Vec<f32>) -> RGBAColor {
        MandelbrotHSL::get_color(input_values[0]).to_rgba()
    }

    fn get_color_mandelbrodt_inverse(input_values: Vec<f32>) -> RGBAColor {
        Self::get_color_mandelbrodt(vec![1.-input_values[0]])
    }

    fn get_color_gray(input_values: Vec<f32>) -> RGBAColor {
        GREY.to_rgba()
    }

}