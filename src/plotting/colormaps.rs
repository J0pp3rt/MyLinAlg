use std::marker::PhantomData;
use plotters::prelude::full_palette::GREY;

use crate::plotting::{*};

#[derive(Clone, Debug)]
pub enum PlotBuilderColorMaps {
    Palette99(f64),
    Palette99ReversedOrder(f64),
    Viridis(f64),
    ViridisInverse(f64),
    Copper(f64),
    CopperInverse(f64),
    Bone(f64),
    BoneInverser(f64),
    Mandelbrodt(f64),
    MandelBrodtInverse(f64),
    ConstantGray(f64),
    ConstantRed(f64),
    Vulcano(f64),
    VulcanoInverse(f64)
}

pub struct ColorMaps {}

impl ColorMaps{
/// input variables are normalized value and value it is normalized to
/// if more are included these are ignored, values are also not always used

    pub fn get_color_from_map_simple(input_value: f32, specified_colormap: PlotBuilderColorMaps) -> RGBAColor {
        match specified_colormap {
            PlotBuilderColorMaps::Palette99(_) => {panic!("Simple get colormap is not supported for pallete colormaps!")},
            PlotBuilderColorMaps::Palette99ReversedOrder(_) => {panic!("Simple get colormap is not supported for pallete colormaps!")},
            _ => {Self::get_color_from_map(vec![input_value, 0.], specified_colormap)}
        }

        
    }

    pub fn get_color_from_map(mut input_values: Vec<f32>, specified_colormap: PlotBuilderColorMaps) -> RGBAColor {

        match specified_colormap {
            PlotBuilderColorMaps::Palette99(alpha) => Self::get_color_palette_index_of_total(input_values, alpha),
            PlotBuilderColorMaps::Palette99ReversedOrder(alpha) => Self::get_color_palette_index_of_total_reversed(input_values, alpha),
            PlotBuilderColorMaps::Viridis(alpha) => Self::get_color_viridis(input_values, alpha),
            PlotBuilderColorMaps::ViridisInverse(alpha) => Self::get_color_viridis_inverse(input_values, alpha),
            PlotBuilderColorMaps::Copper(alpha) => Self::get_color_copper(input_values, alpha),
            PlotBuilderColorMaps::CopperInverse(alpha) => Self::get_color_copper_inverse(input_values, alpha),
            PlotBuilderColorMaps::Bone(alpha) => Self::get_color_bone(input_values, alpha),
            PlotBuilderColorMaps::BoneInverser(alpha) => Self::get_color_bone_inverse(input_values, alpha),
            PlotBuilderColorMaps::Mandelbrodt(alpha) => Self::get_color_mandelbrodt(input_values, alpha),
            PlotBuilderColorMaps::MandelBrodtInverse(alpha) => Self::get_color_mandelbrodt_inverse(input_values, alpha),
            PlotBuilderColorMaps::Vulcano(alpha) => Self::get_color_vulcano(input_values, alpha),
            PlotBuilderColorMaps::VulcanoInverse(alpha) => Self::get_color_vulcano_inverse(input_values, alpha),
            PlotBuilderColorMaps::ConstantGray(alpha) => Self::get_color_gray(input_values, alpha),
            PlotBuilderColorMaps::ConstantRed(alpha) => Self::get_color_red(input_values, alpha),
        }
    }

    fn get_color_palette_index_of_total(input_values: Vec<f32>, alpha: f64) -> RGBAColor {
        let mut index = (input_values[0] * input_values[1]) as usize;
        let translation_vector = vec![3, 4, 1, 2, 0, 5, 6, 9, 10, 7, 8];
        /////////////////////////////////////vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        if index <= translation_vector.len()-1 {
            index = translation_vector[index]
        }
        let mut color = Palette99::pick(index).to_rgba();
        color.3 = alpha;
        color

        
    }

    fn get_color_palette_index_of_total_reversed(input_values: Vec<f32>, alpha: f64) -> RGBAColor {
        let index = 1 - (input_values[0] * input_values[1]) as usize;
        let mut color = Palette99::pick(index).mix(0.9).to_rgba();
        color.3 = alpha;
        color
    }

    fn get_color_viridis(input_values: Vec<f32>, alpha: f64) -> RGBAColor {
        let mut color = ViridisRGBA::get_color(input_values[0]);
        color.3 = alpha;
        color
    }

    fn get_color_viridis_inverse(input_values: Vec<f32>, alpha: f64) -> RGBAColor {
        Self::get_color_viridis(vec![1.-input_values[0]], alpha)
    }

    fn get_color_copper(input_values: Vec<f32>, alpha: f64) -> RGBAColor {
        let mut color = Copper::get_color(input_values[0]).to_rgba();
        color.3 = alpha;
        color

    }

    fn get_color_copper_inverse(input_values: Vec<f32>, alpha: f64) -> RGBAColor {
        Self::get_color_copper(vec![1.-input_values[0]], alpha)
    }

    fn get_color_bone(input_values: Vec<f32>, alpha: f64) -> RGBAColor {
        let mut color = Bone::get_color(input_values[0]).to_rgba();
        color.3 = alpha;
        color
    }

    fn get_color_bone_inverse(input_values: Vec<f32>, alpha: f64) -> RGBAColor {
        Self::get_color_bone(vec![1.-input_values[0]], alpha)
    }

    fn get_color_mandelbrodt(input_values: Vec<f32>, alpha: f64) -> RGBAColor {
        let mut color = MandelbrotHSL::get_color(input_values[0]).to_rgba();
        color.3 = alpha;
        color
    }

    fn get_color_mandelbrodt_inverse(input_values: Vec<f32>, alpha: f64) -> RGBAColor {
        Self::get_color_mandelbrodt(vec![1.-input_values[0]], alpha)
    }

    fn get_color_vulcano(input_values: Vec<f32>, alpha: f64) -> RGBAColor {
        let mut color = VulcanoHSL::get_color(input_values[0]).to_rgba();
        color.3 = alpha;
        color
    }

    fn get_color_vulcano_inverse(input_values: Vec<f32>, alpha: f64) -> RGBAColor {
        Self::get_color_vulcano(vec![1.-input_values[0]], alpha)
    }

    fn get_color_gray(input_values: Vec<f32>, alpha: f64) -> RGBAColor {
        let mut color = GREY.to_rgba();
        color.3 = alpha;
        color
    }

    fn get_color_red(input_values: Vec<f32>, alpha: f64) -> RGBAColor {
        let mut color = RED.to_rgba();
        color.3 = alpha;
        color
    }

}