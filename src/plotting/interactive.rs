use crate::plotting::{*};
use crate::egui::Image;
use eframe::CreationContext;
use egui::{ColorImage, Vec2};
use egui::Context;
use image::{*};
use egui::TextureHandle;
use egui::Sense;

const PLOT_RESOLUTION_FACTOR: u32 = 2;

#[derive(Clone)]
pub struct InteractivePlotSettings{
    pub x_range: Range<f64>,
    pub y_range: Range<f64>,
    pub plot_height: u32,
    pub plot_width: u32,
    pub some_input_number: usize,

}

impl Default for InteractivePlotSettings {
    fn default() -> Self {
        Self { 
            x_range: -5. .. 5., 
            y_range: -5. .. 5., 
            plot_height: 1000, 
            plot_width: 1000,
            some_input_number: 10}
    }
}

struct InteractivePlot {
    plot_builder_function: Rc<Box<dyn Fn(InteractivePlotSettings) -> PlotBuilder<f64>>>,
    saved_plot: PlotBuilder<f64>,
    plot_texture_handle: TextureHandle,
    x_range: Range<f64>,
    y_range: Range<f64>,
    plot_height: u32,
    plot_width: u32,
    some_input_number: usize,
}

pub fn make_interactive_plot(function_to_plot: Rc<Box<dyn Fn(InteractivePlotSettings) -> PlotBuilder<f64>>>) {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default(),
        ..Default::default()
    };
    eframe::run_native("interactive plot", options, 
        Box::new(|cc| {
            egui_extras::install_image_loaders(&cc.egui_ctx);
            Box::new(InteractivePlot::new(function_to_plot, cc))
        }
        )
        ).unwrap();
}

impl InteractivePlot {
    fn new(function_to_plot: Rc<Box<dyn Fn(InteractivePlotSettings) -> PlotBuilder<f64>>>, cc: &CreationContext<'_>) -> Self {
        let plot_settings = InteractivePlotSettings::default();
        let mut plot = (function_to_plot.clone())(plot_settings.clone());
        plot.set_plot_size(plot_settings.plot_width * PLOT_RESOLUTION_FACTOR, plot_settings.plot_height * PLOT_RESOLUTION_FACTOR);
        plot.set_x_range(plot_settings.x_range.clone()).set_y_range(plot_settings.y_range.clone());
        let plot_rgb = plot.clone().to_plotters_processor().bitmap_to_rgb();
        let plot_image = image::ImageBuffer::from_raw(plot_settings.plot_width * PLOT_RESOLUTION_FACTOR, plot_settings.plot_height * PLOT_RESOLUTION_FACTOR, plot_rgb).unwrap();
        let plot_buffer = ImageBuffer::<Rgb<u8>, Vec<u8>>::from(plot_image);
        let plot_pixels = plot_buffer.as_flat_samples();
        let plot_color_image = egui::ColorImage::from_rgb(
            [(plot_settings.plot_width * PLOT_RESOLUTION_FACTOR) as usize, (plot_settings.plot_height * PLOT_RESOLUTION_FACTOR) as usize],
            plot_pixels.as_slice()
        );
        let plot_texture_handle = cc.egui_ctx.load_texture("fist plot", plot_color_image, Default::default());

        Self { 
            plot_builder_function: function_to_plot.clone(),
            saved_plot: plot,
            plot_texture_handle,
            x_range: plot_settings.x_range, 
            y_range: plot_settings.y_range, 
            plot_height: plot_settings.plot_height, 
            plot_width: plot_settings.plot_width,
            some_input_number: 1}
    }

    // fn make_plot_settings(&self) -> InteractivePlotSettings {
    //     InteractivePlotSettings { 
    //         x_range: self.x_range.clone(), 
    //         y_range: self.y_range.clone(), 
    //         plot_height: self.plot_height, 
    //         plot_width: self.plot_width }
    // }

    fn reprocess_function(&mut self) {
        let settings = InteractivePlotSettings { 
            x_range: self.x_range.clone(), 
            y_range: self.y_range.clone(), 
            plot_height: self.plot_height, 
            plot_width: self.plot_width, 
            some_input_number: self.some_input_number 
        };
        let new_plot_builder = (self.plot_builder_function)(settings);
        self.saved_plot = new_plot_builder;
    }

    fn update_plot_lazy(&mut self) {
        self.saved_plot.x_range = Option::Some(self.x_range.clone());
        self.saved_plot.y_range = Option::Some(self.y_range.clone());
        self.saved_plot.plot_height = Option::Some((self.plot_height * PLOT_RESOLUTION_FACTOR) as u32);
        self.saved_plot.plot_width = Option::Some((self.plot_width * PLOT_RESOLUTION_FACTOR) as u32);
    }

    fn update_lazy_and_render(&mut self) {
        self.update_plot_lazy();
        let start_render = Instant::now();
        let plot_rgb = self.saved_plot.clone().to_plotters_processor().bitmap_to_rgb();
        let stop_render = start_render.elapsed();
        let start_image = Instant::now();
        let plot_image = image::ImageBuffer::from_raw((self.plot_width* PLOT_RESOLUTION_FACTOR) as u32, (self.plot_height * PLOT_RESOLUTION_FACTOR) as u32, plot_rgb).unwrap();
        let plot_buffer = ImageBuffer::<Rgb<u8>, Vec<u8>>::from(plot_image);
        let plot_pixels = plot_buffer.as_flat_samples();
        let plot_color_image = egui::ColorImage::from_rgb(
            [(self.plot_width * PLOT_RESOLUTION_FACTOR) as usize, (self.plot_height * PLOT_RESOLUTION_FACTOR) as usize],
            plot_pixels.as_slice()
        );
        let stop_image = start_image.elapsed();
        let start_texture = Instant::now();
        self.plot_texture_handle.set(plot_color_image, Default::default());
        let stop_texture = start_texture.elapsed();
        // println!("render: {:?}, image: {:?}, texture: {:?}", stop_render, stop_image, stop_texture);   
    }

}

impl eframe::App for InteractivePlot {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {

        let mut plot_has_been_altered_lazy = false;
        let mut plot_has_changed_reprocess = false;
        // let window_info =  frame.viewport()
        // ctx.available_rect()

        egui::SidePanel::right("the right panel").exact_width(100.).show(ctx, |ui| {
            ui.label("label text");
            if ui.button("itterate").clicked() {
                self.some_input_number += 10;
                plot_has_changed_reprocess = true;
                plot_has_been_altered_lazy = true;
            }
        });
        egui::CentralPanel::default().show(ctx, |ui| {
            let window_size = ui.available_size_before_wrap();
            // println!("windowsize {:?}", window_size);

            if !(window_size.x as u32 == 0) {
                plot_has_been_altered_lazy = true;
                self.plot_width = window_size.x as u32;
            }
            if !(window_size.y as u32 == 0) {
                plot_has_been_altered_lazy = true;
                self.plot_height = window_size.y as u32;
            }

            let plotted_image = ui.add_sized(
                ui.available_size(), 
                egui::Image::from_texture(&self.plot_texture_handle)
                            .maintain_aspect_ratio(true)
                            .shrink_to_fit()
                            .max_size(Vec2::new(self.plot_width as f32, self.plot_height as f32))
                            .sense(Sense::drag()));



            if plotted_image.dragged() {
                plot_has_been_altered_lazy = true;

                let mouse_delta = plotted_image.drag_delta();
                let margins_x = self.saved_plot.plotting_settings.plotters_margin * 2 
                    + self.saved_plot.plotting_settings.plotters_y_label_area_size 
                    + self.saved_plot.plotting_settings.plotters_right_y_label_area_size;
                let margins_y = self.saved_plot.plotting_settings.plotters_margin * 2
                    + self.saved_plot.plotting_settings.plotters_x_label_area_size;
                let relative_drag_x = mouse_delta.x as f64 * PLOT_RESOLUTION_FACTOR as f64
                    / ((self.saved_plot.plot_width.unwrap() - margins_x) as f64);
                let relative_drag_y = mouse_delta.y as f64 * PLOT_RESOLUTION_FACTOR as f64 
                    / ((self.saved_plot.plot_width.unwrap() - margins_y) as f64);

                let (x_range_low, x_range_high) = (self.x_range.start, self.x_range.end);
                let (y_range_low, y_range_high) = (self.y_range.start, self.y_range.end);
                let plotted_x_range = x_range_high - x_range_low;
                let plotted_y_range = y_range_high - y_range_low;
                self.x_range = (x_range_low - plotted_x_range*relative_drag_x) .. (x_range_high - plotted_x_range*relative_drag_x);
                self.y_range = (y_range_low + plotted_y_range*relative_drag_y) .. (y_range_high + plotted_y_range*relative_drag_y);
                
            }

            let delta_scroll: f32 = ui.input(|i|i.scroll_delta.y);
            if !(delta_scroll == 0.) {
                plot_has_been_altered_lazy = true;
                let scroll_factor: f64 = 0.001;
                let zoom_value = delta_scroll as f64 * scroll_factor;

                let (x_range_low, x_range_high) = (self.x_range.start, self.x_range.end);
                let (y_range_low, y_range_high) = (self.y_range.start, self.y_range.end);
                let plotted_x_range = x_range_high - x_range_low;
                let plotted_y_range = y_range_high - y_range_low;
                self.x_range = (x_range_low + plotted_x_range*zoom_value) .. (x_range_high - plotted_x_range*zoom_value);
                self.y_range = (y_range_low + plotted_y_range*zoom_value) .. (y_range_high - plotted_y_range*zoom_value);
            }
        });
        if plot_has_changed_reprocess {
            self.reprocess_function();
        }
        if plot_has_been_altered_lazy {
            self.update_lazy_and_render();
        }
    }
}