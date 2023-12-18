
#[derive(Debug, Clone)]
pub enum LineSearchExtrapolationMethods {
    GoldenRuleExtrapolation
}

#[derive(Clone, Debug)]
pub enum LineSearchInterpolationMethods {
    GoldenRuleInterpolation,
    QuadraticInterpolation,
}

use plotters::element::CoordMapper;

use crate::{*};

pub struct LineSearcher {
    pub cost_controller: CostController,
    pub search_line: SpatialVectorWithBasePointNDof<f64, IsColl>,
    pub lower_bound_scale: Option<f64>,
    pub cost_lower_bound: Option<f64>,
    pub upper_bound_scale: Option<f64>,
    pub cost_upper_bound: Option<f64>,
    pub central_point: Option<f64>,
    pub cost_central_point: Option<f64>,
    pub inner_point_low: Option<f64>,
    pub cost_inner_point_low: Option<f64>,
    pub inner_point_high: Option<f64>,
    pub cost_inner_point_high: Option<f64>,
    pub optimization_settings: OptimizationSettings,
}

impl LineSearcher {
    /// This function immediatly finds the descent direction
    pub fn new(
        search_line: SpatialVectorWithBasePointNDof<f64, IsColl>,
        cost_controller: CostController,
        optimization_settings: OptimizationSettings,
    ) -> Self {

        let n_dof = search_line.n_dof();

        Self { 
            cost_controller, 
            search_line,
            lower_bound_scale: Option::None, 
            cost_lower_bound: Option::None, 
            upper_bound_scale: Option::None, 
            cost_upper_bound: Option::None, 
            central_point: Option::None, 
            cost_central_point: Option::None, 
            inner_point_low: Option::None, 
            cost_inner_point_low: Option::None, 
            inner_point_high: Option::None, 
            cost_inner_point_high: Option::None, 
            optimization_settings 
        }
    }
}

impl LineSearcher {
    pub fn reset_line_searcher(&mut self) -> &mut Self {

        self.lower_bound_scale = Option::None; 
        self.cost_lower_bound = Option::None; 
        self.upper_bound_scale = Option::None; 
        self.cost_upper_bound = Option::None; 
        self.central_point = Option::None; 
        self.cost_central_point = Option::None; 
        self.inner_point_low = Option::None; 
        self.cost_inner_point_low = Option::None; 
        self.inner_point_high = Option::None; 
        self.cost_inner_point_high = Option::None;

        self
    }

    pub fn update_line_searcher(
        &mut self,
        new_search_line: SpatialVectorWithBasePointNDof<f64, IsColl>,
    ) -> &mut Self {

        self.search_line = new_search_line;

        self
    }
}

impl LineSearcher {
    pub fn coordinates_of_lower_bound(&self) -> Vec<f64> {
        self.search_line.base_point_vec()
    }

    pub fn coordinates_of_upper_bound(&self) -> Vec<f64> {
        self.search_line.end_position_vec()
    }

    pub fn coordinates_at_scale(&self, scale: f64) -> Vec<f64> {
        self.search_line.volatile().scale(scale).end_position_vec()
    }

    pub fn cost(&mut self, at_scaling_factor: f64) -> f64 {
        self.cost_controller.cost_at_x(&self.search_line.volatile().scale(at_scaling_factor).end_position_vec())
    }

    pub fn cost_x(&mut self, position: &PosNDof<f64>) -> f64 {
        self.cost_controller.cost_at_pos(&position)
    }

    pub fn bracket_averaged_variables(&self) -> PosNDof<f64> {
        let variables_lower_bound = self.search_line.base_point_vec();
        let variables_upper_bound = self.search_line.end_position_vec();

        let averaged_variables = variables_lower_bound.iter()
            .zip(variables_upper_bound.iter())
            .map(|(x_l, x_u)| (x_l + x_u) / 2.).collect();

        PosNDof::new(averaged_variables)
    }

    pub fn bracket_averaged_variables_vec(&self) -> Vec<f64> {
        self.bracket_averaged_variables().x
    }
}
