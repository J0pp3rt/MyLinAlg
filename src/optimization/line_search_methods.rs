use std::marker::PhantomData;

// use mla::plotting::egui::accesskit::Orientation;

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

pub struct UnkownBracketPointOnly {}
pub struct UnkownBracketWithGradient {}
pub struct Bracketed {}
pub struct QuadraticInterpolation {}
pub struct GoldenRuleInterpolation {}

pub trait IsInterpolation {
    fn itterate_untill_convergence(&mut self) ;
    fn itterate(&mut self);
    fn bracket_averaged_variables(&self) -> PosNDof<f64>;
    fn used_function_evals(&self) -> usize;
}

impl IsInterpolation for LineSearchMethods<GoldenRuleInterpolation, IsColl> {
    fn itterate_untill_convergence(&mut self) {
        self.itterate_untill_convergence();
    }
    fn itterate(&mut self) {
        self.itterate();
    }
    fn bracket_averaged_variables(&self) -> PosNDof<f64> {
        self.bracket_averaged_variables()
    }

    fn used_function_evals(&self) -> usize {
        self.used_function_evals
    }
}

impl IsInterpolation for LineSearchMethods<QuadraticInterpolation, IsColl> {
    fn itterate_untill_convergence(&mut self) {
        self.itterate_untill_convergence();
    }
    fn itterate(&mut self){
        self.itterate();
    }
    fn bracket_averaged_variables(&self) -> PosNDof<f64> {
        self.bracket_averaged_variables()
    }

    fn used_function_evals(&self) -> usize {
        self.used_function_evals
    }
}

pub struct LineSearchMethods<State, Orientation = IsColl> {
    cost_function: Rc<Box<dyn Fn(Vec<f64>) -> f64 + 'static >>,
    pub search_line: SpatialVectorWithBasePointNDof<f64, Orientation>,
    lower_bound_scale: f64,
    cost_lower_bound: f64,
    upper_bound_scale: f64,
    cost_upper_bound: f64,
    pub central_point: Option<f64>,
    pub cost_central_point: Option<f64>,
    inner_point_low: Option<f64>,
    cost_inner_point_low: Option<f64>,
    inner_point_high: Option<f64>,
    cost_inner_point_high: Option<f64>,
    error_margin: f64,
    initial_step_size: f64,
    pub used_function_evals: usize,
    _phantom: PhantomData<State>,
}

impl LineSearchMethods<IsColl, UnkownBracketPointOnly> {

}

impl LineSearchMethods<UnkownBracketWithGradient,IsColl> {
    pub fn bracket_extrapolation_method(mut self, extrapolation_method: LineSearchExtrapolationMethods) -> LineSearchMethods<Bracketed, IsColl> {

        match extrapolation_method {
            LineSearchExtrapolationMethods::GoldenRuleExtrapolation => {
                self.find_bracket_golden_rule_expansion()
            },
        }
    }
    
    pub fn find_bracket_golden_rule_expansion(mut self) -> LineSearchMethods<Bracketed, IsColl> {        

        let mut search_line = self.search_line.clone();
        search_line.scale_to_length(self.initial_step_size);
        let cost_at_initial_guess = self.cost(0.);
        
        match search_line.length() == 0. {
            true => {
                panic!("Search line has no direction (length = 0)");
            },
            _ => {},
        }

        let cost_at_first_step = self.cost(1.);

        let mut step_min_2_cost = cost_at_initial_guess;
        let mut step_min_2_total_distance = 0.;
        let mut at_least_one_step_taken = false;

        let mut step_min_1_cost = cost_at_first_step;
        let mut step_min_1_total_distance = self.initial_step_size;

        let mut step_now_cost: f64 = 0.;
        let mut step_now_total_distance: f64 = 0.;

        let mut inflection_not_yet_found = true;
        while inflection_not_yet_found {

            step_now_total_distance = step_min_1_total_distance * (1. + golden_ratio_conjugate());
            let step_now_x = search_line.volatile().scale_to_length(step_now_total_distance).end_position();
            step_now_cost = self.cost_x(step_now_x);

            match step_now_cost > step_min_1_cost {
                false => {
                    at_least_one_step_taken = true;

                    step_min_2_cost = step_min_1_cost;
                    step_min_2_total_distance = step_min_1_total_distance;

                    step_min_1_cost = step_now_cost;
                    step_min_1_total_distance = step_now_total_distance;
                },
                true => {
                    inflection_not_yet_found = false;
                }
            }
        }
        
        let lenght_after_transforamtion = step_now_total_distance - step_min_2_total_distance;
        let mut search_line = search_line.clone();

        if at_least_one_step_taken {
            search_line
                .scale_to_length(step_min_2_total_distance)
                .move_base_to_end_position()
                .scale_to_length(lenght_after_transforamtion);
        } else {
            search_line
                .scale_to_length(lenght_after_transforamtion);
        }
        
        let lower_bound_scale = 0.;
        let cost_lower_bound = step_min_2_cost;
        let upper_bound_scale = 1.;
        let cost_upper_bound = step_now_cost;
        let inner_point_low: Option<f64> = Option::None;;
        let cost_inner_point_low: Option<f64> = Option::None;;
        let inner_point_high: Option<f64> = Option::None;
        let cost_inner_point_high: Option<f64> = Option::None;

        LineSearchMethods::<Bracketed, IsColl> {
            cost_function: self.cost_function,
            search_line,
            lower_bound_scale,
            cost_lower_bound,
            upper_bound_scale,
            cost_upper_bound,
            central_point: Option::None,
            cost_central_point: Option::None,
            inner_point_low,
            cost_inner_point_low,
            inner_point_high,
            cost_inner_point_high,
            error_margin: self.error_margin,
            used_function_evals: self.used_function_evals,
            initial_step_size: self.initial_step_size,
            _phantom: PhantomData::<Bracketed>
        }
    }
}

impl LineSearchMethods<Bracketed, IsColl> {

    pub fn interpolation_method(self, interpolation_method: LineSearchInterpolationMethods) -> Box<dyn IsInterpolation> {
        
        match interpolation_method {
            LineSearchInterpolationMethods::GoldenRuleInterpolation => {
                Box::new(self.use_golden_section_interpolation())
            },
            LineSearchInterpolationMethods::QuadraticInterpolation => {
                Box::new(self.use_quadratic_interpolation())
            },
        }
    }

    pub fn use_golden_section_interpolation(self) -> LineSearchMethods<GoldenRuleInterpolation>{
        LineSearchMethods::<GoldenRuleInterpolation> {
            cost_function: self.cost_function,
            search_line: self.search_line,
            lower_bound_scale: self.lower_bound_scale,
            cost_lower_bound: self.cost_lower_bound,
            upper_bound_scale: self.upper_bound_scale,
            cost_upper_bound: self.cost_upper_bound,
            central_point: self.central_point,
            cost_central_point: self.cost_central_point,
            inner_point_low: self.inner_point_low,
            cost_inner_point_low: self.cost_inner_point_low,
            inner_point_high: self.inner_point_high,
            cost_inner_point_high: self.cost_inner_point_high,
            error_margin: self.error_margin,
            initial_step_size: self.initial_step_size,
            used_function_evals: self.used_function_evals,
            _phantom: PhantomData::<GoldenRuleInterpolation>
        }
    }

    pub fn use_quadratic_interpolation(self) -> LineSearchMethods<QuadraticInterpolation>{
        LineSearchMethods::<QuadraticInterpolation, IsColl> {
            cost_function: self.cost_function,
            search_line: self.search_line,
            lower_bound_scale: self.lower_bound_scale,
            cost_lower_bound: self.cost_lower_bound,
            upper_bound_scale: self.upper_bound_scale,
            cost_upper_bound: self.cost_upper_bound,
            central_point: self.central_point,
            cost_central_point: self.cost_central_point,
            inner_point_low: self.inner_point_low,
            cost_inner_point_low: self.cost_inner_point_low,
            inner_point_high: self.inner_point_high,
            cost_inner_point_high: self.cost_inner_point_high,
            error_margin: self.error_margin,
            initial_step_size: self.initial_step_size,
            used_function_evals: self.used_function_evals,
            _phantom: PhantomData::<QuadraticInterpolation>
        }
    }
}

impl LineSearchMethods<QuadraticInterpolation, IsColl> {
    pub fn itterate_untill_convergence(&mut self) -> &mut Self {

        while self.is_completed().not() {
            self.itterate();
        }
        self
    }

    pub fn itterate(&mut self) -> &mut Self {

        let central_point: f64;
        let cost_central_point: f64;

        match (self.central_point, self.cost_central_point) {
            (Option::Some(location), Option::Some(cost)) => {
                central_point = location;
                cost_central_point = cost;
            }
            _ => {
                self.central_point = Option::Some(0.5);
                central_point = 0.5;
                cost_central_point = self.cost(0.5);
            }
        }
        
        // fitted to cost = c_1 + c_2 x + c_3 x^2
        let c_1 = self.cost_lower_bound;
        let c_3 = 
            (cost_central_point - c_1) / (central_point * (central_point - 1.)) 
            + (c_1 - self.cost_upper_bound) / (central_point - 1.);
        let c_2 = 
            self.cost_upper_bound - self.cost_lower_bound - c_3;
        
        let lowest_point = - c_2 / (2. * c_3);
        let cost_lowest_point = self.cost(lowest_point);
        #[derive(Clone)]
        enum CostIs {
            LowerThenCentral,
            HigherThenCentral
        }

        enum LowestPoint {
            LeftOfCentral,
            RightOfCentral
        }

        let position_lowest: LowestPoint;
        let comparitive_cost: CostIs;
        if lowest_point < central_point {
            position_lowest = LowestPoint::LeftOfCentral;
        } else {
            position_lowest = LowestPoint::RightOfCentral;
        }

        if cost_lowest_point < cost_central_point {
            comparitive_cost = CostIs::LowerThenCentral;
        } else {
            comparitive_cost = CostIs::HigherThenCentral;
        }

        match (position_lowest, comparitive_cost.clone()) {
            (LowestPoint::LeftOfCentral, CostIs::LowerThenCentral) => {
                let length_after_transformation = central_point;
                let length_to_new_central_point = lowest_point / length_after_transformation;
                self.central_point = Option::Some(length_to_new_central_point);
                self.cost_central_point = Option::Some(cost_lowest_point);
                self.upper_bound_scale = 1.;
                self.cost_upper_bound = cost_central_point;
                self.search_line.scale(central_point);
            },
            (LowestPoint::LeftOfCentral, CostIs::HigherThenCentral) => {
                let length_after_transformation = 1. -  lowest_point;
                let length_to_new_central_point = (central_point - lowest_point) / length_after_transformation;
                let scaling_factor_after_transform = (1. - lowest_point) / lowest_point;
                self.central_point = Option::Some(length_to_new_central_point);
                self.cost_central_point = Option::Some(cost_central_point);
                self.lower_bound_scale = 0.;
                self.cost_lower_bound = cost_lowest_point;
                self.search_line.scale(lowest_point).move_base_to_end_position().scale(scaling_factor_after_transform);
            }, 
            (LowestPoint::RightOfCentral, CostIs::LowerThenCentral) => {
                let length_after_transformation = 1. - central_point;
                let length_to_new_central_point = (lowest_point - central_point) / length_after_transformation;
                let scaling_factor_after_transform = (1. - central_point) / central_point;
                self.central_point = Option::Some(length_to_new_central_point);
                self.cost_central_point = Option::Some(cost_lowest_point);
                self.lower_bound_scale = 0.;
                self.cost_lower_bound = cost_central_point;
                self.search_line.scale(central_point);
                self.search_line.move_base_to_end_position();
                self.search_line.scale(scaling_factor_after_transform);
                // println!("change lower bound")
            },
            (LowestPoint::RightOfCentral, CostIs::HigherThenCentral) => {
                let length_after_transformation = lowest_point;
                let length_to_new_central_point = central_point / length_after_transformation;
                self.central_point = Option::Some(length_to_new_central_point);
                self.cost_central_point = Option::Some(cost_central_point);
                self.upper_bound_scale = 1.;
                self.cost_upper_bound = cost_lowest_point;
                self.search_line.scale(lowest_point);
            }, 
        }
        
        let stuck_condition_3 = match comparitive_cost {
                CostIs::LowerThenCentral => {false},
                CostIs::HigherThenCentral => {true},
            };

        println!("central_point {}, lowest_point {}, is higher then central {}", central_point, lowest_point, stuck_condition_3);
        let stuck_condition_1 = central_point > 0.8;
        let stuck_condition_2 = lowest_point > 0.9;

        if stuck_condition_1 && stuck_condition_2 {
            self.cost_central_point = Option::None;
            self.central_point = Option::None;
        }

        let stuck_condition_1 = central_point < 0.2;
        let stuck_condition_2 = lowest_point < 0.1;

        if stuck_condition_1 && stuck_condition_2 {
            self.cost_central_point = Option::None;
            self.central_point = Option::None;
        }

        // if it is moving only to one side, it is probably safe to move the lower bound a bit to the right
        // if stuck_condition_1 && stuck_condition_2 && stuck_condition_3 {
        //     println!("expected_cost_at_expected: {}, cost at central {}, cost at upper {}", , cost_central_point, self.cost_upper_bound);
        //     let length_after_transformation = 1. -  0.1; // a choosen value
        //     let cost_at_new_lower_bound = self.cost(0.1);
        //     self.cost_lower_bound = cost_at_new_lower_bound;
        //     let scaling_factor_to_upper_after_transform = (1. - length_after_transformation) / length_after_transformation;

        //     match (self.central_point, self.cost_central_point) {
        //         (Option::Some(central_point), Option::Some(_)) => {
        //             let length_to_central_point_after_transform = central_point / length_after_transformation;
        //             self.central_point = Option::Some(length_to_central_point_after_transform)
        //         }
        //         _ => {}
        //     }
        //     self.lower_bound_scale = 0.;
        //     self.search_line.scale(0.4).move_base_to_end_position().scale(scaling_factor_to_upper_after_transform);
        //     println!("Applied the anti-get-stuck-on-quadratic rule!");
        //     println!("cost at central {}, cost at upper {}", self.cost(self.central_point.unwrap()), self.cost(1.));
        // } 

        println!("central_point {}, lowest_point {}, is higher then central {}", central_point, lowest_point, stuck_condition_3);

        self
    }
}

impl LineSearchMethods<GoldenRuleInterpolation, IsColl> {
    pub fn itterate_untill_convergence(&mut self) -> &mut Self {

        while self.is_completed().not() {
            self.itterate();
        }
        self
    }

    pub fn itterate(&mut self) -> &mut Self {

        let inner_point_low: f64;
        let cost_inner_point_low: f64;
        match (self.inner_point_low, self.cost_inner_point_low) {
            (Option::Some(_), Option::Some(cost)) => {
                cost_inner_point_low = cost;
            }
            _ => {
                inner_point_low = 1. - golden_ratio_conjugate();
                cost_inner_point_low = self.cost(inner_point_low);
            }
        }

        let inner_point_high: f64;
        let cost_inner_point_high: f64;
        match (self.inner_point_high, self.cost_inner_point_high) {
            (Option::Some(_), Option::Some(cost)) => {
                cost_inner_point_high = cost;
            }
            _ => {
                inner_point_high = golden_ratio_conjugate();
                cost_inner_point_high = self.cost(inner_point_high);
            }
        }

        enum CostHigherOn {
            InnerPointLow,
            InnerPointHigh,
        }

        let highest_value_on: CostHigherOn;
        if cost_inner_point_low > cost_inner_point_high {
            highest_value_on = CostHigherOn::InnerPointLow;
        } else {
            highest_value_on = CostHigherOn::InnerPointHigh;
        }

        let lenght_after_transforamtion = self.search_line.length() * golden_ratio_conjugate();

        match highest_value_on {
            CostHigherOn::InnerPointLow => {
                self.search_line
                    .scale(1.-golden_ratio_conjugate() )
                    .move_base_to_end_position()
                    .scale_to_length(lenght_after_transforamtion);
                self.lower_bound_scale = 0.;
                self.cost_lower_bound = cost_inner_point_low;
                self.inner_point_low = Option::Some(1. - golden_ratio_conjugate());
                self.cost_inner_point_low = Option::Some(cost_inner_point_high);
                self.inner_point_high = Option::None; 
                self.cost_inner_point_high = Option::None;
            },
            CostHigherOn::InnerPointHigh => {
                self.search_line
                    .scale_to_length(lenght_after_transforamtion);
                self.upper_bound_scale = 1.;
                self.cost_upper_bound = cost_inner_point_high;
                self.inner_point_low = Option::None;
                self.cost_inner_point_low = Option::None;
                self.inner_point_high = Option::Some(golden_ratio_conjugate());
                self.cost_inner_point_high = Option::Some(cost_inner_point_low);
            },
        }
        self
    }
}

impl<State> LineSearchMethods<State, IsColl> {
    pub fn coordinates_of_lower_bound(&self) -> Vec<f64> {
        self.search_line.base_point_vec()
    }

    pub fn coordinates_of_upper_bound(&self) -> Vec<f64> {
        self.search_line.end_position_vec()
    }

    pub fn coordinates_at_scale(&self, scale: f64) -> Vec<f64> {
        self.search_line.volatile().scale(scale).end_position_vec()
    }

    pub fn is_completed(&self) -> bool {
        // this function assumes that the vector is normalized in between lower and upper bound.
        let coord_distance = 
            self.search_line.length();
        coord_distance < self.error_margin 
    }

    pub fn cost(&mut self, at_scaling_factor: f64) -> f64 {
        self.used_function_evals += 1;
        (self.cost_function)(self.search_line.volatile().scale(at_scaling_factor).end_position_vec())
    }

    pub fn cost_x(&mut self, position: PosNDof<f64>) -> f64 {
        self.used_function_evals += 1;
        (self.cost_function)(position.x)
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


impl LineSearchMethods<Bracketed, IsColl> {
    pub fn new_from_bracket( 
            cost_function: Rc<Box<dyn Fn(Vec<f64>) -> f64>>,
            mut search_line: SpatialVectorWithBasePointNDof<f64, IsColl>,
            lower_bound_scale: f64,
            upper_bound_scale: f64,
            error_margin: f64,
        ) -> LineSearchMethods<Bracketed, IsColl> {

        println!("using new_from_bracket, i haven't tested thois function properly!");
        let lenght_after_transforamtion = upper_bound_scale - upper_bound_scale;
        search_line
            .scale(lower_bound_scale)
            .move_base_to_end_position()
            .scale_to_length(lenght_after_transforamtion);
        let cost_lower_bound = (cost_function)(search_line.volatile().scale(0.).end_position_vec());
        let cost_upper_bound = (cost_function)(search_line.volatile().scale(1.).end_position_vec());

        LineSearchMethods::<Bracketed, IsColl> { 
            cost_function,
            search_line,
            lower_bound_scale, 
            cost_lower_bound,
            upper_bound_scale, 
            cost_upper_bound,
            central_point: Option::None,
            cost_central_point: Option::None,
            inner_point_low: Option::None,
            cost_inner_point_low: Option::None,
            inner_point_high: Option::None,
            cost_inner_point_high: Option::None,
            error_margin,
            initial_step_size: 1.,
            used_function_evals: 2,
            _phantom: PhantomData::<Bracketed>
        }
    }
}

impl LineSearchMethods<UnkownBracketWithGradient, IsColl> {
    pub fn new_with_search_line(
            cost_function: Rc<Box<dyn Fn(Vec<f64> ) -> f64 + 'static>>,
            search_line: SpatialVectorWithBasePointNDof<f64, IsColl>,
            error_margin: f64,
        ) -> LineSearchMethods<UnkownBracketWithGradient, IsColl> {

        let inital_step_size = 0.01;

        Self::new_with_search_line_and_initial_stepsize(        
            cost_function,
            &search_line,
            inital_step_size,
            error_margin,
        )
    }

    pub fn new_with_search_line_and_initial_stepsize(
            cost_function: Rc<Box<dyn Fn(Vec<f64> ) -> f64 + 'static>>,
            search_line: &SpatialVectorWithBasePointNDof<f64, IsColl>,
            initial_step_size: f64,
            error_margin: f64,
        ) -> LineSearchMethods<UnkownBracketWithGradient, IsColl> {


        LineSearchMethods::<UnkownBracketWithGradient, IsColl> { 
            cost_function,
            search_line: search_line.clone(),
            lower_bound_scale: 0., 
            cost_lower_bound: 0.,
            upper_bound_scale: 0., 
            cost_upper_bound: 0.,
            central_point: Option::None,
            cost_central_point: Option::None,
            inner_point_low: Option::None,
            cost_inner_point_low: Option::None,
            inner_point_high: Option::None,
            cost_inner_point_high: Option::None,
            error_margin,
            initial_step_size,
            used_function_evals: 0,
            _phantom: PhantomData::<UnkownBracketWithGradient>
        }
    }

    fn new_initial_point_and_stepsize(        
            cost_function: Rc<Box<dyn Fn(Vec<f64> ) -> f64 + 'static>>,
            search_point: &PosNDof<f64>,
            initial_step_size: f64,
            error_margin: f64,
        ) -> LineSearchMethods<UnkownBracketPointOnly, IsColl> {

        let n_dof = search_point.n_dof();

        LineSearchMethods::<UnkownBracketPointOnly, IsColl> {
            cost_function,
            search_line: SpatialVectorWithBasePointNDof::new_with_base( &search_point.x(), &vec![0.; n_dof]),
            lower_bound_scale: 0.,
            cost_lower_bound: 0.,
            upper_bound_scale: 0.,
            cost_upper_bound: 0.,
            central_point: Option::None,
            cost_central_point: Option::None,
            inner_point_low: Option::None,
            cost_inner_point_low: Option::None,
            inner_point_high: Option::None,
            cost_inner_point_high: Option::None,
            error_margin,
            initial_step_size,
            used_function_evals: 0,
            _phantom: PhantomData::<UnkownBracketPointOnly>
        }
        }
}