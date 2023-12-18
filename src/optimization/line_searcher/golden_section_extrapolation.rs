use crate::{*};

impl LineSearcher {
    pub fn find_bracket_golden_rule_extrapolation(&mut self) -> &mut Self {        

        self.search_line.scale_to_length(self.optimization_settings.line_search_initial_step_size);
        let cost_at_initial_guess = self.cost(0.);
        
        match self.search_line.length() == 0. {
            true => {
                panic!("Search line has no direction (length = 0)");
            },
            _ => {},
        }

        let cost_at_first_step = self.cost(1.);

        let mut step_min_2_cost = cost_at_initial_guess;
        let mut step_min_2_scale = 0.;
        let mut at_least_one_step_taken = false;

        let mut step_min_1_cost = cost_at_first_step;
        let mut step_min_1_scale = 1.;

        let mut step_now_cost: f64 = 0.;
        let mut step_now_scale: f64 = 1.;

        let mut inflection_not_yet_found = true;
        while inflection_not_yet_found {

            step_now_scale = step_now_scale * (1. + golden_ratio_conjugate());
            step_now_cost = self.cost(step_now_scale);

            match step_now_cost > step_min_1_cost {
                false => {
                    at_least_one_step_taken = true;

                    step_min_2_cost = step_min_1_cost;
                    step_min_2_scale = step_min_1_scale;

                    step_min_1_cost = step_now_cost;
                    step_min_1_scale = step_now_scale;
                },
                true => {

                    inflection_not_yet_found = false;
                }
            }
        }
        
        if at_least_one_step_taken {
            let normalization_scale_after_transform = (step_now_scale - step_min_2_scale) / step_min_2_scale;
            self.search_line
                .scale(step_min_2_scale)
                .move_base_to_end_position()
                .scale(normalization_scale_after_transform);
        } else {
            self.search_line
                .scale(step_now_scale);
        }
        
        self.lower_bound_scale = Option::Some(0.);
        self.cost_lower_bound = Option::Some(step_min_2_cost);
        self.upper_bound_scale = Option::Some(1.);
        self.cost_upper_bound = Option::Some(step_now_cost);
        self.inner_point_low = Option::None;
        self.cost_inner_point_low = Option::None;
        self.inner_point_high = Option::None;
        self.cost_inner_point_high = Option::None;

        self
    }
}