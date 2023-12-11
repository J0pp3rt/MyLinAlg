use crate::{*};
#[derive(Clone, Debug)]
pub enum DescentMethodsEnum {
    SteepestDescent,
    ConjugateDescent(ConjugateGradientMethods),
    QuasiNewton(QuasiNewtonMethods)
}
#[derive(Clone, Debug)]
pub enum ConjugateGradientMethods {
    FletcherReeves,
    PolakRibiere
}
#[derive(Clone, Debug)]
pub enum QuasiNewtonMethods {
    BFGS,
    BFGSBeforeInverse,
    BFGSBeforeInverseMoreStable,
    DFP,
    DFPMoreStable
}

#[derive(Clone, Debug)]
pub enum ConstraintMeritFunctions {
    ExteriorQuadratic,
    InteriorLog,
    InteriorInverseBarrier,
}

pub struct PointChoosen {}
pub struct SearchLineFound {}

pub struct DescentMethods<State = PointChoosen, Orientation = IsColl> {
    cost_function:  Rc<Box<dyn Fn(Vec<f64>) -> f64 + 'static >>,
    constraint_functions: Vec<Rc<Box<dyn Fn(Vec<f64>) -> f64 + 'static >>>,
    previous_coordinates: Option<PosNDof<f64>>,
    current_coordinates: PosNDof<f64>,
    previous_gradient: Option<SpatialVectorNDof<f64, IsColl>>,
    current_gradient: SpatialVectorNDof<f64, IsColl>,
    previous_search_line: Option<SpatialVectorWithBasePointNDof<f64, IsColl>>,
    current_search_line: Option<SpatialVectorWithBasePointNDof<f64, IsColl>>,
    previous_hessian_approx: Option<Matrix<f64>>,
    current_hessian_approx: Option<Matrix<f64>>,
    pub used_function_evals: usize,
    differentiation_step_size: f64,
    _current_state: PhantomData<State>,
    _orrientations: PhantomData<Orientation>
}

impl DescentMethods<IsColl> {

    pub fn new(
            cost_function: Rc<Box<dyn Fn(Vec<f64>) -> f64 + 'static >>, 
            constraint_functions: Vec<Rc<Box<dyn Fn(Vec<f64>) -> f64 + 'static >>>, 
            search_point: &PosNDof<f64>,
            differentiation_step_size: f64
        ) -> DescentMethods<PointChoosen> {

        let local_gradient = Self::set_local_gradient_uninit(cost_function.clone(), search_point, differentiation_step_size);
        let n_dof = search_point.n_dof();

        DescentMethods::<PointChoosen, IsColl> {
            cost_function,
            constraint_functions,
            previous_coordinates: Option::None,
            current_coordinates: search_point.clone(),
            previous_gradient: Option::None,
            current_gradient: local_gradient,
            previous_search_line: Option::None,
            current_search_line: Option::None,
            previous_hessian_approx: Option::None,
            current_hessian_approx: Option::Some(Matrix::new_square_eye(n_dof, 1.)),
            used_function_evals: 2*n_dof,
            differentiation_step_size,
            _current_state: PhantomData::<PointChoosen>,
            _orrientations: PhantomData::<IsColl>
        }
    }

    pub fn new_unconstrained(
            cost_function: Rc<Box<dyn Fn(Vec<f64>) -> f64 + 'static >>, 
            search_point: &PosNDof<f64>,
            differentiation_step_size: f64
        ) -> DescentMethods<PointChoosen> {

        let constrained_vec = Vec::<Rc<Box<dyn Fn(Vec<f64>) -> f64 + 'static >>>::new();
        
        Self::new(cost_function, constrained_vec, search_point, differentiation_step_size)
    }

    fn set_local_gradient_uninit(
            cost_function: Rc<Box<dyn Fn(Vec<f64>) -> f64 + 'static >>, 
            search_point: &PosNDof<f64>,
            differentiation_step_size: f64
        ) ->  SpatialVectorNDof<f64, IsColl> {

        let n_degrees_of_freedom = search_point.n_dof();
        let mut derivative_per_dof = Vec::<f64>::with_capacity(n_degrees_of_freedom);
        let search_vector = SpatialVectorWithBasePointNDof::new_with_base(&search_point.clone().x, &vec![0.; n_degrees_of_freedom]);
        let step_size = differentiation_step_size;

        for i_dof in 0.. n_degrees_of_freedom {
            let cost_function_at_small_step_forward = (cost_function)(search_vector.volatile().set_vector_i_value(i_dof, step_size).end_position_vec());
            let cost_function_at_small_step_bacward = (cost_function)(search_vector.volatile().set_vector_i_value(i_dof, -step_size).end_position_vec());
            let cost_difference = cost_function_at_small_step_forward - cost_function_at_small_step_bacward;
            derivative_per_dof.push(cost_difference / (2. * step_size));
        }

        SpatialVectorNDof::new_from_direction(derivative_per_dof.clone())
    }
}

impl<State> DescentMethods<State, IsColl> {

    fn n_dof(&self) -> usize {
        self.current_coordinates.n_dof()
    }

    fn self_to_point_choosen(self) -> DescentMethods<PointChoosen> {
        DescentMethods::<PointChoosen> {
            cost_function: self.cost_function,
            constraint_functions: self.constraint_functions,
            previous_coordinates: self.previous_coordinates,
            current_coordinates: self.current_coordinates,
            previous_gradient: self.previous_gradient,
            current_gradient: self.current_gradient,
            previous_search_line: self.previous_search_line,
            current_search_line: self.current_search_line,
            previous_hessian_approx: self.previous_hessian_approx,
            current_hessian_approx: self.current_hessian_approx,
            used_function_evals: self.used_function_evals,
            differentiation_step_size: self.differentiation_step_size,
            _current_state: PhantomData::<PointChoosen>,
            _orrientations: PhantomData::<IsColl>
        }
    }

    fn self_to_search_line_found(self) -> DescentMethods<SearchLineFound> {

        DescentMethods::<SearchLineFound> {
            cost_function: self.cost_function,
            constraint_functions: self.constraint_functions,
            previous_coordinates: self.previous_coordinates,
            current_coordinates: self.current_coordinates,
            previous_gradient: self.previous_gradient,
            current_gradient: self.current_gradient,
            previous_search_line: self.previous_search_line,
            current_search_line: self.current_search_line,
            previous_hessian_approx: self.previous_hessian_approx,
            current_hessian_approx: self.current_hessian_approx,
            used_function_evals: self.used_function_evals,
            differentiation_step_size: self.differentiation_step_size,
            _current_state: PhantomData::<SearchLineFound>,
            _orrientations: PhantomData::<IsColl>
        }
    }


    pub fn cost_value(&mut self, coordinates: Vec<f64>) -> f64 {

        self.used_function_evals += 1;
        (self.cost_function)(coordinates)
    }

    pub fn cost_value_constraint(&mut self, coordinates: Vec<f64>, ) -> f64 {

        self.used_function_evals += 1;
        (self.cost_function)(coordinates)
    }

    fn itterate_values(&mut self) -> &mut Self {
        // not supposed to be called by user

        self.previous_coordinates = Option::Some(self.current_coordinates.clone());

        self.previous_gradient = Option::Some(self.current_gradient.clone());

        if let Option::Some(_) = self.current_search_line {
            self.previous_search_line = self.current_search_line.clone();
            self.current_search_line = Option::None;
        }
        if let Option::Some(_) = self.current_hessian_approx {
            self.previous_hessian_approx = self.current_hessian_approx.clone();
            self.current_hessian_approx = Option::None;
        }

        self
    }

    pub fn update_search_point_and_local_gradient(mut self, new_search_point: PosNDof<f64>) -> DescentMethods<PointChoosen> {

        self.itterate_values();
        self.current_coordinates = new_search_point;
        self.set_local_gradient();
        self.self_to_point_choosen()
    }

    fn set_local_gradient(&mut self) ->  &mut Self {

        let n_degrees_of_freedom = self.current_coordinates.n_dof();
        let mut derivative_per_dof = Vec::<f64>::with_capacity(n_degrees_of_freedom);
        let search_vector = SpatialVectorWithBasePointNDof::new_with_base(&self.current_coordinates.x, &vec![0.; n_degrees_of_freedom]);
        let step_size = self.differentiation_step_size;

        for i_dof in 0.. n_degrees_of_freedom {
            let cost_function_at_small_step_forward = self.cost_value(search_vector.volatile().set_vector_i_value(i_dof, step_size).end_position_vec());
            let cost_function_at_small_step_bacward = self.cost_value(search_vector.volatile().set_vector_i_value(i_dof, -step_size).end_position_vec());
            let cost_difference = cost_function_at_small_step_forward - cost_function_at_small_step_bacward;
            derivative_per_dof.push(cost_difference / (2. * step_size));
        }

        self.current_gradient = SpatialVectorNDof::new_from_direction(derivative_per_dof);
        self
    }

    pub fn local_gradient(&self) -> SpatialVectorNDof<f64, IsColl> {
        self.current_gradient.clone()
    }
}

impl DescentMethods<PointChoosen, IsColl> {

    pub fn descent_method(mut self, descent_method: DescentMethodsEnum) -> DescentMethods<SearchLineFound> {
        match descent_method {
            DescentMethodsEnum::SteepestDescent => {
                self.use_steepest_descent()
            },
            DescentMethodsEnum::ConjugateDescent(conjugate_method) => {
                self.use_conjugate_descent(conjugate_method)
            },
            DescentMethodsEnum::QuasiNewton(quasi_newton_method) => {
                self.use_quasi_newton_method(quasi_newton_method)
            },
        }
    }


    pub fn use_steepest_descent (mut self) -> DescentMethods<SearchLineFound> {

        let local_gradient = self.current_gradient.clone();
        let search_direction = -1. * local_gradient;
        self.current_search_line = Option::Some(
            SpatialVectorWithBasePointNDof::new_with_base(&self.current_coordinates.x, &search_direction.vector)
        );

        self.self_to_search_line_found()
    }


    pub fn use_conjugate_descent (mut self, conjugate_method: ConjugateGradientMethods) -> DescentMethods<SearchLineFound>  {

        let current_gradient = self.current_gradient.clone();
        let previous_gradient = self.previous_gradient.clone().unwrap();

        let previous_search_line_direction = self.previous_search_line.clone().unwrap().vector();

        let conjugate_gradient_factor: f64;
        let spiraling_condition = current_gradient.transpose()* previous_search_line_direction > 0.2 * current_gradient.transpose() * &current_gradient;
        match spiraling_condition {
            false => {
                match conjugate_method {
                    ConjugateGradientMethods::FletcherReeves => {
                        conjugate_gradient_factor = current_gradient.transpose() * &current_gradient / (previous_gradient.transpose() * &previous_gradient);
                    },
                    ConjugateGradientMethods::PolakRibiere => {
                        conjugate_gradient_factor = (&current_gradient - &previous_gradient).transpose()*&current_gradient / (previous_gradient.transpose()*&previous_gradient)
                    },
                }
            },
            true => {
                println!("Spiraling detected!");
                conjugate_gradient_factor = 0.;
            }
        }

        let previous_decent_step_scaled = conjugate_gradient_factor * self.previous_search_line.clone().unwrap().vector();
        let conjugate_decent_gradient = -1. * &current_gradient + previous_decent_step_scaled;

        self.current_search_line = Option::Some(
            SpatialVectorWithBasePointNDof::new_with_base(&self.current_coordinates.x, &conjugate_decent_gradient.vector)
        );

        self.self_to_search_line_found()
    }

    pub fn use_quasi_newton_method(mut self, quasi_newton_method: QuasiNewtonMethods) -> DescentMethods<SearchLineFound, IsColl> {
        assert!(if let Option::Some(_) = self.previous_hessian_approx {true} else {false}, "No Hessian approximation available from last itteration");

        let n_dof = self.n_dof();

        let coordinates_previous_step = self.previous_coordinates.clone().unwrap().as_vector();
        let coordinates_current_step = self.current_coordinates.as_vector();
        let dx = coordinates_current_step - coordinates_previous_step;

        let gradient_previous_step = self.previous_gradient.clone().unwrap();
        let gradient_current_step = self.current_gradient.clone();
        let y = gradient_current_step.clone() - gradient_previous_step;

        let hessian_approx = self.previous_hessian_approx.clone().unwrap();
        let sigma = 1. / (y.transpose()*&dx);

        let search_line: SpatialVectorWithBasePointNDof<f64, mla::IsColl>;
        match quasi_newton_method {
            QuasiNewtonMethods::BFGS => {
                let i = Matrix::new_square_eye(n_dof, 1.);
        
                let next_hessian = (&i - sigma*&dx*y.transpose()) * hessian_approx * (&i - sigma*&y*dx.transpose()) + sigma*&dx*dx.transpose();
                
                self.current_hessian_approx = Option::Some(next_hessian.clone());
                let search_direction = -1. * next_hessian * gradient_current_step;
                search_line = SpatialVectorWithBasePointNDof::new_with_base(&self.current_coordinates.x, &search_direction.vector);
            },
            QuasiNewtonMethods::BFGSBeforeInverse => {                
                let da = 
                    sigma * &y*y.transpose() 
                    -  &hessian_approx*&dx*dx.transpose()*&hessian_approx / (dx.transpose() * &hessian_approx * &dx);
        
                let next_hessian = hessian_approx + da;
        
                self.current_hessian_approx = Option::Some(next_hessian.clone());
                let next_hessian_inverse = next_hessian.inverse_through_gauss();
                let search_direction = -1. * next_hessian_inverse * gradient_current_step;
                search_line = SpatialVectorWithBasePointNDof::new_with_base(&self.current_coordinates.x, &search_direction.vector);
            },
            QuasiNewtonMethods::BFGSBeforeInverseMoreStable => {
                let da = 
                    (1. + sigma * dx.transpose()*&hessian_approx*&dx)*(sigma * &y*y.transpose())
                    - sigma * &y*dx.transpose()*&hessian_approx 
                    - sigma * &hessian_approx*&dx*y.transpose();
        
                let next_hessian = hessian_approx + da;
        
                self.current_hessian_approx = Option::Some(next_hessian.clone());
                let next_hessian_inverse = next_hessian.inverse_through_gauss();
                let search_direction = -1. * next_hessian_inverse * gradient_current_step;
                search_line = SpatialVectorWithBasePointNDof::new_with_base(&self.current_coordinates.x, &search_direction.vector);
            },
            QuasiNewtonMethods::DFP => {
                let db = 
                    sigma * &dx*dx.transpose()
                    - &hessian_approx*&y*y.transpose()*&hessian_approx / (y.transpose()*&hessian_approx*&y);
        
                let next_hessian = hessian_approx + db;
        
                self.current_hessian_approx = Option::Some(next_hessian.clone());
                let search_direction = -1. * next_hessian * gradient_current_step;
                search_line = SpatialVectorWithBasePointNDof::new_with_base(&self.current_coordinates.x, &search_direction.vector);
            },
            QuasiNewtonMethods::DFPMoreStable => {
                let db = 
                    (1. + sigma * y.transpose()*&hessian_approx*&y) * (sigma * &dx*dx.transpose())
                    - sigma * &dx*y.transpose()*&hessian_approx
                    - sigma * &hessian_approx*&y*dx.transpose();
    
                let next_hessian = hessian_approx + db;
    
                self.current_hessian_approx = Option::Some(next_hessian.clone());
                let search_direction = -1. * next_hessian * gradient_current_step;
                search_line = SpatialVectorWithBasePointNDof::new_with_base(&self.current_coordinates.x, &search_direction.vector);
            },
        }

        self.current_search_line = Option::Some(search_line);
        self.self_to_search_line_found()

    }


}

impl DescentMethods<SearchLineFound, IsColl> {
    pub fn search_line(&self) -> SpatialVectorWithBasePointNDof<f64, IsColl> {
        match self.current_search_line.clone() {
            Option::Some(search_line) => {
                search_line
            },
            Option::None => {
                panic!("You should not have been able to call this function without the search line being set");
            }
        }
    }
}