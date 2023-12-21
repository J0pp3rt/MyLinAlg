
use mla::{*};
use mshio::MshFile;
use core::panic;
use std::error::Error;
use std::fmt::format;
use std::fs;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::io::Lines;
use std::process::Command;
use std::ops::Not;


////
/// 1 read mesh and analyse
/// 2 build up matrices per triangle
/// 3 solve so that u t+1 = H * u t
/// plot

type ExtractedMsh = MshFile<u64, i32, f64>;

fn main() -> Result<(), Box<dyn Error>> {
    // triangle_to_matrix();
    let msh_parsed = parse_mesh_smartboy()?;

    let physical_tags = get_physical_tag_info("main.msh");

    let mesh = extract_mesh_information(msh_parsed, physical_tags);


    simulation(mesh);
    // dbg!(mesh);


    // let mut plot = PlotBuilder::new();

    // plot.add_simple_polygon(PolygonElement::new_with_z(vec![0., 10., 10.], vec![0., 10., 0.], 0.));
    // plot.add_simple_polygon(PolygonElement::new_with_z(vec![0., 10., 0.], vec![0., 10., 10.], 1.));

    // plot.to_plotters_processor().bitmap_to_file("test_triangle.png")

    Ok(())
}

fn get_physical_tag_info(path: &str) -> Vec<PhysicalTagPrimitive> {
    let mut mesh = Mesh::new();

    let mesh_file = File::open(path).unwrap();
    let mut mesh_file_lines = io::BufReader::new(mesh_file).lines();
    
    if let Option::Some(first_line) = mesh_file_lines.next() {
        if (first_line.unwrap() == "$MeshFormat").not() {
            panic!("Not a proper mesh file!")
        }
    }
    
    let version_line = mesh_file_lines.next().expect("EOF too early!").expect("EOF too early!");
    
    let mut byte_itterator = version_line.split_ascii_whitespace();

    let version_byte = byte_itterator.next().expect("Error in mesh version line!");
    if (version_byte == "4.1").not() {
        println!("Warning! parser was made for mesh format 4.1!");
    } 
    let file_type = byte_itterator.next().expect("Error in mesh version line!");
    if (file_type == "0").not() {
        panic!("Parser not made for binary format");
    }
    let data_size = byte_itterator.next().expect("Error in mesh version line!").parse::<usize>().unwrap();

    scan_mesh_file_for("$EndMeshFormat", &mut mesh_file_lines);
 
    ////////////////////////////////////////////////////////////////////
    // end mesh format
    ////////////////////////////////////////////////////////////////////

    //////////////////////////////////////////////////////////////////
    // begin physical names
    /////////////////////////////////////////////////////////////////

    scan_mesh_file_for("$PhysicalNames", &mut mesh_file_lines);

    let number_of_physical_tags = mesh_file_lines.next().expect("EOF too early!").expect("EOF too early!").parse::<usize>().unwrap();

    let mut physical_tags_primitive = Vec::<PhysicalTagPrimitive>::with_capacity(number_of_physical_tags);
    for _ in 0..number_of_physical_tags {
        let line = mesh_file_lines.next().expect("EOF too early!").expect("EOF too early!");
        let mut byte_itterator = line.split_ascii_whitespace();

        let dimension_size = byte_itterator.next().expect("EOF too early!").parse::<usize>().unwrap();
        let unique_number = byte_itterator.next().expect("EOF too early!").parse::<usize>().unwrap();
        let name = byte_itterator.next().expect("EOF too early!");

        let primitive_tag = PhysicalTagPrimitive {
            dimension_size,
            unique_number,
            name: name.to_string().replace("\"", "")
        };
        physical_tags_primitive.push(primitive_tag);
    }

    let mut ordered_list = Vec::<PhysicalTagPrimitive>::with_capacity(number_of_physical_tags);
    for ordered_index in 0..number_of_physical_tags {
        let mut no_tag_added = true;
        for tag_index in 0..number_of_physical_tags {
            if physical_tags_primitive[tag_index].unique_number == ordered_index +1{
                ordered_list.push(physical_tags_primitive[tag_index].clone());
                no_tag_added = false;
                break
            }
        }
        if no_tag_added {
            panic!("Missing tag numbers!")
        }
    }

    /////
    
    ordered_list
}

struct MeshPoint {
    pos: Pos2<f64>,
    physical_tag: PhysicalTag,
}

struct MeshCurve {
    physical_tag: PhysicalTag,
}

struct MeshSurface {
    physical_tag: PhysicalTag,
}

fn get_tag_type(primitive_tag : Vec<PhysicalTagPrimitive>, tag_id: usize) -> PhysicalTag {
    let name = primitive_tag[tag_id-1].name.as_str();
    match name {
        "interior" => PhysicalTag::Interior,
        "isolated" => PhysicalTag::Boundry(BoundryType::Neumann),
        "heated" => PhysicalTag::Boundry(BoundryType::Dirchlet),
        _ => panic!("non standart tag used!")
    }
}

fn extract_mesh_information(parsed_msh: ExtractedMsh, physical_tags: Vec<PhysicalTagPrimitive>) -> Mesh {
    let entities = parsed_msh.data.entities.unwrap().clone();
    let number_of_points = entities.points.len();
    let number_of_curves = entities.curves.len();
    let number_of_surfaces = entities.surfaces.len();


    let mut ordered_points = Vec::<MeshPoint>::with_capacity(number_of_points);
    for index_ordered in 0..number_of_points {
        let mut added_no_entity = true;
        for index_points in 0..number_of_points {
            if entities.points[index_points].tag -1 == index_ordered as i32 {
                added_no_entity = false;
                let point = entities.points[index_points].clone();
                let mesh_point = MeshPoint {
                    pos: Pos2 {
                        x: point.x,
                        y: point.y
                    },
                    physical_tag: get_tag_type(physical_tags.clone(), point.physical_tags[0] as usize)
                };
                ordered_points.push(mesh_point);
            }
        }
        if added_no_entity {
            panic!("Points do not have proper numbering!")
        }
    }

    let mut ordered_curves = Vec::<MeshCurve>::with_capacity(number_of_curves);
    for index_ordered in 0..number_of_curves {
        let mut added_no_entity = true;
        for index_curves in 0..number_of_curves {
            if entities.curves[index_curves].tag -1 == index_ordered as i32 {
                added_no_entity = false;
                let curve = entities.curves[index_curves].clone();
                let mesh_curve = MeshCurve {
                    physical_tag: get_tag_type(physical_tags.clone(), curve.physical_tags[0] as usize)
                };
                ordered_curves.push(mesh_curve);
            }
        }
        if added_no_entity {
            panic!("Curves do not have proper numbering!")
        }
    }

    let mut ordered_surfaces = Vec::<MeshSurface>::with_capacity(number_of_surfaces);
    for index_ordered in 0..number_of_surfaces {
        let mut added_no_entity = true;
        for index_surface in 0..number_of_surfaces {
            if entities.surfaces[index_surface].tag -1 == index_ordered as i32 {
                added_no_entity = false;
                let surface = entities.curves[index_surface].clone();
                let mesh_surface = MeshSurface {
                    physical_tag: get_tag_type(physical_tags.clone(), surface.physical_tags[0] as usize)
                };
                ordered_surfaces.push(mesh_surface);
            }
        }
        if added_no_entity {
            panic!("Surfaces do not have proper numbering!")
        }
    }

    let nodes_parsed = parsed_msh.data.nodes.expect("No nodes detected!");
    let number_of_nodes = nodes_parsed.num_nodes;
    let mut nodes_vec = Vec::<Node>::with_capacity(number_of_nodes as usize);
    if (nodes_parsed.num_nodes == nodes_parsed.max_node_tag).not() && (nodes_parsed.min_node_tag == 1).not() {
        panic!("Nodes are not properly (simple: 1,2,4...,N) ordered!");
    }

    for node_block in nodes_parsed.node_blocks {
        let node_block_dimension = node_block.entity_dim as usize;
        let entity_tag = node_block.entity_tag as usize;
        let physical_tag: PhysicalTag;
        match node_block_dimension {
            0 => {
                physical_tag = ordered_points[entity_tag-1].physical_tag;
            },
            1 => {
                physical_tag = ordered_curves[entity_tag-1].physical_tag;
            },
            2 => {
                physical_tag = ordered_curves[entity_tag-1].physical_tag;
            },
            3 => {
                todo!("Volumes not yet supported! (this is Mesh extraction script)")
            },
            _ => {
                panic!("Multiverse elements not supported at the moment!")
            }
        }
        for node in node_block.nodes {
            nodes_vec.push(
                Node { 
                    pos: Pos2 { 
                        x: node.x, 
                        y: node.y 
                    }, 
                    physical_tag: physical_tag 
                }
            )
        }
    }

    let elements_parsed = parsed_msh.data.elements.expect("No elements detected!");
    let number_of_elements = elements_parsed.num_elements;
    let mut triangles_vec = Vec::<Triangle>::with_capacity(number_of_elements as usize);
    if (elements_parsed.num_elements == elements_parsed.max_element_tag).not() && (elements_parsed.min_element_tag == 1).not() {
        panic!("Elements are not properly (simple: 1,2,4...,N) ordered!");
    }
    let mut node_is_used = vec![false; number_of_nodes as usize];
    for element_block in elements_parsed.element_blocks {
        type ParserElementTypes = mshio::mshfile::ElementType;

        //stateguard to only get the traingles
        match element_block.element_type {
            ParserElementTypes::Tri3 => {},
            _ => {continue;}
        }

        for triangle in element_block.elements {
            for node in &triangle.nodes {
                node_is_used[*node as usize-1] = true;
            }
            let new_triangle = Triangle {
                nodes: [
                    triangle.nodes[0] as usize -1,
                    triangle.nodes[1] as usize -1,
                    triangle.nodes[2] as usize -1,
                ],
            };
            triangles_vec.push(new_triangle);
        }

    }

    let number_of_used_nodes: usize = node_is_used.iter().filter(|is_used| **is_used).map(|_| {1}).sum();
    let used_nodes: Vec<Node> = node_is_used.iter().zip(nodes_vec.iter()).filter(|(is_used, _)| **is_used).map(|(_, node)| *node).collect();
    
    let mut collective_reducer: Vec<usize> = vec![0; number_of_nodes as usize];
    let mut current_reducer = 0;
    for (index, is_used) in (0..number_of_nodes as usize).zip(node_is_used.iter()) {
        if is_used.not() {
            current_reducer += 1;
        }
        collective_reducer[index] = current_reducer;
    }

    for traingle in triangles_vec.iter_mut() {
        for node in traingle.nodes.iter_mut() {
            *node = *node - collective_reducer[*node];

        }
    }

    Mesh { 
        nodes: used_nodes, 
        triangles: triangles_vec 
    }
}

fn simulation(mesh: Mesh) {
    let mut m_matrix = Matrix::new_square_with_constant_values( mesh.nodes.len(), 0.);
    let mut r_matrix = Matrix::new_square_with_constant_values(mesh.nodes.len(), 0.);

    for triangle in &mesh.triangles {

        let node_0 = triangle.nodes[0];
        let node_1 = triangle.nodes[1];
        let node_2 = triangle.nodes[2];

        let node_matrix_triangle = Matrix::new_from_vector_rows(vec![
            vec![mesh.nodes[node_0].pos.x, mesh.nodes[node_0].pos.y, 1.], 
            vec![mesh.nodes[node_1].pos.x, mesh.nodes[node_1].pos.y, 1.],
            vec![mesh.nodes[node_2].pos.x, mesh.nodes[node_2].pos.y, 1.]
        ]);

        let basis_matrix: Matrix<f64>;
        let basis_derivatives_matrix: Matrix<f64>;
        (basis_matrix, basis_derivatives_matrix) = find_basis_function_matrix(node_matrix_triangle);
        // basis_matrix.printer();
        // basis_derivatives_matrix.printer();
        // println!("~~~~~~~~~~~~~~~~:");
        // println!("~~~~~~~~~~~~~~~~:");
        // for i in 0..=2 {
        //     println!("sum row {}", basis_derivatives_matrix[i].cells.iter().map(|x| x).sum::<f64>())
        // }
        // println!("~~~~~~~~~~~~~~~~:");
        // println!("~~~~~~~~~~~~~~~~:");
        for j in 0..=2 {
            for k in 0..=j {
                m_matrix[triangle.nodes[j]][triangle.nodes[k]] += (basis_matrix[j][k]).abs();
                r_matrix[triangle.nodes[j]][triangle.nodes[k]] += basis_derivatives_matrix[j][k];

                if (j==k).not() {
                    m_matrix[triangle.nodes[k]][triangle.nodes[j]] += (basis_matrix[j][k]).abs();
                    r_matrix[triangle.nodes[k]][triangle.nodes[j]] += basis_derivatives_matrix[j][k];
                }
            }
        }
        // for i in 0..r_matrix.len() {
            // println!("sum row M {}", m_matrix[i].cells.iter().map(|x| x).sum::<f64>())
        // }
        // println!("m matrix:");
        // m_matrix.printer();
        // println!("r matrix:");
        // r_matrix.printer();
        // println!("over:");
    }

    // m_matrix[2][0] = m_matrix[2][0]*-1.; /////////////////////////////////////////////////////////////////////////////////////////////

    // println!("m matrix:");
    // m_matrix.printer();
    // println!("r matrix:");
    // r_matrix.printer();

    let alpha = 1.11*(10f64).powi(-4);

    r_matrix = -alpha* r_matrix;

    // for i in 0..r_matrix.len() {
    //     println!("sum row {}", r_matrix[i].cells.iter().map(|x| x).sum::<f64>())
    // }

    let liner_solver = Solver2D {
        A_matrix: m_matrix,
        B_matrix: r_matrix,
        solver: Solver2DStrategy::Guass,
    };

    let linear_solved = Matrix::new_from_solver(liner_solver);

    // linear_solved.A_matrix.printer();
    // println!("q matrix:");
    // linear_solved.B_matrix.printer();
    // for row in linear_solved.B_matrix.rows.iter() {
    //     println!("{:?}", row.cells)
    // }
    // for i in 0..linear_solved.B_matrix.len() {
    //     println!("sum row {}", linear_solved.B_matrix[i].cells.iter().map(|x| x).sum::<f64>())
    // }
    
    let q_matrix = linear_solved.B_matrix; 

    let dt = 0.001;
    let time_steps = 100000;
    let i = Matrix::new_square_eye(q_matrix.len(), 1.);
    println!("forming time integration matrix:");
    let p_matrix = 
        &i
        + dt * &q_matrix 
        + 1./2. * dt.powi(2) * &q_matrix*&q_matrix 
        + 1./6. * dt.powi(3) * &q_matrix*&q_matrix*&q_matrix 
        + 1./24. *dt.powi(4) * &q_matrix*&q_matrix*&q_matrix*&q_matrix;
    // for row in p_matrix.rows.iter() {
    //     println!("{:?}", row.cells)
    // }
    
    // let time_steps = 5000;

    let mut temperatures_over_time = Matrix::new_with_constant_values(q_matrix.len(), time_steps, 5.);
    // let mut current_temperatures = Collumn::new_form_vec(vec![100., 0., 0., 0., 0., 0., 0.]);
    let mut current_temperatures = Collumn::new_form_vec(vec![5.;q_matrix.len()]);
    // current_temperatures[0] = 50.;
    // current_temperatures[1] = -20.;
    // temperatures_over_time[0][0] = current_temperatures[0];
    // temperatures_over_time[1][0] = current_temperatures[1];

    let dirchlet_temperature = 100.;

    for (index,node) in mesh.nodes.iter().enumerate() {
        match node.physical_tag {
            PhysicalTag::Interior => {},
            PhysicalTag::Boundry(boundry_type) => {
                match boundry_type {
                    BoundryType::Dirchlet => {
                        current_temperatures[index] = dirchlet_temperature;
                        temperatures_over_time[index][0] = dirchlet_temperature;
                    },
                    BoundryType::Neumann => {},
                }
            },
        }
    }

    // current_temperatures[0] = 5.;
    // temperatures_over_time[0][0] = current_temperatures[0];

    // temperatures_over_time[1][0] = current_temperatures[1];
    // temperatures_over_time[2][0] = 0.;
    // temperatures_over_time[3][0] = 0.;
    // temperatures_over_time[4][0] = 0.;
    // temperatures_over_time[5][0] = 0.;
    // temperatures_over_time[6][0] = 0.;
    println!("starting time integration:");
    let mut plot_counter = 0;
    for t_i in 1..time_steps {

        let mut new_temperatures = &p_matrix * &current_temperatures;
        // let dT = dt*&q_matrix * &current_temperatures;
        // let mut new_temperatures = dT + current_temperatures;
        // new_temperatures[0] = 50.;
        // new_temperatures[1] = -20.;
        // new_temperatures[1] = 5.;
        // new_temperatures[2] = 5.;
        // new_temperatures[3] = 5.;
        // new_temperatures[4] = 5.;
        // new_temperatures[6] = 5.;
        for (index,node) in mesh.nodes.iter().enumerate() {
            match node.physical_tag {
                PhysicalTag::Interior => {},
                PhysicalTag::Boundry(boundry_type) => {
                    match boundry_type {
                        BoundryType::Dirchlet => {
                            new_temperatures[index] = dirchlet_temperature;
                            temperatures_over_time[index][t_i] = dirchlet_temperature;
                        },
                        BoundryType::Neumann => {},
                    }
                },
            }
        }
        // new_temperatures[1] = 100. -100.*(t_i  as f64/ time_steps as f64);
        // new_temperatures[0] = 10. ;
        for node_i in 0..q_matrix.len() {
            
            temperatures_over_time[node_i][t_i] = new_temperatures[node_i];
        }
        current_temperatures = new_temperatures;

        if t_i % 100 == 0 {
            plot_counter += 1;
            mesh_plotter(&mesh, current_temperatures.to_vec(), plot_counter, t_i as f64 * dt)
        }
    }

    // let mut plot = PlotBuilder::new();
    // for node_i in 0..q_matrix.len() {
    //     // if (node_i == 1).not() {
    //         plot.add_simple_2d_line_y_only(&temperatures_over_time[node_i].cells);
    //     // }
        
    // }

    // plot.scale_plot(2.);
    // plot.plotting_settings.legend_show = false;
    // plot.plotting_settings.show_grid_major = true;
    // plot.set_title("first mesh test");
    // plot.set_x_label("Itteratie");
    // plot.set_y_label("Temperatuur");
    // plot.plotting_settings.legend_location = LegendPosition::East;
    // plot.to_plotters_processor().bitmap_to_file("FEM_test_meshed.png");

    println!("making video");
    Command::new("sh").arg("-c").arg("ffmpeg -y -framerate 10 -i animation/%04d.png video.mp4").output().expect("video failed");
}

fn mesh_plotter(mesh: &Mesh, temperatures: Vec<f64>, time_index: usize, time: f64) {
    let mut plot_triangles = PlotBuilder::new();
    plot_triangles.scale_plot(4.);

    let mut triangles_plot = PolygonSet::new();

    for triangle in &mesh.triangles {
        let trianle_node_0 = triangle.nodes[0];
        let trianle_node_1 = triangle.nodes[1];
        let trianle_node_2 = triangle.nodes[2];

        let t_1 = temperatures[trianle_node_0];
        let t_2 = temperatures[trianle_node_1];
        let t_3 = temperatures[trianle_node_2];

        let x_0 = &mesh.nodes[trianle_node_0];
        let x_1 = &mesh.nodes[trianle_node_1];
        let x_2 = &mesh.nodes[trianle_node_2];

        let average_temperature = (t_1 + t_2 + t_3) / 3.;
        let triangle_plot = PolygonElement::new_with_z(vec![x_0.pos.x, x_1.pos.x, x_2.pos.x], vec![x_0.pos.y, x_1.pos.y, x_2.pos.y], average_temperature);
        triangles_plot.add_polygon(triangle_plot);
    }

    let title = "t = ".to_owned() + &format!("{:?}",&time.to_string()) + " s";
    let number_width = 4;
    let path_number = format!("{:0number_width$}", time_index, number_width = number_width);
    let path = "animation/".to_owned() + &path_number + ".png";
    plot_triangles.plotting_settings.color_map_restricter_lower_bound = 0.;
    plot_triangles.plotting_settings.color_map_restricter_upper_bound = 1.;
    plot_triangles.set_z_range(0. .. 100.);
    plot_triangles.add_polygon_set(triangles_plot).set_title(&title);
    plot_triangles.to_plotters_processor().bitmap_to_file(&path)
}

fn triangle_to_matrix() {



    let mut nodes = Vec::<Pos2<f64>>::new();
    nodes.push(Pos2::new(0., 0.));
    nodes.push(Pos2::new(0.31, 0.));

    nodes.push(Pos2::new(0., 0.2));
    nodes.push(Pos2::new(0.15, 0.2));
    nodes.push(Pos2::new(0.31, 0.2));

    nodes.push(Pos2::new(0., 0.4));
    nodes.push(Pos2::new(0.3, 0.4));

    let mut triangles = Vec::<Vec<usize>>::new();
    // triangles.push(vec![1, 2, 0]);
    
    triangles.push(vec![0, 1, 3]);

    triangles.push(vec![0, 2, 3]);
    triangles.push(vec![1, 3, 4]);

    triangles.push(vec![2, 3, 5]);
    triangles.push(vec![3, 5, 6]);
    triangles.push(vec![3, 4, 6]);

    let mut m_matrix = Matrix::new_square_with_constant_values( nodes.len(), 0.);
    let mut r_matrix = Matrix::new_square_with_constant_values(nodes.len(), 0.);

    for triangle in &triangles {

        let node_matrix_triangle = Matrix::new_from_vector_rows(vec![
            vec![nodes[triangle[0]].x, nodes[triangle[0]].y, 1.], 
            vec![nodes[triangle[1]].x, nodes[triangle[1]].y, 1.],
            vec![nodes[triangle[2]].x, nodes[triangle[2]].y, 1.]
        ]);

        let basis_matrix: Matrix<f64>;
        let basis_derivatives_matrix: Matrix<f64>;
        (basis_matrix, basis_derivatives_matrix) = find_basis_function_matrix(node_matrix_triangle);
        basis_matrix.printer();
        basis_derivatives_matrix.printer();
        println!("~~~~~~~~~~~~~~~~:");
        println!("~~~~~~~~~~~~~~~~:");
        for i in 0..=2 {
            println!("sum row {}", basis_derivatives_matrix[i].cells.iter().map(|x| x).sum::<f64>())
        }
        println!("~~~~~~~~~~~~~~~~:");
        println!("~~~~~~~~~~~~~~~~:");
        for j in 0..=2 {
            for k in 0..=j {
                m_matrix[triangle[j]][triangle[k]] += (basis_matrix[j][k]).abs();
                r_matrix[triangle[j]][triangle[k]] += basis_derivatives_matrix[j][k];

                if (j==k).not() {
                    m_matrix[triangle[k]][triangle[j]] += (basis_matrix[j][k]).abs();
                    r_matrix[triangle[k]][triangle[j]] += basis_derivatives_matrix[j][k];
                }
            }
        }
        for i in 0..r_matrix.len() {
            println!("sum row M {}", m_matrix[i].cells.iter().map(|x| x).sum::<f64>())
        }
        println!("m matrix:");
        m_matrix.printer();
        println!("r matrix:");
        r_matrix.printer();
        println!("over:");
    }

    // m_matrix[2][0] = m_matrix[2][0]*-1.; /////////////////////////////////////////////////////////////////////////////////////////////

    println!("m matrix:");
    m_matrix.printer();
    println!("r matrix:");
    r_matrix.printer();

    let alpha = 1.172*(10f64).powi(-5);

    r_matrix = -alpha* r_matrix;

    for i in 0..r_matrix.len() {
        println!("sum row {}", r_matrix[i].cells.iter().map(|x| x).sum::<f64>())
    }

    let liner_solver = Solver2D {
        A_matrix: m_matrix,
        B_matrix: r_matrix,
        solver: Solver2DStrategy::Guass,
    };

    let linear_solved = Matrix::new_from_solver(liner_solver);

    linear_solved.A_matrix.printer();
    println!("q matrix:");
    // linear_solved.B_matrix.printer();
    for row in linear_solved.B_matrix.rows.iter() {
        println!("{:?}", row.cells)
    }
    for i in 0..linear_solved.B_matrix.len() {
        println!("sum row {}", linear_solved.B_matrix[i].cells.iter().map(|x| x).sum::<f64>())
    }
    
    let q_matrix = linear_solved.B_matrix; 

    let dt = 0.005;
    let i = Matrix::new_square_eye(q_matrix.len(), 1.);
    println!("p matrix:");
    let p_matrix = 
        &i
        + dt * &q_matrix 
        + 1./2. * dt.powi(2) * &q_matrix*&q_matrix 
        + 1./6. * dt.powi(3) * &q_matrix*&q_matrix*&q_matrix 
        + 1./24. *dt.powi(4) * &q_matrix*&q_matrix*&q_matrix*&q_matrix;
    for row in p_matrix.rows.iter() {
        println!("{:?}", row.cells)
    }
    let time_steps = 20000;
    // let time_steps = 5000;

    let mut temperatures_over_time = Matrix::new_with_constant_values(q_matrix.len(), time_steps, 5.);
    // let mut current_temperatures = Collumn::new_form_vec(vec![100., 0., 0., 0., 0., 0., 0.]);
    let mut current_temperatures = Collumn::new_form_vec(vec![5.;q_matrix.len()]);
    // current_temperatures[0] = 50.;
    // current_temperatures[1] = -20.;
    // temperatures_over_time[0][0] = current_temperatures[0];
    // temperatures_over_time[1][0] = current_temperatures[1];

    current_temperatures[0] = 5.;
    temperatures_over_time[0][0] = current_temperatures[0];

    // temperatures_over_time[1][0] = current_temperatures[1];
    // temperatures_over_time[2][0] = 0.;
    // temperatures_over_time[3][0] = 0.;
    // temperatures_over_time[4][0] = 0.;
    // temperatures_over_time[5][0] = 0.;
    // temperatures_over_time[6][0] = 0.;

    for t_i in 1..time_steps {

        let mut new_temperatures = &p_matrix * &current_temperatures;
        // let dT = dt*&q_matrix * &current_temperatures;
        // let mut new_temperatures = dT + current_temperatures;
        // new_temperatures[0] = 50.;
        // new_temperatures[1] = -20.;
        // new_temperatures[1] = 5.;
        // new_temperatures[2] = 5.;
        // new_temperatures[3] = 5.;
        // new_temperatures[4] = 5.;
        // new_temperatures[6] = 5.;

        new_temperatures[0] = 5. + 5.*(t_i  as f64/ time_steps as f64);
        // new_temperatures[1] = 100. -100.*(t_i  as f64/ time_steps as f64);
        // new_temperatures[0] = 10. ;
        for node_i in 0..q_matrix.len() {
            
            temperatures_over_time[node_i][t_i] = new_temperatures[node_i];
        }
        current_temperatures = new_temperatures;
    }

    let mut plot = PlotBuilder::new();
    for node_i in 0..q_matrix.len() {
        // if (node_i == 1).not() {
            plot.add_simple_2d_line_y_only(&temperatures_over_time[node_i].cells);
        // }
        
    }

    plot.scale_plot(2.);
    plot.plotting_settings.legend_show = true;
    plot.plotting_settings.show_grid_major = true;
    plot.set_title("Verkeerde invloed");
    plot.set_x_label("Itteratie");
    plot.set_y_label("Temperatuur");
    plot.plotting_settings.legend_location = LegendPosition::East;
    plot.to_plotters_processor().bitmap_to_file("FEM_test.png");

    let mut plot_triangles = PlotBuilder::new();
    plot_triangles.scale_plot(4.);

    let mut triangles_plot = PolygonSet::new();

    let time_index = temperatures_over_time.len() -1;
    for triangle in &triangles {
        let trianle_node_0 = triangle[0];
        let trianle_node_1 = triangle[1];
        let trianle_node_2 = triangle[2];

        let t_1 = temperatures_over_time[trianle_node_0][time_index];
        let t_2 = temperatures_over_time[trianle_node_1][time_index];
        let t_3 = temperatures_over_time[trianle_node_2][time_index];

        let x_0 = &nodes[trianle_node_0];
        let x_1 = &nodes[trianle_node_1];
        let x_2 = &nodes[trianle_node_2];

        let average_temperature = (t_1 + t_2 + t_3) / 3.;
        let triangle_plot = PolygonElement::new_with_z(vec![x_0.x, x_1.x, x_2.x], vec![x_0.y, x_1.y, x_2.y], average_temperature);
        triangles_plot.add_polygon(triangle_plot);
    }

    plot_triangles.add_polygon_set(triangles_plot);
    plot_triangles.to_plotters_processor().bitmap_to_file("grid_plot.png")
    
    
}

#[derive(Clone, Copy)]
enum PhysicalTag {
    Interior,
    Boundry(BoundryType)
}

#[derive(Clone)]
struct PhysicalTagPrimitive {
    dimension_size: usize,
    unique_number: usize,
    name: String,
}

#[derive(Clone, Copy)]
enum BoundryType {
    Dirchlet,
    Neumann
}

#[derive(Clone, Copy)]
struct Node {
    pos: Pos2<f64>,
    physical_tag: PhysicalTag
}

struct Triangle {
    nodes: [usize; 3],
}

struct Mesh {
    nodes: Vec<Node>,
    triangles: Vec<Triangle>,
}

impl Mesh {
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
            triangles: Vec::new()
        }
    }
}


fn parse_mesh_smartboy() -> Result<MshFile<u64, i32, f64>, Box<dyn Error>> {
    let msh_bytes = fs::read("main.msh")?;
    let mesh = mshio::parse_msh_bytes(msh_bytes.as_slice()).unwrap();

    Ok(mesh)
}



fn scan_mesh_file_for(header: &str, line_itterator: &mut Lines<BufReader<File>>) {
    let error_message = "Error: did not find: ".to_owned() + header + " in mesh file!";
    while (header == line_itterator.next().expect(&error_message).expect(&error_message)).not() {}
}








fn find_basis_function_matrix(mut node_matrix: Matrix<f64>) -> (Matrix<f64>, Matrix<f64>) {

    let solution_matrix = Matrix::new_from_vector_is_collumn(vec![1., 0., 0.]);

    let mut linear_solver = Solver2D {
        A_matrix: node_matrix.clone(),
        B_matrix: solution_matrix,
        solver: Solver2DStrategy::Guass,
    };

    let coefficients_1 = Matrix::new_from_solver(linear_solver.clone()).B_matrix.get_collumn(0).to_vec();
    linear_solver.B_matrix[0][0] =  0.;
    linear_solver.B_matrix[1][0] = 1.;
    let coefficients_2 = Matrix::new_from_solver(linear_solver.clone()).B_matrix.get_collumn(0).to_vec();
    linear_solver.B_matrix[1][0] =  0.;
    linear_solver.B_matrix[2][0] = 1.;
    let coefficients_3 = Matrix::new_from_solver(linear_solver.clone()).B_matrix.get_collumn(0).to_vec();

    let coefficients = Matrix::new_from_vector_rows(vec![coefficients_1, coefficients_2, coefficients_3]);

    let mut basis_matrix_derivatives = Matrix::new_square_with_constant_values(3, 0.);

    // let x_0 = node_matrix[0][0];
    // let y_0 = node_matrix[0][1];

    // let x_1 = node_matrix[1][0];
    // let y_1 = node_matrix[1][1];

    // let x_2 = node_matrix[2][0];
    // let y_2 = node_matrix[2][1];

    // let determinant = x_0 * (y_1 - y_2) + x_1 * (y_2 - y_0) + x_2 * (y_0 - y_1);
    // let area = 0.5*determinant.abs();

    // let dl0dx = (y_1 - y_2)/determinant;
    // let dl1dx = (y_2 - y_0)/determinant;
    // let dl2dx = (y_0 - y_1)/determinant;

    // let dl0dy = (x_2 - x_1)/determinant;
    // let dl1dy = (x_0 - x_2)/determinant;
    // let dl2dy = (x_1 - y_0)/determinant;

    // basis_matrix_derivatives[0][0] = (dl0dx*dl0dx + dl0dy*dl0dy)*area;
    // basis_matrix_derivatives[1][0] = (dl1dx*dl0dx + dl1dy*dl0dy)*area;
    // basis_matrix_derivatives[2][0] = (dl2dx*dl0dx + dl2dy*dl0dy)*area;

    // basis_matrix_derivatives[1][1] = (dl1dx*dl1dx + dl1dy*dl1dy)*area;
    // basis_matrix_derivatives[2][1] = (dl2dx*dl1dx + dl2dy*dl1dy)*area;

    // basis_matrix_derivatives[2][2] = (dl2dx*dl2dx + dl2dy*dl2dy)*area;

    // for j in 0..=2 {
    //     for k in 0..j {
    //         basis_matrix_derivatives[k][j] = basis_matrix_derivatives[j][k]
    //     }
    // }


    ////
    /// 
    /// 
    ///     
        let angle_vertex_1 = (node_matrix[1][1] - node_matrix[0][1]).atan2(node_matrix[1][0] - node_matrix[0][0]);
        let angle_vertex_2 = (node_matrix[2][1] - node_matrix[0][1]).atan2(node_matrix[2][0] - node_matrix[0][0]);
        let angle_inside = angle_vertex_2-angle_vertex_1;
        let height_triangle = ((node_matrix[2][0]-node_matrix[0][0]).powi(2) + (node_matrix[2][1]-node_matrix[0][1]).powi(2)).sqrt() * angle_inside.sin();
        let base_triangle = ((node_matrix[1][0]-node_matrix[0][0]).powi(2) + (node_matrix[1][1]-node_matrix[0][1]).powi(2)).sqrt();
        let area = (1./2. * height_triangle * base_triangle).abs();
        for j in 0..=2 {

            let alpha_j = coefficients[j][0];
            let beta_j = coefficients[j][1];
            let delta_j = coefficients[j][2];

            for k in 0..=j {

                let alpha_k = coefficients[k][0];
                let beta_k = coefficients[k][1];
                let delta_k = coefficients[k][2];

                basis_matrix_derivatives[j][k] = 
                    (alpha_j*alpha_k + beta_j*beta_k)*area;

                if (j==k).not() {
                    basis_matrix_derivatives[k][j] = basis_matrix_derivatives[j][k];
                }
            }
        }
    ///
    /// 
    /// 
    /// 
    /// 
    let x_0 = node_matrix[0][0];
    let y_0 = node_matrix[0][1];

    //translating node_0 to center (shouldn't matter for the result)
    for i in 0..=2 {
        node_matrix[i][0] += -x_0;
        node_matrix[i][1] += -y_0;
    }

    let transformation_angle = node_matrix[1][1].atan2(node_matrix[1][0]);

    // println!("rotated {}", -node_matrix[2][0]*transformation_angle.sin() + node_matrix[2][1]*transformation_angle.cos());
    // println!("-node_matrix[2][0].sin() = {}", -node_matrix[2][0]*transformation_angle.sin());
    // println!("node_matrix[2][1].cos() = {}", node_matrix[2][1]*transformation_angle.cos());
    // let proper_node_chosen = -node_matrix[2][0]*transformation_angle.sin() + node_matrix[2][1]*transformation_angle.cos() > 0.;
    // println!("rotated {}", -node_matrix[2][0].sin() + node_matrix[2][1].cos() > 0.);

    // if proper_node_chosen.not() {
    //     println!("triangle is upside down!");
    //     let node_1_copy = node_matrix[1].clone();
    //     node_matrix[1] = node_matrix[2].clone();
    //     node_matrix[2] = node_1_copy;
    //     transformation_angle = node_matrix[1][1].atan2(node_matrix[1][0]);
    // }
    
    
    let mut rotation_matrix = Matrix::new_square_with_constant_values(6, 0.);
    for i in 0..=1 {
        rotation_matrix[i][i] = 1.;
    }
    for i in 2..=5 {
        rotation_matrix[i][i] = transformation_angle.cos();
    }
    rotation_matrix[3][2] = -transformation_angle.sin();
    rotation_matrix[5][4] = -transformation_angle.sin();
    rotation_matrix[2][3] = transformation_angle.sin();
    rotation_matrix[4][5] = transformation_angle.sin();

    let mut coordinates = Vec::new();
    for i in 0..=2 {
        coordinates.push(node_matrix[i][0]);
        coordinates.push(node_matrix[i][1]);
    }

    let coordinates = Collumn::new_form_vec(coordinates);

    let rotated_values = &rotation_matrix * coordinates;
    for i in 0..=2 {
        node_matrix[i][0] = rotated_values[i*2];
        node_matrix[i][1] = rotated_values[i*2+1];
    }
    
    // if proper_node_chosen.not() {
    //     node_matrix[2][1] = node_matrix[2][1]*-1.;
    // }

    let solution_matrix = Matrix::new_from_vector_is_collumn(vec![1., 0., 0.]);

    let mut linear_solver = Solver2D {
        A_matrix: node_matrix.clone(),
        B_matrix: solution_matrix,
        solver: Solver2DStrategy::Guass,
    };

    let coefficients_1 = Matrix::new_from_solver(linear_solver.clone()).B_matrix.get_collumn(0).to_vec();
    linear_solver.B_matrix[0][0] =  0.;
    linear_solver.B_matrix[1][0] = 1.;
    let coefficients_2 = Matrix::new_from_solver(linear_solver.clone()).B_matrix.get_collumn(0).to_vec();
    linear_solver.B_matrix[1][0] =  0.;
    linear_solver.B_matrix[2][0] = 1.;
    let coefficients_3 = Matrix::new_from_solver(linear_solver.clone()).B_matrix.get_collumn(0).to_vec();

    let coefficients = Matrix::new_from_vector_rows(vec![coefficients_1, coefficients_2, coefficients_3]);


    let dx0 = node_matrix[2][0] - node_matrix[0][0];
    let dxdy0: f64;
    let dn0: f64;
    match dx0 == 0. {
        false => {
            dxdy0 = (node_matrix[2][1] - node_matrix[0][1]) / dx0;
            dn0 = 1. / dxdy0;
        },
        true => {
            dxdy0 = f64::MAX;
            dn0 = 0.;
        }
    }

    let dx1 = node_matrix[2][0] - node_matrix[1][0];
    let dxdy1: f64;
    let dn1: f64;
    let phi: f64;
    match dx1 == 0. {
        false => {
            dxdy1 = (node_matrix[2][1] - node_matrix[1][1]) / dx1;
            dn1 = 1. / dxdy1;
            phi = node_matrix[1][0] ; 
        },
        true => {
            dxdy1 = f64::MAX;
            dn1 = 0.;
            phi = node_matrix[1][0]
        }
    }

    let mut basis_matrix = Matrix::new_square_with_constant_values(3, 0.);

    for j in 0..=2 {
        let alpha_j = coefficients[j][0];
        let beta_j = coefficients[j][1];
        let delta_j = coefficients[j][2];
        for k in 0..=j {
            let alpha_k = coefficients[k][0];
            let beta_k = coefficients[k][1];
            let delta_k = coefficients[k][2];
            
            let theta =  1./3. * (alpha_k*alpha_j);
            let psi = 1./2. * (alpha_k*beta_j + alpha_j*beta_k);
            let epsilon = 1./2. * (alpha_k*delta_j + alpha_j*delta_k);
            let eta = beta_k*beta_j;
            let rho = beta_k*delta_j + beta_j*delta_k;
            let sigma = delta_k * delta_j;

            let a = dn1.powi(3);
            let b = 3. * phi * dn1.powi(2);
            let c = 3. * phi.powi(2)*dn1;
            let d = 1. * phi.powi(3);
            let e = dn0.powi(3);

            let n_3 = 
                theta*(a-e) 
                + psi*(dn1.powi(2) - dn0.powi(2)) 
                + eta * (dn1 - dn0);

            let n_2 = 
                theta*b 
                + 2.*psi*phi*dn1 
                + epsilon*(dn1.powi(2) - dn0.powi(2)) 
                + eta*phi 
                + rho*(dn1 - dn0);

            let n_1 = 
                theta*c 
                + psi*phi.powi(2) 
                + 2. * epsilon*phi*dn1 
                + rho*phi 
                + sigma*(dn1-dn0);

            let n_0 = 
                theta*d 
                + epsilon*phi.powi(2) 
                + sigma*phi;

            let y_2 = node_matrix[2][1];

            if j == k {
                basis_matrix[j][k] = 1. * area /3.;
            } else {
                basis_matrix[j][k] = 0.
            }

            // basis_matrix[j][k] = 
            //     1./4. *n_3*y_2.powi(4) 
            //     + 1./3. *n_2*y_2.powi(3) 
            //     + 1./2. * n_1*y_2.powi(2) 
            //     + n_0*y_2;

            // if (j==k).not() {
            //     basis_matrix[k][j] = basis_matrix[j][k];
            // }


        }
    }


    // todo!()
    // println!("m matrix:");
    // basis_matrix.printer();
    // println!("r matrix:");
    // basis_matrix_derivatives.printer();

    // basis_matrix_derivatives = rotation_matrix.transpose_square() * basis_matrix_derivatives;

    // println!("r matrix:");
    // basis_matrix_derivatives.printer();

    // if proper_node_chosen.not() { // swapping rows back
    //     let basis_matrix_row_1_copy = basis_matrix[1].clone();
    //     basis_matrix[1] = basis_matrix[2].clone();
    //     basis_matrix[2] = basis_matrix_row_1_copy;
    //     let top_row_copy = basis_matrix[0][1];
    //     basis_matrix[0][1] = basis_matrix[0][2];
    //     basis_matrix[0][2] = top_row_copy;
    // }
    // println!("m matrix:");
    // basis_matrix.printer();
    // println!("r matrix:");
    // basis_matrix_derivatives.printer();
    // println!("m matrix:");
    (basis_matrix, basis_matrix_derivatives)
}
