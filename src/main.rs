use std::f32::consts::E;
use std::{cmp::Ordering, f64::consts::PI, io::Error};
// use rayon::prelude::*;
// use std::sync::Mutex;
// use ndarray::{Array1, Array2, ArrayBase, Data, Ix1};
use nalgebra::{Matrix4, Vector2};
use ode_solvers::{Rk4, SVector, System, Vector3, Vector4, Dop853}; //, dop853::Dop853
use roots::{find_root_brent, SimpleConvergency};
use plotters::prelude::*;
use plotters::style::ShapeStyle;



const ETA_ETA_MATRIX: Matrix4<f64> = Matrix4::new(
    1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0,
);

const INV_GEV_IN_SEC: f64 = 1.0 / (1.52e24);
const INV_GEV_IN_CM: f64 = 1.9732697880000002e-14;

const EPSILON: f64 = 1e6;
const INV_GEV_TO_NANO_SEC: f64 = 1e9 / (1.52e24);

pub static ATLAS_RADII_PIXEL: [f64; 8] = [3.10, 5.05, 8.85, 12.25, 29.9, 37.1, 44.3, 51.4];
const ATLAS_Z_RANGE: [f64; 8] = [32.0, 40.0, 40.0, 40.0, 74.9, 74.9, 74.9, 74.9];

// static LENGTH_IN_PHI: [f64; 8] = [0.005, 0.005, 0.005, 0.005, 0.0034, 0.0034, 0.0034, 0.0034];

static PHI_SIZE: [f64; 8] = [
    0.0016129,
    0.000990099,
    0.000564972,
    0.000408163,
    0.000113712,
    0.0000916442,
    0.0000767494,
    0.0000661479,
];

static Z_SIZE: [f64; 8] = [0.025, 0.04, 0.04, 0.04, 0.116, 0.116, 0.116, 0.116];

const NA: f64 = 6.022e23;
const ME: f64 = 0.511e-3;
const RE: f64 = 2.81794033e-15 * 100.0; // fm to m conversion and then to cm
const EV2GEV: f64 = 10e-9;

// static HBAR_OMEGA_P_SI: f64 = 31.05;
static STERNHEIMER_A_SI: f64 = 0.1492;
static STERNHEIMER_K_SI: f64 = 3.2546;
static STERNHEIMER_X0_SI: f64 = 0.2015;
static STERNHEIMER_X1_SI: f64 = 2.8716;
static STERNHEIMER_I_SI: f64 = 173.0;
static STERNHEIMER_CBAR_SI: f64 = 4.4355;
static STERNHEIMER_DELTA0_SI: f64 = 0.14;

static Z_SI: i32 = 14;

static A_SI: f64 = 28.085;

static RHO_SI: f64 = 2.33;

// static X0_SI: f64 = 9.37;

// static LAMBDA_SI: f64 = 46.52;

// static LAMBDAC_SI: f64 = 30.16;

// static EEHOLE_SI: f64 = 3.6;

const THICKNESSES: [f64; 8] = [0.023, 0.025, 0.025, 0.025, 0.0285, 0.0285, 0.0285, 0.0285];
const DELTA_R: [f64; 8] = THICKNESSES;

fn beta(gamma: f64) -> f64 {
    (1.0 - 1.0 / gamma.powi(2)).sqrt()
}

fn eta(gamma: f64) -> f64 {
    (gamma.powi(2) - 1.0).sqrt()
}

// fn tmax(ee: f64, mm: f64, me: f64) -> f64 {
//     let eta_ee_mm = eta(ee / mm);
//     let beta_ee_mm = beta(ee / mm);
//     2.0 * me * eta_ee_mm.powi(2)
//         / (1.0 + 2.0 * eta_ee_mm / beta_ee_mm * me / mm + (me / mm).powi(2))
// }

fn delta(eta: f64) -> f64 {
    let x0 = STERNHEIMER_X0_SI;
    let x1 = STERNHEIMER_X1_SI;
    let cbar = STERNHEIMER_CBAR_SI;
    let a = STERNHEIMER_A_SI;
    let k = STERNHEIMER_K_SI;
    let delta0 = STERNHEIMER_DELTA0_SI;

    let log_eta = eta.log10();
    let out: f64;
    if log_eta >= x1 {
        out = 2.0 * std::f64::consts::LN_10 * log_eta - cbar
    } else if log_eta >= x0 {
        out = 2.0 * std::f64::consts::LN_10 * log_eta - cbar + a * (x1 - log_eta).powf(k)
    } else {
        out = delta0 * 10f64.powf(2.0 * (log_eta - x0))
    }
    // println!("{:}", out);
    out
}

fn delta_p_pdg(x: f64, gamma: f64, z: f64) -> f64 {
    // TODO: Ensure this function works as expected!
    

    // 1.0 * Rho_Si * 4 * np.pi * NA * re**2 * me * z**2 *0.5 * Z_Si * x / (A_Si * beta(gamma)**2) * 
    //              (np.log(2 * me * eta(gamma)**2 / (eV2GeV * SternheimerI_Si)) + 
    //               np.log((1.0 * Rho_Si * 4 * np.pi * NA * re**2 * me * z**2) * 
    //                      0.5 * Z_Si * x / (A_Si * beta(gamma)**2) / (eV2GeV * SternheimerI_Si)) - 
    //              beta(gamma)**2 - Delta(eta(gamma), Si) + 0.2)
    

    let output: f64 = 1.0 * RHO_SI * 4.0 * PI * NA * RE.powi(2) * ME * z.powi(2) * 0.5 * (Z_SI as f64) * x / 
                 (A_SI * beta(gamma).powi(2)) * (
                 (2.0 * ME * eta(gamma).powi(2) / (EV2GEV * STERNHEIMER_I_SI)).ln() + 
                 ((1.0 * RHO_SI * 4.0 * PI * NA * RE.powi(2) * ME * z.powi(2) * 
                   0.5 * (Z_SI as f64) * x / (A_SI * beta(gamma).powi(2)) / 
                   (EV2GEV * STERNHEIMER_I_SI)).ln()) - 
                 beta(gamma).powi(2) - delta(eta(gamma)) + 0.2
                 );

    // println!("{:}", output);
    output

}

fn de(gamma: f64, delta_r: f64) -> f64 {
    1e3 * delta_p_pdg(delta_r, gamma, 1.0)
}

fn s(x1: &Vector3<f64>, x2: &Vector3<f64>) -> Vector3<f64> {
    (x1 - x2).normalize()
}

fn vpar(v: &Vector3<f64>, x1: &Vector3<f64>, x2: &Vector3<f64>) -> Vector3<f64> {
    let s_dir = s(x1, x2);
    s_dir * v.dot(&s_dir)
}

fn vper(v: &Vector3<f64>, x1: &Vector3<f64>, x2: &Vector3<f64>) -> Vector3<f64> {
    v - vpar(v, x1, x2)
}

fn gammasc(v: f64) -> f64 {
    1.0 / (1.0 - v.powi(2)).sqrt()
}

fn gamma(v: &Vector3<f64>) -> f64 {
    1.0 / (1.0 - v.norm_squared()).sqrt()
}

fn lorentz(p4: &Vector4<f64>) -> Matrix4<f64> {
    let p4_1_4: Vector3<f64> = Vector3::new(p4.y, p4.z, p4.w);
    let v: f64 = p4_1_4.norm() / p4.x;
    let n: Vector3<f64> = p4_1_4 / p4_1_4.norm();
    let gscv = gammasc(v);

    Matrix4::new(
        gscv, -gscv * v * n.x, -gscv * v * n.y, -gscv * v * n.z,
        -gscv * v * n.x, 1.0 + (gscv - 1.0) * n.x.powi(2), (gscv - 1.0) * n.x * n.y, (gscv - 1.0) * n.x * n.z,
        -gscv * v * n.y, (gscv - 1.0) * n.y * n.x, 1.0 + (gscv - 1.0) * n.y.powi(2), (gscv - 1.0) * n.y * n.z,
        -gscv * v * n.z, (gscv - 1.0) * n.z * n.x, (gscv - 1.0) * n.z * n.y, 1.0 + (gscv - 1.0) * n.z.powi(2),
    )

    // Matrix4::identity()
}


fn yzw(p4: &Vector4<f64>) -> Vector3<f64> {
    Vector3::new(p4.y, p4.z, p4.w)
}

fn dp_dt(
    v: &Vector3<f64>,
    q_e: f64,
    q_m: f64,
    e: &Vector3<f64>,
    b: &Vector3<f64>,
) -> Vector3<f64> {
    q_e * (e + v.cross(b)) + q_m * (b - v.cross(e))
}

struct ParticleSystem {
    m: f64,
    q_e: f64,
    q_m: f64,
}

type State = ode_solvers::SVector<f64, 15>;
type Time = f64;

impl System<State> for ParticleSystem {
    fn system(&self, _t: Time, _y: &State, _dy: &mut State) {
        let gammabeta1 = _y.fixed_rows::<3>(0).into_owned();
        let gammabeta2 = _y.fixed_rows::<3>(3).into_owned();
        let x1 = _y.fixed_rows::<3>(6).into_owned();
        let x2 = _y.fixed_rows::<3>(9).into_owned();
        let b = _y.fixed_rows::<3>(12).into_owned();

        let dgammabeta1dt = (1.0 / self.m)
            * dp_dt(
                &(gammabeta1 / (1.0 + gammabeta1.norm_squared()).sqrt()),
                self.q_e,
                self.q_m,
                &Vector3::new(0.0, 0.0, 0.0),
                &b
            );
        let dgammabeta2dt = (1.0 / self.m)
            * dp_dt(
                &(gammabeta2 / (1.0 + gammabeta2.norm_squared()).sqrt()),
                self.q_e,
                self.q_m,
                &Vector3::new(0.0, 0.0, 0.0),
                &b
            );

        let dx1dt = gammabeta1 / (1.0 + gammabeta1.norm_squared()).sqrt();
        let dx2dt = gammabeta2 / (1.0 + gammabeta2.norm_squared()).sqrt();

        _dy.fixed_rows_mut::<3>(0).copy_from(&dgammabeta1dt);
        _dy.fixed_rows_mut::<3>(3).copy_from(&dgammabeta2dt);
        _dy.fixed_rows_mut::<3>(6).copy_from(&dx1dt);
        _dy.fixed_rows_mut::<3>(9).copy_from(&dx2dt);
        _dy.fixed_rows_mut::<3>(12).copy_from(&Vector3::new(0.0, 0.0, 0.0));
    }
}

fn find_tracks(
    vec1: &Vector4<f64>,
    vec2: &Vector4<f64>,
    q_e: f64,
    q_m: f64,
    b: &Vector3<f64>,
) -> Result<(Interpolator, Interpolator), Error> {

    let m: f64 = (vec1.dot(&(ETA_ETA_MATRIX * vec1))).sqrt();

    // let mut tmax = false;


    let system = ParticleSystem {
        m,
        q_e,
        q_m
    };

    let mut y0: State = State::zeros();

    let gammabeta1_0: Vector3<f64> = (1.0 / m) * yzw(vec1);
    let gammabeta2_0: Vector3<f64> = (1.0 / m) * yzw(vec2);
    let x1_0: Vector3<f64> = Vector3::new(0.0, 10.0_f64.powi(-5), 0.0);
    let x2_0: Vector3<f64> = Vector3::new(0.0, -10.0_f64.powi(-5), 0.0);


    // Set initial conditions in y0
    y0.fixed_rows_mut::<3>(0).copy_from(&gammabeta1_0);
    y0.fixed_rows_mut::<3>(3).copy_from(&gammabeta2_0);
    y0.fixed_rows_mut::<3>(6).copy_from(&x1_0);
    y0.fixed_rows_mut::<3>(9).copy_from(&x2_0);
    y0.fixed_rows_mut::<3>(12).copy_from(b);

    let total_time = 1e19;
    let num_steps = 10000000;
    let step_size = total_time / (num_steps as f64); // Approximate step size


    let event_fn: Box<dyn Fn(Time, &State) -> bool> = Box::new(
        |time, state| {
            let mult_vec: Vector4<f64> = Vector4::new(INV_GEV_IN_SEC, INV_GEV_IN_CM, INV_GEV_IN_CM, INV_GEV_IN_CM);
            let sumx12: Vector3<f64> = state.fixed_rows::<3>(6).into_owned() + state.fixed_rows::<3>(9).into_owned();
            let norm: f64 = (yzw(&mult_vec.component_mul(&Vector4::new(time, sumx12.x, sumx12.y, sumx12.z)))).norm();
            // println!("norm: {:}", norm);
            norm > 100.0
        },
    );

    // let mut stepper = Dop853::new(system, 0.1, 10e19, step_size, y0, 1e-5, 1e12, boost_back).with_event_fn(event_fn);

    let mut stepper = Rk4::new(system, 0.1, y0, total_time, step_size).with_event_fn(event_fn);



    let res = stepper.integrate();


    // Handle result
    let mut sol0_values: Vec<Vector4<f64>> = Vec::new();
    let mut sol1_values: Vec<Vector4<f64>> = Vec::new();

    match res {
        Ok(_stats) => {
            let time_points: &Vec<f64> = stepper.x_out();
            let sol_states: &Vec<SVector<f64, 15>> = stepper.y_out();

            let conversion_vec: Vector4<f64> = Vector4::new(INV_GEV_IN_SEC, INV_GEV_IN_CM, INV_GEV_IN_CM, INV_GEV_IN_CM);
            for i in 0..time_points.len() {
                let sol0: Vector4<f64> = conversion_vec.component_mul(&Vector4::new(time_points[i], sol_states[i][6], sol_states[i][7], sol_states[i][8]));
                let sol1: Vector4<f64> = conversion_vec.component_mul(&Vector4::new(time_points[i], sol_states[i][9], sol_states[i][10], sol_states[i][11]));
                sol0_values.push(sol0);
                sol1_values.push(sol1);
            }

            return Ok((
                Interpolator::new(time_points.to_owned(), sol0_values.to_owned()),
                Interpolator::new(time_points.to_owned(), sol1_values.to_owned()),
            ));
        }
        Err(_) => {
            println!("An error occured.");
            Err(Error::new(std::io::ErrorKind::Other, "An error occured integrating in FindTracks."))
        }
    }
}


struct Interpolator {
    t: Vec<f64>,
    vector4: Vec<Vector4<f64>>,
}

impl Interpolator {
    fn new(t: Vec<f64>, vector4: Vec<Vector4<f64>>) -> Self {
        Interpolator { t, vector4 }
    }

    fn interpolate(&self, t_value: f64) -> Vector4<f64> {
        let n = self.t.len();
        if n == 0 {
            return Vector4::new(0.0, 0.0, 0.0, 0.0);
        }
        if n == 1 {
            return self.vector4[0];
        }

        if t_value <= self.t[0] {
            // Linear extrapolation using the first two points
            let slope = (self.vector4[1] - self.vector4[0]) / (self.t[1] - self.t[0]);
            return self.vector4[0] + slope * (t_value - self.t[0]);
        }

        if t_value >= self.t[n - 1] {
            // Linear extrapolation using the last two points
            let slope = (self.vector4[n - 1] - self.vector4[n - 2]) / (self.t[n - 1] - self.t[n - 2]);
            return self.vector4[n - 1] + slope * (t_value - self.t[n - 1]);
        }

        for i in 0..n - 1 {
            if self.t[i] <= t_value && t_value <= self.t[i + 1] {
                let t0 = self.t[i];
                let t1 = self.t[i + 1];
                let y0 = &self.vector4[i];
                let y1 = &self.vector4[i + 1];
                let factor = (t_value - t0) / (t1 - t0);
                return y0 * (1.0 - factor) + y1 * factor;
            }
        }

        // Fallback, should not reach here if t is sorted
        self.vector4[n - 1]
    }
}

fn beta_beta(traj_map: &Interpolator, t0: f64) -> f64 {
    // return ((np.linalg.norm(trajMap([t0 + epsilon])[0][1:4]) - np.linalg.norm(trajMap([t0])[0][1:4])) / (trajMap([t0 + epsilon])[0][0] - trajMap([t0])[0][0])) / (3 * 10**10)

    let traj_map_t0_plus_epsilon: Vector4<f64> = traj_map.interpolate(t0 + EPSILON);
    let traj_map_t0: Vector4<f64> = traj_map.interpolate(t0);

    // return ((np.linalg.norm(trajMap([t0 + epsilon])[0][1:4]) - np.linalg.norm(trajMap([t0])[0][1:4])) / (trajMap([t0 + epsilon])[0][0] - trajMap([t0])[0][0])) / (3 * 10**10)

    (yzw(&traj_map_t0_plus_epsilon).norm() - yzw(&traj_map_t0).norm())
        / (traj_map_t0_plus_epsilon.x - traj_map_t0.x)
        / 3.0e10
}

fn gamma_gamma(traj_map: &Interpolator, t0: f64) -> f64 {
    return 1.0 / (1.0 - beta_beta(traj_map, t0).powi(2)).sqrt();
}

fn find_intersections(traj: &Interpolator) -> Vec<(i32, f64, f64, f64, f64, f64)> {
    let t_values = traj.t.clone();
    let r_values: Vec<f64> = t_values
        .iter()
        .map(|&t| {
            let traj_t = traj.interpolate(t);
            Vector3::new(traj_t[1], traj_t[2], 0.0).norm()
        })
        .collect();

    let coarse_list: Vec<(f64, f64, f64, f64)> = t_values
        .iter()
        .zip(&r_values)
        .zip(t_values.iter().skip(1).zip(&r_values).skip(1))
        .map(|((t0, r0), (t1, r1))| (*t0, *r0, *t1, *r1))
        .collect();

    let mut final_list = Vec::new();

    for layer in 0..8 {
        let layer_radius = ATLAS_RADII_PIXEL[layer];
        let phi_size_layer = PHI_SIZE[layer];
        let z_size_layer = Z_SIZE[layer];

        let layer_list: Vec<(f64, f64, f64, f64)> = coarse_list
            .iter()
            .filter(|row| (row.1 - layer_radius) * (row.3 - layer_radius) < 0.0)
            .cloned()
            .collect();

        for (t_low, _, t_high, _) in layer_list {
            let root = find_root_brent(
                t_low,
                t_high,
                |t| {
                    let traj_t = traj.interpolate(t);
                    Vector3::new(traj_t[1], traj_t[2], 0.0).norm() - layer_radius
                },
                &mut SimpleConvergency {
                    eps: 1e-12,
                    max_iter: 1000,
                },
            );

            if let Ok(t) = root {
                let traj_t = traj.interpolate(t);
                let z = traj_t[3];
                let phi = traj_t[2].atan2(traj_t[1]);
                let gamma = gamma_gamma(&traj, t);
                let beta = beta_beta(&traj, t);

                final_list.push((
                    (layer + 1) as i32,
                    (z / Z_SIZE[1]).round() * Z_SIZE[1],
                    (phi / phi_size_layer).round() * phi_size_layer,
                    de(gamma, DELTA_R[layer]),
                    beta.abs() * gamma,
                    INV_GEV_TO_NANO_SEC * gamma * t,
                ));
            }
        }
    }

    final_list
}


fn plot_trajectories(sol1: &Interpolator, sol2: &Interpolator) -> Result<(), Box<dyn std::error::Error>> {
    // Extracting trajectory data
    let sol1_points: Vec<(f64, f64)> = sol1.vector4.iter().map(|p| (p[1], p[2])).collect();
    let sol2_points: Vec<(f64, f64)> = sol2.vector4.iter().map(|p| (p[1], p[2])).collect();

    let rad = *ATLAS_RADII_PIXEL.last().unwrap_or(&0.0);
    let x_range = (-rad - 5.0, rad + 5.0);
    let y_range = (-rad - 5.0, rad + 5.0);

    let root_area = BitMapBackend::new("plot.png", (800, 800)).into_drawing_area();
    root_area.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root_area)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .margin(5)
        .build_cartesian_2d(x_range.0..x_range.1, y_range.0..y_range.1)?;

    chart.configure_mesh().draw()?;

    chart
        .draw_series(LineSeries::new(
            sol1_points.into_iter(),
            BLUE.stroke_width(1),
        ))?
        .label("Solution 1")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart
        .draw_series(LineSeries::new(
            sol2_points.into_iter(),
            RED.stroke_width(1),
        ))?
        .label("Solution 2")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

        for &radius in &ATLAS_RADII_PIXEL {
            let num_points = 3600; // Increase the number of points for a smoother circle
            let circle_data: Vec<(f64, f64)> = (0..num_points)
                .map(|i| {
                    let angle = i as f64 * std::f64::consts::PI * 2.0 / num_points as f64;
                    (radius * angle.cos(), radius * angle.sin())
                })
                .collect();
        
            chart.draw_series(LineSeries::new(circle_data.into_iter(), &BLACK))?;
        }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperLeft)
        // .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    Ok(())
}

fn export_trajectories(
    filename: &str,
    sol1: &Interpolator,
    sol2: &Interpolator,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create(filename)?;

    writeln!(file, "monopole#,t,x,y,z")?;

    for (i, sol) in sol1.vector4.iter().enumerate() {
        writeln!(file, "1,{}, {}, {}, {}", sol[0], sol[1], sol[2], sol[3])?;
    }

    for (i, sol) in sol2.vector4.iter().enumerate() {
        writeln!(file, "2,{}, {}, {}, {}", sol[0], sol[1], sol[2], sol[3])?;
    }

    Ok(())
}

pub fn run_point(
    vec1: &Vector4<f64>,
    vec2: &Vector4<f64>,
    plotflag: bool,
    exportflag: bool,
) -> Result<Vec<(i32, f64, f64, f64, f64, f64, i32)>, Error> {

    match find_tracks(vec1, vec2, 0.0, 1.0, &Vector3::new(0.0, 0.0, 1.18314e-16)) {
            Ok((sol1, sol2)) => {
                if plotflag {
                    if let Err(e) = plot_trajectories(&sol1, &sol2) {
                        println!("Error plotting trajectories: {:?}", e);
                    }
                }

                if exportflag {
                    // Export the trajectories to a file
                    if let Err(e) = export_trajectories("trajectories.csv", &sol1, &sol2) {
                        println!("Error exporting trajectories: {:?}", e);
                    }
                }

                let intersections1 = find_intersections(&sol1);
                let intersections2 = find_intersections(&sol2);


                let mut combined_intersections: Vec<(i32, f64, f64, f64, f64, f64, i32)> = intersections1
            .iter()
            .map(|&(a, b, c, d, e, f)| (a, b, c, d, e, f, 1))
            .chain(
                intersections2
                    .iter()
                    .map(|&(a, b, c, d, e, f)| (a, b, c, d, e, f, 2)),
            )
            .collect();

        combined_intersections
            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        return Ok(combined_intersections);
            }
            Err(err) => {
                // Handle the error case if needed
                println!("Error: {:?}", err);
                // Return an appropriate value or propagate the error
                return Err(err);
            }
        }

}

fn main() {
    let vec1: Vector4<f64> = Vector4::new(421.69956147, 258.12146064, 154.10248991, -254.86516886);
    let vec2: Vector4<f64> = Vector4::new(202.65928421, -123.22431566, 12.442253162, 56.848428683);

    let aa = run_point(&vec1, &vec2, false, true);
    println!("{:?}", aa);
}