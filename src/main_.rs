mod knappen;
mod quirk;
mod background;

use clap::{App, Arg};
use nalgebra::Vector4;
use std::env;


fn main() {
    let matches = App::new("Quirk and Background Simulation")
        .version("1.0")
        .author("Mark Zakharyan & Shelley Tong, w/ credit to Max Fieg for the original Mathematica code")
        .about("Run simulation with quirk and background processes.")
        .arg(Arg::with_name("quirk_inputPathP")
            .long("quirk_inputPathP")
            .value_names(&vec!["PATH"])
            .help("The path of the first Quirk CSV file.")
            .takes_value(true))
        .arg(Arg::with_name("quirk_inputPathA")
            .long("quirk_inputPathA")
            .value_names(&vec!["PATH"])
            .help("The path of the second Quirk CSV file.")
            .takes_value(true))
        .arg(Arg::with_name("quirk_path_output")
            .long("quirk_path_output")
            .value_names(&vec!["PATH"])
            .help("Output Quirk CSV file path.")
            .takes_value(true))
        .arg(Arg::with_name("background_path_input")
            .long("background_path_input")
            .value_names(&vec!["PATH"])
            .help("Input background CSV path.")
            .takes_value(true))
        .arg(Arg::with_name("background_path_output")
            .long("background_path_output")
            .value_names(&vec!["PATH"])
            .help("Output background CSV file path.")
            .takes_value(true))
        .arg(Arg::with_name("Lambda")
            .short('L')
            .long("Lambda")
            .value_names(&vec!["FLOAT"])
            .help("Lambda used for Quirk string tension (root_sigma).")
            .default_value("20.0")
            .takes_value(true))
        .arg(Arg::with_name("mass")
            .short('m')
            .long("mass")
            .value_names(&vec!["FLOAT"])
            .help("Mass used for quirks (GeV).")
            .default_value("500.0")
            .takes_value(true))
        .arg(Arg::with_name("pid")
            .short('p')
            .long("pid")
            .value_names(&vec!["INT"])
            .help("Positive PID for quirks (should be 15, weird if you change this)")
            .default_value("15")
            .takes_value(true))
        .get_matches();

    let lambda: f64 = matches.value_of("Lambda").unwrap_or("20.0").parse().unwrap();
    let mass: f64 = matches.value_of("mass").unwrap_or("500.0").parse().unwrap();
    let pid: i32 = matches.value_of("pid").unwrap_or("20.0").parse().unwrap();

    let current_directory = env::current_dir().expect("Failed to get current directory");

    
    let quirk_input_path_p = matches.value_of("quirk_inputPathP")
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("{}/4vector_{}GeV_PID{}_1jet.csv", current_directory.display(), mass, pid));

    let quirk_input_path_a = matches.value_of("quirk_inputPathA")
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("{}/4vector_{}GeV_PID{}_1jet.csv", current_directory.display(), mass, -pid));

    let quirk_path_output = matches.value_of("quirk_path_output")
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("{}/HitFiles/QuirkMass_{}_Lambda_{}.csv", current_directory.display(), mass, lambda));

    let background_path_input = matches.value_of("background_path_input")
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("{}/4vector_pionbgd_wCuts.csv", current_directory.display()));

    let background_path_output = matches.value_of("background_path_output")
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("{}/HitFiles/Bgd/Bgd_{}_1jet_wCuts.csv", current_directory.display(), mass));


    let passed = quirk::process(lambda, &quirk_input_path_p, &quirk_input_path_a, &quirk_path_output);
    // let passed: Vec<i32> = vec![3, 4, 10, 14, 15, 19, 20, 21, 26, 31, 32, 33, 40, 41, 43, 47, 50, 51, 52, 55, 56, 57, 58, 59, 61, 66, 67, 68, 69, 70, 79, 81, 82, 83, 87, 88, 94, 95, 98, 100, 103, 104, 105, 111, 113, 114, 116, 117, 118, 119, 127, 128, 131, 134, 136, 139, 140, 141, 142, 143, 144, 150, 152, 154, 155, 165, 166, 171, 173, 174, 176, 177, 178, 180, 181, 184, 186, 187, 188, 189, 192, 194, 195, 197, 198, 199, 203, 205, 208, 211, 212, 220, 222, 224, 228, 229, 234, 236, 237, 240];
    println!("passed: {:?}", passed);
    background::process(lambda, &passed, &background_path_input, &background_path_output);

    // let vec1: Vector4<f64> = Vector4::<f64>::new(620.27, -270.96,-234.9, -78.2);
    // let vec2: Vector4<f64> = Vector4::<f64>::new(615.9, 235.12, 233.8, -139.26);
    // let aa: Vec<(i32, f64, f64, f64, f64, f64, i32)> = match knappen::run_point(&vec1, &vec2, 500.0, false, false) {
    //     Ok(aa) => aa,
    //     Err(e) => {
    //         println!("Error: {:?}", e);
    //         vec![]
    //     }
    // };
    // println!("{:?}", aa);
}
