use crate::knappen::{run_point, ATLAS_RADII_PIXEL};
use csv::{WriterBuilder, Writer};
use nalgebra::Vector4;
use serde::Deserialize;
use std::env;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::time::Instant;

#[derive(Debug, Deserialize)]
struct Particle {
    E: f64,
    px: f64,
    py: f64,
    pz: f64,
}

pub fn process(lambda: f64, input_path_p: &str, input_path_a: &str, output_path: &str) -> Vec<i32> {
    let mut rdr_p = csv::ReaderBuilder::new().has_headers(true).from_path(input_path_p).unwrap();
    let mut rdr_a = csv::ReaderBuilder::new().has_headers(true).from_path(input_path_a).unwrap();

    let particles_p: Vec<Particle> = rdr_p.deserialize().map(|result| result.unwrap()).collect();
    let particles_a: Vec<Particle> = rdr_a.deserialize().map(|result| result.unwrap()).collect();

    std::fs::create_dir_all(Path::new(output_path).parent().unwrap()).unwrap();
    let file = File::create(output_path).unwrap();
    let mut writer = BufWriter::new(file);
    let mut csv_writer = Writer::from_writer(writer);

    // Write headers
    csv_writer.write_record(&["EventID", "PID", "Layer", "r[cm]", "phi", "z[cm]", "E[GeV]", "px[GeV]", "py[GeV]", "pz[GeV]"]).unwrap();


    let mut passed = Vec::new();
    let mut buffer: Vec<Vec<String>> = Vec::new();


    for (event_id, (particle_p, particle_a)) in particles_p.iter().zip(particles_a.iter()).enumerate().take(500) {
        let vec1 = Vector4::new(particle_a.E, particle_a.px, particle_a.py, particle_a.pz);
        let vec2 = Vector4::new(particle_p.E, particle_p.px, particle_p.py, particle_p.pz);

        let vecs = [vec1, vec2];

        match run_point(&vec1, &vec2, lambda, false, true) {
            Ok(aa) => {
                if aa.len() == 16 {
                    passed.push((event_id+1) as i32);
                    for hit in aa {
                        let record = vec![
                            (event_id + 1).to_string(),
                            hit.0.to_string(),
                            hit.6.to_string(),
                            ATLAS_RADII_PIXEL[hit.0 as usize - 1].to_string(),
                            hit.2.to_string(),
                            hit.1.to_string(),
                            vecs[hit.6 as usize - 1][0].to_string(),
                            vecs[hit.6 as usize - 1][1].to_string(),
                            vecs[hit.6 as usize - 1][2].to_string(),
                            vecs[hit.6 as usize - 1][3].to_string(),
                        ];
                        buffer.push(record);
                        if buffer.len() >= 10 {
                            // Write buffered records to file
                            for record in buffer.drain(..) {
                                csv_writer.write_record(&record).unwrap();
                            }
                        }
                    }
                }
            }
            Err(e) => {
                continue;
            }
        }

        println!("Event ID: {}", event_id + 1);
        
        // layer = AA[jj][0] - 1
        //         r = ATLASradiiPixel[layer]
        //         z = AA[jj][1]
        //         phi = AA[jj][2]
        //         QID = AA[jj][6] - 1
        //         listhere = [EventID+1, layer+1, QID+1, r, phi, z, vecs[QID][0], vecs[QID][1], vecs[QID][2], vecs[QID][3]] # ensure the +1s are correct???
        //         data.append(listhere)
        
    }
    
    for record in buffer {
        csv_writer.write_record(&record).unwrap();
    }

    csv_writer.flush().unwrap();

    passed
}

pub fn _default_quirk() {
    let lambda = 20.0;
    let pid = 15;
    let mass = 500;

    let current_directory = env::current_dir().unwrap();
    let input_path_p = format!("{}/4vector_{}GeV_PID{}_1jet.csv", current_directory.display(), mass, pid);
    let input_path_a = format!("{}/4vector_{}GeV_PID{}_1jet.csv", current_directory.display(), mass, -pid);
    let output_path = format!("{}/HitFiles/QuirkMass_{}_Lambda_{}.csv", current_directory.display(), mass, lambda);

    let start_time = Instant::now();
    let passed = process(lambda, &input_path_p, &input_path_a, &output_path);
    let duration = start_time.elapsed();

    println!("(Rust Quirk Regular) Time taken: {:?} seconds", duration.as_secs_f32());
    println!("passed: {:?}", passed);
}

// (Quirk Regular) Time taken: 53.58206 seconds
// rs passed = [3, 4, 6, 7, 8, 9, 10, 11, 17, 19, 20, 24, 25, 26, 31, 32, 41, 43, 45, 46, 48, 49, 50, 51, 53, 55, 56, 57, 59, 69, 72, 74, 76, 78, 80, 81, 83, 84, 86, 88, 91, 94, 95, 101, 105, 106, 107, 108, 109, 111, 114, 121, 122, 125, 126, 127, 128, 130, 131, 138, 139, 146, 154, 157, 158, 159, 163, 169, 171, 172, 173, 174, 179, 181, 182, 183, 184, 187, 189, 190, 194, 196, 198, 200, 204, 205, 210, 214, 217, 219, 221, 222, 224, 225, 226, 230, 233, 234, 239, 240, 245, 247, 248, 253, 254, 256, 259, 260, 261, 265, 267, 274, 276, 277, 280, 281, 282, 283, 284, 287, 290, 293, 295, 299, 300, 305, 306, 307, 308, 309, 316, 318, 320, 332, 339, 343, 346, 349, 351, 352, 358, 359, 362, 364, 369, 370, 372, 374, 377, 379, 381, 383, 386, 387, 391, 398, 400, 402, 406, 408, 415, 422, 423, 428, 435, 436, 440, 443, 445, 448, 450, 456, 457, 458, 460, 461, 464, 466, 467, 470, 475, 480, 484, 490, 492, 496, 498, 499]
// py passed = [3, 4, 10, 14, 15, 19, 20, 21, 26, 31, 32, 33, 40, 41, 43, 47, 50, 51, 52, 55, 56, 57, 58, 59, 61, 66, 67, 68, 69, 70, 79, 81, 82, 83, 87, 88, 94, 95, 98, 100, 103, 104, 105, 111, 113, 114, 116, 117, 118, 119, 127, 128, 131, 134, 136, 139, 140, 141, 142, 143, 144, 150, 152, 154, 155, 165, 166, 171, 173, 174, 176, 177, 178, 180, 181, 184, 186, 187, 188, 189, 192, 194, 195, 197, 198, 199, 203, 205, 208, 211, 212, 220, 222, 224, 228, 229, 234, 236, 237, 240]
