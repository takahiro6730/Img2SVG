// use std::fs::File;
// use std::io::Write;

// fn main() {
//     let width = 400;
//     let height = 200;

//     let mut svg = String::new();

//     // Write SVG header
//     svg.push_str(&format!(
//         r#"<?xml version="1.0" encoding="UTF-8" standalone="no"?>
// <svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">"#,
//         width, height
//     ));

//     // Draw a rectangle
//     svg.push_str(&format!(
//         r#"<rect x="50" y="50" width="300" height="100" fill="blue" stroke="black" stroke-width="2"/>"#
//     ));

//     // Write SVG footer
//     svg.push_str("</svg>");

//     // Write SVG string to a file
//     let mut file = File::create("output.svg").expect("Unable to create file");
//     file.write_all(svg.as_bytes()).expect("Unable to write to file");

// }

use image::GenericImageView;

fn main() {
    // Open the PNG file
    if let Ok(img) = image::open("samplePictures/1PM.png") {
        let (width, height) = img.dimensions();
        println!("Image dimensions: {} x {}", width, height);

        for (x, y, pixel) in img.pixels() {

            let rgba = pixel.0;
            println!("Pixel at ({}, {}): {:?}", x, y, rgba);
        }
    } else {
        println!("Failed to open the PNG file");
    }
}