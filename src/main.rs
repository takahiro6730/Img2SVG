use image::GenericImageView;

fn main() {

    if let Ok(img) = image::open("samplePictures/1PM.png") {
        let (width, height) = img.dimensions();
        println!("Image dimensions: {} x {}", width, height);

        let img_data = img.into_rgba8();

        let gray_image_data = convert_to_grayscale(&img_data);

        let sigma = 1.0; 
        let blurred_image = apply_gaussian_blur(&gray_image_data, sigma);
        let (gradient_x, gradient_y) = compute_gradients(&blurred_image);


        let suppressed_image = non_maximum_suppression(&gradient_x, &gradient_y);

        let thresholded_image = apply_hysteresis_thresholding(&suppressed_image, 150, 200);

        thresholded_image.save("thresholded_image.png").expect("Failed to save image");
   
    } else {
        println!("Failed to open the PNG file");
    }
}
fn convert_to_grayscale(image_data: &image::RgbaImage) -> image::GrayImage {
    let mut gray_image_data = image::GrayImage::new(image_data.width(), image_data.height());
    for (x, y, pixel) in image_data.enumerate_pixels() {
        let r = pixel[0] as f32;
        let g = pixel[1] as f32;
        let b = pixel[2] as f32;
        let luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b).round() as u8;
        gray_image_data.put_pixel(x, y, image::Luma([luminance]));
    }
    gray_image_data
}

fn apply_gaussian_blur(image: &image::GrayImage, sigma: f32) -> image::GrayImage {

    let kernel = gaussian_kernel(sigma);
    let mut blurred_image = image.clone();
    for y in 1..image.height() - 1 {
        for x in 1..image.width() - 1 {
            let mut sum = 0.0;
            for ky in -1..=1 {
                for kx in -1..=1 {
                    let (nx, ny) = (x as i32 + kx, y as i32 + ky);
                    if nx >= 0 && ny >= 0 && nx < image.width() as i32 && ny < image.height() as i32 {
                        let pixel_value = image.get_pixel(nx as u32, ny as u32)[0] as f32;
                        let kernel_value = kernel[(ky + 1) as usize][(kx + 1) as usize];
                        sum += pixel_value * kernel_value;
                        }
                }
            }
            let blurred_value = sum.round() as u8;
            blurred_image.put_pixel(x as u32, y as u32, image::Luma([blurred_value]));
        }
    }

    blurred_image
}

fn gaussian_kernel(sigma: f32) -> [[f32; 3]; 3] {
    let sigma_sq = sigma * sigma;
    let mut kernel = [[0.0; 3]; 3];
    let coefficient = 1.0 / (2.0 * std::f32::consts::PI * sigma_sq);
    let denominator = 2.0 * sigma_sq;

    for y in -1..=1 {
        for x in -1..=1 {
            let exponent = -(x * x + y * y) as f32 / denominator;
            kernel[(y + 1) as usize][(x + 1) as usize] = coefficient * exponent.exp();
        }
    }

    let sum: f32 = kernel.iter().flat_map(|row| row.iter()).sum();
    for row in &mut kernel {
        for value in row {
            *value /= sum;
        }
    }

    kernel
}

fn compute_gradients(image: &image::GrayImage) -> (image::GrayImage, image::GrayImage) {
    // Sobel operator kernels
    let sobel_x_kernel = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]];
    let sobel_y_kernel = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]];

    // Convolve the image with the Sobel kernels to compute gradients
    let gradient_x = convolve(image, &sobel_x_kernel);
    let gradient_y = convolve(image, &sobel_y_kernel);

    (gradient_x, gradient_y)
}

fn convolve(image: &image::GrayImage, kernel: &[[i32; 3]; 3]) -> image::GrayImage {
    let mut result = image::GrayImage::new(image.width(), image.height());

    for y in 1..image.height() - 1 {
        for x in 1..image.width() - 1 {
            let mut sum = 0;
            for ky in 0..3 {
                for kx in 0..3 {
                    let px = x + kx - 1;
                    let py = y + ky - 1;

                        let kernel_value = kernel[ky as usize][kx as usize];
                        let pixel_value = image.get_pixel(px as u32, py as u32)[0] as i32;
                        sum += kernel_value * pixel_value;

                }
            }
            result.put_pixel(x, y, image::Luma([sum as u8]));
        }
    }

    result
}

fn non_maximum_suppression(gradient_magnitude: &image::GrayImage, gradient_direction: &image::GrayImage) -> image::GrayImage {
    let mut suppressed_image = image::GrayImage::new(gradient_magnitude.width(), gradient_magnitude.height());

    for y in 1..gradient_magnitude.height() - 1 {
        for x in 1..gradient_magnitude.width() - 1 {
            let magnitude = gradient_magnitude.get_pixel(x, y)[0] as f32;
            let direction = gradient_direction.get_pixel(x, y)[0] as f32;

            let (dx, dy) = direction_to_deltas(direction);

            let (x1, y1) = ((x as i32 + dx) as u32, (y as i32 + dy) as u32);
            let (x2, y2) = ((x as i32 - dx) as u32, (y as i32 - dy) as u32);

            let mag1 = gradient_magnitude.get_pixel(x1, y1)[0] as f32;
            let mag2 = gradient_magnitude.get_pixel(x2, y2)[0] as f32;

            if magnitude >= mag1 && magnitude >= mag2 {
                suppressed_image.put_pixel(x, y, image::Luma([magnitude as u8]));
            } else {
                suppressed_image.put_pixel(x, y, image::Luma([0]));
            }
        }
    }

    suppressed_image
}

fn direction_to_deltas(direction: f32) -> (i32, i32) {
    let angle = direction.to_radians();
    let dx = angle.cos();
    let dy = angle.sin();
    let dx = if dx.abs() < 1e-6 { 0.0 } else { dx };
    let dy = if dy.abs() < 1e-6 { 0.0 } else { dy };
    (dx.round() as i32, dy.round() as i32)
}

fn apply_hysteresis_thresholding(image: &image::GrayImage, low_threshold: u8, high_threshold: u8) -> image::GrayImage {
    let mut result = image::GrayImage::new(image.width(), image.height());

    for y in 0..image.height() {
        for x in 0..image.width() {
            let pixel_value = image.get_pixel(x, y)[0];

            if pixel_value >= high_threshold {
                // Strong edge pixel
                result.put_pixel(x, y, image::Luma([255]));
            } else if pixel_value >= low_threshold {
                // Weak edge pixel
                // Check neighboring pixels for strong edges
                let has_strong_neighbor = has_strong_neighbor(image, x, y, high_threshold);
                if has_strong_neighbor {
                    result.put_pixel(x, y, image::Luma([255]));
                } else {
                    result.put_pixel(x, y, image::Luma([0]));
                }
            } else {
                // Non-edge pixel
                result.put_pixel(x, y, image::Luma([0]));
            }
        }
    }

    result
}

fn has_strong_neighbor(image: &image::GrayImage, x: u32, y: u32, high_threshold: u8) -> bool {
    let neighbors = [
        (x.saturating_sub(1), y.saturating_sub(1)),
        (x, y.saturating_sub(1)),
        (x.saturating_add(1), y.saturating_sub(1)),
        (x.saturating_sub(1), y),
        (x.saturating_add(1), y),
        (x.saturating_sub(1), y.saturating_add(1)),
        (x, y.saturating_add(1)),
        (x.saturating_add(1), y.saturating_add(1)),
    ];

    for &(nx, ny) in &neighbors {
        if nx < image.width() && ny < image.height() {
            let pixel_value = image.get_pixel(nx, ny)[0];
            if pixel_value >= high_threshold {
                return true;
            }
        }
    }

    false
}