use anyhow::{ensure, Context, Result};
use libdebayer::{DebayerAlgorithm, DebayerInputImage};
use opencv::core::{Mat, Scalar, Vec3b, CV_8UC1, CV_8UC3};
use opencv::imgcodecs::{imread, IMREAD_COLOR};
use opencv::prelude::*;
use std::path::PathBuf;
use std::time::Instant;

fn convert_to_bggr(image: &Mat) -> Result<Mat> {
    ensure!(
        image.typ() == CV_8UC3 && !image.empty(),
        "Expected a nonempty BGR image"
    );
    let mut bayer =
        Mat::new_rows_cols_with_default(image.rows(), image.cols(), CV_8UC1, Scalar::all(0.0))?;
    for y in 0..image.rows() {
        for x in 0..image.cols() {
            let channel = if y % 2 != x % 2 {
                1
            } else if y % 2 == 0 {
                0
            } else {
                2
            };
            *bayer.at_2d_mut::<u8>(y, x)? = image.at_2d::<Vec3b>(y, x)?[channel];
        }
    }
    Ok(bayer)
}

fn validate_images(original: &Mat, processed: &Mat) -> Result<()> {
    ensure!(
        !original.empty() && original.typ() == CV_8UC3 && processed.typ() == CV_8UC3,
        "PSNR requires nonempty CV_8UC3 images"
    );
    ensure!(
        original.size()? == processed.size()?,
        "PSNR image dimensions differ"
    );
    Ok(())
}

fn psnr_from_mse(mse: f64) -> f64 {
    if mse == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (255.0 * 255.0 / mse).log10()
    }
}

fn calculate_psnr(original: &Mat, processed: &Mat) -> Result<f64> {
    validate_images(original, processed)?;
    let mut sum = 0.0;
    for y in 0..original.rows() {
        for x in 0..original.cols() {
            let original = original.at_2d::<Vec3b>(y, x)?;
            let processed = processed.at_2d::<Vec3b>(y, x)?;
            for channel in 0..3 {
                let delta = original[channel] as f64 - processed[channel] as f64;
                sum += delta * delta;
            }
        }
    }
    Ok(psnr_from_mse(sum / (original.total() as f64 * 3.0)))
}

// Missing G samples at R/B sites of RGGB or BGGR, cropped by four pixels.
fn calculate_green_psnr_at_red_blue(original: &Mat, processed: &Mat) -> Result<f64> {
    validate_images(original, processed)?;
    ensure!(
        original.rows() > 8 && original.cols() > 8,
        "Image is too small for a four-pixel metric crop"
    );
    let mut sum = 0.0;
    let mut count = 0usize;
    for y in 4..original.rows() - 4 {
        for x in 4..original.cols() - 4 {
            if y % 2 != x % 2 {
                continue;
            }
            let delta = original.at_2d::<Vec3b>(y, x)?[1] as f64
                - processed.at_2d::<Vec3b>(y, x)?[1] as f64;
            sum += delta * delta;
            count += 1;
        }
    }
    ensure!(count > 0, "No missing-green samples in metric crop");
    Ok(psnr_from_mse(sum / count as f64))
}

fn process(image: &Mat) -> Result<Mat> {
    let mut input = DebayerInputImage::try_from(image)?;
    Ok(Mat::try_from(
        input.debayer(DebayerAlgorithm::Menon2007Bggr2Bgr)?,
    )?)
}

fn main() -> Result<()> {
    let folder = std::env::var_os("KODAK_FOLDER_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../benchmark/kodak"));
    let mut files = Vec::new();
    for entry in folder
        .read_dir()
        .with_context(|| format!("Cannot read {}", folder.display()))?
    {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if entry.file_type()?.is_file()
            && name.starts_with("kodim")
            && name.ends_with(".png")
            && !name.ends_with(".out.png")
        {
            files.push(entry.path());
        }
    }
    files.sort();
    ensure!(
        !files.is_empty(),
        "No Kodak PNG images found in {}",
        folder.display()
    );
    println!(
        "Menon2007 BGGR; full-image all-channel BGR PSNR; missing-green PSNR uses a 4-pixel crop"
    );
    let mut psnr_sum = 0.0;
    let mut green_sum = 0.0;
    let mut processed = 0usize;
    for file in files {
        let filename = file
            .to_str()
            .context("OpenCV requires a UTF-8 image path")?;
        let image = imread(filename, IMREAD_COLOR)?;
        ensure!(!image.empty(), "Could not decode {}", file.display());
        let bayer = convert_to_bggr(&image)?;
        let _ = process(&bayer)?;
        let start = Instant::now();
        let result = process(&bayer)?;
        let elapsed = start.elapsed();
        let psnr = calculate_psnr(&image, &result)?;
        let green = calculate_green_psnr_at_red_blue(&image, &result)?;
        psnr_sum += psnr;
        green_sum += green;
        processed += 1;
        println!("{}: PSNR {psnr} dB; missing-green PSNR {green} dB; warm host-to-host time {} us (allocation + copies + kernels + synchronization)",
            file.display(), elapsed.as_micros());
    }
    println!(
        "Arithmetic mean of {processed} image PSNRs: {} dB; missing-green PSNR: {} dB",
        psnr_sum / processed as f64,
        green_sum / processed as f64
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_channels_contribute_and_identical_is_infinite() -> Result<()> {
        let reference = Mat::new_rows_cols_with_default(10, 10, CV_8UC3, Scalar::all(0.0))?;
        for channel in 0..3 {
            let mut value = Scalar::all(0.0);
            value[channel] = 30.0;
            let changed = Mat::new_rows_cols_with_default(10, 10, CV_8UC3, value)?;
            assert!((calculate_psnr(&reference, &changed)? - psnr_from_mse(300.0)).abs() < 1e-10);
        }
        assert!(calculate_psnr(&reference, &reference)?.is_infinite());
        Ok(())
    }

    #[test]
    fn missing_green_mask_excludes_measured_green() -> Result<()> {
        let reference = Mat::new_rows_cols_with_default(10, 10, CV_8UC3, Scalar::all(0.0))?;
        let mut changed = reference.try_clone()?;
        changed.at_2d_mut::<Vec3b>(4, 4)?[1] = 12;
        changed.at_2d_mut::<Vec3b>(4, 5)?[1] = 200;
        assert!(
            (calculate_green_psnr_at_red_blue(&reference, &changed)? - psnr_from_mse(72.0)).abs()
                < 1e-10
        );
        Ok(())
    }
}
