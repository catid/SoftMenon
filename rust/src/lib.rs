use libdebayer_sys::*;
use opencv::boxed_ref::BoxedRef;
use opencv::core::{Mat, MatTraitConst};
use opencv::prelude::*;
use std::ffi::c_void;
use std::rc::Rc;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CudaError {
    #[error("cudaMallocPitch failed with error code: {0}")]
    CudaMallocFailed(u32),
    #[error("cudaMemset2D failed with error code: {0}")]
    CudaMemset2DFailed(u32),
    #[error("cudaMemcpy2DAsync failed with error code: {0}")]
    CudaMemcpy2DAsyncFailed(u32),
    #[error("cudaStreamCreate failed with error code: {0}")]
    CudaStreamCreateFailed(u32),
    #[error("cudaStreamSynchronize failed with error code: {0}")]
    CudaStreamSynchronizeFailed(u32),
    #[error("debayer kernel launch failed with error code: {0}")]
    CudaKernelFailed(u32),
}

#[derive(Error, Debug)]
pub enum DebayerError {
    #[error("OpenCV error: {0}")]
    Opencv(#[from] opencv::error::Error),
    #[error("CUDA error: {0}")]
    Cuda(#[from] CudaError),
    #[error("Invalid input image: {0}")]
    InvalidInput(&'static str),
}

#[derive(Clone, Copy, Debug)]
pub enum DebayerAlgorithm {
    BilinearRggb2Bgr,
    BilinearBggr2Bgr,
    Malvar2004Rggb2Bgr,
    Malvar2004Bggr2Bgr,
    Menon2007Rggb2Bgr,
    Menon2007Bggr2Bgr,
    SoftMenonRggb2Bgr,
    SoftMenonBggr2Bgr,
}

pub enum DebayerImageType {
    Input,
    Output,
}

struct CudaStream(cudaStream_t);

impl CudaStream {
    fn new() -> Result<Rc<Self>, CudaError> {
        let mut stream = std::ptr::null_mut();
        let ret = unsafe { cudaStreamCreate(&mut stream) };
        if ret != cudaError::cudaSuccess {
            unsafe {
                let _ = cudaGetLastError();
            }
            return Err(CudaError::CudaStreamCreateFailed(ret));
        }
        Ok(Rc::new(Self(stream)))
    }

    fn synchronize(&self) -> Result<(), CudaError> {
        let ret = unsafe { cudaStreamSynchronize(self.0) };
        if ret != cudaError::cudaSuccess {
            unsafe {
                let _ = cudaGetLastError();
            }
            return Err(CudaError::CudaStreamSynchronizeFailed(ret));
        }
        Ok(())
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        unsafe {
            let _ = cudaStreamDestroy(self.0);
        }
    }
}

fn padded_size(size: usize) -> Result<usize, DebayerError> {
    let pad = SARONIC_DEBAYER_PAD as usize;
    let block = KERNEL_BLOCK_SIZE as usize;
    size.checked_add(pad)
        .and_then(|x| x.checked_add(block - 1))
        .map(|x| x / block * block)
        .and_then(|x| x.checked_add(pad))
        .ok_or(DebayerError::InvalidInput("image dimensions overflow"))
}

struct CudaImage {
    width: usize,
    height: usize,
    pitch: usize,
    raw_data: *mut c_void,
    stream: Rc<CudaStream>,
}

impl CudaImage {
    fn new(
        width: usize,
        height: usize,
        channels: usize,
        stream: Rc<CudaStream>,
    ) -> Result<Self, DebayerError> {
        let padded_width = padded_size(width)?
            .checked_mul(channels)
            .ok_or(DebayerError::InvalidInput("image row size overflows"))?;
        let padded_height = padded_size(height)?;
        let mut image = Self {
            width,
            height,
            pitch: 0,
            raw_data: std::ptr::null_mut(),
            stream,
        };
        unsafe {
            let ret = cudaMallocPitch(
                &mut image.raw_data,
                &mut image.pitch,
                padded_width,
                padded_height,
            );
            if ret != cudaError::cudaSuccess {
                let _ = cudaGetLastError();
                return Err(CudaError::CudaMallocFailed(ret).into());
            }
            // The RAII owner already exists, including on memset failure.
            let ret = cudaMemset2D(image.raw_data, image.pitch, 0, padded_width, padded_height);
            if ret != cudaError::cudaSuccess {
                let _ = cudaGetLastError();
                return Err(CudaError::CudaMemset2DFailed(ret).into());
            }
        }
        Ok(image)
    }
}

impl Drop for CudaImage {
    fn drop(&mut self) {
        if !self.raw_data.is_null() {
            // Input can be dropped while an output still owns this stream.
            // Finish kernels that reference this allocation before freeing it.
            let _ = self.stream.synchronize();
            unsafe {
                let _ = cudaFree(self.raw_data);
            }
        }
    }
}

pub struct DebayerInputImage {
    img: CudaImage,
}
pub struct DebayerOutputImage {
    img: CudaImage,
}

impl DebayerInputImage {
    /// Launch reconstruction on the input's stream. The output independently
    /// owns the stream and remains valid after this input is dropped.
    pub fn debayer(
        &mut self,
        algorithm: DebayerAlgorithm,
    ) -> Result<DebayerOutputImage, DebayerError> {
        let output = CudaImage::new(
            self.img.width,
            self.img.height,
            3,
            Rc::clone(&self.img.stream),
        )?;
        let stream = self.img.stream.0;
        let width = self.img.width as i32;
        let height = self.img.height as i32;
        let input = self.img.raw_data as *mut u8;
        unsafe {
            let ret = debayer_mirror_image(stream, width, height, self.img.pitch, input);
            if ret != cudaError::cudaSuccess {
                return Err(CudaError::CudaKernelFailed(ret).into());
            }
            let kernel = match algorithm {
                DebayerAlgorithm::BilinearRggb2Bgr => debayer_rggb2bgr_bilinear,
                DebayerAlgorithm::BilinearBggr2Bgr => debayer_bggr2bgr_bilinear,
                DebayerAlgorithm::Malvar2004Rggb2Bgr => debayer_rggb2bgr_malvar2004,
                DebayerAlgorithm::Malvar2004Bggr2Bgr => debayer_bggr2bgr_malvar2004,
                DebayerAlgorithm::Menon2007Rggb2Bgr => debayer_rggb2bgr_menon2007,
                DebayerAlgorithm::Menon2007Bggr2Bgr => debayer_bggr2bgr_menon2007,
                DebayerAlgorithm::SoftMenonRggb2Bgr => debayer_rggb2bgr_softmenon,
                DebayerAlgorithm::SoftMenonBggr2Bgr => debayer_bggr2bgr_softmenon,
            };
            let ret = kernel(
                stream,
                width,
                height,
                self.img.pitch,
                output.pitch,
                input,
                output.raw_data as *mut u8,
            );
            if ret != cudaError::cudaSuccess {
                return Err(CudaError::CudaKernelFailed(ret).into());
            }
        }
        Ok(DebayerOutputImage { img: output })
    }
}

impl TryFrom<DebayerOutputImage> for Mat {
    type Error = DebayerError;

    /// Copy to a CV_8UC3 BGR Mat and wait for the transfer to complete.
    fn try_from(image: DebayerOutputImage) -> Result<Self, Self::Error> {
        let mut output = Mat::new_rows_cols_with_default(
            image.img.height as i32,
            image.img.width as i32,
            opencv::core::CV_8UC3,
            opencv::core::Scalar::all(0.0),
        )?;
        let pitch = output.step1(0)?;
        unsafe {
            let source = (image.img.raw_data as *mut u8).add(
                SARONIC_DEBAYER_PAD as usize * image.img.pitch + SARONIC_DEBAYER_PAD as usize * 3,
            );
            let ret = cudaMemcpy2DAsync(
                output.data_mut() as *mut c_void,
                pitch,
                source as *const c_void,
                image.img.pitch,
                image.img.width * 3,
                image.img.height,
                cudaMemcpyKind::cudaMemcpyDeviceToHost,
                image.img.stream.0,
            );
            if ret != cudaError::cudaSuccess {
                let _ = cudaGetLastError();
                return Err(CudaError::CudaMemcpy2DAsyncFailed(ret).into());
            }
        }
        image.img.stream.synchronize()?;
        Ok(output)
    }
}

fn cv_to_debayer<T: MatTraitConst>(image: &T) -> Result<DebayerInputImage, DebayerError> {
    if image.typ() != opencv::core::CV_8UC1 {
        return Err(DebayerError::InvalidInput("expected CV_8UC1 sensor mosaic"));
    }
    if image.dims() != 2 || image.cols() < 2 || image.rows() < 2 || image.data().is_null() {
        return Err(DebayerError::InvalidInput(
            "expected a two-dimensional image at least 2 by 2",
        ));
    }
    let width = image.cols() as usize;
    let height = image.rows() as usize;
    let pitch = image.step1(0)?;
    if pitch < width {
        return Err(DebayerError::InvalidInput(
            "row stride is shorter than image width",
        ));
    }
    let stream = CudaStream::new()?;
    let input = CudaImage::new(width, height, 1, stream)?;
    unsafe {
        let destination = (input.raw_data as *mut u8)
            .add(SARONIC_DEBAYER_PAD as usize * input.pitch + SARONIC_DEBAYER_PAD as usize);
        let ret = cudaMemcpy2DAsync(
            destination as *mut c_void,
            input.pitch,
            image.data() as *const c_void,
            pitch,
            width,
            height,
            cudaMemcpyKind::cudaMemcpyHostToDevice,
            input.stream.0,
        );
        if ret != cudaError::cudaSuccess {
            let _ = cudaGetLastError();
            return Err(CudaError::CudaMemcpy2DAsyncFailed(ret).into());
        }
    }
    // The returned object does not borrow the Mat. Even pinned host memory must
    // be safe to modify or release immediately after this conversion returns.
    input.stream.synchronize()?;
    Ok(DebayerInputImage { img: input })
}

impl TryFrom<&Mat> for DebayerInputImage {
    type Error = DebayerError;
    fn try_from(image: &Mat) -> Result<Self, Self::Error> {
        cv_to_debayer(image)
    }
}

impl TryFrom<&BoxedRef<'_, Mat>> for DebayerInputImage {
    type Error = DebayerError;
    fn try_from(image: &BoxedRef<'_, Mat>) -> Result<Self, Self::Error> {
        cv_to_debayer(image)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::core::{Rect, Scalar, Vec3b};

    #[test]
    fn rejects_empty_small_and_non_mosaic_images() -> Result<(), DebayerError> {
        for (rows, cols, kind) in [
            (0, 0, opencv::core::CV_8UC1),
            (1, 8, opencv::core::CV_8UC1),
            (8, 8, opencv::core::CV_16UC1),
            (8, 8, opencv::core::CV_8UC3),
        ] {
            let image = Mat::new_rows_cols_with_default(rows, cols, kind, Scalar::all(0.0))?;
            assert!(matches!(
                DebayerInputImage::try_from(&image),
                Err(DebayerError::InvalidInput(_))
            ));
        }
        assert!(padded_size(usize::MAX).is_err());
        Ok(())
    }

    #[test]
    fn strided_upload_and_output_survive_input_drop() -> Result<(), DebayerError> {
        let output = {
            let image =
                Mat::new_rows_cols_with_default(11, 15, opencv::core::CV_8UC1, Scalar::all(77.0))?;
            let roi = Mat::roi(&image, Rect::new(2, 2, 9, 7))?;
            let mut input = DebayerInputImage::try_from(&roi)?;
            input.debayer(DebayerAlgorithm::BilinearRggb2Bgr)?
        };
        let output = Mat::try_from(output)?;
        assert_eq!((output.rows(), output.cols()), (7, 9));
        for y in 0..output.rows() {
            for x in 0..output.cols() {
                assert_eq!(*output.at_2d::<Vec3b>(y, x)?, Vec3b::from([77, 77, 77]));
            }
        }
        Ok(())
    }
}
