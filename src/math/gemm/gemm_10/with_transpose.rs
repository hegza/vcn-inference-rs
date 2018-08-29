use super::params::Gemm10CompileParameters;
use super::*;

/// Cnugteren's gemm version 10. A and B are column-major but B will be transposed by a kernel.
pub struct Gemm10WithBTransposeKernel {
    gemm10: Gemm10Kernel,
    b_untransposed_buf: Buffer<f32>,
    transpose_kernel: Kernel,
}

impl OclGemm<Gemm10WithBTransposeKernel> for Gemm10WithBTransposeKernel {
    fn uninitialized(
        m: usize,
        n: usize,
        k: usize,
        out: &mut [f32],
        device: DeviceType,
    ) -> Gemm10WithBTransposeKernel {
        // Make sure enough space is reserved for the output buffer
        debug_assert_eq!(out.len(), m * n);

        // If Device uses RAM, use_host_ptr and mapping via address translation may be faster
        let use_host_ptr = device == DeviceType::CPU;

        let src_transpose = String::from_utf8_lossy(include_bytes!("../cl/transpose.cl"));
        let src_padding = String::from_utf8_lossy(include_bytes!("../cl/padding.cl"));
        let src_mtx_mul = String::from_utf8_lossy(include_bytes!("../cl/10_incomplete_tiles.cl"));

        let params = Gemm10WithBTransposeCompileParameters::choose(m, n, k, device);
        let mut params_list: Vec<String> = params.clone().into();
        params_list.push("-I./src/math/gemm/cl".to_owned());

        let (queue, program, _context) = cl_util::init_from_sources::<f32>(
            &[&src_transpose, &src_padding, &src_mtx_mul],
            &params_list.iter().map(AsRef::as_ref).collect::<Vec<&str>>(),
            Some(device),
        );

        let (a_buf, a_padded_buf) = (
            Buffer::<f32>::builder()
                .queue(queue.clone())
                .flags(flags::MEM_READ_ONLY)
                .len(k * m)
                .build()
                .unwrap(),
            Buffer::<f32>::builder()
                .queue(queue.clone())
                .flags(flags::MEM_READ_WRITE)
                .len(params.padded_k() * params.padded_m())
                .build()
                .unwrap(),
        );
        let (b_untransposed_buf, b_transposed_buf, b_padded_buf) = (
            Buffer::<f32>::builder()
                .queue(queue.clone())
                .flags(flags::MEM_READ_ONLY)
                .len(n * k)
                .build()
                .unwrap(),
            Buffer::<f32>::builder()
                .queue(queue.clone())
                .flags(flags::MEM_READ_WRITE)
                .len(k * n)
                .build()
                .unwrap(),
            Buffer::<f32>::builder()
                .queue(queue.clone())
                .flags(flags::MEM_READ_WRITE)
                .len(params.padded_k() * params.padded_n())
                .build()
                .unwrap(),
        );
        let (c_padded_buf, c_buf) = unsafe {
            (
                Buffer::<f32>::builder()
                    .queue(queue.clone())
                    .flags(flags::MEM_READ_WRITE)
                    .len(params.padded_m() * params.padded_n())
                    .build()
                    .unwrap(),
                Buffer::<f32>::builder()
                    .queue(queue.clone())
                    .flags(flags::MEM_WRITE_ONLY)
                    .len(m * n)
                    .use_host_slice(&out)
                    .build()
                    .unwrap(),
            )
        };

        // Build kernel for transposing B
        let transpose_kernel = Kernel::builder()
            .program(&program)
            .name("transpose")
            .queue(queue.clone())
            .local_work_size(SpatialDims::Two(params.transpose.0, params.transpose.1))
            .global_work_size(SpatialDims::Two(k, n))
            .arg(k as i32)
            .arg(n as i32)
            .arg_named("b_untransposed", &b_untransposed_buf)
            .arg_named("b_transposed", &b_transposed_buf)
            .build()
            .unwrap();

        let pad_a_kernel = Kernel::builder()
            .program(&program)
            .name("paddingAddZeroes")
            .queue(queue.clone())
            .local_work_size(params.pad_lws())
            .global_work_size(params.pad_a_gws())
            .arg(m as i32)
            .arg(k as i32)
            .arg_named("a", &a_buf)
            .arg(params.padded_m() as i32)
            .arg(params.padded_k() as i32)
            .arg_named("a_padded", &a_padded_buf)
            .build()
            .unwrap();
        let pad_b_kernel = Kernel::builder()
            .program(&program)
            .name("paddingAddZeroes")
            .queue(queue.clone())
            .local_work_size(params.pad_lws())
            .global_work_size(params.pad_b_gws())
            .arg(n as i32)
            .arg(k as i32)
            .arg_named("b_transposed", &b_transposed_buf)
            .arg(params.padded_n() as i32)
            .arg(params.padded_k() as i32)
            .arg_named("b_transposed_padded", &b_padded_buf)
            .build()
            .unwrap();
        let gemm10 =
                // Build kernel for mtx_mul
                Kernel::builder()
                    .program(&program)
                    .name("myGEMM10")
                    .queue(queue.clone())
                    .local_work_size(params.lws())
                    .global_work_size(params.gws())
                    .arg(params.padded_m() as i32)
                    .arg(params.padded_n() as i32)
                    .arg(params.padded_k() as i32)
                    .arg_named("a_padded", &a_buf)
                    .arg_named("b_transposed_padded", &b_transposed_buf)
                    .arg_named("c_padded", &c_padded_buf)
                    .build()
                    .unwrap();
        let unpad_c_kernel = Kernel::builder()
            .program(&program)
            .name("paddingRemoveZeroes")
            .queue(queue.clone())
            .local_work_size(params.pad_lws())
            .global_work_size(params.unpad_c_gws())
            .arg(params.padded_m() as i32)
            .arg(params.padded_n() as i32)
            .arg_named("c_padded", &c_padded_buf)
            .arg(m as i32)
            .arg(n as i32)
            .arg_named("c", &c_buf)
            .build()
            .unwrap();
        queue.finish().unwrap();

        Gemm10WithBTransposeKernel {
            gemm10: Gemm10Kernel {
                kernel: gemm10,
                pad_a_kernel,
                pad_b_kernel,
                unpad_c_kernel,
                queue: queue,
                a_buf: a_buf,
                b_transposed_buf,
                use_host_ptr,
            },
            b_untransposed_buf,
            transpose_kernel,
        }
    }
    fn from_slices(
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b_untransposed: &[f32],
        c: &mut [f32],
        device: DeviceType,
    ) -> Gemm10WithBTransposeKernel {
        debug_assert_eq!(a.len(), k * m);
        debug_assert_eq!(b_untransposed.len(), n * k);

        let mut kernel = Gemm10WithBTransposeKernel::uninitialized(m, n, k, c, device);
        {
            let queue = &kernel.gemm10.queue;

            // Re-create buffers as use-host-ptr if necessary
            if kernel.gemm10.use_host_ptr {
                unsafe {
                    kernel.gemm10.a_buf = Buffer::<f32>::builder()
                        .queue(queue.clone())
                        .flags(flags::MEM_READ_ONLY)
                        .len(k * m)
                        .use_host_slice(&a)
                        .build()
                        .unwrap();
                    kernel.b_untransposed_buf = Buffer::<f32>::builder()
                        .queue(queue.clone())
                        .flags(flags::MEM_READ_ONLY)
                        .len(n * k)
                        .use_host_slice(&b_untransposed)
                        .build()
                        .unwrap();
                }
                kernel
                    .gemm10
                    .kernel
                    .set_arg("a", &kernel.gemm10.a_buf)
                    .unwrap();
                kernel
                    .transpose_kernel
                    .set_arg("b_untransposed", &kernel.b_untransposed_buf)
                    .unwrap();
            } else {
                kernel.set_buffers_from_slices(&a, &b_untransposed);
            }
        }

        kernel
    }

    // TODO: comment is misleading; is this how it should be implemented though?
    /// Sets input and output buffers from the slices a, b and c. Essentially free on CPU and
    /// devices with shared GPU/CPU memory.
    ///
    /// a and b are column-major (b will be transposed automatically into row-major by the algorithm)
    fn set_buffers_from_slices(&self, a: &[f32], b: &[f32]) {
        match self.gemm10.use_host_ptr {
            true => unsafe {
                cl_util::map_to_buf(&self.gemm10.a_buf, a).unwrap();
                cl_util::map_to_buf(&self.b_untransposed_buf, b).unwrap();
            },
            false => {
                self.gemm10.a_buf.write(a).enq().unwrap();
                self.b_untransposed_buf.write(b).enq().unwrap();
            }
        }
    }

    fn calculate(&self) {
        unsafe {
            self.transpose_kernel
                .cmd()
                .queue(&self.gemm10.queue)
                .enq()
                .unwrap();
            self.gemm10.calculate();
        }
    }

    fn queue(&self) -> &Queue {
        &self.gemm10.queue
    }
}

#[derive(Copy, Clone)]
struct Gemm10WithBTransposeCompileParameters {
    gemm10_params: Gemm10CompileParameters,
    transpose: (usize, usize),
}

impl Gemm10WithBTransposeCompileParameters {
    fn choose(
        m: usize,
        n: usize,
        k: usize,
        device: DeviceType,
    ) -> Gemm10WithBTransposeCompileParameters {
        let gemm10_params = Gemm10CompileParameters::choose(m, n, k, device);

        let max_lws = if device == DeviceType::CPU {
            1
        } else {
            let device = cl_util::resolve_device(Some(device));
            let dev_max_lws = device.max_wg_size().unwrap();
            dev_max_lws
        };

        // Optimal tile-size is as close to the preferred maximum work-group-size while still
        // fitting into the max work group size on GPU and 1 on CPU because no autovectorization is
        // possible for this kernel. cnugteren used hard-coded 32x32.
        let local_work_side = (max_lws as f64).sqrt() as usize;

        // Dimensions for local memory optimization
        let transpose = (min(local_work_side, k), min(local_work_side, n));

        Gemm10WithBTransposeCompileParameters {
            gemm10_params,
            transpose,
        }
    }
    fn lws(&self) -> SpatialDims {
        self.gemm10_params.lws
    }
    fn gws(&self) -> SpatialDims {
        self.gemm10_params.gws
    }
    fn padded_m(&self) -> usize {
        self.gemm10_params.padded_m
    }
    fn padded_n(&self) -> usize {
        self.gemm10_params.padded_n
    }
    fn padded_k(&self) -> usize {
        self.gemm10_params.padded_k
    }
    fn pad_lws(&self) -> SpatialDims {
        self.gemm10_params.pad_lws
    }
    fn pad_a_gws(&self) -> SpatialDims {
        self.gemm10_params.pad_a_gws
    }
    fn pad_b_gws(&self) -> SpatialDims {
        self.gemm10_params.pad_b_gws
    }
    fn unpad_c_gws(&self) -> SpatialDims {
        self.gemm10_params.unpad_c_gws
    }
    fn padding_x(&self) -> usize {
        self.gemm10_params.padding.0
    }
    fn padding_y(&self) -> usize {
        self.gemm10_params.padding.1
    }
}

impl Into<Vec<String>> for Gemm10WithBTransposeCompileParameters {
    fn into(self) -> Vec<String> {
        let mut inner: Vec<String> = self.gemm10_params.into();
        inner.push(format!("-D TRANSPOSEX={}", self.transpose.0).to_owned());
        inner.push(format!("-D TRANSPOSEY={}", self.transpose.1).to_owned());
        inner
    }
}
