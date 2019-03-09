use super::*;

/// Cnugteren's gemm version 6. A and B are column-major but B will be transposed by a kernel.
/// Minimum matrix size: 32x32
pub struct Gemm6WithBTransposeKernel {
    gemm6: Gemm6Kernel,
    b_untransposed_buf: Buffer<f32>,
    transpose_kernel: Kernel,
}

/// Cnugteren's gemm version 6. A is column-major and B is row-major. This is the more efficient
/// implementation.
/// Minimum matrix size: 32x32
pub struct Gemm6Kernel {
    kernel: Kernel,
    queue: Queue,
    a_buf: Buffer<f32>,
    b_transposed_buf: Buffer<f32>,
    use_host_ptr: bool,
}

impl OclGemm<Gemm6WithBTransposeKernel> for Gemm6WithBTransposeKernel {
    fn uninitialized(
        m: usize,
        n: usize,
        k: usize,
        out: &mut [f32],
        device: DeviceType,
    ) -> Gemm6WithBTransposeKernel {
        // Make sure the correct amount of space is reserved for the output buffer
        debug_assert_eq!(out.len(), m * n);

        // Make sure the matrix dimensions are powers of two
        debug_assert!((m & (m - 1)) == 0);
        debug_assert!((n & (n - 1)) == 0);
        debug_assert!((k & (k - 1)) == 0);

        // If Device uses RAM, use_host_ptr and mapping via address translation may be faster
        let use_host_ptr = device == DeviceType::CPU;

        let src_transpose = String::from_utf8_lossy(include_bytes!("cl/transpose.cl"));
        let src_mtx_mul = String::from_utf8_lossy(include_bytes!("cl/6_register_tiling.cl"));

        let c_params = Gemm6WithBTransposeCompileParameters::choose(m, n, device);
        let mut c_params_list: Vec<String> = c_params.into();
        c_params_list.push("-I./src/math/gemm/cl".to_owned());

        let (queue, program, _context) = cl_util::init_from_sources::<f32>(
            &[&src_transpose, &src_mtx_mul],
            &c_params_list
                .iter()
                .map(AsRef::as_ref)
                .collect::<Vec<&str>>(),
            Some(device),
        );

        let (a_buf, b_untransposed_buf, b_transposed_buf) = (
            Buffer::<f32>::builder()
                .queue(queue.clone())
                .flags(flags::MEM_READ_ONLY)
                .len(k * m)
                .build()
                .unwrap(),
            Buffer::<f32>::builder()
                .queue(queue.clone())
                .flags(flags::MEM_READ_ONLY)
                .len(n * k)
                .build()
                .unwrap(),
            Buffer::<f32>::builder()
                .queue(queue.clone())
                .flags(flags::MEM_READ_WRITE)
                .len(n * k)
                .build()
                .unwrap(),
        );
        let c_buf = unsafe {
            Buffer::<f32>::builder()
                .queue(queue.clone())
                .flags(flags::MEM_WRITE_ONLY)
                .len(m * n)
                .use_host_slice(&out)
                .build()
                .unwrap()
        };

        // Build kernel for transposing B
        let transpose_kernel = Kernel::builder()
            .program(&program)
            .name("transpose")
            .queue(queue.clone())
            .local_work_size(SpatialDims::Two(c_params.transpose.0, c_params.transpose.1))
            .global_work_size(SpatialDims::Two(k, n))
            .arg(k as i32)
            .arg(n as i32)
            .arg_named("b_untransposed", &b_untransposed_buf)
            .arg_named("b_transposed", &b_transposed_buf)
            .build()
            .unwrap();

        let lws = c_params.lws();
        let gws = c_params.gws();
        let gemm6 =
                // Build kernel for mtx_mul
                Kernel::builder()
                    .program(&program)
                    .name("myGEMM6")
                    .queue(queue.clone())
                    .local_work_size(lws)
                    .global_work_size(gws)
                    .arg(m as i32)
                    .arg(n as i32)
                    .arg(k as i32)
                    .arg_named("a", &a_buf)
                    .arg_named("b_transposed", &b_transposed_buf)
                    .arg_named("c", &c_buf)
                    .build()
                    .unwrap();
        queue.finish().unwrap();

        Gemm6WithBTransposeKernel {
            gemm6: Gemm6Kernel {
                kernel: gemm6,
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
        b: &[f32],
        c: &mut [f32],
        device: DeviceType,
    ) -> Gemm6WithBTransposeKernel {
        debug_assert_eq!(a.len(), k * m);
        debug_assert_eq!(b.len(), n * k);

        let mut kernel = Gemm6WithBTransposeKernel::uninitialized(m, n, k, c, device);
        {
            let queue = &kernel.gemm6.queue;

            // Re-create buffers as use-host-ptr if necessary
            if kernel.gemm6.use_host_ptr {
                unsafe {
                    kernel.gemm6.a_buf = Buffer::<f32>::builder()
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
                        .use_host_slice(&b)
                        .build()
                        .unwrap();
                }
                kernel
                    .gemm6
                    .kernel
                    .set_arg("a", &kernel.gemm6.a_buf)
                    .unwrap();
                kernel
                    .transpose_kernel
                    .set_arg("b_untransposed", &kernel.b_untransposed_buf)
                    .unwrap();
            } else {
                kernel.set_buffers_from_slices(&a, &b);
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
        if self.gemm6.use_host_ptr {
            unsafe {
                cl_util::map_to_buf(&self.gemm6.a_buf, a).unwrap();
                cl_util::map_to_buf(&self.b_untransposed_buf, b).unwrap();
            }
        } else {
            self.gemm6.a_buf.write(a).enq().unwrap();
            self.b_untransposed_buf.write(b).enq().unwrap();
        }
    }

    fn calculate(&self) {
        unsafe {
            self.transpose_kernel
                .cmd()
                .queue(&self.gemm6.queue)
                .enq()
                .unwrap();
            self.gemm6.calculate();
        }
    }

    fn queue(&self) -> &Queue {
        &self.gemm6.queue
    }
}

impl OclGemm<Gemm6Kernel> for Gemm6Kernel {
    fn uninitialized(
        m: usize,
        n: usize,
        k: usize,
        out: &mut [f32],
        device: DeviceType,
    ) -> Gemm6Kernel {
        // Make sure enough space is reserved for the output buffer
        debug_assert_eq!(out.len(), m * n);

        // Make sure the matrix dimensions are powers of two
        debug_assert!((m & (m - 1)) == 0);
        debug_assert!((n & (n - 1)) == 0);
        debug_assert!((k & (k - 1)) == 0);

        // If Device uses RAM, use_host_ptr and mapping via address translation may be faster
        let use_host_ptr = device == DeviceType::CPU;

        let src_mtx_mul = String::from_utf8_lossy(include_bytes!("cl/6_register_tiling.cl"));

        let c_params = Gemm6CompileParameters::choose(m, n, device);
        let mut c_params_list: Vec<String> = c_params.into();
        c_params_list.push("-I./src/math/gemm/cl".to_owned());

        let (queue, program, _context) = cl_util::init_from_sources::<f32>(
            &[&src_mtx_mul],
            &c_params_list
                .iter()
                .map(AsRef::as_ref)
                .collect::<Vec<&str>>(),
            Some(device),
        );

        let (a_buf, b_transposed_buf) = (
            Buffer::<f32>::builder()
                .queue(queue.clone())
                .flags(flags::MEM_READ_ONLY)
                .len(k * m)
                .build()
                .unwrap(),
            Buffer::<f32>::builder()
                .queue(queue.clone())
                .flags(flags::MEM_READ_ONLY)
                .len(n * k)
                .build()
                .unwrap(),
        );
        let c_buf = unsafe {
            Buffer::<f32>::builder()
                .queue(queue.clone())
                .flags(flags::MEM_WRITE_ONLY)
                .len(m * n)
                .use_host_slice(&out)
                .build()
                .unwrap()
        };

        let lws = c_params.lws;
        let gws = c_params.gws;
        let gemm6 =
                // Build kernel for mtx_mul
                Kernel::builder()
                    .program(&program)
                    .name("myGEMM6")
                    .queue(queue.clone())
                    .local_work_size(lws)
                    .global_work_size(gws)
                    .arg(m as i32)
                    .arg(n as i32)
                    .arg(k as i32)
                    .arg_named("a", &a_buf)
                    .arg_named("b_transposed", &b_transposed_buf)
                    .arg_named("c", &c_buf)
                    .build()
                    .unwrap();
        queue.finish().unwrap();

        Gemm6Kernel {
            kernel: gemm6,
            queue: queue,
            a_buf: a_buf,
            b_transposed_buf,
            use_host_ptr,
        }
    }
    fn from_slices(
        m: usize,
        n: usize,
        k: usize,
        a: &[f32],
        b_transposed: &[f32],
        c: &mut [f32],
        device: DeviceType,
    ) -> Gemm6Kernel {
        debug_assert_eq!(a.len(), k * m);
        debug_assert_eq!(b_transposed.len(), k * n);

        let mut kernel = Gemm6Kernel::uninitialized(m, n, k, c, device);
        {
            let queue = &kernel.queue;

            // Re-create buffers as use-host-ptr if necessary
            if kernel.use_host_ptr {
                unsafe {
                    kernel.a_buf = Buffer::<f32>::builder()
                        .queue(queue.clone())
                        .flags(flags::MEM_READ_ONLY)
                        .len(k * m)
                        .use_host_slice(&a)
                        .build()
                        .unwrap();
                    kernel.b_transposed_buf = Buffer::<f32>::builder()
                        .queue(queue.clone())
                        .flags(flags::MEM_READ_ONLY)
                        .len(k * n)
                        .use_host_slice(&b_transposed)
                        .build()
                        .unwrap();
                }
                kernel.kernel.set_arg("a", &kernel.a_buf).unwrap();
                kernel
                    .kernel
                    .set_arg("b_transposed", &kernel.b_transposed_buf)
                    .unwrap();
            } else {
                kernel.set_buffers_from_slices(&a, &b_transposed);
            }
        }

        kernel
    }

    /// a is column-major and b_transposed is row-major.
    fn set_buffers_from_slices(&self, a: &[f32], b_transposed: &[f32]) {
        if self.use_host_ptr {
            unsafe {
                cl_util::map_to_buf(&self.a_buf, a).unwrap();
                cl_util::map_to_buf(&self.b_transposed_buf, b_transposed).unwrap();
            }
        } else {
            self.a_buf.write(a).enq().unwrap();
            self.b_transposed_buf.write(b_transposed).enq().unwrap();
        }
    }

    fn calculate(&self) {
        unsafe {
            self.kernel.cmd().queue(&self.queue).enq().unwrap();
        }
    }

    fn queue(&self) -> &Queue {
        &self.queue
    }
}

#[derive(Copy, Clone)]
struct Gemm6CompileParameters {
    // The tile-size in dimension m
    tsm: usize,
    // The tile-size in dimension n
    tsn: usize,
    // The tile-size in dimension k
    tsk: usize,
    // The amount of work-per-work-item in dimension m
    wpwim: usize,
    // The amount of work-per-work-item in dimension n
    wpwin: usize,
    lws: SpatialDims,
    gws: SpatialDims,
}

#[derive(Copy, Clone)]
struct Gemm6WithBTransposeCompileParameters {
    gemm6_params: Gemm6CompileParameters,
    transpose: (usize, usize),
}

impl Gemm6CompileParameters {
    fn choose(m: usize, n: usize, device: DeviceType) -> Gemm6CompileParameters {
        // TODO: the main performance reciprocate here seems to be the amount of local memory used by the kernel
        // TODO: get local memory size (32768) on this device and fit the amount of memory used by the kernel into that
        // local_memory_bytes = 4 * TSK * TSM + 4 * (TSK + 2) * TSN

        let cache_line_size = if device == DeviceType::CPU {
            1
        } else {
            let device = cl_util::resolve_device(Some(device));
            let cache_line_size = match device
                .info(ocl::enums::DeviceInfo::GlobalMemCachelineSize)
                .unwrap()
            {
                ocl::enums::DeviceInfoResult::GlobalMemCachelineSize(x) => x,
                _ => panic!("ocl API returned incorrect result"),
            };
            cache_line_size as usize
        };

        // Optimal tile-size is as close to the preferred maximum work-group-size while still
        // fitting into the max work group size on GPU. cnugteren used hard-coded 128x128.
        // TODO: separate m and n here like in gemm 10, sa. test.rs for a related test case TODO
        let c_len = m * n;
        let c_side = (c_len as f64).sqrt() as usize;
        let ts = min(cache_line_size, c_side);

        let tsm: usize = ts;
        let tsn: usize = ts;
        let tsk: usize = 16;
        let wpwim: usize = min(4, tsm);
        let wpwin: usize = min(4, tsn);

        Gemm6CompileParameters {
            tsm,
            tsn,
            tsk,
            wpwim,
            wpwin,
            lws: SpatialDims::Two(tsn / wpwim, tsn / wpwin),
            gws: SpatialDims::Two(m / wpwim, n / wpwin),
        }
    }
}

impl Gemm6WithBTransposeCompileParameters {
    fn choose(m: usize, n: usize, device: DeviceType) -> Gemm6WithBTransposeCompileParameters {
        let gemm6_params = Gemm6CompileParameters::choose(m, n, device);

        let max_lws = if device == DeviceType::CPU {
            1
        } else {
            let device = cl_util::resolve_device(Some(device));
            let dev_max_lws = device.max_wg_size().unwrap().min(m * n);
            (dev_max_lws as f64).sqrt() as usize
        };

        let transpose = {
            // Optimal tile-size is as close to the preferred maximum work-group-size while still
            // fitting into the max work group size on GPU and 1 on CPU because no autovectorization is
            // possible for this kernel. cnugteren used hard-coded 32x32.
            let local_work_side = (max_lws as f64).sqrt() as usize;

            // Dimensions for local memory optimization
            (local_work_side, local_work_side)
        };

        Gemm6WithBTransposeCompileParameters {
            gemm6_params,
            transpose,
        }
    }
    fn lws(&self) -> SpatialDims {
        self.gemm6_params.lws
    }
    fn gws(&self) -> SpatialDims {
        self.gemm6_params.gws
    }
}

impl Into<Vec<String>> for Gemm6CompileParameters {
    fn into(self) -> Vec<String> {
        vec![
            format!("-D TSM={}", self.tsm).to_owned(),
            format!("-D TSN={}", self.tsn).to_owned(),
            format!("-D TSK={}", self.tsk).to_owned(),
            format!("-D WPWIM={}", self.wpwim).to_owned(),
            format!("-D WPWIN={}", self.wpwin).to_owned(),
        ]
    }
}

impl Into<Vec<String>> for Gemm6WithBTransposeCompileParameters {
    fn into(self) -> Vec<String> {
        let mut inner: Vec<String> = self.gemm6_params.into();
        inner.push(format!("-D TRANSPOSEX={}", self.transpose.0).to_owned());
        inner.push(format!("-D TRANSPOSEY={}", self.transpose.1).to_owned());
        inner
    }
}
