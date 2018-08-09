use cl_util;
use flags::DeviceType;
use layers::Coeff;
use ocl;
use ocl::{flags, Buffer, Context, Device, Kernel, OclPrm, Platform, Program, Queue, SpatialDims};
use std::cmp::min;
use std::mem::size_of;

#[derive(Copy, Clone)]
pub struct Gemm10CompileParameters {
    // The vector-width (in number of floats)
    pub width: usize,
    // The tile-size in dimension M
    pub tsm: usize,
    // The tile-size in dimension N
    pub tsn: usize,
    // The tile-size in dimension K
    pub tsk: usize,
    // The amount of work-per-thread in dimension M
    pub wpwim: usize,
    // The amount of work-per-thread in dimension N
    pub wpwin: usize,
    pub padded_m: usize,
    pub padded_n: usize,
    pub padded_k: usize,
    pub padding: (usize, usize),
    pub pad_lws: SpatialDims,
    pub pad_a_gws: SpatialDims,
    pub pad_b_gws: SpatialDims,
    pub unpad_c_gws: SpatialDims,
    pub lws: SpatialDims,
    pub gws: SpatialDims,
}

fn ceil_div(x: usize, y: usize) -> usize {
    (x + y - 1) / y
}

impl Gemm10CompileParameters {
    pub fn choose(m: usize, n: usize, k: usize, device: DeviceType) -> Gemm10CompileParameters {
        trace!("GEMM 10 M, N, K: {}, {}, {}", m, n, k);

        // TODO: the main performance reciprocate here seems to be the amount of local memory used by the kernel
        // TODO: get local memory size (32768) on this device and fit the amount of memory used by the kernel into that

        let padding = (8, 8);

        let cache_line_size = if device == DeviceType::CPU {
            1
        } else {
            let device = cl_util::select_device(Some(device));
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
        let tsm: usize = min(cache_line_size, m);
        let tsn: usize = min(cache_line_size, n);
        let tsk: usize = 16;
        trace!("GEMM 10 tile-sizes MxNxK: {}x{}x{}", tsm, tsn, tsk);

        let wpwim: usize = min(4, tsm);
        let wpwin: usize = min(4, tsn);
        let width: usize = 16;

        let padded_k = ceil_div(k, tsk) * tsk;
        let padded_m = ceil_div(m, tsm) * tsm;
        let padded_n = ceil_div(n, tsn) * tsn;

        let pad_lws = SpatialDims::Two(padding.0, padding.1);
        let pad_a_gws = SpatialDims::Two(padded_m, padded_k);
        let pad_b_gws = SpatialDims::Two(padded_n, padded_k);
        let unpad_c_gws = SpatialDims::Two(m, n);

        let lws = SpatialDims::Two(tsm / wpwim, tsn / wpwin);
        trace!("GEMM 10 LWS: {:?}", lws);
        let gws = SpatialDims::Two(padded_m / wpwim, padded_n / wpwin);
        trace!("GEMM 10 GWS: {:?}", gws);

        let local_mem_size = {
            let device = cl_util::select_device(Some(device));
            let local_size = match device.info(ocl::enums::DeviceInfo::LocalMemSize).unwrap() {
                ocl::enums::DeviceInfoResult::LocalMemSize(x) => x,
                _ => panic!("ocl API returned incorrect result"),
            };
            local_size as usize
        };
        let local_mem_usage = 4 * 2 * tsk * tsm + 4 * tsk * tsn * 2;
        trace!(
            "GEMM 10 local memory usage: ~{}/{}",
            local_mem_usage,
            local_mem_size
        );

        Gemm10CompileParameters {
            // The vector-width (in number of floats)
            width,
            // The tile-size in dimension M
            tsm,
            // The tile-size in dimension N
            tsn,
            // The tile-size in dimension K
            tsk,
            // The amount of work-per-thread in dimension M
            wpwim,
            // The amount of work-per-thread in dimension N
            wpwin,
            padded_m,
            padded_n,
            padded_k,
            padding,
            pad_lws,
            pad_a_gws,
            pad_b_gws,
            unpad_c_gws,
            lws,
            gws,
        }
    }
}

impl Into<Vec<String>> for Gemm10CompileParameters {
    fn into(self) -> Vec<String> {
        vec![
            format!("-D WIDTH={}", self.width).to_owned(),
            format!("-D TSM={}", self.tsm).to_owned(),
            format!("-D TSN={}", self.tsn).to_owned(),
            format!("-D TSK={}", self.tsk).to_owned(),
            format!("-D WPTM={}", self.wpwim).to_owned(),
            format!("-D WPTN={}", self.wpwin).to_owned(),
            format!("-D PADDINGX={}", self.padding.0).to_owned(),
            format!("-D PADDINGY={}", self.padding.1).to_owned(),
        ]
    }
}
