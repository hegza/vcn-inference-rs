// TODO: rename to ClType, functions as-is
pub trait ClTypeName {
    fn cl_type_name() -> &'static str;
}
// TODO: rename to ClVecType, functions as-is
pub trait ClVecTypeName: ClTypeName {
    fn cl_vec2_type_name() -> &'static str;
    fn cl_vec4_type_name() -> &'static str;
    fn cl_vec8_type_name() -> &'static str;
    fn cl_vec16_type_name() -> &'static str;
}

macro_rules! cl_type {
    ($NativeType:ty, $cl_data_type:tt) => {
        impl ClTypeName for $NativeType {
            fn cl_type_name() -> &'static str {
                $cl_data_type
            }
        }
    };
}

macro_rules! cl_vec_type {
    ($NativeType:ty, $cl_data_type:tt) => {
        cl_type!($NativeType, $cl_data_type);
        impl ClVecTypeName for $NativeType {
            fn cl_vec2_type_name() -> &'static str {
                concat!($cl_data_type, "2")
            }
            fn cl_vec4_type_name() -> &'static str {
                concat!($cl_data_type, "4")
            }
            fn cl_vec8_type_name() -> &'static str {
                concat!($cl_data_type, "8")
            }
            fn cl_vec16_type_name() -> &'static str {
                concat!($cl_data_type, "16")
            }
        }
    };
}

cl_vec_type!(i8, "char");
cl_vec_type!(u8, "uchar");
cl_vec_type!(i16, "short");
cl_vec_type!(u16, "ushort");
cl_vec_type!(i32, "int");
cl_vec_type!(u32, "uint");
cl_vec_type!(i64, "long");
cl_vec_type!(u64, "ulong");
cl_vec_type!(f32, "float");
cl_vec_type!(f64, "double");
cl_type!(usize, "size_t");
cl_type!((), "void");
