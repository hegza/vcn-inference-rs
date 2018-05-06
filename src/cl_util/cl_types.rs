pub trait ClTypeName {
    fn cl_type_name() -> &'static str;
}

/*
macro_rules! cl_type {
    ( $( $t:ty ),* ) => { ... };
}

cl_type(i8, char, true)
*/

impl ClTypeName for i8 {
    fn cl_type_name() -> &'static str {
        "char"
    }
}

impl ClTypeName for u8 {
    fn cl_type_name() -> &'static str {
        "uchar"
    }
}

impl ClTypeName for i16 {
    fn cl_type_name() -> &'static str {
        "short"
    }
}

impl ClTypeName for u16 {
    fn cl_type_name() -> &'static str {
        "ushort"
    }
}

impl ClTypeName for i32 {
    fn cl_type_name() -> &'static str {
        "int"
    }
}

impl ClTypeName for u32 {
    fn cl_type_name() -> &'static str {
        "uint"
    }
}

impl ClTypeName for i64 {
    fn cl_type_name() -> &'static str {
        "long"
    }
}

impl ClTypeName for u64 {
    fn cl_type_name() -> &'static str {
        "ulong"
    }
}

impl ClTypeName for f32 {
    fn cl_type_name() -> &'static str {
        "float"
    }
}

impl ClTypeName for f64 {
    fn cl_type_name() -> &'static str {
        "double"
    }
}

impl ClTypeName for usize {
    fn cl_type_name() -> &'static str {
        "size_t"
    }
}

impl ClTypeName for () {
    fn cl_type_name() -> &'static str {
        "void"
    }
}
