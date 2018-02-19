use std::fmt;
use std::str::FromStr;

#[derive(Debug, PartialEq, Eq)]
pub enum Class {
    Bus,
    NormalCar,
    Truck,
    Van,
}

impl fmt::Display for Class {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::Class::*;
        let c = match *self {
            Bus => "bus",
            NormalCar => "normalcar",
            Truck => "truck",
            Van => "van",
        };
        write!(f, "{}", c)
    }
}

impl FromStr for Class {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Class, &'static str> {
        use self::Class::*;
        match s {
            "bus" => Ok(Bus),
            "normalcar" => Ok(NormalCar),
            "truck" => Ok(Truck),
            "van" => Ok(Van),
            _ => Err("cannot convert input to any of existing classes"),
        }
    }
}
