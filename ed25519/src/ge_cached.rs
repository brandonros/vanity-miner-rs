use super::fe::Fe;

#[repr(C, align(16))]
#[derive(Debug, Clone, Copy)]
pub struct GeCached {
    y_plus_x: Fe,
    y_minus_x: Fe,
    z: Fe,
    t2d: Fe,
}

impl GeCached {
    pub fn new(y_plus_x: Fe, y_minus_x: Fe, z: Fe, t2d: Fe) -> GeCached {
        GeCached { y_plus_x, y_minus_x, z, t2d }
    }

    pub const fn from_bytes_const(s: &[[u8; 32]; 4]) -> Self {
        GeCached {
            y_plus_x: Fe::from_bytes_const(&s[0]),
            y_minus_x: Fe::from_bytes_const(&s[1]),
            z: Fe::from_bytes_const(&s[2]),
            t2d: Fe::from_bytes_const(&s[3]),
        }
    }

    pub fn maybe_set(&mut self, other: &GeCached, do_swap: u8) {
        self.y_plus_x.maybe_set(&other.y_plus_x, do_swap);
        self.y_minus_x.maybe_set(&other.y_minus_x, do_swap);
        self.z.maybe_set(&other.z, do_swap);
        self.t2d.maybe_set(&other.t2d, do_swap);
    }

    pub fn y_plus_x(&self) -> Fe {
        self.y_plus_x
    }

    pub fn y_minus_x(&self) -> Fe {
        self.y_minus_x
    }

    pub fn z(&self) -> Fe {
        self.z
    }

    pub fn t2d(&self) -> Fe {
        self.t2d
    }
}