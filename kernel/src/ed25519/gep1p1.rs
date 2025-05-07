use super::fe::Fe;
use super::gep3::GeP3;


#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub struct GeP1P1 {
    x: Fe,
    y: Fe,
    z: Fe,
    t: Fe,
}

impl GeP1P1 {
    pub fn new(x: Fe, y: Fe, z: Fe, t: Fe) -> GeP1P1 {
        GeP1P1 { x, y, z, t }
    }

    pub fn to_p3(&self) -> GeP3 {
        let x = self.x * self.t;
        let y = self.y * self.z;
        let z = self.z * self.t;
        let t = self.x * self.y;
        GeP3::new(x, y, z, t)
    }
}
