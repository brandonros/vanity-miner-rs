use super::ge_p1_p1::GeP1P1;
use super::fe::Fe;
#[repr(C, align(16))]

#[derive(Clone, Copy)]
pub struct GeP2 {
    x: Fe,
    y: Fe,
    z: Fe,
}

impl GeP2 {
    pub fn new(x: Fe, y: Fe, z: Fe) -> GeP2 {
        GeP2 { x, y, z }
    }

    pub fn dbl(&self) -> GeP1P1 {
        let xx = self.x.square();
        let yy = self.y.square();
        let b = self.z.square_and_double();
        let a = self.x + self.y;
        let aa = a.square();
        let y3 = yy + xx;
        let z3 = yy - xx;
        let x3 = aa - y3;
        let t3 = b - z3;

        GeP1P1::new(x3, y3, z3, t3)
    }
}