//#![feature(static_mutex)]
#![feature(float_extras)]
#![allow(improper_ctypes)]

extern crate libc;
use libc::{c_double, c_char, c_int};
use std::ffi::{CString, CStr};
use std::ops::{Add, Mul, Sub, Div, Neg};
use std::f64::NEG_INFINITY as NINF;
use std::mem;
extern crate simd;

extern crate mpi;
use mpi::traits::*;
use mpi::datatype::UserDatatype;



pub type CInterval = simd::x86::sse2::f64x2;

// Structure holding a GAOL interval.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct gaol_int {
    pub data: CInterval
}

// Functions exported from the C GAOL wrapper.
#[link(name="rustgaol", kind="dylib")]
#[link(name="gaol", kind="dylib")]
#[link(name="gdtoa", kind="dylib")]
#[link(name="crlibm", kind="dylib")]
extern {
    // All constructors and non inplace functions return an allocated
    // interval. These resources must be freed.

    // Constructs an interval from two doubles
    fn make_interval_dd(a: c_double, b: c_double, out: *mut gaol_int);

    // Constructs an interval from a point
    fn make_interval_d(x: c_double, out: *mut gaol_int);

    // Constructs an interval from two strings
    fn make_interval_ss(inf: *const c_char,
                        sup: *const c_char,
                        out: *mut gaol_int, success: *mut c_char);
    // Constructs an interval from one string, using the single string
    // constructor. See the GAOL documentation.

    fn make_interval_s(x: *const c_char, out: *mut gaol_int,
                       success: *mut c_char);
    // Creates a clone of a GAOL interval

    //fn make_interval_i(x: *const gaol_int, out: *mut gaol_int);

    // Returns the empty GAOL interval

    fn make_interval_e() -> gaol_int;

    // Prints a GAOL interval using C++.
//    fn print(a: *const gaol_int);

    // Deletes a GAOL interval.
//    fn del_int(a: gaol_int);

    // Returns the sum of a and b as a new interval.

    fn add(a: *const gaol_int, b: *const gaol_int, out: *mut gaol_int);

    // Returns a = a + b

    fn iadd(a: *mut gaol_int, b: *const gaol_int);

    // Returns a - b as a new interval.

    fn sub(a: *const gaol_int, b: *const gaol_int, out: *mut gaol_int);

    // Returns a = a - b.

    fn isub(a: *mut gaol_int, b: *const gaol_int);

    // Returns a * b as a new interval.

    fn mul(a: *const gaol_int, b: *const gaol_int, out: *mut gaol_int);

    // Returns a = a * b.

    fn imul(a: *mut gaol_int, b: *const gaol_int);

    // Returns a / b as a new interval.

    fn div_g(a: *const gaol_int, b: *const gaol_int, out: *mut gaol_int);

    // Returns a = a / b.

    fn idiv_g(a: *mut gaol_int, b: *const gaol_int);

    // Returns -a as a new interval.

    fn neg_g(a: *const gaol_int, out: *mut gaol_int);

    // Returns a = -a.

    fn ineg_g(a: *mut gaol_int);

    // Returns sin(a) as a new interval.

    fn sin_g(a: *const gaol_int, out: *mut gaol_int);
    // Returns a = sin(a).
    fn asin_g(a: *const gaol_int, out: *mut gaol_int);
    // Returns a = asin(a).

    fn isin_g(a: *mut gaol_int);
    fn iasin_g(a: *mut gaol_int);

    fn sqrt_g(a: *const gaol_int, out: *mut gaol_int);
    // Returns a = sin(a).

    fn isqrt_g(a: *mut gaol_int);
    // Returns cos(a) as a new interval.

    fn cos_g(a: *const gaol_int, out: *mut gaol_int);
    // Returns a = cos(a).

    fn acos_g(a: *const gaol_int, out: *mut gaol_int);
    // Returns a = acos(a).

    fn icos_g(a: *mut gaol_int);
    fn iacos_g(a: *mut gaol_int);


    // Returns tan(a) as a new interval.
    fn tan_g(a: *const gaol_int, out: *mut gaol_int);
    // Returns a = tan(a).

    fn atan_g(a: *const gaol_int, out: *mut gaol_int);
    // Returns a = atan(a).

    fn itan_g(a: *mut gaol_int);
    fn iatan_g(a: *mut gaol_int);

    // Returns e^a as a new interval.

    fn exp_g(a: *const gaol_int, out: *mut gaol_int);
    // Returns a = e^a.

    fn iexp_g(a: *mut gaol_int);

    // Returns ln(a) as a new interval.

    fn log_g(a: *const gaol_int, out: *mut gaol_int);
    // Returns a = ln(a).

    fn ilog_g(a: *mut gaol_int);


    fn abs_g(x: *const gaol_int, out: *mut gaol_int);


    fn iabs_g(x: *mut gaol_int);

    fn dabs_g(x: *const gaol_int, out: *mut gaol_int);


    fn idabs_g(x: *mut gaol_int);

    // Returns a^b as a new interval.

    fn pow_ig(a: *const gaol_int, b: c_int, out: *mut gaol_int);
    // Returns a = a^b.

    fn ipow_ig(a: *mut gaol_int, b: c_int);

    // Returns a^b with b an interval as a new interval.

    fn pow_vg(a: *const gaol_int, b: *const gaol_int, out: *mut gaol_int);
    // Returns a = a^b where be is an interval.

    fn ipow_vg(a: *mut gaol_int, b: *const gaol_int);


    // Hyperbolic functions
    fn sinh_g(a: *const gaol_int, out: *mut gaol_int);
    // Returns a = sinh(a).
    fn asinh_g(a: *const gaol_int, out: *mut gaol_int);
    // Returns a = asinh(a).

    fn isinh_g(a: *mut gaol_int);
    fn iasinh_g(a: *mut gaol_int);


    fn cosh_g(a: *const gaol_int, out: *mut gaol_int);
    // Returns a = cosh(a).
    fn acosh_g(a: *const gaol_int, out: *mut gaol_int);
    // Returns a = acosh(a).

    fn icosh_g(a: *mut gaol_int);
    fn iacosh_g(a: *mut gaol_int);


    fn tanh_g(a: *const gaol_int, out: *mut gaol_int);
    // Returns a = tanh(a).
    fn atanh_g(a: *const gaol_int, out: *mut gaol_int);
    // Returns a = atanh(a).

    fn itanh_g(a: *mut gaol_int);
    fn iatanh_g(a: *mut gaol_int);

    fn fp2_g(a: *const gaol_int, out: *mut gaol_int);
    fn ifp2_g(a: *mut gaol_int);

    fn symint_g(a: *const gaol_int, out: *mut gaol_int);
    fn isymint_g(a: *mut gaol_int);

    fn sub2_g(a: *const gaol_int, b: *const gaol_int, out: *mut gaol_int);
    fn isub2_g(a: *const gaol_int, b: *const gaol_int);

    // Returns the supremum of a.

    fn upper_g(a: *const gaol_int) -> c_double;
    // Returns the infimum of a.

    fn lower_g(a: *const gaol_int) -> c_double;
    // Returns the width of a

    fn width_g(a: *const gaol_int) -> c_double;
    // Returns the midpoint of a.

    fn midpoint_g(a: *const gaol_int) -> gaol_int;
    // Sets out_1 and out_2 to two intervals of input split.

    fn split_g(input: *const gaol_int, out_1: *mut gaol_int,
               out_2: *mut gaol_int);

    // Determine if interval is empty.
    fn is_empty_g(x: *const gaol_int) -> c_char;

    // Interval predicates
    fn straddles_zero_g(x: *const gaol_int) -> c_char;
    fn is_canonoical_g(x: *const gaol_int) -> c_char;

    // Returns a string representation of the interval.
    fn to_str(a: *const gaol_int) -> *const c_char;
}

#[derive(Copy, Clone)]
pub struct GI {
    pub data: gaol_int
}

impl Equivalence for GI {
    type Out = UserDatatype;
    fn equivalent_datatype() -> Self::Out {
        UserDatatype::vector(1, std::mem::size_of::<GI>() as i32,
                             0, &u8::equivalent_datatype())
    }
}


impl GI {
    pub fn new_d(inf: f64, sup: f64) -> GI {
        let mut result = GI{data: gaol_int{data: CInterval::new(0.0, 0.0)}};
        unsafe{make_interval_dd(inf as c_double, sup as c_double, &mut result.data)};
        result
    }

    pub fn new_c(x: &str) -> Result<GI, String> {
        let mut result = GI{data: gaol_int{data: CInterval::new(0.0, 0.0)}};
        let mut success = 1 as i8;
        unsafe {
            make_interval_s(CString::new(x).unwrap().as_ptr(), &mut result.data, &mut success)
        };
        if success == 1{
            Ok(result)
        }
        else {
            Err(format!("Error evaluating expression: {}", x))
        }
    }

    pub fn new_p(x: f64) -> GI {
        let mut result = GI{data: gaol_int{data: CInterval::new(0.0, 0.0)}};
        unsafe{make_interval_d(x as c_double, &mut result.data)};
        result
    }

    pub fn new_ss(inf: &str, sup: &str) -> Result<GI, String> {
        let mut result = GI{data: gaol_int{data: CInterval::new(0.0, 0.0)}};
        let mut success = 1 as i8;
        unsafe {
            make_interval_ss(CString::new(inf).unwrap().as_ptr(),
                             CString::new(sup).unwrap().as_ptr(),
                             &mut result.data, &mut success)
        };
        if success == 1 {
            Ok(result)
        }
        else {
            Err(format!("Error evaluating expression: {}, {}", inf, sup))
        }
    }

    pub fn new_e() -> GI {
        GI{data: unsafe{make_interval_e()}}
    }

    pub fn assign(&mut self, inf: f64, sup: f64) {
        unsafe{make_interval_dd(inf as c_double, sup as c_double, &mut self.data)};
    }

    pub fn neg(&mut self) {
        unsafe{ineg_g(&mut self.data)};
    }

    pub fn add(&mut self, other: GI) {
        unsafe{iadd(&mut self.data, &other.data)};
    }

    pub fn sub(&mut self, other: GI) {
        unsafe{isub(&mut self.data, &other.data)};
    }

    pub fn mul(&mut self, other: GI) {
        unsafe{imul(&mut self.data, &other.data)};
    }

    pub fn div(&mut self, other: GI) {
        unsafe{idiv_g(&mut self.data, &other.data)};
    }

    pub fn abs(&mut self) {
        unsafe{iabs_g(&mut self.data)};
    }

    pub fn dabs(&mut self) {
        unsafe{idabs_g(&mut self.data)};
    }

    pub fn pow(&mut self, exp: i32) {
        unsafe{ipow_ig(&mut self.data, exp)};
    }

    pub fn powi(&mut self, exp: GI) {
        unsafe{ipow_vg(&mut self.data, &exp.data)};
    }

    pub fn exp(&mut self) {
        unsafe{iexp_g(&mut self.data)};
    }

    pub fn log(&mut self) {
        unsafe{ilog_g(&mut self.data)};
    }

    pub fn sin(&mut self) {
        unsafe{isin_g(&mut self.data)};
    }

    pub fn asin(&mut self) {
        unsafe{iasin_g(&mut self.data)};
    }

    pub fn sqrt(&mut self) {
        unsafe{isqrt_g(&mut self.data)};
    }

    pub fn cos(&mut self) {
        unsafe{icos_g(&mut self.data)};
    }

    pub fn tan(&mut self) {
        unsafe{itan_g(&mut self.data)};
    }

    pub fn acos(&mut self) {
        unsafe{iacos_g(&mut self.data)};
    }

    pub fn atan(&mut self) {
        unsafe{iatan_g(&mut self.data)};
    }

    // Hyperbolic functions
    pub fn sinh(&mut self) {
        unsafe{isinh_g(&mut self.data)};
    }

    pub fn asinh(&mut self) {
        unsafe{iasinh_g(&mut self.data)};
    }

    pub fn cosh(&mut self) {
        unsafe{icosh_g(&mut self.data)};
    }

    pub fn acosh(&mut self) {
        unsafe{iacosh_g(&mut self.data)};
    }

    pub fn tanh(&mut self) {
        unsafe{itanh_g(&mut self.data)};
    }

    pub fn atanh(&mut self) {
        unsafe{iatanh_g(&mut self.data)};
    }

    pub fn floor_power2(&mut self) {
        unsafe{ifp2_g(&mut self.data)};
    }

    pub fn sym_interval(&mut self) {
        unsafe{isymint_g(&mut self.data)};
    }

    pub fn sub2(&mut self, other: GI) {
        unsafe{isub2_g(&mut self.data, &other.data)};
    }

    // Auxiliary functions
    fn split(&self, out1: &mut GI, out2: &mut GI) {
        unsafe{split_g(&self.data, &mut out1.data, &mut out2.data)};
    }

    pub fn upper(&self) -> f64 {
        unsafe{upper_g(&self.data) as f64}
    }

    pub fn lower(&self) -> f64 {
        unsafe{lower_g(&self.data) as f64}
    }

    fn midpoint(&self) -> GI {
        GI{data: unsafe{midpoint_g(&self.data)}}
    }

    pub fn width(&self) -> f64 {
        unsafe{width_g(&self.data) as f64}
    }

    pub fn is_empty(&self) -> bool {
        unsafe{is_empty_g(&self.data) == 1}
    }

    pub fn straddles_zero(&self) -> bool {
        unsafe{straddles_zero_g(&self.data) == 1}
    }
    pub fn is_canonical(&self) -> bool {
        unsafe{is_canonoical_g(&self.data) == 1}
    }
}


impl Add for GI {
    type Output = GI;
    fn add(self, other: GI) -> Self{
        let mut result = GI{data: gaol_int{data: CInterval::new(0.0, 0.0)}};
        unsafe{add(&self.data, &other.data, &mut result.data)};
        result
    }
}

impl Sub for GI {
    type Output = GI;
    fn sub(self, other: GI) -> Self{
        let mut result = GI{data: gaol_int{data: CInterval::new(0.0, 0.0)}};
        unsafe{sub(&self.data, &other.data, &mut result.data)};
        result
    }
}

impl Mul for GI {
    type Output = GI;
    fn mul(self, other: GI) -> Self{
        let mut result = GI{data: gaol_int{data: CInterval::new(0.0, 0.0)}};
        unsafe{mul(&self.data, &other.data, &mut result.data)};
        result
    }
}

impl Div for GI {
    type Output = GI;
    fn div(self, other: GI) -> Self{
        let mut result = GI{data: gaol_int{data: CInterval::new(0.0, 0.0)}};
        unsafe{div_g(&self.data, &other.data, &mut result.data)};
        result
    }
}

impl Neg for GI {
    type Output = GI;
    fn neg(self) -> Self {
        let mut result = GI{data: gaol_int{data: CInterval::new(0.0, 0.0)}};
        unsafe{neg_g(&self.data, &mut result.data)};
        result
    }
}


impl ToString for GI {
    fn to_string(&self) -> String {
        let x: *const c_char = unsafe {to_str(&self.data)};
        let z = unsafe {
            let y = CStr::from_ptr(x).to_bytes();
            String::from_utf8(y.to_vec()).unwrap()
        };
        unsafe{libc::free(x as *mut libc::c_void);};
        z
    }
}


pub fn abs(x: GI) -> GI {
    let mut result = GI{data: gaol_int{data: CInterval::new(0.0, 0.0)}};
    unsafe{abs_g(&x.data, &mut result.data)};
    result
}

pub fn dabs(x: GI) -> GI {
    let mut result = GI{data: gaol_int{data: CInterval::new(0.0, 0.0)}};
    unsafe{dabs_g(&x.data, &mut result.data)};
    result
}

pub fn pow(base: GI, exp: i32) -> GI {
    let mut result = GI{data: gaol_int{data: CInterval::new(0.0, 0.0)}};
    unsafe{pow_ig(&base.data, exp, &mut result.data)};
    result
}

pub fn powi(base: GI, exp: GI) -> GI {
    let mut result = GI{data: gaol_int{data: CInterval::new(0.0, 0.0)}};
    unsafe{pow_vg(&base.data, &exp.data, &mut result.data)};
    result
}

pub fn sin(x: GI) -> GI {
    let mut result = GI{data: gaol_int{data: CInterval::new(0.0, 0.0)}};
    unsafe{sin_g(&x.data, &mut result.data)};
    result
}

pub fn asin(x: GI) -> GI {
    let mut result = GI{data: gaol_int{data: CInterval::new(0.0, 0.0)}};
    unsafe{asin_g(&x.data, &mut result.data)};
    result
}

pub fn sqrt(x: GI) -> GI {
    let mut result = GI{data: gaol_int{data: CInterval::new(0.0, 0.0)}};
    unsafe{sqrt_g(&x.data, &mut result.data)};
    result
}

pub fn cos(x: GI) -> GI {
    let mut result = GI{data: gaol_int{data: CInterval::new(0.0, 0.0)}};
    unsafe{cos_g(&x.data, &mut result.data)};
    result
}

pub fn acos(x: GI) -> GI {
    let mut result = GI{data: gaol_int{data: CInterval::new(0.0, 0.0)}};
    unsafe{acos_g(&x.data, &mut result.data)};
    result
}

pub fn tan(x: GI) -> GI {
    let mut result = GI{data: gaol_int{data: CInterval::new(0.0, 0.0)}};
    unsafe{tan_g(&x.data, &mut result.data)};
    result
}

pub fn atan(x: GI) -> GI {
    let mut result = GI{data: gaol_int{data: CInterval::new(0.0, 0.0)}};
    unsafe{atan_g(&x.data, &mut result.data)};
    result
}

// Hyperbolic functions
pub fn sinh(x: GI) -> GI {
    let mut result = GI{data: gaol_int{data: CInterval::new(0.0, 0.0)}};
    unsafe{sinh_g(&x.data, &mut result.data)};
    result
}

pub fn asinh(x: GI) -> GI {
    let mut result = GI{data: gaol_int{data: CInterval::new(0.0, 0.0)}};
    unsafe{asinh_g(&x.data, &mut result.data)};
    result
}

pub fn cosh(x: GI) -> GI {
    let mut result = GI{data: gaol_int{data: CInterval::new(0.0, 0.0)}};
    unsafe{cosh_g(&x.data, &mut result.data)};
    result
}

pub fn acosh(x: GI) -> GI {
    let mut result = GI{data: gaol_int{data: CInterval::new(0.0, 0.0)}};
    unsafe{acosh_g(&x.data, &mut result.data)};
    result
}

pub fn tanh(x: GI) -> GI {
    let mut result = GI{data: gaol_int{data: CInterval::new(0.0, 0.0)}};
    unsafe{tanh_g(&x.data, &mut result.data)};
    result
}

pub fn atanh(x: GI) -> GI {
    let mut result = GI{data: gaol_int{data: CInterval::new(0.0, 0.0)}};
    unsafe{atanh_g(&x.data, &mut result.data)};
    result
}

pub fn floor_power2(x: GI) -> GI {
    let mut result = GI{data: gaol_int{data: CInterval::new(0.0, 0.0)}};
    unsafe{fp2_g(&x.data, &mut result.data)};
    result
}

pub fn sym_interval(x: GI) -> GI {
    let mut result = GI{data: gaol_int{data: CInterval::new(0.0, 0.0)}};
    unsafe{symint_g(&x.data, &mut result.data)};
    result
}

pub fn sub2(x: GI, y: GI) -> GI {
    let mut result = GI{data: gaol_int{data: CInterval::new(0.0, 0.0)}};
    unsafe{sub2_g(&x.data, &y.data, &mut result.data)};
    result
}

pub fn exp(x: GI) -> GI {
    let mut result = GI{data: gaol_int{data: CInterval::new(0.0, 0.0)}};
    unsafe{exp_g(&x.data, &mut result.data)};
    result
}

pub fn log(x: GI) -> GI {
    let mut result = GI{data: gaol_int{data: CInterval::new(0.0, 0.0)}};
    unsafe{log_g(&x.data, &mut result.data)};
    result
}

pub fn eps_tol(widest: GI, tol: f64) -> bool {
    #[allow(deprecated)]
    let (_,exp_x,_) = widest.lower().integer_decode();
    #[allow(deprecated)]
    let (_,exp_y,_) = widest.upper().integer_decode();
    let exp = {if exp_x < exp_y {exp_y} else {exp_x}} as i64;
    let wideste = ((exp+1023) << 52) & 0x7FF0000000000000;
    let d: f64 = unsafe{mem::transmute(wideste)};
    let ww = widest.width();
    ww <= tol || ww <= d
}

pub fn widest_index(_x: &Vec<GI>) -> usize {
    let mut w = NINF;
    let mut w_index = 0;
    for i in 0.._x.len() {
        let wid = _x[i].width();
        if wid > w {
            w = wid;
            w_index = i;
        }
    }
    w_index
}


pub fn width_box(_x: &Vec<GI>, tol: f64) -> bool {
    let mut w = NINF;
    let mut widest = GI::new_e();
    for a in _x {
        let wid = a.width();
        if wid > w {
            w = wid;
            widest = *a;
        }
    }
    eps_tol(widest, tol)
}

pub fn midpoint_box(_x: &Vec<GI>) -> Vec<GI> {
    let mut result = _x.clone();
    for i in 0.._x.len() {
        result[i] = _x[i].midpoint();
    }
    result
}

fn get_next_binade(current: f64)
                   -> f64 {
    let float_bits: u64 = unsafe{mem::transmute(current)};
    let osign = float_bits >> 63;
    let oexp = (float_bits >> 52) & 0x7FF;
    let sign = if oexp==0 && osign==1 { 0 } else { osign };
    let exp = if sign==1 { oexp-1 } else { oexp+1 };
    let bin = (sign << 63) | (exp << 52);
    let d: f64 = unsafe{mem::transmute(bin)};
    d
}

pub fn split_box(_x: &Vec<GI>) -> (Vec<Vec<GI>>, bool) {
    let mut w = NINF;
    let mut w_ind: usize = 0;
    for i in 0.._x.len() {
        let wid = _x[i].width();
        if wid > w {
            w = wid;
            w_ind = i;
        }
    }

    let mut a = _x.clone();
    let mut b = _x.clone();
    let nb = get_next_binade(_x[w_ind].midpoint().lower());
    if nb < _x[w_ind].upper() {
        a[w_ind] = GI::new_d(_x[w_ind].lower(), nb);
        b[w_ind] = GI::new_d(nb, _x[w_ind].upper());
    } else {
        _x[w_ind].split(&mut a[w_ind], &mut b[w_ind]);
    }
    if a[w_ind].lower() == b[w_ind].lower() &&
        a[w_ind].upper() == b[w_ind].upper() {
            (vec![a], false)
    }
    else {
        (vec![a, b], true)
    }
}

pub fn is_empty(x: &GI) -> bool {
    x.is_empty()
}

pub fn straddles_zero(x: &GI) -> bool {
    x.straddles_zero()
}

pub fn is_canonical(x: &GI) -> bool {
    x.is_canonical()
}

pub fn func(_x: &Vec<GI>) -> GI {
    GI::new_d(0.0, 0.0)
}
