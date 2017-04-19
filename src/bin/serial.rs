/*
  Implements the basic IBBA maximization algorithm with a priority queue.
*/

use std::collections::BinaryHeap;

extern crate gelpia_utils;
use gelpia_utils::{Quple, INF, NINF, Flt};

extern crate gr;
use gr::{GI, width_box, split_box, midpoint_box, eps_tol};

extern crate function;
use function::FuncObj;

// BEGIN: MOVE THESE INTO OPTIONS PARSING FRAMEWORK
use std::env;

extern crate getopts;
use getopts::Options;
// END: MOVE THESE INTO OPTIONS PARSING FRAMEWORK


fn ibba(x_0: &Vec<GI>, e_x: Flt, e_f: Flt, f: FuncObj) -> (Flt, Vec<GI>) {
    let mut f_best_high = NINF;
    let mut f_best_low  = NINF;
    let mut best_x = x_0.clone();
    let mut q = BinaryHeap::new();
    let mut i: u32 = 0;

    let (fc, dfc) = f.call(&x_0);
    q.push(Quple{p: INF, pf: i, data: x_0.clone(), fdata: fc, dfdata: dfc, dead: false});
    while q.len() != 0 {
        let v = q.pop();
        let (ref x, fx) =
            match v {
                Some(y) => (y.data, y.fdata),
                None => panic!("wtf")
            };

//        let xw = width_box(x);
        //let fx = f.call(x);
//        let fw = fx.width();

        if fx.upper() < f_best_low ||
            width_box(x, e_x) ||
            eps_tol(fx, e_f) //|| (!contains_zero && !on_boundary)
        {
                if f_best_high < fx.upper() {
                    f_best_high = fx.upper();
                    best_x = x.clone();
                }
                continue;
            }
        else {
            let (x_s, is_split) = split_box(&x);
            for sx in x_s {
                let est = f.call(&midpoint_box(&sx)).0;
                if f_best_low < est.lower()  {
                    f_best_low = est.lower();
                }
                i += 1;
                if is_split {
                    let (fc, dfc) = f.call(&x_0);
                    q.push(Quple{p: est.upper(),
                                 pf: i,
                                 data: sx.clone(),
                                 fdata: fc,
                                 dfdata: dfc,
                                 dead: false});
                }
            }
        }
    }
//    println!("{:?}", i);
    (f_best_high, best_x)
}

// BEGIN: MOVE THESE INTO OPTIONS PARSING FRAMEWORK
fn proc_consts(consts: &String) -> Vec<GI> {
    let mut result = vec![];
    for inst in consts.split('|') {
        if inst == "" {
            continue;
        }
        result.push(GI::new_c(inst).unwrap());
    }
    result
}

// END: MOVE THESE INTO OPTIONS PARSING FRAMEWORK

fn main() {
    // BEGIN: MOVE THESE INTO OPTIONS PARSING FRAMEWORK
    let args: Vec<String> = env::args().collect();

    let mut opts = Options::new();
    opts.reqopt("c", "constants", "", "");
    opts.reqopt("f", "function", "", "");
    opts.reqopt("i", "input", "", "");
    let matches = match opts.parse(&args[1..]) {
        Ok(m) => {m},
        Err(f) => {panic!(f.to_string())}
    };
    let const_string = matches.opt_str("c").unwrap();
    let input_string = matches.opt_str("i").unwrap();
    let func_string = matches.opt_str("f").unwrap();

    let args = proc_consts(&input_string.to_string());
    let fo = FuncObj::new(&proc_consts(&const_string.to_string()),
                              &func_string.to_string(), false, "".to_string());
    
    // END: MOVE THESE INTO OPTIONS PARSING FRAMEWORK

    let (max, interval) = ibba(&args, 1e-4, 1e-4, fo.clone());
    println!("Max: {:?}", max);
    for i in 0..interval.len()  {
        println!("X{}: {}", i, interval[i].to_string());
    }
        
}
