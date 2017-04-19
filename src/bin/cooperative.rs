// Cooperative optimization solver
use std::collections::BinaryHeap;
use std::io::Write;
extern crate rand;

#[macro_use(max)]
extern crate gelpia_utils;
extern crate ga;
extern crate gr;

use ga::{ea, Individual};

use gelpia_utils::{Quple, INF, NINF, Flt, Parameters, eps_tol, check_diff};

use gr::{GI, width_box, split_box, midpoint_box};

use std::sync::{Barrier, RwLock, Arc, RwLockWriteGuard};

use std::sync::atomic::{AtomicBool, Ordering};

use std::thread;

use std::time::Duration;

extern crate function;
use function::FuncObj;

extern crate args;
use args::{process_args};

extern crate time;

extern crate nix;

extern crate rayon;
use rayon::prelude::*;

use std::cmp::Ord;

#[derive(PartialEq, PartialOrd, Debug)]
pub struct f64_o(f64);

impl f64_o {
    pub fn new(value: f64) -> f64_o {
        f64_o(value)
    }
}

impl Eq for f64_o {}

impl Ord for f64_o {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.partial_cmp(other) {
            None => panic!("Comparison of {:?} with {:?} failed unexpectedly", self, other),
            Some(result) => result
        }
    }
}



/// Returns the guaranteed upperbound for the algorithm
/// from the queue.
fn get_upper_bound(q: &Vec<Item>,
                   f_best_high: f64) -> f64{
    let mut max = f_best_high;
    for qi in q.iter() {
        max = max!{max, qi.fx.upper()};
    }
    max
}

fn log_max(q: &Vec<Item>,
           f_best_low: f64,
           f_best_high: f64) {
    let max = get_upper_bound(q, f_best_high);
    let _ = writeln!(&mut std::io::stderr(),
                     "lb: {}, possible ub: {}, guaranteed ub: {}",
                     f_best_low,
                     f_best_high,
                     max);
}

#[allow(dead_code)]
fn print_q(q: &RwLockWriteGuard<BinaryHeap<Quple>>) {
    let mut lq: BinaryHeap<Quple> = (*q).clone();
    while lq.len() != 0 {
        let qi = lq.pop().unwrap();
        let (gen, v, _) = (qi.pf, qi.p, qi.fdata);
        print!("[{}, {}, {}], ", v, gen, qi.fdata.to_string());
    }
    println!("\n");
}

/// Returns a tuple (function_estimate, eval_interval)
/// # Arguments
/// * `f` - The function to evaluate with
/// * `input` - The input domain
fn est_func(f: &FuncObj, input: &Vec<GI>) -> (Flt, GI, Option<Vec<GI>>) {
    let mid = midpoint_box(input);
    let (est_m, _) = f.call(&mid);
    let (fsx, dfsx) = f.call(&input);
    let (fsx_u, _) = f.call(&input.iter()
                            .map(|&si| GI::new_p(si.upper()))
                            .collect::<Vec<_>>());
    let (fsx_l, _) = f.call(&input.iter()
                            .map(|&si| GI::new_p(si.lower()))
                            .collect::<Vec<_>>());
    let est_max = est_m.lower().max(fsx_u.lower()).max(fsx_l.lower());
    (est_max, fsx, dfsx)
}

#[derive(Clone)]
pub struct Item {
    pub x: Vec<GI>,
    pub fx: GI,
    pub dfx: Option<Vec<GI>>,
    pub iter_est: f64,
    pub dead: bool,
}

fn ibba_threadpool_wrapper(x_0: Vec<GI>, e_x: Flt, e_f: Flt, e_f_r: Flt,
                           f_bestag: Arc<RwLock<Flt>>,
                           f_best_shared: Arc<RwLock<Flt>>,
                           x_bestbb: Arc<RwLock<Vec<GI>>>,
                           b1: Arc<Barrier>, b2: Arc<Barrier>,
                           q: Arc<RwLock<BinaryHeap<Quple>>>,
                           sync: Arc<AtomicBool>, stop: Arc<AtomicBool>,
                           f: FuncObj,
                           logging: bool, max_iters: u32, threads: usize)
                           -> (Flt, Flt, Vec<GI>, Vec<Item>) {
    let config = rayon::Configuration::new().num_threads(threads);
    let pool = rayon::ThreadPool::new(config).unwrap();
    pool.install(|| ibba(x_0, e_x, e_f, e_f_r,
                         f_bestag, f_best_shared,
                         x_bestbb,
                         b1, b2, q, sync, stop, f, logging, max_iters))
}


// Returns the upper bound, the domain where this bound occurs and a status
// flag indicating whether the answer is complete for the problem.
fn ibba(x_0: Vec<GI>, e_x: Flt, e_f: Flt, e_f_r: Flt,
        f_bestag: Arc<RwLock<Flt>>,
        f_best_shared: Arc<RwLock<Flt>>,
        x_bestbb: Arc<RwLock<Vec<GI>>>,
        b1: Arc<Barrier>, b2: Arc<Barrier>,
        q: Arc<RwLock<BinaryHeap<Quple>>>,
        sync: Arc<AtomicBool>, stop: Arc<AtomicBool>,
        f: FuncObj,
        logging: bool, max_iters: u32)
        -> (Flt, Flt, Vec<GI>, Vec<Item>) {
    let mut best_x = x_0.clone();

    let mut iters: u32 = 0;
    let mut update_iters: u32 = 0;
    let (est_max, first_val, _) = est_func(&f, &x_0);

    let mut qvec = Vec::new();
    qvec.push(Item{x: x_0.clone(), fx: first_val, dfx: None, iter_est:est_max, dead: false});

    let mut thislen = 1;

    let mut f_best_low = est_max;
    let mut f_best_high = est_max;

    while thislen != 0 && !stop.load(Ordering::Acquire) {
        let oldlen = thislen;

        if max_iters != 0 && iters >= max_iters {
            break;
        }
        if sync.load(Ordering::Acquire) {
            // Ugly: Update the update thread's view of the best branch bound.
            unreachable!();
            *f_best_shared.write().unwrap() = f_best_low;
            b1.wait();
            b2.wait();
        }

        let fbl_orig = f_best_low;
        f_best_low = max!(f_best_low, *f_bestag.read().unwrap());

        if update_iters > 2048 {
            update_iters = 0;
            let guaranteed_bound = get_upper_bound(&qvec, f_best_high);
            if (guaranteed_bound - f_best_high).abs() < e_f {
                f_best_high = guaranteed_bound;
                break;
            }
        }

        if logging && fbl_orig != f_best_low {
            log_max(&qvec, f_best_low, f_best_high);
        }

        qvec.par_iter_mut().for_each(|it| mark_dead(it, &x_0, f_best_low, e_x, e_f, e_f_r));

        {
            let dead_iter = qvec.par_iter().filter(|it| it.dead);

            // update f_best_high
            if let Some(new_best_high_it) = dead_iter.max_by_key(|it| f64_o::new(it.fx.upper()) ) {
                if new_best_high_it.fx.upper() > f_best_high {
                    f_best_high = new_best_high_it.fx.upper();
                    best_x = new_best_high_it.x.clone();
                    if logging {
                        log_max(&qvec, f_best_low, f_best_high);
                    }
                }
            }
        }

        let split_vec: Vec<Item> = qvec.par_iter().filter(|it| !it.dead).flat_map(|it| split(it, &f)).collect();

        if let Some(new_best_low_it) = split_vec.par_iter().max_by_key(|it| f64_o::new(it.iter_est) ) {
            if new_best_low_it.iter_est > f_best_low {
                f_best_low = new_best_low_it.iter_est;
                *x_bestbb.write().unwrap() = new_best_low_it.x.clone();
            }
        }

        let temp: Vec<Item> = split_vec.par_iter().filter(|it| !it.dead).cloned().collect();
        temp.par_iter().cloned().collect_into(&mut qvec);
        thislen = qvec.len();

        iters += oldlen as u32;
        update_iters += oldlen as u32;
    }

    stop.store(true, Ordering::Release);
    (f_best_low, f_best_high, best_x, qvec)
}


fn mark_dead(it: &mut Item, x_0: &Vec<GI>, f_best_low: f64, e_x: f64, e_f: f64, e_f_r: f64) -> () {
    if check_diff(it.dfx.clone(), &it.x, x_0) ||
        it.fx.upper() < f_best_low ||
        width_box(&it.x, e_x) ||
        eps_tol(it.fx, it.iter_est, e_f, e_f_r)
    {
        it.dead = true;
    }
}


fn split(it: &Item, f: &FuncObj) -> Vec<Item> {
    if ! it.dead {
        let (x_s, is_split) = split_box(&it.x);
        let m = x_s.iter().map(|sx| {
            let (est_max, fsx, dfsx) = est_func(&f, &sx.clone());
            Item {x:(*sx).clone(), fx:fsx, dfx:dfsx, iter_est:est_max, dead:!is_split}
        });

        m.collect()
    } else {
        Vec::new()
    }
}

fn distance(p: &Vec<GI>, x: &Vec<GI>) -> Flt {
    let mut result = 0.0;
    for i in 0..x.len() {
        let dx = max!(x[i].lower() - p[i].lower(),
                      0.0, &p[i].lower() - x[i].upper());
        result += dx*dx;
    }
    result.sqrt()
}

fn update(q: Arc<RwLock<BinaryHeap<Quple>>>, population: Arc<RwLock<Vec<Individual>>>,
          f_best_shared: Arc<RwLock<Flt>>,
          stop: Arc<AtomicBool>, sync: Arc<AtomicBool>,
          b1: Arc<Barrier>, b2: Arc<Barrier>,
          f: FuncObj,
          upd_interval: u32,
          timeout: u32) {
    let start = time::get_time();
    let one_sec = Duration::new(1, 0);
    'out: while !stop.load(Ordering::Acquire) {
        // Timer code...
        let last_update = time::get_time();
        while upd_interval == 0 ||
            (time::get_time() - last_update).num_seconds() <= upd_interval as i64 {
                thread::sleep(one_sec);
                if timeout > 0 &&
                    (time::get_time() - start).num_seconds() >= timeout as i64 {
                        let _ = writeln!(&mut std::io::stderr(), "Stopping early...");
                        stop.store(true, Ordering::Release);
                        break 'out;
                    }
                if stop.load(Ordering::Acquire) { // Check if we've already stopped
                    break 'out;
                }
            }
        // Signal EA and IBBA threads.
        sync.store(true, Ordering::Release);

        // Wait for EA and IBBA threads to stop.
        b1.wait();
        // Do update bizness.
        let mut q_u = q.write().unwrap();
        let mut q = Vec::new();
        for qi in q_u.iter() {
            q.push(qi.clone());
        }
        let mut pop = population.write().unwrap();
        let fbest = f_best_shared.read().unwrap();

        for i in 0..pop.len() {
            let mut d_min = INF;
            let mut j = 0;
            let mut x: Vec<GI>;
                           let px: Flt = NINF;
                           let mut x_c: Vec<GI> = q[0].data.clone();
                           while j < q.len() && d_min != 0.0 {
                               x = q[j].data.clone();
                               if f.call(&x).0.upper() < *fbest {
                                   q.remove(j);
                                   continue;
                               }
                               let d = distance(&pop[i].solution, &x);

                               if d == 0.0 {d_min = 0.0; continue;}
                               else {
                                   if d < d_min {d_min = d; x_c = x.clone();}
                               }
                               j += 1;
                           }
                           if d_min == 0.0 {
                               let fp = f.call(&pop[i].solution).0.lower();
                               if px < fp {
                                   q[j] = Quple{p: fp, pf: q[j].pf,
                                                data: q[j].data.clone(),
                                                fdata: q[j].fdata,
                                                dfdata: q[j].dfdata.clone()}.clone();
                               }
                           }
                           else {
                               // Individual is outside the search region.
                               // Project it back into the nearest current search space.
                               project(&mut pop[i], &x_c, f.clone());
                           }
            }
            // Restore the q for the ibba thread.
            (*q_u) = BinaryHeap::from(q);
            // Clear sync flag.
            sync.store(false, Ordering::SeqCst);

            // Resume EA and IBBA threads.
            b2.wait();
        }
    }

    /* Projects the box x into the box x_c */
    fn project(p: &mut Individual, x_c: &Vec<GI>, f: FuncObj) {
        for i in 0..x_c.len() {
            if p.solution[i].lower() < x_c[i].lower()
                                       || p.solution[i].lower() > x_c[i].upper() {
                if x_c[i].upper() < p.solution[i].lower()
                                    {p.solution[i] = GI::new_d(x_c[i].upper(),
                                                               x_c[i].upper());}
                                    else {p.solution[i] = GI::new_d(x_c[i].lower(),
                                                                    x_c[i].lower());}
                }
            }
            p.fitness = f.call(&p.solution).0.lower();
        }

        fn main() {
            let args = process_args();

            let ref x_0 = args.domain;
            let ref fo = args.function;
            let x_err = args.x_error;
            let y_err = args.y_error;
            let y_rel = args.y_error_rel;
            let seed = args.seed;

            // Early out if there are no input variables...
            if x_0.len() == 0 {
                let result = fo.call(&x_0).0;
                println!("[[{},{}], {{}}]", result.lower(), result.upper());
                return
            }

            let q_inner: BinaryHeap<Quple> = BinaryHeap::new();
            let q = Arc::new(RwLock::new(q_inner));

            let population_inner: Vec<Individual> = Vec::new();
            let population = Arc::new(RwLock::new(population_inner));

            let b1 = Arc::new(Barrier::new(3));
            let b2 = Arc::new(Barrier::new(3));

            let sync = Arc::new(AtomicBool::new(false));
            let stop = Arc::new(AtomicBool::new(false));

            let f_bestag: Arc<RwLock<Flt>> = Arc::new(RwLock::new(NINF));
            let f_best_shared: Arc<RwLock<Flt>> = Arc::new(RwLock::new(NINF));

            let x_e = x_0.clone();
            let x_i = x_0.clone();

            let x_bestbb = Arc::new(RwLock::new(x_0.clone()));

            let ibba_thread =
            {
                let q = q.clone();
                let b1 = b1.clone();
                let b2 = b2.clone();
                let f_bestag = f_bestag.clone();
                let f_best_shared = f_best_shared.clone();
                let x_bestbb = x_bestbb.clone();
                let sync = sync.clone();
                let stop = stop.clone();
                let fo_c = fo.clone();
                let logging = args.logging;
                let iters= args.iters;
                thread::Builder::new().name("IBBA".to_string()).spawn(move || {
                    ibba_threadpool_wrapper(x_i, x_err, y_err, y_rel,
                                           f_bestag, f_best_shared,
                                           x_bestbb,
                                           b1, b2, q, sync, stop, fo_c, logging, iters,
                                           8)// testing now, needs to be an arg
                })};

            let ea_thread =
            {
                let population = population.clone();
                let f_bestag = f_bestag.clone();
                let x_bestbb = x_bestbb.clone();
                let sync = sync.clone();
                let stop = stop.clone();
                let b1 = b1.clone();
                let b2 = b2.clone();
                let fo_c = fo.clone();
                let factor = x_e.len();
                thread::Builder::new().name("EA".to_string()).spawn(move || {
                    ea(x_e, Parameters{population: 50*factor, //1000,
                                       selection: 8, //4,
                                       elitism: 5, //2,
                                       mutation: 0.4_f64,//0.3_f64,
                                       crossover: 0.0_f64, // 0.5_f64
                                       seed:  seed,
                                       threads: 8, // testing now, needs to be an arg
                    },
                       population,
                       f_bestag,
                       x_bestbb,
                       b1, b2,
                       stop, sync, fo_c)
                })};

            // pending finding out how to kill threads
            //let update_thread =
            {
                let q = q.clone();
                let population = population.clone();
                let f_best_shared = f_best_shared.clone();
                let sync = sync.clone();
                let stop = stop.clone();
                let b1 = b1.clone();
                let b2 = b2.clone();
                let fo_c = fo.clone();
                let to = args.timeout.clone();
                let ui = args.update_interval.clone();
                let _ = thread::Builder::new().name("Update".to_string()).spawn(move || {
                    update(q, population, f_best_shared, stop, sync, b1, b2, fo_c, ui, to)
                });};

            let result = ibba_thread.unwrap().join();


            if result.is_ok() {
                let (min, mut max, mut interval, mut qvec) = result.unwrap();
                // Go through all remaining intervals from IBBA to find the true
                // max
                            // update f_best_high
                if let Some(new_best_high_it) = qvec.par_iter().max_by_key(|it| f64_o::new(it.fx.upper()) ) {
                    if new_best_high_it.fx.upper() > max {
                        max = new_best_high_it.fx.upper();
                        interval = new_best_high_it.x.clone();
                    }
                }
                
                // let ref lq = q.read().unwrap();
                // for i in qvec.iter() {
                //     let ref top = *i;
                //     let (ub, dom) = (top.fdata.upper(), &top.data);
                //     if ub > max {
                //     max = ub;
                //     interval = dom.clone();
                // }
            // }
            println!("[[{},{}], {{", min, max);
            for i in 0..args.names.len() {
                println!("'{}' : {},", args.names[i], interval[i].to_string());
            }
            println!("}}]");

        }
        else {println!("error")}

        nix::sys::signal::kill(nix::unistd::getpid(), nix::sys::signal::SIGINT);

        // We don't need an answer from this...
        let _ea_result = ea_thread.unwrap().join();
    }
