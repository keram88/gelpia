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

use std::sync::atomic::{AtomicBool, Ordering, AtomicUsize};

use std::thread;

use std::time::Duration;

extern crate function;
use function::FuncObj;

extern crate args;
use args::{process_args};

extern crate threadpool;
use threadpool::ThreadPool;
use std::sync::mpsc::channel;
extern crate time;

/// Returns the guaranteed upperbound for the algorithm
/// from the queue.
fn get_upper_bound(q: &RwLockWriteGuard<Vec<Quple>>,
                   f_best_high: f64) -> f64{
    let mut max = f_best_high;
    for qi in q.iter() {
        max = max!{max, qi.fdata.upper()};
    }
    max
}

fn log_max(q: &RwLockWriteGuard<Vec<Quple>>,
           f_best_low: f64,
           f_best_high: f64) {
    let max = get_upper_bound(q, f_best_high);
    let _ = writeln!(&mut std::io::stderr(),
                     "lb: {}, possible ub: {}, guaranteed ub: {}",
                     f_best_low,
                     f_best_high,
                     max);
}

fn print_q(q: &RwLockWriteGuard<BinaryHeap<Quple>>) {
    let mut lq: BinaryHeap<Quple> = (*q).clone();
    while lq.len() != 0 {
        let qi = lq.pop().unwrap();
        let (gen, v, fx) = (qi.pf, qi.p, qi.fdata);
        print!("[{}, {}, {}], ", v, gen, qi.fdata.to_string());
    }
    println!("\n");
}

/// Returns a tuple (function_estimate, eval_interval)
/// # Arguments
/// * `f` - The function to evaluate with
/// * `input` - The input domain
fn est_func(f: &FuncObj, input: &Vec<GI>) -> (Flt, GI, Option <Vec<GI>>) {
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

// Returns the upper bound, the domain where this bound occurs and a status
// flag indicating whether the answer is complete for the problem.
fn ibba(x_0: Vec<GI>, e_x: Flt, e_f: Flt, e_f_r: Flt,
        f_bestag: Arc<RwLock<Flt>>,
        f_best_shared: Arc<RwLock<Flt>>,
        x_bestbb: Arc<RwLock<Vec<GI>>>,
        b1: Arc<Barrier>, b2: Arc<Barrier>,
        q: Arc<RwLock<Vec<Quple>>>,
        sync: Arc<AtomicBool>, stop: Arc<AtomicBool>,
        f: FuncObj,
        logging: bool, max_iters: u32)
        -> (Flt, Flt, Vec<GI>) {
    let mut best_x = x_0.clone();

    let iters = Arc::new(AtomicUsize::new(0));
    let (est_max, first_val, _) = est_func(&f, &x_0);
    {
        q.write().unwrap().push(Quple{p: est_max, pf: 0, data: x_0.clone(),
                                      fdata: first_val, dfdata: None, dead:false});
    }
    let mut f_best_low = est_max;
    let mut f_best_high = est_max;

    let n_workers = 11;
    let n_jobs = n_workers;
    let pool = ThreadPool::new(n_workers);

    while q.read().unwrap().len() != 0 && !stop.load(Ordering::Acquire) {
        if max_iters != 0 && iters.load(Ordering::Acquire) as u32 >= max_iters {
            break;
        }
        if sync.load(Ordering::Acquire) {
            // Ugly: Update the update thread's view of the best branch bound.
            *f_best_shared.write().unwrap() = f_best_low;
            b1.wait();
            b2.wait();
        }
        {
            // Take q as writable during an iteration
            let q = q.write().unwrap();

            let fbl_orig = f_best_low;
            f_best_low = max!(f_best_low, *f_bestag.read().unwrap());

            if iters.load(Ordering::Acquire) % 2048 == 0 {
                let guaranteed_bound =
                    get_upper_bound(&q, f_best_high);
                if (guaranteed_bound - f_best_high).abs() < e_f {
                    f_best_high = guaranteed_bound;
                    break;
                }
            }

            if logging && fbl_orig != f_best_low {
                log_max(&q, f_best_low, f_best_high);
            }
        }

        let p_q_len = {
            let mut q = q.write().unwrap();
            q.sort();
            q.len()/n_workers + 1
        };

        let outer_barr = Arc::new(Barrier::new(n_workers + 1));

        let (qtx, qrx) = channel();
        let (htx, hrx) = channel();
        let (ltx, lrx) = channel();

        for i in 0..n_workers {
            let inner_barr = outer_barr.clone();
//            let elems = p_q[i].clone();
            let _f = f.clone();
            let qtx = qtx.clone();
            let htx = htx.clone();
            let ltx = ltx.clone();
            let f_bestag = f_bestag.clone();
            let iters = iters.clone();
            let lqi = q.clone();
            let x_0 = x_0.clone();
            pool.execute(move || {
                let mut l_f_best_high = f_best_high;
                let mut l_best_x = vec![];

                let mut l_f_best_low = f_best_low;
                let mut l_best_low_x = vec![];

                let mut lqo = vec![];
                let mut used = false;
                let lqi = lqi.read().unwrap();

                for j in 0..p_q_len {
                    if i + j*n_workers >= lqi.len() { break };
                    used = true;
                    let ref elem = lqi[i + j*n_workers];
                    let ref x = elem.data;
                    let ref iter_est = elem.p;
                    let ref fx = elem.fdata;
                    let ref gen = elem.pf;
                    let ref dfx = elem.dfdata;

                    if check_diff(dfx.clone(), x, &x_0) {
                        continue;
                    }

                    
                    if fx.upper() < l_f_best_low ||
                        width_box(&x, e_x) ||
                        eps_tol(*fx, *iter_est, e_f, e_f_r) {
                            if l_f_best_high < fx.upper() {
                                l_f_best_high = fx.upper();
                                l_best_x = x.clone();
                            }
                        }
                    else {
                        let (x_s, is_split) = split_box(&x);
                        for sx in x_s {
                            let (est_max, fsx, dfsx) = est_func(&_f, &sx);
                            if l_f_best_low < est_max {
                                l_f_best_low = est_max;
                                l_best_low_x = sx.clone();
                                // ltx.send((est_max, sx.clone())).unwrap();
                            }
                            iters.fetch_add(1, Ordering::Release);
                            if is_split && fsx.upper() > f_best_low &&
                                fsx.upper() > *f_bestag.read().unwrap() {
                                    lqo.push(Quple{p: est_max,
                                                   pf: gen+1,
                                                   data: sx,
                                                   fdata: fsx,
                                                   dfdata: dfsx,
                                    dead: false});
                                }
                        }
                    }
                }
                ltx.send((l_f_best_low, l_best_low_x, used)).unwrap();
                htx.send((l_f_best_high, l_best_x, used)).unwrap();
                lqo.sort();
                qtx.send(lqo).unwrap();
                inner_barr.wait();
            });
        }
        outer_barr.wait();
        drop(qtx);
        drop(htx);
        drop(ltx);

        for li in lrx.iter() {
            let (lb, lx, non_empty) = li;
            if non_empty && f_best_low < lb {
                f_best_low = lb;
                *x_bestbb.write().unwrap() = lx.clone();
            }
        }

        for hi in hrx.iter() {
            let (ub, ux, non_empty) = hi;
            if non_empty && f_best_high < ub {
                f_best_high = ub;
                best_x = ux.clone();
            }
        }
        {
            let mut lq = q.write().unwrap();
            *lq = vec![];
            for qis in qrx.iter() {
                for qi in &qis {
                    if qi.fdata.upper() > f_best_low {
                        lq.push(qi.clone());
                    }
                }
            }
        }
    }
    println!("{}", iters.load(Ordering::Acquire));
    stop.store(true, Ordering::Release);
    (f_best_low, f_best_high, best_x)
}

fn update(stop: Arc<AtomicBool>, _sync: Arc<AtomicBool>,
          _b1: Arc<Barrier>, _b2: Arc<Barrier>,
          _f: FuncObj,
          timeout: u32) {
    let start = time::get_time();
    let one_sec = Duration::new(1, 0);
    'out: while !stop.load(Ordering::Acquire) {
        // Timer code...
        thread::sleep(one_sec);
        if timeout > 0 &&
            (time::get_time() - start).num_seconds() >= timeout as i64 {
                let _ = writeln!(&mut std::io::stderr(), "Stopping early...");
                stop.store(true, Ordering::Release);
                break 'out;
            }
    }
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

    let q_inner: Vec<Quple> = Vec::new();
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
            ibba(x_i, x_err, y_err, y_rel,
                 f_bestag, f_best_shared,
                 x_bestbb,
                 b1, b2, q, sync, stop, fo_c, logging, iters)
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
                               threads: 8,
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
        let sync = sync.clone();
        let stop = stop.clone();
        let b1 = b1.clone();
        let b2 = b2.clone();
        let fo_c = fo.clone();
        let to = args.timeout.clone();
        let _ = thread::Builder::new().name("Update".to_string()).spawn(move || {
            update(stop, sync, b1, b2, fo_c, to)
        });};

    let result = ibba_thread.unwrap().join();
    let ea_result = ea_thread.unwrap().join();


    // Join EA and Update here pending stop signaling.
    if result.is_ok() {
        let (min, mut max, mut interval) = result.unwrap();
        // Go through all remaining intervals from IBBA to find the true
        // max
        let n_workers = 11;
        let n_jobs = n_workers;
        let pool = ThreadPool::new(n_workers);
        let (mtx, mrx) = channel();
        let outer_barr = Arc::new(Barrier::new(n_workers + 1));
        let p_q_len = {
            let q = q.read().unwrap();
            q.len()/n_workers + 1
        };

        
        for i in 0..n_workers {
            let inner_barr = outer_barr.clone();
            let mtx = mtx.clone();
            let q = q.clone();
            pool.execute(move || {
                let lq = q.read().unwrap();
                let mut lmax = max;
                let mut ldom: Option<Vec<GI>> = None;
                let q = q.clone();
                for j in 0..p_q_len {
                    if i + j*n_workers >= lq.len() { break };
                    let ref xn = lq[i+j*n_workers];
                    let (ub, dom) = (xn.fdata.upper(), &xn.data);
                    if ub > lmax {
                        lmax = ub;
                        ldom = Some(dom.clone());
                    }
                }
                mtx.send((lmax, ldom));
                inner_barr.wait();
            });
        }
        outer_barr.wait();
        drop(mtx);
        for x in mrx.iter() {
            let (ub, dom) = x;
            if ub > max && !dom.is_none() {
                max = ub;
                interval = dom.unwrap().clone();
            }
        }
        
        
/*        let mut lq = q.write().unwrap();
        while lq.len() != 0 {
            let ref top = lq.pop().unwrap();
            let (ub, dom) = (top.fdata.upper(), &top.data);
            if ub > max {
            max = ub;
            interval = dom.clone();
    }*/
    println!("[[{},{}], {{", min, max);
    for i in 0..args.names.len() {
        println!("'{}' : {},", args.names[i], interval[i].to_string());
    }
    println!("}}]");

}
else {println!("error")}
}
