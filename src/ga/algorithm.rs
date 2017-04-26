use std::thread;
use std::sync::{Barrier, RwLock, Arc};
use std::sync::atomic::{AtomicBool};
use std::sync::atomic::Ordering as AtOrd;

extern crate rand;
use rand::{Rng, SeedableRng, XorShiftRng};
use rand::distributions::{IndependentSample, Range};

use std::time::Duration;

extern crate mpi;
use mpi::traits::*;
use mpi::datatype::UserDatatype;
use mpi::topology::*;
use mpi::ffi::{MPI_Abort, MPI_Comm};
use mpi::raw::{AsRaw};
use mpi::request::WaitGuard;

extern crate gr;
use gr::*;

extern crate gelpia_utils;
use gelpia_utils::{Flt, Parameters};
extern crate function;
use function::FuncObj;

type GARng = XorShiftRng;

extern crate rayon;
use rayon::prelude::*;


#[derive(Clone)]
pub struct Individual {
    pub solution: Vec<GI>,
    pub fitness: Flt,
}

impl Equivalence for Individual {
    type Out = UserDatatype;
    fn equivalent_datatype() -> Self::Out {
        UserDatatype::vector(1, std::mem::size_of::<Individual>() as i32,
                             0, &u8::equivalent_datatype())
    }
}

/// Recieves broadcasted promising domains from root
fn get_x_best(x_best: Arc<RwLock<Vec<GI>>>, x_i: Vec<GI>,
              world: SystemCommunicator) {
    let root_process = world.process_at_rank(0);
    loop {
        let mut x: Vec<GI> = x_i.clone();
        root_process.broadcast_into(&mut x[..]);
        {
            *x_best.write().unwrap() = x.clone();
        }
    }
}

fn migrate(pop: &Vec<Individual>, src: i32, dest: i32,
           count: usize, comm: &SystemCommunicator,
           fo_c: &FuncObj)
           -> Vec<Individual> {
    let mut migrants_flat = Vec::new();
    let size = pop[0].solution.len();
    
    // Send flattened
    for i in 0..count {
        let ref sol = pop[i].solution;
        for x in sol {
            migrants_flat.push(x.clone());
        }
    }
    let _sreq = WaitGuard::from(comm.process_at_rank(dest)
                                .immediate_send(&migrants_flat[..]));
    let (new_migrants_flat, _) = comm.process_at_rank(src)
        .receive_vec::<GI>();

    // Expand migrants
    let mut new_migrants: Vec<Individual> = Vec::new();
    for i in 0..count {
        let mut new_sol: Vec<GI> = Vec::new();
        for j in 0..size {
            new_sol.push(new_migrants_flat[i*size + j].clone());
        }
        let fitness = fo_c.call(&new_sol).0.lower();
        new_migrants.push(Individual{ solution: new_sol, fitness: fitness });
    }
    
    new_migrants
}

fn send_hint(hint: Flt, comm: &SystemCommunicator) {
    comm.process_at_rank(0).send(&hint);
}

pub fn ea(x_e: Vec<GI>,
          param: Parameters,
          fo_c: FuncObj,
          world: &SystemCommunicator)
          -> () {
    // Constant function
    if x_e.len() == 0 {
        return;
    }

    let config = rayon::Configuration::new().num_threads(param.threads);
    let pool = rayon::ThreadPool::new(config).unwrap();
    let x_bestbb = Arc::new(RwLock::new(x_e.clone()));
    {
        let x_bestbb = x_bestbb.clone();
        let world = world.clone();
        let x_i = x_e.clone();
        thread::Builder::new()
            .name("X-Best-RX".to_string())
            .spawn(move || get_x_best(x_bestbb, x_i, world));
    }
    pool.install(|| ea_core(&x_e, &param, &fo_c, &x_bestbb, world));
}


fn ea_core(x_e: &Vec<GI>,
           param: &Parameters,
           fo_c: &FuncObj,
           x_bestbb: &Arc<RwLock<Vec<GI>>>,
           world: &SystemCommunicator)
           -> () {
    let rank = world.rank();
    let size = world.size();
    let next_rank = if rank < world.size() - 1 { rank + 1 } else { 1 };
    let prev_rank = if rank == 1 { world.size() -1 } else { rank - 1 };
    let seed: u32 =
        match param.seed {
            0 => 3735928579,
            1 => rand::thread_rng().next_u32(),
            other => other,
        } + ((rank*17) as u32);
    
    let seed_split: [u32; 4] = [(seed & 0xFF000000) >> 24,
                                (seed & 0xFF0000) >> 16,
                                (seed & 0xFF00) >> 8 ,
                                (seed & 0xFF)];
    
    let mut rng: GARng = GARng::from_seed(seed_split);

    let mut rngs = [0..param.population].iter().map(|_| {
        let seed = rng.next_u32();
        let seed_split: [u32; 4] = [(seed & 0xFF000000) >> 24,
                                    (seed & 0xFF0000) >> 16,
                                    (seed & 0xFF00) >> 8 ,
                                    (seed & 0xFF)];
        GARng::from_seed(seed_split)
    }).collect();

    let dimension = Range::new(0, x_e.len());
    let ranges = x_e.iter().map(|g| Range::new(g.lower(), g.upper())).collect();
    let mut population = Vec::new();
    let mut elites = Vec::new();

    let addition_size = param.population - population.len();

    for _ in 0..addition_size {
        let ind = rand_individual(fo_c, &ranges, &mut rng);
        population.push(ind.clone());
    }

    population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

    for i in 0..param.elitism {
        let ind = population[i].clone();
        elites.push(ind);
    }
    let mut counter = 1;
    loop {
        {
            elites.par_iter_mut()
                .zip(population.par_iter().take(param.elitism))
                .for_each(|(dest, src)| *dest = src.clone())
        }
        let kept = param.elitism;

        let kept_new = kept + param.selection;
        inplace_new_addition(&mut population, &mut rngs,
                             kept, kept_new, fo_c, &ranges);

        let kept_new_bred = param.population;
        inplace_next_generation(&mut population, &mut rngs, &elites,
                                kept_new, kept_new_bred, fo_c, param.mutation,
                                param.crossover, &dimension, &ranges);

        population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

        send_hint(population[0].fitness, world);

        // Report fittest of the fit.
        {
            // let mut fbest = f_bestag.write().unwrap();

            // *fbest = if *fbest < population[0].fitness {
            //     population[0].fitness
            // } else {
            //     *fbest
            // }
        }

        if counter % 1000 == 0 && size > 2 {
            let mut migrants = migrate(&population, prev_rank,
                                   next_rank, 7, world, &fo_c);
            population.append(&mut migrants);
            population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
            population.truncate(param.population);
        }

        // Kill worst of the worst
        {
            let bestbb = x_bestbb.read().unwrap();
            let ftg = bestbb.iter().map(|g| Range::new(g.lower(), g.upper())).collect();
            let worst_ind = population.len() - 1;
            population[worst_ind] = rand_individual(fo_c, &ftg, &mut rng);
        }

        population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
        counter += 1;
    }

    return;
}


fn inplace_new_addition(population: &mut Vec<Individual>, rngs: &mut Vec<GARng>,
                        low: usize, high: usize,
                        fo_c: &FuncObj, ranges: &Vec<Range<f64>>)
                        -> () {
    let span = high - low;
    let pop_slice = population.par_iter_mut().skip(low).take(span);
    let rng_slice = rngs.par_iter_mut().skip(low).take(span);
    pop_slice.zip(rng_slice).for_each(|(dest, rng)| {
        *dest = rand_individual(fo_c, ranges, rng);
    });
}

fn rand_individual(fo_c: &FuncObj, ranges: &Vec<Range<f64>>, rng: &mut GARng)
                   -> Individual {
    let new_sol = ranges.iter()
        .map(|r| GI::new_p(r.ind_sample(rng)))
        .collect();
    let (fitness_i, _) = fo_c.call(&new_sol);
    let fitness = fitness_i.lower();

    Individual{solution:new_sol, fitness:fitness}
}


fn inplace_next_generation(population: &mut Vec<Individual>, rngs: &mut Vec<GARng>, elites: &Vec<Individual>,
                           low: usize, high: usize,
                           fo_c: &FuncObj, mut_rate: f64, crossover: f64,
                           dimension: &Range<usize>, ranges: &Vec<Range<f64>>)
                           -> () {
    let span = high - low;
    let pop_slice = population.par_iter_mut().skip(low).take(span);
    let rng_slice = rngs.par_iter_mut().skip(low).take(span);
    pop_slice.zip(rng_slice).for_each(|(dest, rng)| {
        *dest =
            if rng.gen::<f64>() < crossover {
                breed(rng.choose(elites).unwrap(), rng.choose(elites).unwrap(),
                      fo_c, dimension, rng)
            } else {
                mutate(rng.choose(&elites).unwrap(),
                       fo_c, mut_rate, ranges, rng)
            }
    });
}


fn mutate(input: &Individual, fo_c: &FuncObj, mut_rate: f64,
          ranges: &Vec<Range<f64>>, rng: &mut GARng)
          -> Individual {
    let mut output_sol = Vec::new();

    for (r, &ind) in ranges.iter().zip(input.solution.iter()) {
        output_sol.push(
            if rng.gen::<f64>() < mut_rate {
                ind
            } else {
                GI::new_p(r.ind_sample(&mut *rng))
            });
    }

    let (fitness_i, _) = fo_c.call(&output_sol);
    let fitness = fitness_i.lower();

    Individual{solution: output_sol, fitness: fitness}
}


fn breed(parent1: &Individual, parent2: &Individual, fo_c: &FuncObj,
         dimension: &Range<usize>, rng: &mut GARng) -> (Individual) {
    let mut child = parent1.clone();
    let crossover_point = dimension.ind_sample(rng);
    child.solution.truncate(crossover_point);
    let mut rest = parent2.clone().solution.split_off(crossover_point);
    child.solution.append(&mut rest);
    let (fitness_i, _) = fo_c.call(&child.solution);
    child.fitness = fitness_i.lower();
    child
}
