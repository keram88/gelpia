use std::sync::{Barrier, RwLock, Arc};
use std::sync::atomic::{AtomicBool};
use std::sync::atomic::Ordering as AtOrd;
extern crate rand;
use rand::{Rng, SeedableRng, XorShiftRng};
use rand::distributions::{IndependentSample, Range};


extern crate gr;
use gr::{GI};

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

pub fn ea(x_e: Vec<GI>,
          param: Parameters,
          population: Arc<RwLock<Vec<Individual>>>,
          f_bestag: Arc<RwLock<Flt>>,
          x_bestbb: Arc<RwLock<Vec<GI>>>,
          b1: Arc<Barrier>,
          b2: Arc<Barrier>,
          stop: Arc<AtomicBool>,
          sync: Arc<AtomicBool>,
          fo_c: FuncObj) {
    // Constant function
    if x_e.len() == 0 {
        return;
    }
    let seed = param.seed;
    ea_core(&x_e, &param, &stop, &sync, &b1, &b2, &f_bestag,
            &x_bestbb, population, &fo_c, seed);
    return;
}


fn ea_core(x_e: &Vec<GI>, param: &Parameters, stop: &Arc<AtomicBool>,
           sync: &Arc<AtomicBool>, b1: &Arc<Barrier>, b2: &Arc<Barrier>,
           f_bestag: &Arc<RwLock<Flt>>,
           x_bestbb: &Arc<RwLock<Vec<GI>>>,
           population: Arc<RwLock<Vec<Individual>>>, fo_c: &FuncObj,
           seed: u32) {
    let rng_seed: u32 =
        match seed {
            0 => 3735928579,
            1 => rand::thread_rng().next_u32(),
            _ => seed,
        };
    let seed_r: [u32; 4] = [(rng_seed & 0xFF000000) >> 24,
                            (rng_seed & 0xFF0000) >> 16,
                            (rng_seed & 0xFF00) >> 8 ,
                            rng_seed & 0xFF];
    let mut rng: GARng = GARng::from_seed(seed_r);

    let dimension = Range::new(0, x_e.len());
    let mut ranges = Vec::new();
    for g in x_e {
        ranges.push(Range::new(g.lower(), g.upper()));
    }

    loop {//!stop.load(AtOrd::Acquire) {
        if sync.load(AtOrd::Acquire) {
            unreachable!();
            b1.wait();
            b2.wait();
        }
        let ref mut population = *population.write().unwrap();
        let mut extension = sample(param.population, population, fo_c, &ranges, &mut rng, stop);
        population.append(&mut extension);
        population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

        for _ in 0..100 {
            population.truncate(param.elitism);

            let mut sel_extension = sample(param.selection + param.elitism, population, fo_c, &ranges, &mut rng, stop);
            population.append(&mut sel_extension);

            let mut gen_extension = next_generation(param.population, population, fo_c, param.mutation, param.crossover, &dimension, &ranges, &mut rng);
            population.append(&mut gen_extension);

            population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

            // Report fittest of the fit.
            {
                let mut fbest = f_bestag.write().unwrap();

                *fbest = if *fbest < population[0].fitness {
                    population[0].fitness
                } else {
                    *fbest
                }
            }

            // Kill worst of the worst
            let mut ftg = Vec::new();
            {
                let bestbb = x_bestbb.read().unwrap();
                // From The Gods
                for i in 0..bestbb.len() {
                    ftg.push(Range::new(bestbb[i].lower(), bestbb[i].upper()));
                }
            }
            let worst_ind = population.len() - 1;
            population[worst_ind] = rand_individual(fo_c, &ftg, &mut rng);

            population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
        }
    }

    return;
}


fn sample(population_size: usize, population: &Vec<Individual>,
          fo_c: &FuncObj, ranges: &Vec<Range<f64>>, rng: &mut GARng,
          stop: &Arc<AtomicBool>)
          -> Vec<Individual> {
    let extension_size = population_size - population.len();
    let mut seeds = Vec::with_capacity(extension_size);
    unsafe { seeds.set_len(extension_size); }

    for i in 0..extension_size {
        seeds[i] = rng.next_u32();
    }

    seeds.par_iter().map(|s| rand_seeded_individual(fo_c, ranges, *s)).collect()
}


fn rand_seeded_individual(fo_c: &FuncObj, ranges: &Vec<Range<f64>>, seed: u32)
                          -> (Individual) {
    let seed_r: [u32; 4] = [(seed & 0xFF000000) >> 24,
                            (seed & 0xFF0000) >> 16,
                            (seed & 0xFF00) >> 8 ,
                            seed & 0xFF];
    let mut rng: GARng = GARng::from_seed(seed_r);

    rand_individual(fo_c, ranges, &mut rng)
}


fn rand_individual(fo_c: &FuncObj, ranges: &Vec<Range<f64>>, rng: &mut GARng)
                   -> (Individual) {
    let mut new_sol = Vec::new();
    for r in ranges {
        new_sol.push(GI::new_p(r.ind_sample(rng)));
    }
    let (fitness_i, _) = fo_c.call(&new_sol);
    let fitness = fitness_i.lower();

    Individual{solution:new_sol, fitness:fitness}
}


fn next_generation(population_size:usize, population: &mut Vec<Individual>,
                   fo_c: &FuncObj, mut_rate: f64, crossover: f64,
                   dimension: &Range<usize>, ranges: &Vec<Range<f64>>,
                   rng: &mut GARng)
                   -> Vec<Individual> {

    let elites = population.clone();

    let extension_size = population_size - population.len();
    let mut seeds = Vec::with_capacity(extension_size);
    unsafe { seeds.set_len(extension_size); }

    for i in 0..extension_size {
        seeds[i] = rng.next_u32();
    }

    seeds.par_iter().map(|s| seeded_next_generation_individual(&elites, fo_c, mut_rate, crossover, dimension, ranges, *s)).collect()
}


fn seeded_next_generation_individual(elites: &Vec<Individual>,
                                     fo_c: &FuncObj, mut_rate: f64, crossover: f64,
                                     dimension: &Range<usize>, ranges: &Vec<Range<f64>>,
                                     seed: u32)
                                     -> Individual {
    let seed_r: [u32; 4] = [(seed & 0xFF000000) >> 24,
                            (seed & 0xFF0000) >> 16,
                            (seed & 0xFF00) >> 8 ,
                            seed & 0xFF];
    let mut rng: GARng = GARng::from_seed(seed_r);

    if rng.gen::<f64>() < crossover {
        breed(rng.choose(&elites).unwrap(),
              rng.choose(&elites).unwrap(),
              fo_c,
              dimension,
              &mut rng)
    } else {
        mutate(rng.choose(&elites).unwrap(),
               fo_c,
               mut_rate,
               ranges,
               &mut rng)
    }
}



fn mutate(input: &Individual, fo_c: &FuncObj, mut_rate: f64,
          ranges: &Vec<Range<f64>>, rng: &mut GARng)
          -> (Individual) {
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
