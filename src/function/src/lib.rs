// External libraries
use std::option::Option;
use std::cmp;

use std::fmt;

// Internal libraries
extern crate gr;
use gr::*;

#[derive(Clone)]
enum OpType {
    Func(String),
    Const(usize),
    Var(usize),
    UVar(usize),
    Op(String),
    Pow(i32)
}

impl fmt::Display for OpType {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {

        let outstring = match self {
            &OpType::Func(ref s) => format!("f{}", s),
            &OpType::Const(i) => format!("c{}", i),
            &OpType::Var(i) => format!("i{}", i),
            &OpType::UVar(i) => format!("v{}", i),
            &OpType::Op(ref s) => format!("o{}", s),
            &OpType::Pow(i) => format!("p{}", i)
        };
        write!(f, "{}", outstring)
    }

}

#[derive(Clone)]
pub struct FuncObj {
    user_vars: Vec<GI>,
    constants: Vec<GI>,
    instructions: Vec<OpType>,
    max_stack: usize,
}

impl fmt::Display for FuncObj {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut s: String = String::new();
        let ref x = self.instructions;
        for i in x {
            s += format!("{}", i).as_str();
            s += " ";
        }
        write!(f, "Fo: {} {}", s, self.max_stack)
    }
}

impl FuncObj {
    pub fn call(&self, _x: &Vec<GI>) -> (GI, Option<Vec<GI>>) {
        self.interpreted(_x, &self.constants)
    }

    fn interpreted(&self, _x: &Vec<GI>, _c: &Vec<GI>) -> (GI, Option<Vec<GI>>) {
        let mut stack: Vec<GI> = Vec::with_capacity(self.max_stack);
        for inst in &self.instructions {
            assert!(stack.len() <= self.max_stack);
            match inst {
                &OpType::Func(ref s) => {
                    let op = stack.last_mut().unwrap();
                    match s.as_str() {
                        "abs" => op.abs(),
                        "sin" => op.sin(),
                        "asin"=> op.asin(),
                        "cos" => op.cos(),
                        "acos" => op.acos(),
                        "tan" => op.tan(),
                        "atan" => op.atan(),
                        "exp" => op.exp(),
                        "log" => op.log(),
                        "neg" => op.neg(),
                        "sqrt" => op.sqrt(),
                        "sinh" => op.sinh(),
                        "cosh" => op.cosh(),
                        "tanh" => op.tanh(),
                        "asinh" => op.asinh(),
                        "acosh" => op.acosh(),
                        "atanh" => op.atanh(),
                        "floor_power2" => op.floor_power2(),
                        "sym_interval" => op.sym_interval(),
                        _     => unreachable!()
                    };
                },
                &OpType::Const(i) => {
                    stack.push(self.constants[i]);
                },
                &OpType::Var(i) => {
                    stack.push(_x[i]);
                },
                &OpType::UVar(i) => {
                    stack.push(self.user_vars[i]);
                },
                &OpType::Op(ref s) => {
                    let right = stack.pop().unwrap();
                    let left = stack.last_mut().unwrap();
                    match s.as_str() {
                        "+" => left.add(right),
                        "-" => left.sub(right),
                        "*" => left.mul(right),
                        "/" => left.div(right),
                        "p" => left.powi(right),
                        "sub2" => left.sub2(right),
                        _   => unreachable!()
                    };
                },
                &OpType::Pow(exp) => {
                    let arg = stack.last_mut().unwrap();
                    arg.pow(exp);
                }
            }
        }
        if stack.len() == 0 {
            println!("Uh oh");
        }
        (stack[0], None)
    }

    pub fn new(consts: &Vec<GI>, instructions: &String, debug: bool, suffix: String) -> FuncObj {
        let mut insts = vec![];
        let mut max_stack = 0_usize;
        let mut stack_size = 0_usize;
        for inst in instructions.split(',') {
            max_stack = cmp::max(max_stack, stack_size);
            let dummy = inst.trim().to_string();
            let (first, rest) = dummy.split_at(1);
            insts.push(match first {
                "c" => { stack_size += 1;
                         OpType::Const(rest.to_string().parse::<usize>().unwrap()) },
                "i" => { stack_size += 1;
                         OpType::Var(rest.to_string().parse::<usize>().unwrap()) },
                "v" => { stack_size += 1;
                         OpType::UVar(rest.to_string().parse::<usize>().unwrap()) },
                "o" => { stack_size -= 1;
                         OpType::Op(rest.to_string()) },
                "f" => OpType::Func(rest.to_string()),
                "p" => OpType::Pow(rest.to_string().parse::<i32>().unwrap()),
                _   => panic!()
            });
        }
        FuncObj{user_vars: vec![],
                constants: consts.clone(),
                instructions: insts,
                max_stack: max_stack}
    }
}
