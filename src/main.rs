extern crate "nalgebra" as na;

use std::io::BufferedReader;
use std::num::Float;
use std::num::pow;
use std::num;
use std::collections::HashMap;
use std::io::File;
use std::rand::distributions::{Exp, IndependentSample};
use na::{DVec, dot};


#[deriving(Clone)]
struct DataEntry {
    features: Vec<f64>,
    target: f64,
    feature_vector: DVec<f64>
}

struct LogisticRegression {
    dataset: Vec<DataEntry>,
    weights: DVec<f64>
}

pub fn sigmoid(x : f64) -> f64 {
    let minus_x = -x;
    1.0 / (1.0 + minus_x.exp())
}

impl LogisticRegression {
    pub fn from_data_entries(dataset: &Vec<DataEntry>) -> LogisticRegression {
        let first = dataset.get(0).unwrap();
        let len = first.features.len() + 1;
        let mut weights : DVec<f64> = DVec::new_zeros(len);
        for i in range(0, len) {
            weights[i] = 1.0;
        }

        LogisticRegression{
            dataset: dataset.clone(),
            weights: weights
        }
    }

    pub fn train(&mut self, epochs: uint) {

        for i in range(0u, epochs) {
            // Calculate Hypothesys h_i = g(w * x_i)
            let all_hypothesis : DVec<f64> = self.dataset.iter().map(|el| {
                let result : f64 = dot(&self.weights, &el.feature_vector);
                sigmoid(result)
            }).collect();

            if i % 100 == 0 {
                let all_errors : f64 = range(0, self.dataset.len()).map(|i| {
                    let h_i = all_hypothesis[i];
                    let y_i = self.dataset.get(i).unwrap().target;

                    y_i * h_i.log2() + (1.0 - y_i) * (1.0 - h_i).log2()
                }).fold(0.0, {|a, b|  a + b }) / self.dataset.len() as f64;
                println!("{}", all_errors);
            }

            // Calculate per-track delta error d = (h_i - y_i)
            let delta_errors : DVec<f64> = range(0u, self.dataset.len()).map(|i| {
                let h_i = all_hypothesis[i];
                let y_i = self.dataset.get(i).unwrap().target;
                h_i - y_i
            }).collect();

            // Calculate weight delta w_i = alpha * (d * x_i)
            let weight_updates : DVec<f64> = range(0u, self.weights.len()).map(|i| {
                let x_i : DVec<f64> = self.dataset.iter().map(|ds| {
                    ds.feature_vector[i]
                }).collect();

                0.001 * dot(&x_i, &delta_errors)
            }).collect();

            // Update weights
            self.weights = self.weights - weight_updates;
        }
    }
}

impl DataEntry {
    pub fn new(splitted_data: &Vec<&str>, target: f64) -> DataEntry {
        let features: Vec<f64> = splitted_data.iter().filter_map(|el| {
            from_str(*el)
        }).collect();

        let mut vec : DVec<f64> = DVec::new_zeros(features.len() + 1);
        vec[0] = 1.0;
        for f in range(1u, features.len()) {
            vec[f] = *features.get(f).unwrap();
        }

        DataEntry{
            features: features,
            target: target,
            feature_vector: vec,
        }
    }
}


fn main() {

    let path = Path::new("iris.data");
    let mut file = BufferedReader::new(File::open(&path));

    let mut dataset : Vec<DataEntry> = Vec::new();

    {
        let mut counter: f64 = 0.0;
        let mut target_map : HashMap<String, f64> = HashMap::new();

        for line in file.lines() {
            let data = line.unwrap();
            let trimmed_data = data.trim();
            let mut splitted_data : Vec<&str> = trimmed_data.as_slice().split(',').collect();

            let class_type = splitted_data.pop().unwrap().to_string();
            let class_target = {
                if target_map.contains_key(&class_type) {
                    target_map[class_type]
                } else {
                    target_map.insert(class_type.clone(), counter);
                    counter += 1.0;
                    target_map[class_type]
                }
            };

            {
                let kls = class_type.as_slice();
                if kls == "Iris-setosa" || kls == "Iris-versicolor" {
                    dataset.push(DataEntry::new(&splitted_data, class_target));
                }
            }
        }
    }

    let mut log_regr = LogisticRegression::from_data_entries(&dataset);
    log_regr.train(50000);
}
