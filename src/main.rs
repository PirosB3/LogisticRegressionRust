extern crate "nalgebra" as na;

use std::io::BufferedReader;
use std::collections::HashMap;
use std::io::File;
use std::rand::distributions::{Exp, IndependentSample};
use na::DVec;


#[deriving(Clone)]
struct DataEntry {
    features: Vec<f64>,
    target: f64
}

struct LogisticRegression {
    dataset: Vec<DataEntry>,
    weights: DVec<f64>
}

pub fn vector_vector_mul(data_entry: &[f64], weights: &[f64]) -> f64 {
    let mut total: f64 = weights[0];

    let len_weights = weights.len() - 1;
    let slice_of_weights = weights.slice(1, len_weights);

    for i in range(0u, len_weights) {
        total += (data_entry[i] * weights[i]);
    }
    total
}

pub fn sigmoid(x : f64) -> f64 {
    let minus_x = -x;
    1.0 / (1.0 + minus_x.exp2())
}

impl LogisticRegression {
    pub fn from_data_entries(dataset: &Vec<DataEntry>) -> LogisticRegression {
        let first = dataset.get(0).unwrap();
        let len = first.features.len() + 1;
        let mut weights : Vec<f64> = Vec::new();
        for i in range(0, len) {
            weights.push(1.0);
        }

        LogisticRegression{
            dataset: dataset.clone(),
            weights: weights
        }
    }

    pub fn train(&mut self, epochs: int) {
        /*
        let mut last_error : Option<f64> = None;
        for i in range(0i, epochs) {
            {
                // Calculate hypothsies
                let all_hypothesies : Vec<f64> = self.dataset.iter().map(|ds| {
                    let reality = vector_vector_mul(
                        ds.feature_vector(),
                        self.weights.as_slice()
                    );
                    sigmoid(reality)
                }).collect();

                // Calculate error
                if epochs % 100 == 0 {
                    let error = range(0u, all_hypothesies.len()).map(|i| {
                        let h_i = all_hypothesies.get(i).unwrap();
                        let y_i = self.dataset.get(i).unwrap().target;
                        y_i * h_i.log2() + (1.0 - y_i) * (1.0 - *h_i).log2()
                    }).fold(0.0, |a, b| { a + b});
                    println!("Error is: {}", error / all_hypothesies.len() as f64);
                }

                // Calculate derivatives
                let weight_gradients : Vec<f64> = range(0u, self.weights.len()).map(|weight_id| {

                    range(0u, all_hypothesies.len()).map(|dataset_id| {
                        let h_id = all_hypothesies.get(dataset_id).unwrap();
                        let y_id = self.dataset.get(dataset_id).unwrap().target;

                        let x_weight_id = {
                            if weight_id == 0 {
                                1.0
                            } else {
                                *self.dataset.get(dataset_id).unwrap().feature_vector().get(weight_id-1).unwrap()
                            }
                        };

                        x_weight_id * (*h_id - y_id)
                    }).fold(0.0, |a, b| { a + b })
                }).collect();

                // Update weights
                for i in range(0u, self.weights.len()) {
                    let mut weight = self.weights.get_mut(i).unwrap();
                    *weight = *weight - (0.0001 * *weight_gradients.get(i).unwrap());
                }
            }
        }*/
    }
}

impl DataEntry {
    pub fn new(splitted_data: &Vec<&str>, target: f64) -> DataEntry {
        let features: Vec<f64> = splitted_data.iter().filter_map(|el| {
            from_str(*el)
        }).collect();
        DataEntry{
            features: features,
            target: target
        }
    }

    pub fn feature_vector(&self) -> &DVec<f64> {
        let vec : DVec<f64> = DVec::new_zeros(self.features.len() + 1);
        vec[0] = 1.0;
        for f in range(1u, self.features.len() + 1) {
            vec[f] = self.features.get(f).unwrap();
        }
        return &vec;
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
    log_regr.train(10000);
}
