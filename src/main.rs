use csv::Reader;
use std::fs::File;
use ndarray::{ Array, Array1, Array2};
use linfa::Dataset;
use linfa_trees::DecisionTree;
use linfa::prelude::*;


fn get_dataset(db_path:  String) -> Dataset<f32, usize, ndarray::Dim<[usize;1]>> {
    let mut reader = Reader::from_path(db_path).unwrap();
    
    let headers = get_headers(&mut reader);
    let data = get_data(&mut reader);
    let target_index = headers.len() - 1;

    let features = headers[0..target_index].to_vec();
    let records = get_records(&data, target_index);
    let targets = get_targets(&data, target_index);

    return Dataset::new(records, targets).with_feature_names(features);
}

fn get_headers(reader: &mut Reader<File>) -> Vec<String> {
    return reader.headers().unwrap().iter().map(|r| r.to_owned()).collect();
}

fn get_records(data: &Vec<Vec<f32>>, target_index: usize) -> Array2<f32> {
    let mut records: Vec<f32> = vec![];
    for record in data.iter() {
        records.extend_from_slice( &record[0..target_index] );
    }
    return Array::from(records).into_shape((data.iter().len(),target_index)).unwrap();
}

fn get_targets(data: &Vec<Vec<f32>>, target_index: usize) -> Array1<usize> {
    let targets = data.iter().map(|record| record[target_index] as usize).collect::<Vec<usize>>();
    return Array::from( targets );
}

fn get_data(reader: &mut Reader<File>) -> Vec<Vec<f32>> {
    return reader.records().map(|r| r.unwrap().iter().map(|field| field.parse::<f32>().unwrap()).collect::<Vec<f32>>()).collect::<Vec<Vec<f32>>>();
}

fn main() {
    println!("Loading the dataset.");
    let dataset = get_dataset(String::from("./src/Database/heart.csv"));
    
    let (train, test) = dataset.split_with_ratio(0.9);
    
    let model = DecisionTree::params().fit(&train).unwrap();

    let predictions = model.predict(&test);

    println!("{:?}", predictions);
    println!("{:?}", test.targets);

}
