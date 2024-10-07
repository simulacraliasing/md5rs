use std::array;

use ndarray::Array;

fn main() {
    // let frame = vec![
    //     255., 0., 0., 0., 255., 0.,
    //     0., 0., 255., 255., 0., 0.
    // ];
    let frame = vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
    let height = 2;
    let width = 2;
    let channels = 3;
    let array = Array::from_shape_vec((height, width, channels), frame).unwrap();
    let array = array.permuted_axes([2, 0, 1]);
    // let mut array = Array::zeros((3, 2, 2));
    // for h in 0..height {
    //     for w in 0..width {
    //         for c in 0..channels {
    //             array[[c, h, w]] = frame[h * width * channels + w * channels + c];
    //         }
    //     }
    // }
    print!("{:?}", array[[2, 1, 1]]);
}
