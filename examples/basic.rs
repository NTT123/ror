use ror::ROR;

fn main() -> () {
    let net = ROR::new("models/mnist.onnx");

    println!("Hello, world! {:?}", net);
}
