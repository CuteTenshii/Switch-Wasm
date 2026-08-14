use std::path::PathBuf;

fn main() {
    let nro = switch_core::demo::demo_nro();
    let out = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../web/assets/demo.nro");
    std::fs::create_dir_all(out.parent().unwrap()).unwrap();
    std::fs::write(&out, &nro).unwrap();
    println!(
        "wrote {} bytes to {}",
        nro.len(),
        out.canonicalize().unwrap().display()
    );
}
