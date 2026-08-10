fn main() {
    embed_resource::compile("four.rc", embed_resource::NONE)
        .manifest_required()
        .expect("failed to embed Windows application icon");
}
