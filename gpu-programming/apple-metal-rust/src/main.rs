use metal::*;
use objc2::rc::autoreleasepool;

fn main() {
    autoreleasepool(|_pool| {
        let device = Device::system_default().expect("No Metal device found");

        // Metal kernel code as a string embedded directly in Rust
        let metal_src = r#"
        #include <metal_stdlib>
        using namespace metal;

        kernel void gpu_info_kernel(device int* out [[ buffer(0) ]],
                                    uint id [[ thread_position_in_grid ]]) {
            // Just write back the thread id as demo info
            out[id] = int(id);
        }
        "#;

        let compile_options = CompileOptions::new();

        // Compile the embedded Metal kernel
        let library = device.new_library_with_source(metal_src, &compile_options)
            .expect("Failed to compile Metal kernel");

        let function = library.get_function("gpu_info_kernel", None)
            .expect("Failed to find kernel function");

        // Create pipeline
        let pipeline_descriptor = ComputePipelineDescriptor::new();
        pipeline_descriptor.set_compute_function(Some(&function));
        let pipeline_state = device.new_compute_pipeline_state(&pipeline_descriptor)
            .expect("Failed to create pipeline state");

        // Create buffers and other command queue/buffer setup as usual
        let n = 4;
        let output_buffer = device.new_buffer((n * std::mem::size_of::<i32>()) as u64, MTLResourceOptions::StorageModeShared);

        let command_queue = device.new_command_queue();
        let command_buffer = command_queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline_state);
        encoder.set_buffer(0, Some(&output_buffer), 0);

        let grid_size = MTLSize::new(n as u64, 1, 1);
        let threadgroup_size = MTLSize::new(1, 1, 1);

        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let result_ptr = output_buffer.contents() as *const i32;
        let result_slice = unsafe { std::slice::from_raw_parts(result_ptr, n) };

        println!("GPU info kernel output: {:?}", result_slice);
    });
}
