use wgpu::{
    Backends, BindGroupDescriptor, BindGroupEntry,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, BufferBindingType,
    BufferDescriptor, BufferUsages, CommandEncoderDescriptor, ComputePassDescriptor,
    ComputePipelineDescriptor, DeviceDescriptor, Features, InstanceDescriptor, Limits,
    Maintain, MapMode, PipelineLayoutDescriptor, PowerPreference, RequestAdapterOptions,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

use crate::network::{IN_SIZE, PARAM_COUNT};

pub struct GpuInference {
    device:      wgpu::Device,
    queue:       wgpu::Queue,
    pipeline:    wgpu::ComputePipeline,
    bind_group:  wgpu::BindGroup,
    params_buf:  wgpu::Buffer,
    obs_buf:     wgpu::Buffer,
    actions_buf: wgpu::Buffer,
    readback_buf: wgpu::Buffer,
    pub n:       usize,
}

impl GpuInference {
    pub async fn new(n: usize) -> Option<Self> {
        let instance = wgpu::Instance::new(InstanceDescriptor {
            backends: Backends::VULKAN,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await?;

        let info = adapter.get_info();
        println!("  wgpu: {} ({:?})", info.name, info.backend);

        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: None,
                    required_features: Features::empty(),
                    required_limits: Limits::default(),
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .ok()?;

        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                bgl_entry(0, true),
                bgl_entry(1, true),
                bgl_entry(2, false),
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        });

        let params_buf = device.create_buffer(&BufferDescriptor {
            label: Some("params"),
            size: (n * PARAM_COUNT * 4) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let obs_buf = device.create_buffer(&BufferDescriptor {
            label: Some("obs"),
            size: (n * IN_SIZE * 4) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let actions_buf = device.create_buffer(&BufferDescriptor {
            label: Some("actions"),
            size: (n * 4) as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let readback_buf = device.create_buffer(&BufferDescriptor {
            label: Some("readback"),
            size: (n * 4) as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[
                BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: obs_buf.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: actions_buf.as_entire_binding() },
            ],
        });

        Some(Self {
            device,
            queue,
            pipeline,
            bind_group,
            params_buf,
            obs_buf,
            actions_buf,
            readback_buf,
            n,
        })
    }

    /// Upload all individuals' weights. Call once per generation before the loop.
    pub fn upload_params(&self, params: &[Vec<f32>]) {
        let flat: Vec<f32> = params.iter().flat_map(|p| p.iter().copied()).collect();
        self.queue.write_buffer(&self.params_buf, 0, bytemuck::cast_slice(&flat));
    }

    /// Run one batch inference step. `obs_flat` is N × IN_SIZE contiguous f32s.
    /// Returns N actions (dead snakes get action 0 — caller ignores them).
    pub fn infer(&self, obs_flat: &[f32]) -> Vec<usize> {
        self.queue.write_buffer(&self.obs_buf, 0, bytemuck::cast_slice(obs_flat));

        let mut enc = self.device.create_command_encoder(&CommandEncoderDescriptor::default());
        {
            let mut pass = enc.begin_compute_pass(&ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups((self.n as u32 + 63) / 64, 1, 1);
        }
        enc.copy_buffer_to_buffer(
            &self.actions_buf, 0,
            &self.readback_buf, 0,
            (self.n * 4) as u64,
        );
        self.queue.submit(std::iter::once(enc.finish()));

        // Synchronous readback
        let slice = self.readback_buf.slice(..);
        slice.map_async(MapMode::Read, |_| {});
        self.device.poll(Maintain::Wait);

        let data = slice.get_mapped_range();
        let actions: Vec<usize> = bytemuck::cast_slice::<u8, u32>(&data)
            .iter()
            .map(|&a| a as usize)
            .collect();
        drop(data);
        self.readback_buf.unmap();
        actions
    }
}

fn bgl_entry(binding: u32, read_only: bool) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
            ty: BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
