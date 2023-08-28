struct ValueBuffer{
    values: array<f32>,
}
@group(0) @binding(0) var<storage, read_write> valueBuffer : ValueBuffer;

struct IntParams {
    resolution: u32,
    funcSelection: u32,
};
@group(0) @binding(1) var<uniform> ips : IntParams;

struct FloatParams {
    animateTime: f32,
}

@group(0) @binding(2) var<uniform> fps : FloatParams;

fn positionAt(index : vec3u) -> vec3f {
    let dr = getDataRange(ips.funcSelection);
    let vmin = vec3(dr.xRange[0], dr.yRange[0], dr.zRange[0]);
    let vmax = vec3(dr.xRange[1], dr.yRange[1], dr.zRange[1]);
    let vstep = (vmax - vmin)/(f32(ips.resolution) - 1.0);
    return vmin + (vstep * vec3<f32>(index.xyz));
}

fn getIdx(id: vec3u) -> u32 {
    return id.x + ips.resolution * ( id.y + id.z * ips.resolution);
}

@compute @workgroup_size(8, 8, 8)
fn cs_main(@builtin(global_invocation_id) id : vec3u) {
    let position = positionAt(id);
    let idx = getIdx(id);
    valueBuffer.values[idx] = implicitFunc(position.x, position.y, position.z, fps.animateTime, ips.funcSelection);
}