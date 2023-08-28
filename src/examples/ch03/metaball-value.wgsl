struct ValueBuffer{
    values: array<f32>,
};
@group(0) @binding(0) var<storage, read_write> valueBuffer : ValueBuffer;

struct IntParams {
    resolution: u32,
    metaballCount: u32,
};
@group(0) @binding(1) var<uniform> ips : IntParams;

struct Metaball {
    position: vec3f,
    radius: f32,
    strength: f32,
    subtract: f32,
};
@group(0) @binding(2) var<storage> metaballs : array<Metaball>;

fn positionAt(index : vec3u) -> vec3f {
    let vmin = vec3(-4.0, -4.0, -4.0);
    let vmax = vec3(4.0, 4.0, 4.0);
    let vstep = (vmax - vmin)/(f32(ips.resolution) - 1.0);
    return vmin + (vstep * vec3<f32>(index.xyz));
}

fn surfaceFunc(position : vec3f) -> f32 {
    var res = 0.0;
    for(var i = 0u; i < ips.metaballCount; i = i + 1u){
        let ball = metaballs[i];
        let dist = distance(position, ball.position);
        let val = ball.strength / (0.000001 + dist * dist) - ball.subtract;
        if(val > 0.0){
            res = res + val;
        }
    }   
    return res;
}

fn getIdx(id: vec3u) -> u32 {
    return id.x + ips.resolution * ( id.y + id.z * ips.resolution);
}

@compute @workgroup_size(4, 4, 4)
fn cs_main(@builtin(global_invocation_id) id : vec3u) {
    let position = positionAt(id);
    let idx = getIdx(id);
    valueBuffer.values[idx] = surfaceFunc(position);
}