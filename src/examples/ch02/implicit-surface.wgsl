struct Tables {
    edges: array<u32, 256>,
    tris: array<i32, 4096>,
};
@group(0) @binding(0) var<storage> tables : Tables;

struct ValueBuffer{
    values: array<f32>,
}
@group(0) @binding(1) var<storage, read> valueBuffer: ValueBuffer;

struct PositionBuffer {
    values : array<f32>,
};
@group(0) @binding(2) var<storage, read_write> positionsOut : PositionBuffer;

struct NormalBuffer {
    values : array<f32>,
};
@group(0) @binding(3) var<storage, read_write> normalsOut : NormalBuffer;

struct ColorBuffer{
    values: array<f32>,
}
@group(0) @binding(4) var<storage, read_write> colorBuffer: ColorBuffer;

struct IndexBuffer {
    tris : array<u32>,
};
@group(0) @binding(5) var<storage, read_write> indicesOut : IndexBuffer;

struct IndirectParams {
    vc : u32,
    vertexCount : atomic<u32>,
    firstVertex : u32,
    indexCount : atomic<u32>,
};
@group(0) @binding(6) var<storage, read_write> indirect : IndirectParams;

struct IntParams {
    resolution: u32,
    funcSelection: u32,
    colormapSelection: u32,
    colormapDirection: u32,
    colormapReverse: u32,
};
@group(0) @binding(7) var<uniform> ips : IntParams;

struct FloatParams {
    isolevel: f32,
    scale: f32,
};
@group(0) @binding(8) var<uniform> fps : FloatParams;

var<private> vmin: vec3f;
var<private> vmax: vec3f;
var<private> vstep: vec3f;

fn getIdx(id: vec3u) -> u32 {
    return id.x + ips.resolution * ( id.y + id.z * ips.resolution);
}

fn valueAt(index : vec3u) -> f32 {
    if (any(index > vec3(ips.resolution, ips.resolution, ips.resolution))) { return 0.0; }
    let idx = getIdx(index);
    return valueBuffer.values[idx];
}

fn positionAt(index : vec3u) -> vec3f {
    let dr = getDataRange(ips.funcSelection);
    vmin = vec3(dr.xRange[0], dr.yRange[0], dr.zRange[0]);
    vmax = vec3(dr.xRange[1], dr.yRange[1], dr.zRange[1]);
    vstep = (vmax - vmin)/(f32(ips.resolution) - 1.0);
    return vmin + (vstep * vec3<f32>(index.xyz));
}

fn normalAt(index : vec3u) -> vec3f {
    return vec3<f32>(
      valueAt(index - vec3(1u, 0u, 0u)) - valueAt(index + vec3(1u, 0u, 0u)),
      valueAt(index - vec3(0u, 1u, 0u)) - valueAt(index + vec3(0u, 1u, 0u)),
      valueAt(index - vec3(0u, 0u, 1u)) - valueAt(index + vec3(0u, 0u, 1u))
    );
}

var<private> positions : array<vec3f, 12>;
var<private> normals : array<vec3f, 12>;
var<private> colors : array<vec3f, 12>;
var<private> indices : array<u32, 12>;
var<private> cubeVerts : u32 = 0u;

fn interpX(index : u32, i : vec3u, va : f32, vb : f32) {
    let mu = (fps.isolevel - va) / (vb - va);
    positions[cubeVerts] = positionAt(i) + vec3(vstep.x * mu, 0.0, 0.0);

    let na = normalAt(i);
    let nb = normalAt(i + vec3(1u, 0u, 0u));
    normals[cubeVerts] = mix(na, nb, vec3(mu, mu, mu));

    indices[index] = cubeVerts;
    cubeVerts = cubeVerts + 1u;
}

fn interpY(index : u32, i : vec3u, va : f32, vb : f32) {
    let mu = (fps.isolevel - va) / (vb - va);
    positions[cubeVerts] = positionAt(i) + vec3(0.0, vstep.y * mu, 0.0);

    let na = normalAt(i);
    let nb = normalAt(i + vec3(0u, 1u, 0u));
    normals[cubeVerts] = mix(na, nb, vec3(mu, mu, mu));

    indices[index] = cubeVerts;
    cubeVerts = cubeVerts + 1u;
}

fn interpZ(index : u32, i : vec3u, va : f32, vb : f32) {
    let mu = (fps.isolevel - va) / (vb - va);
    positions[cubeVerts] = positionAt(i) + vec3(0.0, 0.0, vstep.z * mu);

    let na = normalAt(i);
    let nb = normalAt(i + vec3(0u, 0u, 1u));
    normals[cubeVerts] = mix(na, nb, vec3(mu, mu, mu));

    indices[index] = cubeVerts;
    cubeVerts = cubeVerts + 1u;
}

@compute @workgroup_size(8, 8, 8)
fn cs_main(@builtin(global_invocation_id) global_id : vec3u) {
    let i0 = global_id;
    let i1 = global_id + vec3(1u, 0u, 0u);
    let i2 = global_id + vec3(1u, 1u, 0u);
    let i3 = global_id + vec3(0u, 1u, 0u);
    let i4 = global_id + vec3(0u, 0u, 1u);
    let i5 = global_id + vec3(1u, 0u, 1u);
    let i6 = global_id + vec3(1u, 1u, 1u);
    let i7 = global_id + vec3(0u, 1u, 1u);

    let v0 = valueAt(i0);
    let v1 = valueAt(i1);
    let v2 = valueAt(i2);
    let v3 = valueAt(i3);
    let v4 = valueAt(i4);
    let v5 = valueAt(i5);
    let v6 = valueAt(i6);
    let v7 = valueAt(i7);

    var cubeIndex = 0u;
    if (v0 < fps.isolevel) { cubeIndex = cubeIndex | 1u; }
    if (v1 < fps.isolevel) { cubeIndex = cubeIndex | 2u; }
    if (v2 < fps.isolevel) { cubeIndex = cubeIndex | 4u; }
    if (v3 < fps.isolevel) { cubeIndex = cubeIndex | 8u; }
    if (v4 < fps.isolevel) { cubeIndex = cubeIndex | 16u; }
    if (v5 < fps.isolevel) { cubeIndex = cubeIndex | 32u; }
    if (v6 < fps.isolevel) { cubeIndex = cubeIndex | 64u; }
    if (v7 < fps.isolevel) { cubeIndex = cubeIndex | 128u; }

    let edges = tables.edges[cubeIndex];
    if ((edges & 1u) != 0u) { interpX(0u, i0, v0, v1); }
    if ((edges & 2u) != 0u) { interpY(1u, i1, v1, v2); }
    if ((edges & 4u) != 0u) { interpX(2u, i3, v3, v2); }
    if ((edges & 8u) != 0u) { interpY(3u, i0, v0, v3); }
    if ((edges & 16u) != 0u) { interpX(4u, i4, v4, v5); }
    if ((edges & 32u) != 0u) { interpY(5u, i5, v5, v6); }
    if ((edges & 64u) != 0u) { interpX(6u, i7, v7, v6); }
    if ((edges & 128u) != 0u) { interpY(7u, i4, v4, v7); }
    if ((edges & 256u) != 0u) { interpZ(8u, i0, v0, v4); }
    if ((edges & 512u) != 0u) { interpZ(9u, i1, v1, v5); }
    if ((edges & 1024u) != 0u) { interpZ(10u, i2, v2, v6); }
    if ((edges & 2048u) != 0u) { interpZ(11u, i3, v3, v7); }

    let triTableOffset = (cubeIndex << 4u) + 1u;
    let indexCount = u32(tables.tris[triTableOffset - 1u]);
    var firstVertex = atomicAdd(&indirect.vertexCount, cubeVerts);
    let bufferOffset = getIdx(global_id);
    let firstIndex = bufferOffset * 15u;

    var rmin = vmin.x;
    var rmax = vmax.x;
    if(ips.colormapDirection == 1){
        rmin = vmin.y;
        rmax = vmax.y;
    } else if(ips.colormapDirection == 2){
        rmin = vmin.z;
        rmax = vmax.z;
    } else if(ips.colormapDirection == 3){
        rmin = 0.0;
        rmax = max(max(vmax.x, vmax.y), vmax.z);
    }
    rmin *= fps.scale;
    rmax *= fps.scale;
    
    var color:vec3f;
    for (var i = 0u; i < cubeVerts; i = i + 1u) {
        positions[i] = fps.scale * positions[i];
        positionsOut.values[firstVertex*3u + i*3u] = positions[i].x;
        positionsOut.values[firstVertex*3u + i*3u + 1u] = positions[i].y;
        positionsOut.values[firstVertex*3u + i*3u + 2u] = positions[i].z;

        normalsOut.values[firstVertex*3u + i*3u] = normals[i].x;
        normalsOut.values[firstVertex*3u + i*3u + 1u] = normals[i].y;
        normalsOut.values[firstVertex*3u + i*3u + 2u] = normals[i].z;

        if(ips.colormapDirection < 3){
            color = colorLerp(ips.colormapSelection, rmin, rmax, 
                positions[i][ips.colormapDirection], ips.colormapReverse);
        } else {
            let p = positions[i];
            let r = sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
            color = colorLerp(ips.colormapSelection, rmin, rmax, r, ips.colormapReverse);
        }
        colorBuffer.values[firstVertex * 3u + i * 3u] = color.x;
        colorBuffer.values[firstVertex * 3u + i * 3u + 1u] = color.y;
        colorBuffer.values[firstVertex * 3u + i * 3u + 2u] = color.z;  
    }

    for (var i = 0u; i < indexCount; i = i + 1u) {
      let index = tables.tris[triTableOffset + i];
      indicesOut.tris[firstIndex + i] = firstVertex + indices[index];
    }

    for (var i = indexCount; i < 15u; i = i + 1u) {
      indicesOut.tris[firstIndex + i] = firstVertex;
    }
}