// vertex shader
struct Uniforms {   
    vpMat : mat4x4f,
    modelMat : mat4x4f,           
    normalMat : mat4x4f,            
};
@binding(0) @group(0) var<uniform> uniforms : Uniforms;

struct Input {
    @location(0) position: vec4f, 
    @location(1) normal: vec4f, 
    @location(2) color: vec4f,
}

struct Output {
    @builtin(position) position : vec4f,
    @location(0) vPosition : vec4f,
    @location(1) vNormal : vec4f,
    @location(2) vColor: vec4f,
};

@vertex
fn vs_main(in:Input) -> Output {    
    var output: Output;            
    let mPosition = uniforms.modelMat * in.position; 
    output.vPosition = mPosition;                  
    output.vNormal =  uniforms.normalMat * in.normal;
    output.position = uniforms.vpMat * mPosition; 
    output.vColor = in.color;              
    return output;
}