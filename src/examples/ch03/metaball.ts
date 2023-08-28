import vsShader from '../../common/shader-vert.wgsl';
import fsShader from '../../common/shader-frag.wgsl';
import csColormapShader from '../../common/colormap.wgsl';
import csMetaballShader from './metaball.wgsl';
import csValueShader from './metaBall-value.wgsl';
import { EdgeTable, TriTable } from '../../common/marching-cubes-table'
import { getIdByColormapName } from '../../common/colormap-selection';
import * as ws from 'webgpu-simplified';
import { mat4 } from 'gl-matrix';

let maxMetaballs = 200;
let resolution = 96;
let marchingCubeCells = (resolution - 1) * (resolution - 1) * (resolution - 1);
let vertexCount = 3 * 12 * marchingCubeCells;
let vertexBufferSize = Float32Array.BYTES_PER_ELEMENT * vertexCount;
let indexCount = 15 * marchingCubeCells;
let indexBufferSize = Uint32Array.BYTES_PER_ELEMENT * indexCount;
let indirectArray:Uint32Array;

interface IMetaballPos {
    x: number
    y: number
    z: number
    vx: number
    vy: number
    vz: number
    speed: number
}
var ballPositions:IMetaballPos[] = [];
var metaballArray:Float32Array;
let strength = 1;
let strengthTarget = 1;
let subtract = 1;
let subtractTarget = 1;

const createPipeline = async (init:ws.IWebGPUInit): Promise<ws.IPipeline> => {
    const descriptor = ws.createRenderPipelineDescriptor({
        init, vsShader, fsShader,
        buffers: ws.setVertexBuffers(['float32x3', 'float32x3', 'float32x3']),//pos, norm, col 
    })
    const pipeline = await init.device.createRenderPipelineAsync(descriptor);

    // uniform buffer for transform matrix
    const  vertUniformBuffer = ws.createBuffer(init.device, 192);

    // uniform buffer for light 
    const lightUniformBuffer = ws.createBuffer(init.device, 48);

    // uniform buffer for material
    const materialUniformBuffer = ws.createBuffer(init.device, 16);
    
    // uniform bind group for vertex shader
    const vertBindGroup = ws.createBindGroup(init.device, pipeline.getBindGroupLayout(0), [vertUniformBuffer]);
    
    // uniform bind group for fragment shader
    const fragBindGroup = ws.createBindGroup(init.device, pipeline.getBindGroupLayout(1), 
        [lightUniformBuffer, materialUniformBuffer]);

    // create depth view
    const depthTexture = ws.createDepthTexture(init);

    // create texture view for MASS (count = 4)
    const msaaTexture = ws.createMultiSampleTexture(init);

    return {
        pipelines: [pipeline],
        uniformBuffers: [
            vertUniformBuffer,    // for vertex
            lightUniformBuffer,   // for fragmnet
            materialUniformBuffer      
        ],
        uniformBindGroups: [vertBindGroup, fragBindGroup],
        depthTextures: [depthTexture],
        gpuTextures: [msaaTexture],
    };
}

const createComputeValuePipeline = async (device: GPUDevice): Promise<ws.IPipeline> => {    
    const descriptor = ws.createComputePipelineDescriptor(device, csValueShader);
    const csPipeline = await device.createComputePipelineAsync(descriptor);

    const volumeElements = resolution * resolution * resolution;
    const valueBufferSize = Float32Array.BYTES_PER_ELEMENT * volumeElements;
    const valueBuffer = ws.createBuffer(device, valueBufferSize, ws.BufferType.Storage);

    const intParamsBufferSize = 
        1 * 4 + // resolution: u32
        1 * 4 + // metaballCount: u32
        2 * 4 + // padding
        0;      
    const intBuffer = ws.createBuffer(device, intParamsBufferSize);
    
    const singleBallBufferSize = 
        3 * 4 + // position: vec3<f32>
        1 * 4 + // radius f32
        1 * 4 + // strength: f32
        1 * 4 + // subtract: f32
        2 * 4 + // padding
        0;
    const ballBufferSize = singleBallBufferSize * maxMetaballs;
    metaballArray = new Float32Array(ballBufferSize / 4);
    const ballBuffer = ws.createBuffer(device, ballBufferSize, ws.BufferType.Storage);

    ballPositions = new Array(maxMetaballs).fill(null).map((_) => ({
        x: (Math.random() * 2 - 1) * (-4),
        y: (Math.random() * 2 - 1) * (-4),
        z: (Math.random() * 2 - 1) * (-4),
        vx: Math.random() * 1000,
        vy: (Math.random() * 2 - 1) * 10,
        vz: Math.random() * 1000,
        speed: Math.random() * 2 + 0.3,
    }));

   
    const csBindGroup = ws.createBindGroup(device, csPipeline.getBindGroupLayout(0), 
        [valueBuffer, intBuffer, ballBuffer]);
    
    return {
        csPipelines: [csPipeline],
        vertexBuffers: [valueBuffer],
        uniformBuffers: [intBuffer, ballBuffer],
        uniformBindGroups: [csBindGroup],        
    };
}

const createComputePipeline = async (device: GPUDevice, valueBuffer: GPUBuffer): Promise<ws.IPipeline> => {    
    const csShader = csColormapShader.concat(csMetaballShader);
    const descriptor = ws.createComputePipelineDescriptor(device, csShader);
    const csPipeline = await device.createComputePipelineAsync(descriptor);
    
    const tableBuffer = device.createBuffer({
        size: (EdgeTable.length + TriTable.length)*Int32Array.BYTES_PER_ELEMENT,
        usage: GPUBufferUsage.STORAGE,
        mappedAtCreation: true,
    });
    const tableArray = new Int32Array(tableBuffer.getMappedRange());
    tableArray.set(EdgeTable);
    tableArray.set(TriTable, EdgeTable.length);
    tableBuffer.unmap();

    const positionBuffer = ws.createBuffer(device, vertexBufferSize, ws.BufferType.VertexStorage);
    const normalBuffer = ws.createBuffer(device, vertexBufferSize, ws.BufferType.VertexStorage);
    const colorBuffer = ws.createBuffer(device, vertexBufferSize, ws.BufferType.VertexStorage);
    const indexBuffer = ws.createBuffer(device, indexBufferSize, ws.BufferType.IndexStorage);
   
    indirectArray = new Uint32Array(4);
    indirectArray[0] = 500;
    const indirectBuffer = ws.createBuffer(device, indirectArray.byteLength, ws.BufferType.IndirectStorage);
    
    const intParamsBufferSize = 
        1 * 4 + // resolution: u32
        1 * 4 + // colormapSelection: u32
        1 * 4 + // colormapDirection: u32
        1 * 4 + // colormapReverse: u32
        0;      
    const intBuffer = ws.createBuffer(device, intParamsBufferSize);
   
    const floatParamsBufferSize = 
        1 * 4 + // isolevel: f32
        1 * 4 + // scale: f32
        2 * 4 + // padding
        0;      
    const floatBuffer = ws.createBuffer(device, floatParamsBufferSize);

    const csBindGroup = ws.createBindGroup(device, csPipeline.getBindGroupLayout(0), [
        tableBuffer, valueBuffer, positionBuffer, normalBuffer, colorBuffer, indexBuffer, 
        indirectBuffer, intBuffer, floatBuffer
    ]);

    return {
        csPipelines: [csPipeline],
        vertexBuffers: [positionBuffer, normalBuffer, colorBuffer, indexBuffer],
        uniformBuffers: [intBuffer, floatBuffer, indirectBuffer],
        uniformBindGroups: [csBindGroup],        
    };
}

const draw = async (init:ws.IWebGPUInit, p:ws.IPipeline, p2:ws.IPipeline, p3: ws.IPipeline) => {  
    const commandEncoder =  init.device.createCommandEncoder();
    const wsize = 4; // set workgroup size

    // compute pass
    {
        const csPass = commandEncoder.beginComputePass();
        csPass.setPipeline(p2.csPipelines[0]);
        csPass.setBindGroup(0, p2.uniformBindGroups[0]);
        csPass.dispatchWorkgroups(Math.ceil(resolution / wsize), Math.ceil(resolution / wsize), 
            Math.ceil(resolution / wsize));

        csPass.setPipeline(p3.csPipelines[0]);
        csPass.setBindGroup(0, p3.uniformBindGroups[0]);
        csPass.dispatchWorkgroups(Math.ceil(resolution / wsize), Math.ceil(resolution / wsize), 
            Math.ceil(resolution / wsize));
        csPass.end();
    }
    
    // render pass
    {
        const descriptor = ws.createRenderPassDescriptor({
            init,
            depthView: p.depthTextures[0].createView(),
            textureView: p.gpuTextures[0].createView(),
        });
        const renderPass = commandEncoder.beginRenderPass(descriptor);

        // draw surface
        renderPass.setPipeline(p.pipelines[0]);
        renderPass.setBindGroup(0, p.uniformBindGroups[0]);
        renderPass.setBindGroup(1, p.uniformBindGroups[1]);
        renderPass.setVertexBuffer(0, p3.vertexBuffers[0]);
        renderPass.setVertexBuffer(1, p3.vertexBuffers[1]);     
        renderPass.setVertexBuffer(2, p3.vertexBuffers[2]);     
        renderPass.setIndexBuffer(p3.vertexBuffers[3], 'uint32');
        renderPass.drawIndexed(indexCount);

        renderPass.end();
    }

    init.device.queue.submit([commandEncoder.finish()]);
}

const run = async () => {
    const canvas = document.getElementById('canvas-webgpu') as HTMLCanvasElement;
    const deviceDescriptor: GPUDeviceDescriptor = {
        requiredLimits:{
            maxStorageBufferBindingSize: 1024*1024*1024, //1024MB, defaulting to 128MB
            maxBufferSize: 1024*1024*1024, // 1024MB, defaulting to 256MB
        }
    }
    const init = await ws.initWebGPU({canvas, msaaCount: 4}, deviceDescriptor);

    var gui = ws.getDatGui();
    const params = {
        animateSpeed: 1,
        resolution: 96,
        scale: 0.5,
        isolevel: 20,
        colormap: 'jet',
        colormapDirection: 'y',
        colormapReverse: false,
        specularColor: '#aaaaaa',
        ambient: 0.1,
        diffuse: 0.7,
        specular: 0.4,
        shininess: 30,
    };
    
    let colormapSelection = 0;
    let colormapDirection = 1;
    let colormapReverse = 0;
    let resolutionChanged = false;
   
    gui.add(params, 'animateSpeed', 0, 5, 0.1);     
      
    var folder = gui.addFolder('Set Surface Parameters');
    folder.open();
    folder.add(params, 'resolution', 8, 156, 4).onChange(() => {      
        resolutionChanged = true;
    });
    
    folder.add(params, 'scale', 0.1, 2, 0.1); 
    folder.add(params, 'isolevel', 0.1, 50, 0.1); 
    folder.add(params, 'colormap', [
        'autumn', 'black', 'blue', 'bone', 'cool', 'cooper', 'cyan', 'fuchsia', 'green', 'greys', 'hsv',
        'hot', 'jet', 'rainbow', 'rainbow_soft', 'red', 'spring', 'summer', 'white', 'winter', 'yellow'
    ]).onChange((val:string) => {
        colormapSelection = getIdByColormapName(val);
    }); 
    folder.add(params, 'colormapDirection', [
        'x', 'y', 'z', 'r'
    ]).onChange((val:string) => {              
        if(val === 'x') colormapDirection = 0;
        else if(val === 'z') colormapDirection = 2;
        else if(val === 'r') colormapDirection = 3;
        else colormapDirection = 1;
    }); 
    folder.add(params, 'colormapReverse').onChange((val:boolean) => {
        if(val) colormapReverse = 1;
        else colormapReverse = 0;
    });
    
    folder = gui.addFolder('Set Lighting Parameters');
    folder.open();
    folder.add(params, 'ambient', 0, 1, 0.02);  
    folder.add(params, 'diffuse', 0, 1, 0.02);  
    folder.addColor(params, 'specularColor');
    folder.add(params, 'specular', 0, 1, 0.02);  
    folder.add(params, 'shininess', 0, 300, 1);  

    const p = await createPipeline(init);
    let p2 = await createComputeValuePipeline(init.device);   
    let p3 = await createComputePipeline(init.device, p2.vertexBuffers[0]);
    
    let modelMat = mat4.create();
    let normalMat = mat4.create();
    init.device.queue.writeBuffer(p.uniformBuffers[0], 64, modelMat as ArrayBuffer);  
    init.device.queue.writeBuffer(p.uniformBuffers[0], 128, normalMat as ArrayBuffer);   

    let vt = ws.createViewTransform([2, 2, 3]);
    let viewMat = vt.viewMat;   
    let aspect = init.size.width / init.size.height;  
    let projectMat = ws.createProjectionMat(aspect);  
    let vpMat = ws.combineVpMat(viewMat, projectMat);

    var camera = ws.getCamera(canvas, vt.cameraOptions);
    let eyePosition = new Float32Array(vt.cameraOptions.eye);
    let lightDirection = new Float32Array([-0.5, -0.5, -0.5]);
    init.device.queue.writeBuffer(p.uniformBuffers[0], 0, vpMat as ArrayBuffer);

    // write light parameters to buffer 
    init.device.queue.writeBuffer(p.uniformBuffers[1], 0, lightDirection);
    init.device.queue.writeBuffer(p.uniformBuffers[1], 16, eyePosition);
   
    let start = performance.now();
    let stats = ws.getStats();

    const frame = async () => {     
        stats.begin();

        if(resolutionChanged){
            resolution = params.resolution;
            marchingCubeCells = (resolution - 1) * (resolution - 1) * (resolution - 1);
            vertexCount = 3 * 12 * marchingCubeCells;
            vertexBufferSize = Float32Array.BYTES_PER_ELEMENT * vertexCount;
            indexCount = 15 * marchingCubeCells;
            indexBufferSize = Uint32Array.BYTES_PER_ELEMENT * indexCount;
            p2 = await createComputeValuePipeline(init.device);   
            p3 = await createComputePipeline(init.device, p2.vertexBuffers[0]);
            resolutionChanged = false;
        }

        projectMat = ws.createProjectionMat(aspect); 
        if(camera.tick()){
            viewMat = camera.matrix;
            vpMat = ws.combineVpMat(viewMat, projectMat);
            eyePosition = new Float32Array(camera.eye.flat());
            init.device.queue.writeBuffer(p.uniformBuffers[0], 0, vpMat as ArrayBuffer);
            init.device.queue.writeBuffer(p.uniformBuffers[1], 16, eyePosition);
        }
        let time = performance.now();
        var dt = (time - start)/1000;   
        start = time;

        // update uniform buffers for specular light color
        init.device.queue.writeBuffer(p.uniformBuffers[1], 32, ws.hex2rgb(params.specularColor));
        
         // update uniform buffer for material
        init.device.queue.writeBuffer(p.uniformBuffers[2], 0, new Float32Array([
            params.ambient, params.diffuse, params.specular, params.shininess
        ]));
       
        // update compute value pipeline
        subtract += (subtractTarget - subtract) * dt * 4;
        strength += (strengthTarget - strength) * dt * 4;

        for (let i = 0; i < maxMetaballs; i++) {
            const pos = ballPositions[i]
      
            pos.vx += -pos.x * pos.speed * 20
            pos.vy += -pos.y * pos.speed * 20
            pos.vz += -pos.z * pos.speed * 20
      
            pos.x += pos.vx * pos.speed * dt * 0.0001
            pos.y += pos.vy * pos.speed * dt * 0.0001
            pos.z += pos.vz * pos.speed * dt * 0.0001

            const padding = 0.9
            const width = Math.abs(-4) - padding
            const height = Math.abs(-4) - padding
            const depth = Math.abs(-4) - padding
      
            if (pos.x > width) {
              pos.x = width
              pos.vx *= -1
            } else if (pos.x < -width) {
              pos.x = -width
              pos.vx *= -1
            }
      
            if (pos.y > height) {
              pos.y = height
              pos.vy *= -1
            } else if (pos.y < -height) {
              pos.y = -height
              pos.vy *= -1
            }
      
            if (pos.z > depth) {
              pos.z = depth
              pos.vz *= -1
            } else if (pos.z < -depth) {
              pos.z = -depth
              pos.vz *= -1
            }
        }

        for( let i = 0; i < maxMetaballs; i++){
            const position = ballPositions[i];
            const offset = i * 8;
            metaballArray[offset] = position.x;
            metaballArray[offset + 1] = position.y;
            metaballArray[offset + 2] = position.z;
            metaballArray[offset + 3] = Math.sqrt(strength/subtract); // radius
            metaballArray[offset + 4] = strength;
            metaballArray[offset + 5] = subtract;
        }

        init.device.queue.writeBuffer(p2.uniformBuffers[0], 0, new Uint32Array([
            resolution, maxMetaballs, 0, 0
        ]));

        init.device.queue.writeBuffer(p2.uniformBuffers[1], 0, metaballArray);

        // update compute pipeline
        init.device.queue.writeBuffer(p3.uniformBuffers[0], 0, new Uint32Array([
                resolution,
                colormapSelection,
                colormapDirection,
                colormapReverse,
            ])
        );

        init.device.queue.writeBuffer(p3.uniformBuffers[1], 0, new Float32Array([
                params.isolevel,
                params.scale,
                0, // padding
                0, 
            ])
        );
        init.device.queue.writeBuffer(p3.uniformBuffers[2], 0, indirectArray);

        draw(init, p, p2, p3);      

        requestAnimationFrame(frame);
        stats.end();
    };
    frame();

    setInterval(() => {
        subtractTarget = 3 + Math.random() * 3;
        strengthTarget = 3 + Math.random() * 3;
    }, 5000)
}

run();