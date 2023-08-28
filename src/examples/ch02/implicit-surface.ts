import vsShader from '../../common/shader-vert.wgsl';
import fsShader from '../../common/shader-frag.wgsl';
import csColormapShader from '../../common/colormap.wgsl';
import { getIdByColormapName } from '../../common/colormap-selection';
import csSurfaceShader from './implicit-surface.wgsl';
import csValueShader from './implicit-value.wgsl';
import csFuncShader from './implicit-func.wgsl';
import { EdgeTable, TriTable } from '../../common/marching-cubes-table'; 
import * as ws from 'webgpu-simplified';
import { vec3, mat4 } from 'gl-matrix';

let resolution = 96;
let marchingCubeCells = (resolution - 1) * (resolution - 1) * (resolution - 1);
let vertexCount = 3 * 12 * marchingCubeCells;
let vertexBufferSize = Float32Array.BYTES_PER_ELEMENT * vertexCount;
let indexCount = 15 * marchingCubeCells;
let indexBufferSize = Uint32Array.BYTES_PER_ELEMENT * indexCount;
let indirectArray:Uint32Array;

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
    const valueShader = csFuncShader.concat(csValueShader);
    const descriptor = ws.createComputePipelineDescriptor(device, valueShader);
    const csPipeline = await device.createComputePipelineAsync(descriptor);

    const volumeElements = resolution * resolution * resolution;
    const valueBufferSize = Float32Array.BYTES_PER_ELEMENT * volumeElements;
    const valueBuffer = ws.createBuffer(device, valueBufferSize, ws.BufferType.Storage);

    const intParamsBufferSize = 
        1 * 4 + // resolution: u32
        1 * 4 + // funcSelection: u32
        2 * 4 + // padding
        0;      
    const intBuffer = ws.createBuffer(device, intParamsBufferSize);
    
    const floatParamsBufferSize = 
        1 * 4 + // animateTime: f32
        3 * 4 + // padding
        0;      
    const floatBuffer = ws.createBuffer(device, floatParamsBufferSize);
   
    const csBindGroup = ws.createBindGroup(device, csPipeline.getBindGroupLayout(0), 
        [valueBuffer, intBuffer, floatBuffer]);
    
    return {
        csPipelines: [csPipeline],
        vertexBuffers: [valueBuffer],
        uniformBuffers: [intBuffer, floatBuffer],
        uniformBindGroups: [csBindGroup],        
    };
}

const createComputePipeline = async (device: GPUDevice, valueBuffer: GPUBuffer): Promise<ws.IPipeline> => {    
    const csShader = csFuncShader.concat(csColormapShader.concat(csSurfaceShader));
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
        1 * 4 + // funcSelection: u32
        1 * 4 + // colormapSelection: u32
        1 * 4 + // colormapDirection: u32
        1 * 4 + // colormapReverse: u32
        3 * 4 + // padding
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
    const wsize = 8; // set workgroup size

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
            maxComputeInvocationsPerWorkgroup: 512 // defaulting to 256
        }
    }
    const init = await ws.initWebGPU({canvas, msaaCount: 4}, deviceDescriptor);

    var gui = ws.getDatGui();
    const params = {
        rotationSpeed: 1,
        animateSpeed: 1,
        surfaceType: 'blobs',
        resolution: 96,
        scale: 1,
        isolevel: 0,
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
    let surfaceType = 2;
    let colormapDirection = 1;
    let colormapReverse = 0;
    let scale = 1.1;
    let resolutionChanged = false;
   
    gui.add(params, 'surfaceType', [
        'sphere', 'schwartzSurface', 'blobs', 'klein', 'torus', 'chmutov', 'gyroid', 'cubeSphere', 
        'orthoCircle', 'gabrielHorn', 'spiderCage', 'barthSextic', 'laplaceGaussian'
    ]).onChange((val:string) => {
        if(val === 'sphere') { surfaceType = 0; scale = 1.4*params.scale; }
        else if(val === 'schwartzSurface') { surfaceType = 1; scale = 0.4*params.scale; }
        else if(val === 'blobs') { surfaceType = 2; scale = 1.1*params.scale; }
        else if(val === 'klein') { surfaceType = 3; scale = 0.5*params.scale; }
        else if(val === 'torus') { surfaceType = 4; scale = 0.6*params.scale; }
        else if(val === 'chmutov') { surfaceType = 5; scale = 1.3*params.scale; }
        else if(val === 'gyroid') { surfaceType = 6; scale = 0.3*params.scale; }
        else if(val === 'cubeSphere') { surfaceType = 7; scale = 0.7*params.scale; }
        else if(val === 'orthoCircle') { surfaceType = 8; scale = 1.7*params.scale; }
        else if(val === 'gabrielHorn') { surfaceType = 9; scale = 0.4*params.scale; }
        else if(val === 'spiderCage') { surfaceType = 10; scale = 0.5*params.scale; }
        else if(val === 'barthSextic') { surfaceType = 11; scale = 1.7*params.scale; }
        else if(val === 'laplaceGaussian') { surfaceType = 12; scale = 0.05*params.scale; }
    });
    gui.add(params, 'animateSpeed', 0, 5, 0.1);     
    gui.add(params, 'rotationSpeed', 0, 5, 0.1);
   
    var folder = gui.addFolder('Set Surface Parameters');
    folder.open();
    folder.add(params, 'resolution', 8, 152, 8).onChange(() => {      
        resolutionChanged = true;
    });
    
    folder.add(params, 'scale', 0.1, 5, 0.1).onChange(() => {
        scale = params.scale;
        if(params.surfaceType === 'schwartzSurface') scale = 0.4* params.scale;
        else if(params.surfaceType === 'blobs') scale = 1.1 *params.scale;
        else if(params.surfaceType === 'sphere') scale = 1.4 *params.scale;
        else if(params.surfaceType === 'klein') scale = 0.5 *params.scale;
        else if(params.surfaceType === 'torus') scale = 0.6 *params.scale;
        else if(params.surfaceType === 'chmutov') scale = 1.3 *params.scale;
        else if(params.surfaceType === 'gyroid') scale = 0.3 *params.scale;
        else if(params.surfaceType === 'cubeSphere') scale = 0.7 *params.scale;
        else if(params.surfaceType === 'orthoCircle') scale = 1.7 *params.scale;
        else if(params.surfaceType === 'gabrielHorn') scale = 0.4 *params.scale;
        else if(params.surfaceType === 'spiderCage') scale = 0.5 *params.scale;
        else if(params.surfaceType === 'barthSextic') scale = 1.7 *params.scale;
        else if(params.surfaceType === 'laplaceGaussian') scale = 0.05 *params.scale;
    }); 
    folder.add(params, 'isolevel', 0, 2, 0.01); 
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
    let vt = ws.createViewTransform([2, 2, 3]);
    let viewMat = vt.viewMat;
   
    let aspect = init.size.width / init.size.height;  
    let rotation = vec3.fromValues(0, 0, 0);    
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
        var dt = (performance.now() - start)/1000;   
        rotation[0] = Math.sin(dt * params.rotationSpeed);
        rotation[1] = Math.cos(dt * params.rotationSpeed); 
        modelMat = ws.createModelMat([0,0,0], rotation);
        normalMat = ws.createNormalMat(modelMat);
        

        // update uniform buffers for transformation 
        init.device.queue.writeBuffer(p.uniformBuffers[0], 64, modelMat as ArrayBuffer);  
        init.device.queue.writeBuffer(p.uniformBuffers[0], 128, normalMat as ArrayBuffer);  
      
        // update uniform buffers for specular light color
        init.device.queue.writeBuffer(p.uniformBuffers[1], 32, ws.hex2rgb(params.specularColor));
        
         // update uniform buffer for material
        init.device.queue.writeBuffer(p.uniformBuffers[2], 0, new Float32Array([
            params.ambient, params.diffuse, params.specular, params.shininess
        ]));
       
        // update compute value pipeline
        init.device.queue.writeBuffer(p2.uniformBuffers[0], 0, new Uint32Array([
            resolution, surfaceType, 0, 0
        ]));

        init.device.queue.writeBuffer(p2.uniformBuffers[1], 0, new Float32Array([
            params.animateSpeed * dt, 0, 0, 0
        ]));

        // update compute pipeline
        init.device.queue.writeBuffer(p3.uniformBuffers[0], 0, new Uint32Array([
                resolution,
                surfaceType, 
                colormapSelection,
                colormapDirection,
                colormapReverse,
                0, // padding
                0, 
                0, 
            ])
        );

        init.device.queue.writeBuffer(p3.uniformBuffers[1], 0, new Float32Array([
                params.isolevel,
                scale,
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
}

run();