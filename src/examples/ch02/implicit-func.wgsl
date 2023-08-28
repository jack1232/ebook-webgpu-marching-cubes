const pi:f32 = 3.14159265359;

struct DataRange {
    xRange: vec2f,
    yRange: vec2f,
    zRange: vec2f,
}

fn sphere(x:f32, y:f32, z:f32, t:f32) -> f32{
    let a = 1.1 + cos(t);   
	let v = x*x + y*y + z*z - a;
	return v;
}

fn schwartzSurface(x:f32, y:f32, z:f32, t:f32) -> f32{
    let a = 1 + 0.5 * cos(t);   
    let v = cos(x*a) + cos(y*a) + cos(z*a);
    return v;
}

fn blobs(x:f32, y:f32, z:f32, t:f32) -> f32{
    let a = 1.01 + cos(2.0*t);   
    let v = x*x + y*y + z*z + cos(4.0*x) + cos(4.0*y) + cos(4.0*z) - a;
    return v;
}

fn klein(x:f32, y:f32, z:f32, t:f32) -> f32{
    let a = 1.01 + cos(2.0*t);   
    let v = (x*x+y*y+z*z + 2.0*y-a)*((x*x+y*y+z*z - 2.0*y-a)*(x*x+y*y+z*z- 2.0*y-a) 
        - 8.0*z*z)+16.0*x*z*(x*x+y*y+z*z - 2.0*y- a);
    return v;
}

fn torus(x:f32, y:f32, z:f32, t:f32) -> f32{
    let a = 1.5*(1.01 + cos(t));
    let b = 0.5*(1.1 + cos(t));
    let v = (sqrt(x*x+y*y)-a)*(sqrt(x*x+y*y)-a) + z*z - b;
    return v;
}

fn chmutov(x:f32, y:f32, z:f32, t:f32) -> f32{
    let a = 0.35*(1.0 + cos(t));   
    let v = pow(x,4.0) + pow(y,4.0) + pow(z,4.0) - (x*x + y*y + z*z - a);
    return v;
}

fn gyroid(x:f32, y:f32, z:f32, t:f32) -> f32{
    let a = 1.0 + 0.5 * cos(t);   
    let v = cos(x*a) * sin(y*a) + cos(y*a) * sin(z*a) + cos(z*a) * sin(x*a);
    return v;
}

fn cubeSphere(x:f32, y:f32, z:f32, t:f32) -> f32{
    let a = 1.2 + cos(t);   
    let v = pow((1.0/2.3)*(1.0/2.3)*(x*x + y*y + z*z),-6.0) + 
        pow(pow(1.0/2.0,8.0)* (pow(x,8.0) + pow(y,8.0) + pow(z,8.0)),6.0) - a;
    return v;
}

fn orthoCircle(x:f32, y:f32, z:f32, t:f32) -> f32{
    let a = 0.5 * (2.2 + cos(t));   
    let v = ((x*x + y*y - a)*(x*x + y*y - a) + z*z)*((y*y + z*z - a)*(y*y + z*z - a) + 
            x*x)*((z*z + x*x - a)*(z*z + x*x - a) + y*y) - 0.075*0.075 *(1.0 + 3.0*(x*x + y*y + z*z));
    return v;
}

fn gabrielHorn(x:f32, y:f32, z:f32, t:f32) -> f32{
    let a = 1.0 + cos(t);
    let v = a*y*y + a*z*z - (1.0/x)*(1.0/x);
    return v;
}

fn spiderCage(x:f32, y:f32, z:f32, t:f32) -> f32 {
	let a = 0.5 * (0.9 + 0.9 * cos(t));
	let x2 = x*x;
	let z2 = z*z;
	let xz2 = x2 + z2;

	let v =  pow(sqrt(pow((x2 - z2), 2.0) / xz2 + 3.0 * pow(y * sin(a), 2.0)) - 3.0, 2.0) + 
	    6.0 * pow(sqrt(pow(x * z, 2.0) / xz2 + pow(y * cos(a), 2.0)) - 1.5, 2.0) - 0.2;
	return v;
}

fn barthSextic(x:f32, y:f32, z:f32, t:f32) -> f32 {
	let a = 0.5 * (2.2 + 1.036*cos(t));
	let a2 = a*a;	
	let x2 = x*x;
	let y2 = y*y;
	let z2 = z*z;
	var v = 3.0 * (a2 * x2 - z2) * (a2 * z2 - y2) * (a2 * y2 - x2)
        - (1.0 + 2.0 * a) * pow(x2 + y2 + z2 - 1.0, 2.0);
	if(x2 + y2 + z2 > 3.1) { v = 0.0; }
	return v - 0.1;
}

fn laplaceGaussian(x:f32, y:f32, z:f32, t:f32) -> f32 {
	let a = 1.4;
	let b = 0.06;
	let c = 1.0/(pi*pow(a, 4.0));
	let temp = (x*x + z*z)/(2.0 * a * a);
	let r = sqrt(x*x + z*z);
	var v:f32;
	if(r > 8.0) {
		v = y + 28.0 * c * ( 1.0 - temp) * exp(-b*temp);
	} else {
		v = y - 4.5*(2.3 + 0.7 * sin(0.0258*(x*x + z*z)));
	}
	return v;
}


fn implicitFunc(x:f32, y:f32, z:f32, t:f32, funcSelection:u32) -> f32{
	var v = 0.0;
	if (funcSelection == 0u) { 
		v = sphere(x, y, z, t);
	} else if (funcSelection == 1u) { 
		v = schwartzSurface(x, y, z, t);
	} else if (funcSelection == 2u) { 
		v = blobs(x, y, z, t);
	} else if (funcSelection == 3u) { 
		v = klein(x, y, z, t);
	} else if (funcSelection == 4u) { 
		v = torus(x, y, z, t);
	} else if (funcSelection == 5u) { 
		v = chmutov(x, y, z, t);
	} else if (funcSelection == 6u) { 
		v = gyroid(x, y, z, t);
	} else if (funcSelection == 7u) { 
		v = cubeSphere(x, y, z, t);
	} else if (funcSelection == 8u) { 
		v = orthoCircle(x, y, z, t);
	} else if (funcSelection == 9u) { 
		v = gabrielHorn(x, y, z, t);
	} else if (funcSelection == 10u) { 
		v = spiderCage(x, y, z, t);
	} else if (funcSelection == 11u) { 
		v = barthSextic(x, y, z, t);
	} else if (funcSelection == 12u) { 
		v = laplaceGaussian(x, y, z, t);
	}	

	return v;
}

fn getDataRange(funcSelection:u32) -> DataRange{
	var dr:DataRange;
	if (funcSelection == 0u) { // sphere
		dr.xRange = vec2(-2.1, 2.1);
		dr.yRange = vec2(-2.1, 2.1);
		dr.zRange = vec2(-2.1, 2.1);
	} else if (funcSelection == 1u) { // schwartzSurface
		dr.xRange = vec2(-4.0, 4.0);
		dr.yRange = vec2(-4.0, 4.0);
		dr.zRange = vec2(-4.0, 4.0);
	} else if (funcSelection == 2u) { // blobs
		dr.xRange = vec2(-2.0, 2.0);
		dr.yRange = vec2(-2.0, 2.0);
		dr.zRange = vec2(-2.0, 2.0);
	} else if (funcSelection == 3u) { // klein
		dr.xRange = vec2(-3.5, 3.5);
		dr.yRange = vec2(-3.5, 3.5);
		dr.zRange = vec2(-4.5, 4.5);
	} else if (funcSelection == 4u) { // torus
		dr.xRange = vec2(-4.0, 4.0);
		dr.yRange = vec2(-4.0, 4.0);
		dr.zRange = vec2(-1.2, 1.2);
	} else if (funcSelection == 5u) { // chmutov
		dr.xRange = vec2(-1.5, 1.5);
		dr.yRange = vec2(-1.5, 1.5);
		dr.zRange = vec2(-1.5, 1.5);
	} else if (funcSelection == 6u) { // gyroid
	    dr.xRange = vec2(-4.0, 4.0);
		dr.yRange = vec2(-4.0, 4.0);
		dr.zRange = vec2(-4.0, 4.0);
	} else if (funcSelection == 7u) { // cubeSphere
		dr.xRange = vec2(-2.0, 2.0);
		dr.yRange = vec2(-2.0, 2.0);
		dr.zRange = vec2(-2.0, 2.0);
	} else if (funcSelection == 8u) { // orthoCircle
		dr.xRange = vec2(-1.5, 1.5);
		dr.yRange = vec2(-1.5, 1.5);
		dr.zRange = vec2(-1.5, 1.5);
	} else if (funcSelection == 9u) { // gabrielHorn
		dr.xRange = vec2( 0.0, 5.5);
		dr.yRange = vec2(-1.5, 1.5);
		dr.zRange = vec2(-1.5, 1.5);
	} else if (funcSelection == 10u) { // spiderCage
		dr.xRange = vec2(-5.0, 5.0);
		dr.yRange = vec2(-3.0, 3.0);
		dr.zRange = vec2(-5.0, 5.0);
	} else if (funcSelection == 11u) { // barthSextic
		dr.xRange = vec2(-2.0, 2.0);
		dr.yRange = vec2(-2.0, 2.0);
		dr.zRange = vec2(-2.0, 2.0);
	} else if (funcSelection == 12u) { // laplaceGaussian
		dr.xRange = vec2(-40.0, 40.0);
		dr.yRange = vec2(-40.0, 40.0);
		dr.zRange = vec2(-40.0, 40.0);
	}

	return dr;
}