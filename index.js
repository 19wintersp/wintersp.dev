let log = () => {};
if (location.hash === "#debug") {
	const logs = document.createElement("pre");
	logs.style.color = "#f77";
	document.querySelector("main").appendChild(logs);
	log = (data) => logs.innerText += data + "\n";
}

try {
	const canvas = document.getElementById("canvas");

	const gl = canvas.getContext("webgl2");

	const updateCanvas = () => {
		canvas.width = Math.ceil(canvas.clientWidth);
		canvas.height = Math.ceil(canvas.clientHeight);
		gl.viewport(0, 0, canvas.width, canvas.height);
	};

	addEventListener("resize", updateCanvas);
	updateCanvas();

	let targetCursorPos = [0, 0];
	const updateCursor = (event) => {
		targetCursorPos = [event.clientX, canvas.height - event.clientY];
	};

	addEventListener("mousemove", updateCursor);
	addEventListener("mouseenter", updateCursor);
	addEventListener("mouseover", updateCursor);

	const P = [];
	const p = Array(256).fill(0).map((x, y) => x + y);
	while (p.length > 0)
		P.push(p.splice(Math.floor(p.length * Math.random()), 1)[0]);

	const perm = new Uint32Array(512), perm12 = new Uint32Array(256);
	for (let i = 0; i < 256; i++) {
		perm[i] = perm[i | 256] = P[i];
		perm12[i] = P[i] % 12;
	}

	const grad = new Float32Array([
		 1,  1,  0,
		-1,  1,  0,
		 1, -1,  0,
		-1, -1,  0,
		 1,  0,  1,
		-1,  0,  1,
		 1,  0, -1,
		-1,  0, -1,
		 0,  1,  1,
		 0, -1,  1,
		 0,  1, -1,
		 0, -1, -1,
	]);

	const vtxSrc = `
		#version 300 es

		uniform float cell;
		uniform float time;
		uniform vec2 size;

		in vec2 pos;
		out vec2 vxy;
		out vec3 tex;

		void main() {
			gl_Position = vec4(pos.xy, 0.0, 1.0);
			vxy = pos;
			tex = vec3((0.5 * (1.0 + pos) * size) * cell, time / 8.0);
		}
	`.trim();

	const fragSrc = `
		#version 300 es

		#define SKEW_F (1.0 / 3.0)
		#define SKEW_G (1.0 / 6.0)

		precision highp float;

		uniform float cell;
		uniform float time;
		uniform uint perm[512];
		uniform uint perm12[256];
		uniform vec3 grad[12];
		uniform vec2 size;
		uniform vec2 cursor;

		in vec2 vxy;
		in vec3 tex;
		out vec4 fragColor;

		float simplex(vec3 v_in) {
			float s = (v_in.x + v_in.y + v_in.z) * SKEW_F;
			vec3 v_fl = floor(v_in + s);
			float t = (v_fl.x + v_fl.y + v_fl.z) * SKEW_G;
			vec3 v_0 = v_in + t - v_fl;

			uvec3 v_1i, v_2i;
			if (v_0.x >= v_0.y) {
				if (v_0.y >= v_0.z) {
					v_1i = uvec3(1, 0, 0);
					v_2i = uvec3(1, 1, 0);
				} else if (v_0.x >= v_0.z) {
					v_1i = uvec3(1, 0, 0);
					v_2i = uvec3(1, 0, 1);
				} else {
					v_1i = uvec3(0, 0, 1);
					v_2i = uvec3(1, 0, 1);
				}
			} else {
				if (v_0.y < v_0.z) {
					v_1i = uvec3(0, 0, 1);
					v_2i = uvec3(0, 1, 1);
				} else if (v_0.x < v_0.z) {
					v_1i = uvec3(0, 1, 0);
					v_2i = uvec3(0, 1, 1);
				} else {
					v_1i = uvec3(0, 1, 0);
					v_2i = uvec3(1, 1, 0);
				}
			}

			uvec3 v_ii = uvec3(ivec3(v_fl) & 255);
			uint g[4] = uint[4](
				perm12[(v_ii.x + perm[v_ii.y + perm[v_ii.z]]) & 255u],
				perm12[(v_ii.x + perm[v_ii.y + perm[v_ii.z + v_1i.z] + v_1i.y] + v_1i.x) & 255u],
				perm12[(v_ii.x + perm[v_ii.y + perm[v_ii.z + v_2i.z] + v_2i.y] + v_2i.x) & 255u],
				perm12[(v_ii.x + perm[v_ii.y + perm[v_ii.z + 1u]     + 1u]     + 1u) & 255u]
			);

			vec3 v_n[4]; // this cannot be directly initialised due to a browser bug
			v_n[0] = v_0;
			v_n[1] = v_0 - vec3(v_1i) + SKEW_G;
			v_n[2] = v_0 - vec3(v_2i) + SKEW_G * 2.0;
			v_n[3] = v_0 - 1.0        + SKEW_G * 3.0;

			float n = 0.0;
			for (int i = 0; i < 4; i++) {
				vec3 v_ni = v_n[i];
				float t = 0.6 - v_ni.x * v_ni.x - v_ni.y * v_ni.y - v_ni.z * v_ni.z;
				if (t > 0.0) {
					t *= t;
					n += t * t * dot(grad[g[i]], v_n[i]);
				}
			}

			return 32.0 * n;
		}

		int simplex_quant(vec3 v_in) {
			float x = 0.0;
			x += simplex(v_in * 0.5) * 3.0;
			x += simplex(v_in * 2.0) * 0.5;

			float cursorDist = length(v_in.xy / cell - cursor);
			x += 1.0 + cos(min(cursorDist * 0.01, 3.14159));

			return int(floor(x));
		}

		float kernel(vec3 c, float d) {
			int q1 = simplex_quant(c);
			int q2 = simplex_quant(c + vec3(  -d,  0.0, 0.0));
			int q3 = simplex_quant(c + vec3(  -d,    d, 0.0));
			int q4 = simplex_quant(c + vec3( 0.0,    d, 0.0));

			if (q1 == q2 && q1 == q3 && q1 == q4) return 0.0;

			int m = abs(min(min(q1, q2), min(q3, q4)));

			return ((m % 5) == 0) ? 1.0 : 0.5;
		}

		void main() {
			float w = 0.15 * (
				kernel(tex + vec3(-.25, -.25, 0.0) * cell, cell) +
				kernel(tex + vec3( .25, -.25, 0.0) * cell, cell) +
				kernel(tex + vec3(-.25,  .25, 0.0) * cell, cell) +
				kernel(tex + vec3( .25,  .25, 0.0) * cell, cell)
			);

			vec2 screen = tex.xy / cell;

			// fade-in
			w *= sin(min(time / 2.0, 1.0) * 1.5708);

			// secondary scaling
			w *= .8 + .2 * (
				simplex(vec3(screen / 512.0, tex.z)) +
				simplex(vec3(screen / 2048.0, 2.0 * tex.z))
			);

			fragColor = vec4(w, w, w, 1.0);
		}
	`.trim();

	const vtxShader = gl.createShader(gl.VERTEX_SHADER);
	gl.shaderSource(vtxShader, vtxSrc);
	gl.compileShader(vtxShader);

	log(gl.getShaderInfoLog(vtxShader));

	const fragShader = gl.createShader(gl.FRAGMENT_SHADER);
	gl.shaderSource(fragShader, fragSrc);
	gl.compileShader(fragShader);

	log(gl.getShaderInfoLog(fragShader));

	const prog = gl.createProgram();
	gl.attachShader(prog, vtxShader);
	gl.attachShader(prog, fragShader);
	gl.linkProgram(prog);

	log(gl.getProgramInfoLog(prog));

	const posBuf = gl.createBuffer();
	gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
	gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
		-1, -1,
		-1,  1,
		 1, -1,
		 1,  1,
	]), gl.STATIC_DRAW);

	const cellLoc = gl.getUniformLocation(prog, "cell");
	const sizeLoc = gl.getUniformLocation(prog, "size");
	const timeLoc = gl.getUniformLocation(prog, "time");
	const cursorLoc = gl.getUniformLocation(prog, "cursor");

	const posLoc = gl.getAttribLocation(prog, "pos");
	const permLoc = gl.getUniformLocation(prog, "perm");
	const perm12Loc = gl.getUniformLocation(prog, "perm12");
	const gradLoc = gl.getUniformLocation(prog, "grad");

	gl.useProgram(prog);

	gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
	gl.enableVertexAttribArray(posLoc);
	gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

	gl.uniform1f(cellLoc, 1/64);
	gl.uniform1uiv(permLoc, perm);
	gl.uniform1uiv(perm12Loc, perm12);
	gl.uniform3fv(gradLoc, grad);

	const tRef = performance.now();
	let cursorPos;

	const render = (pt) => {
		if (cursorPos) {
			const dt = Math.max(performance.now() - pt, 1);
			const dx = targetCursorPos[0] - cursorPos[0];
			const dy = targetCursorPos[1] - cursorPos[1];
			cursorPos[0] += dx * dt / 20;
			cursorPos[1] += dy * dt / 20;
		} else {
			cursorPos = targetCursorPos;
		}

		gl.useProgram(prog);

		gl.uniform2f(sizeLoc, canvas.clientWidth, canvas.clientHeight);
		gl.uniform1f(timeLoc, (performance.now() - tRef) / 1000);
		gl.uniform2f(cursorLoc, ...cursorPos);

		gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

		requestAnimationFrame(render);
	};

	render();
} catch (err) {
	log(err);
	console.error(err);

	canvas.remove();

	const fallback = document.getElementById("fallback");
	fallback.style.visibility = "visible";
}
